/**
 * Typed client for the support agent API.
 *
 * Requests go through Next's rewrite proxy (`/api/backend/*`) so the browser
 * stays same-origin: no CORS preflight per turn, and no cross-origin buffering
 * of the SSE stream.
 *
 * Note: the backend emits a purpose-built SSE contract (start / node / message /
 * awaiting_human / done), not the Vercel AI SDK data-stream protocol, so this
 * client reads it directly rather than via `useChat`. See docs/DEVIATIONS.md.
 */

export type Channel = "voice" | "sms" | "whatsapp" | "webchat" | "email";
export type CustomerTier = "free" | "standard" | "enterprise";

export interface Citation {
  id: string;
  title: string;
  source: string;
  relevance: number;
}

export interface ChatResponse {
  reply: string | string[];
  conversation_id: string;
  ticket_id: string | null;
  intent: string;
  intent_confidence: number;
  sentiment_score: number;
  resolution_status: string;
  resolution_path: string;
  escalated: boolean;
  escalation_reason: string | null;
  citations: Citation[];
  awaiting_human: boolean;
  stubbed: boolean;
}

export interface ChatRequest {
  message: string;
  customer_id?: string;
  channel?: Channel;
  conversation_id?: string | null;
  customer_tier?: CustomerTier;
}

export interface HealthReport {
  status: "ok" | "degraded";
  environment: string;
  subsystems: Record<string, Record<string, unknown>>;
  policy: Record<string, unknown>;
  degradations: string[];
  /**
   * Live probe result per backing store. A subsystem missing from this map was
   * never configured; `false` means it was configured, answered at startup,
   * and has since stopped — which is the more urgent of the two.
   */
  reachable?: Record<string, boolean>;
}

export interface SlaStatus {
  priority: string;
  target_minutes: number;
  escalated_at: string;
  due_at: string;
  remaining_seconds: number;
  overdue_seconds: number;
  state: "on_track" | "due_soon" | "breached";
  label: string;
}

export interface SlaSummary {
  total: number;
  breached: number;
  due_soon: number;
  on_track: number;
  next_due_seconds: number | null;
}

export interface TicketSummary {
  ticket_id: string;
  customer_id: string;
  channel: string;
  subject: string;
  summary: string;
  status: string;
  priority: string;
  intent: string;
  sentiment_score: number;
  escalated: boolean;
  escalation_reason: string | null;
  citations: Citation[];
  updated_at: number;
  created_at: number;
  // Every channel the conversation has used, in order first seen.
  channels: string[];
  assigned_to: string | null;
  sla: SlaStatus | null;
  age_seconds: number;
  waiting_seconds: number | null;
  created_at_iso: string | null;
  escalated_at_iso: string | null;
  first_responded_at_iso: string | null;
}

export interface HandoffPacket {
  conversation_id?: string;
  ticket_id?: string | null;
  executive_summary?: string;
  escalation_reason?: string;
  churn_risk?: string;
  customer_tier?: string;
  sentiment_trajectory?: number[];
  actions_attempted?: string[];
  recommended_actions?: string[];
  deep_agent_todos?: string[];
  rag_citations?: Citation[];
  transcript?: { role: string; text: string }[];
}

export interface MetricsSummary {
  window_seconds: number | null;
  totals: { turns: number; conversations: number; tickets: number };
  fcr: {
    rate: number | null;
    resolved_first_contact: number;
    conversations: number;
    target: number;
    definition: string;
  };
  escalation: {
    rate: number | null;
    escalated_conversations: number;
    by_reason: Record<string, number>;
  };
  csat: {
    score: number | null;
    mean_rating?: number;
    responses: number;
    response_rate: number;
    note: string;
  };
  sentiment: { mean?: number; negative_share?: number; positive_share?: number };
  latency: {
    turn_total: { p50?: number; p95?: number; max?: number };
    by_node: Record<string, { p50: number; p95: number; max: number; samples: number }>;
  };
  distribution: {
    by_intent: Record<string, number>;
    by_channel: Record<string, number>;
  };
  data_quality: { stubbed_share: number; stubbed_turns: number; warning: string | null };
}

export type MetricsWindow = "1h" | "24h" | "7d" | "all";

export function fetchMetrics(window: MetricsWindow = "24h"): Promise<MetricsSummary> {
  return request<MetricsSummary>(`${BASE}/analytics/summary?window=${window}`);
}

export interface QueueEntry extends TicketSummary {
  handoff_packet: HandoffPacket | null;
}

const BASE = "/api/backend";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * How long any single request may take before we give up on it.
 *
 * fetch() has no default timeout: a request to a stalled backend stays pending
 * until the browser or OS gives up, which can be minutes. That is exactly what
 * happened here -- the tickets endpoint took 260s to answer while the dashboard
 * polled it every 10s, so requests stacked up faster than they drained.
 */
const REQUEST_TIMEOUT_MS = 15_000;

/** A request that exceeded its deadline, as distinct from one that failed. */
export class TimeoutError extends Error {
  constructor() {
    super("The service did not respond in time.");
    this.name = "TimeoutError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = REQUEST_TIMEOUT_MS,
): Promise<T> {
  const timer = new AbortController();
  const expiry = setTimeout(() => timer.abort(), timeoutMs);
  // Honour a caller's own signal alongside the deadline, so an explicit
  // cancellation (navigating away, pressing Stop) is not swallowed.
  const signal = init?.signal
    ? AbortSignal.any([init.signal, timer.signal])
    : timer.signal;

  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      signal,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch (error) {
    if (timer.signal.aborted) throw new TimeoutError();
    throw error;
  } finally {
    clearTimeout(expiry);
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Non-JSON error body; keep the status-based message.
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export function sendMessage(payload: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>(`${BASE}/chat`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function resumeConversation(
  conversationId: string,
  resolution: string,
): Promise<ChatResponse> {
  return request<ChatResponse>(`${BASE}/chat/resume`, {
    method: "POST",
    body: JSON.stringify({ conversation_id: conversationId, resolution }),
  });
}

export function fetchHealth(): Promise<HealthReport> {
  return request<HealthReport>("/api/health");
}

export function fetchQueue(): Promise<{
  count: number;
  queue: QueueEntry[];
  sla_summary: SlaSummary;
}> {
  return request(`${BASE}/tickets/queue`);
}

/** Claim a ticket, which also stops the first-response clock. */
export function assignTicket(ticketId: string, agent: string): Promise<TicketSummary> {
  return request<TicketSummary>(`${BASE}/tickets/${ticketId}/assign`, {
    method: "POST",
    body: JSON.stringify({ agent }),
  });
}

export function fetchTickets(
  status?: string,
): Promise<{ count: number; tickets: TicketSummary[] }> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  return request(`${BASE}/tickets${query}`);
}

/** One ticket with its full transcript and handoff packet. */
export interface TicketDetail extends TicketSummary {
  transcript: { role: string; text: string }[];
  handoff_packet: HandoffPacket | null;
}

export function fetchTicket(ticketId: string): Promise<TicketDetail> {
  return request<TicketDetail>(`${BASE}/tickets/${encodeURIComponent(ticketId)}`);
}

/**
 * Record a customer satisfaction rating.
 *
 * `durable` is false when the rating was kept in memory but could not be
 * written to the database -- worth surfacing rather than reporting success.
 */
export function submitFeedback(payload: {
  ticket_id: string;
  score: number;
  comment?: string;
  conversation_id?: string | null;
}): Promise<{ recorded: boolean; durable: boolean; ticket_id: string; score: number }> {
  return request(`${BASE}/analytics/feedback`, {
    method: "POST",
    body: JSON.stringify({ comment: "", ...payload }),
  });
}

export interface VoiceStatus {
  /** True only when a call could actually be placed: credentials AND a live server. */
  enabled: boolean;
  credentials_configured?: boolean;
  transport: { url: string; configured: boolean; reachable?: boolean };
  stt: { model: string; configured: boolean };
  tts: { voice: string; model_path?: string; model_present?: boolean; configured?: boolean };
  target_ttfa_ms?: number;
  pipeline_process?: string;
  missing_configuration?: string[];
}

export function fetchVoiceStatus(): Promise<VoiceStatus> {
  return request<VoiceStatus>(`${BASE}/voice/status`);
}

export interface VoicePreview {
  original: string;
  spoken: string;
  /** Each is true when the named problem is still present after adaptation. */
  checks: Record<string, boolean>;
  filler_example: string;
}

/** Run text through the same adapter the TTS stage receives. */
export function previewVoice(text: string): Promise<VoicePreview> {
  return request<VoicePreview>(`${BASE}/voice/preview`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

export interface WebhookStatus {
  inbound_enabled: boolean;
  missing_configuration: string[];
  endpoints: Record<string, string>;
  signature_validation: string;
}

export function fetchTwilioStatus(): Promise<WebhookStatus> {
  return request<WebhookStatus>(`${BASE}/webhooks/status`);
}

export interface EmailStatus extends WebhookStatus {
  outbound_enabled: boolean;
  outbound_note: string;
}

export function fetchEmailStatus(): Promise<EmailStatus> {
  return request<EmailStatus>(`${BASE}/webhooks/email/status`);
}

/** One event from the chat SSE stream. */
export type StreamEvent =
  | { type: "start"; conversation_id: string }
  | { type: "node"; node: string }
  // A fragment of the reply as the model produces it. The trailing `message`
  // event carries the authoritative, channel-rendered text, so accumulated
  // deltas are replaced rather than appended to.
  | { type: "delta"; text: string }
  | { type: "awaiting_human" }
  | {
      type: "message";
      reply: string | string[];
      ticket_id: string | null;
      intent: string;
      escalated: boolean;
      citations: Citation[];
      stubbed: boolean;
    }
  | { type: "error"; detail: string }
  | { type: "done" };

/**
 * Stream a turn, yielding each server event as it arrives.
 *
 * EventSource cannot POST, so this parses the SSE framing off a fetch body.
 * Frames are split on the blank-line delimiter and buffered across chunk
 * boundaries, since a chunk can end mid-frame.
 *
 * Deliberately exempt from `REQUEST_TIMEOUT_MS`: a turn is several sequential
 * model calls and legitimately runs past any deadline that would suit a plain
 * request. What bounds it instead is visible progress -- node events, then
 * tokens -- and the caller's `signal`, which the Stop button holds.
 */
export async function* streamMessage(
  payload: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(`Stream failed (${response.status})`, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseFrame(frame);
      if (parsed) yield parsed;
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;

  try {
    const data = JSON.parse(dataLines.join("\n")) as Record<string, unknown>;
    return { type: event, ...data } as StreamEvent;
  } catch {
    return null;
  }
}

/** Human-readable label for a graph node, for the progress indicator. */
export const NODE_LABELS: Record<string, string> = {
  router: "Understanding your message",
  auth: "Verifying your account",
  agentic_rag: "Searching the knowledge base",
  simple_resolution: "Preparing an answer",
  complex_resolution: "Investigating across systems",
  sentiment: "Checking how this is going",
  escalation: "Bringing in a colleague",
  ticketing: "Updating your ticket",
  human_handoff: "Handing over to a human",
};
