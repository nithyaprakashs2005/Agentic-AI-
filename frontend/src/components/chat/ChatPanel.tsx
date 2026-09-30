"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  NODE_LABELS,
  type Channel,
  type Citation,
  type CustomerTier,
  fetchHealth,
  type HealthReport,
  streamMessage,
  submitFeedback,
} from "@/lib/api";

interface Turn {
  id: string;
  role: "customer" | "agent";
  text: string;
  citations?: Citation[];
  escalated?: boolean;
  intent?: string;
  stubbed?: boolean;
  ticketId?: string | null;
  /** True while tokens are still arriving, so the bubble shows a live caret. */
  streaming?: boolean;
}

const CHANNELS: Channel[] = ["webchat", "voice", "sms", "whatsapp", "email"];
const TIERS: CustomerTier[] = ["free", "standard", "enterprise"];

const SUGGESTIONS = [
  "How long do refunds take to appear?",
  "What is the return window for an unopened item?",
  "My authenticator codes are being rejected",
  "Where is my order ORD-84021?",
  "I want to speak to a human",
];

/** Conversations survive a reload; the backend checkpoint already did. */
const STORAGE_KEY = "support-console.chat";
const HEALTH_REFRESH_MS = 30_000;

interface Persisted {
  conversationId: string | null;
  turns: Turn[];
  channel: Channel;
  tier: CustomerTier;
  customerId: string;
}

function loadPersisted(): Persisted | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Persisted) : null;
  } catch {
    return null;
  }
}

function persist(state: Persisted): void {
  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Storage unavailable; the conversation simply will not survive a reload.
  }
}

export function ChatPanel() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [restored, setRestored] = useState(false);

  const [channel, setChannel] = useState<Channel>("webchat");
  const [tier, setTier] = useState<CustomerTier>("standard");
  const [customerId, setCustomerId] = useState("cust_demo");
  const [conversationId, setConversationId] = useState<string | null>(null);

  // Mirrored in a ref so `send` can read the latest id without being
  // re-created (and losing in-flight identity) on every turn.
  const conversationRef = useRef<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  // Lets the Stop button cancel an in-flight turn.
  const abortRef = useRef<AbortController | null>(null);
  // Monotonic ids: two turns in the same millisecond would collide on Date.now().
  const seq = useRef(0);
  const nextId = (prefix: string) => `${prefix}-${(seq.current += 1)}`;
  // The last thing the customer sent, so a failed turn can be retried. State
  // rather than a ref because the retry button renders from it.
  const [lastSent, setLastSent] = useState<string | null>(null);

  // --- restore a conversation across reloads ---
  // Restoring from sessionStorage must happen after hydration: the server
  // cannot read it, so a lazy initialiser would render different markup on
  // each side.
  /* eslint-disable react-hooks/set-state-in-effect -- read after hydration */
  useEffect(() => {
    const saved = loadPersisted();
    if (saved) {
      setTurns(saved.turns ?? []);
      setConversationId(saved.conversationId ?? null);
      conversationRef.current = saved.conversationId ?? null;
      setChannel(saved.channel ?? "webchat");
      setTier(saved.tier ?? "standard");
      setCustomerId(saved.customerId ?? "cust_demo");
      seq.current = (saved.turns ?? []).length;
    }
    setRestored(true);
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (!restored) return;
    persist({ conversationId, turns, channel, tier, customerId });
  }, [restored, conversationId, turns, channel, tier, customerId]);

  // --- health, refreshed rather than read once at mount ---
  useEffect(() => {
    let active = true;
    const check = () =>
      fetchHealth()
        .then((report) => active && setHealth(report))
        .catch(() => active && setHealth(null));
    check();
    const id = setInterval(check, HEALTH_REFRESH_MS);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, progress]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || busy) return;

      setLastSent(message);
      const controller = new AbortController();
      abortRef.current = controller;

      setBusy(true);
      setError(null);
      setDraft("");
      setTurns((prior) => [
        ...prior,
        { id: nextId("c"), role: "customer", text: message },
      ]);

      // The bubble tokens stream into. Created on the first delta so a
      // non-streaming provider does not leave an empty bubble behind.
      let liveId: string | null = null;
      let streamed = "";

      try {
        for await (const event of streamMessage(
          {
            message,
            customer_id: customerId,
            channel,
            customer_tier: tier,
            conversation_id: conversationRef.current,
          },
          controller.signal,
        )) {
          switch (event.type) {
            case "start":
              conversationRef.current = event.conversation_id;
              setConversationId(event.conversation_id);
              break;

            case "node":
              setProgress(NODE_LABELS[event.node] ?? null);
              break;

            case "delta": {
              streamed += event.text;
              if (liveId === null) {
                liveId = nextId("a");
                setProgress(null);
                setTurns((prior) => [
                  ...prior,
                  { id: liveId!, role: "agent", text: streamed, streaming: true },
                ]);
              } else {
                const id = liveId;
                const snapshot = streamed;
                setTurns((prior) =>
                  prior.map((t) => (t.id === id ? { ...t, text: snapshot } : t)),
                );
              }
              break;
            }

            case "message": {
              // Authoritative: the channel adapter may have reformatted the
              // text (voice plaintext, SMS segments), so replace rather than
              // keep what streamed.
              const reply = Array.isArray(event.reply)
                ? event.reply.join("\n")
                : event.reply;
              const settled: Turn = {
                id: liveId ?? nextId("a"),
                role: "agent",
                text: reply,
                citations: event.citations,
                escalated: event.escalated,
                intent: event.intent,
                stubbed: event.stubbed,
                ticketId: event.ticket_id,
                streaming: false,
              };
              setTurns((prior) =>
                liveId !== null
                  ? prior.map((t) => (t.id === liveId ? settled : t))
                  : [...prior, settled],
              );
              break;
            }

            case "error":
              setError(event.detail);
              break;

            case "done":
              setProgress(null);
              break;
          }
        }
      } catch (caught) {
        if (controller.signal.aborted) {
          // Deliberate cancellation: keep whatever streamed, and say so.
          if (liveId !== null) {
            const id = liveId;
            setTurns((prior) =>
              prior.map((t) =>
                t.id === id ? { ...t, text: `${t.text} (stopped)`, streaming: false } : t,
              ),
            );
          }
        } else {
          setError(
            caught instanceof Error && caught.name === "TimeoutError"
              ? "The agent took too long to respond."
              : "Could not reach the support service. Is the API running on port 8000?",
          );
        }
      } finally {
        abortRef.current = null;
        setProgress(null);
        setBusy(false);
      }
    },
    [busy, channel, customerId, tier],
  );

  const reset = useCallback(() => {
    stop();
    conversationRef.current = null;
    setConversationId(null);
    setTurns([]);
    setError(null);
    seq.current = 0;
  }, [stop]);

  const lastAgentTurn = [...turns].reverse().find((t) => t.role === "agent" && !t.streaming);

  return (
    <div className="flex flex-col gap-4">
      {health?.status === "degraded" && (
        <DegradationNotice degradations={health.degradations} />
      )}

      <section className="rounded-xl border border-line bg-surface shadow-sm">
        <Controls
          channel={channel}
          onChannel={setChannel}
          tier={tier}
          onTier={setTier}
          customerId={customerId}
          onCustomerId={setCustomerId}
          disabled={busy}
          onReset={reset}
        />

        <div
          className="h-[26rem] overflow-y-auto px-4 py-4"
          aria-live="polite"
          aria-busy={busy}
        >
          {turns.length === 0 && !progress ? (
            <EmptyState onPick={send} disabled={busy} />
          ) : (
            <ul className="flex flex-col gap-3">
              {turns.map((turn) => (
                <li key={turn.id}>
                  <Bubble turn={turn} />
                </li>
              ))}
              {progress && (
                <li>
                  <ProgressBubble label={progress} />
                </li>
              )}
            </ul>
          )}
          <div ref={bottom} />
        </div>

        {error && (
          <div
            role="alert"
            className="mx-4 mb-3 flex flex-wrap items-center gap-2 rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger"
          >
            <span>{error}</span>
            {lastSent && (
              <button
                type="button"
                onClick={() => void send(lastSent)}
                className="rounded border border-danger/40 px-2 py-0.5 text-xs hover:bg-danger-soft"
              >
                Try again
              </button>
            )}
          </div>
        )}

        <form
          className="flex gap-2 border-t border-line p-3"
          onSubmit={(e) => {
            e.preventDefault();
            void send(draft);
          }}
        >
          <label htmlFor="message" className="sr-only">
            Message
          </label>
          <input
            id="message"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Ask about a refund, a return, an order, or a login problem…"
            disabled={busy}
            autoComplete="off"
            className="flex-1 rounded-lg border border-line-strong bg-surface px-3 py-2 text-sm outline-none disabled:opacity-60"
          />
          {busy ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-lg border border-line-strong px-4 py-2 text-sm font-medium hover:bg-surface-2"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!draft.trim()}
              className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-canvas disabled:opacity-40"
            >
              Send
            </button>
          )}
        </form>
      </section>

      {lastAgentTurn?.ticketId && !busy && (
        <SatisfactionPrompt
          // Remount per ticket, so a new one asks again from a clean slate.
          key={lastAgentTurn.ticketId}
          ticketId={lastAgentTurn.ticketId}
          conversationId={conversationId}
        />
      )}

      {conversationId && (
        <p className="text-xs text-ink-3">
          Conversation <code>{conversationId}</code> — the same thread is reused
          across channels, so switching channel above keeps the context.
        </p>
      )}
    </div>
  );
}

function SatisfactionPrompt({
  ticketId,
  conversationId,
}: {
  ticketId: string;
  conversationId: string | null;
}) {
  const [sent, setSent] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);

  if (sent !== null) {
    return (
      <p className="text-xs text-ink-3">
        Thanks — rated {sent}/5.
        {failed && " (recorded here, but the store was unreachable)"}
      </p>
    );
  }

  const rate = async (score: number) => {
    try {
      const result = await submitFeedback({
        ticket_id: ticketId,
        score,
        conversation_id: conversationId,
      });
      setFailed(!result.durable);
    } catch {
      setFailed(true);
    }
    setSent(score);
  };

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-ink-3">
      <span>Was this helpful?</span>
      {[1, 2, 3, 4, 5].map((score) => (
        <button
          key={score}
          type="button"
          onClick={() => void rate(score)}
          aria-label={`Rate ${score} out of 5`}
          className="rounded border border-line-strong px-2 py-0.5 hover:bg-surface-2"
        >
          {score}
        </button>
      ))}
    </div>
  );
}

function Controls(props: {
  channel: Channel;
  onChannel: (c: Channel) => void;
  tier: CustomerTier;
  onTier: (t: CustomerTier) => void;
  customerId: string;
  onCustomerId: (id: string) => void;
  disabled: boolean;
  onReset: () => void;
}) {
  return (
    <div className="flex flex-wrap items-end gap-3 border-b border-line px-4 py-3 text-xs">
      <Field label="Channel">
        <select
          value={props.channel}
          onChange={(e) => props.onChannel(e.target.value as Channel)}
          disabled={props.disabled}
          className="rounded border border-line-strong bg-surface px-2 py-1"
        >
          {CHANNELS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Tier">
        <select
          value={props.tier}
          onChange={(e) => props.onTier(e.target.value as CustomerTier)}
          disabled={props.disabled}
          className="rounded border border-line-strong bg-surface px-2 py-1"
        >
          {TIERS.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </Field>

      <Field label="Customer">
        <input
          value={props.customerId}
          onChange={(e) => props.onCustomerId(e.target.value)}
          disabled={props.disabled}
          className="w-36 rounded border border-line-strong bg-surface px-2 py-1"
        />
      </Field>

      <button
        type="button"
        onClick={props.onReset}
        className="ml-auto rounded border border-line-strong px-2 py-1 hover:bg-surface-2"
      >
        New conversation
      </button>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-ink-3">{label}</span>
      {children}
    </label>
  );
}

function Bubble({ turn }: { turn: Turn }) {
  const isCustomer = turn.role === "customer";
  return (
    <div className={isCustomer ? "flex justify-end" : "flex justify-start"}>
      <div
        className={[
          "max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm whitespace-pre-wrap",
          isCustomer
            ? "bg-ink text-canvas"
            : "bg-surface-2 text-ink",
        ].join(" ")}
      >
        {turn.text}
        {turn.streaming && (
          <span
            aria-hidden
            className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-ink-3 align-middle"
          />
        )}

        {!isCustomer && !turn.streaming && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            {turn.intent && <Tag>{turn.intent}</Tag>}
            {turn.escalated && <Tag tone="warn">handed to a human</Tag>}
            {turn.stubbed && <Tag tone="info">offline stub, not model output</Tag>}
            {turn.ticketId && <Tag>{turn.ticketId}</Tag>}
          </div>
        )}

        {turn.citations && turn.citations.length > 0 && (
          <details className="mt-2 text-xs">
            <summary className="cursor-pointer text-ink-3">
              {turn.citations.length} source
              {turn.citations.length === 1 ? "" : "s"}
            </summary>
            <ul className="mt-1 flex flex-col gap-1">
              {turn.citations.map((c) => (
                <li key={c.id}>
                  <a
                    href={c.source}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="underline decoration-dotted hover:decoration-solid"
                  >
                    {c.title || c.id}
                  </a>
                  <span className="ml-1 text-ink-3">
                    ({c.relevance.toFixed(4)})
                  </span>
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

function ProgressBubble({ label }: { label: string }) {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-2 rounded-2xl bg-surface-2 px-3.5 py-2.5 text-sm text-ink-3">
        <span className="flex gap-1" aria-hidden>
          {[0, 150, 300].map((delay) => (
            <span
              key={delay}
              className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink-3"
              style={{ animationDelay: `${delay}ms` }}
            />
          ))}
        </span>
        <span role="status">{label}…</span>
      </div>
    </div>
  );
}

function Tag({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "warn" | "info";
}) {
  const tones = {
    neutral: "bg-surface-2 text-ink-2",
    warn: "bg-warn-soft text-warn",
    info: "bg-info-soft text-info",
  } as const;
  return (
    <span className={`rounded px-1.5 py-0.5 text-[11px] ${tones[tone]}`}>{children}</span>
  );
}

function EmptyState({
  onPick,
  disabled,
}: {
  onPick: (text: string) => void;
  disabled: boolean;
}) {
  return (
    <div className="flex h-full flex-col justify-center gap-3">
      <p className="text-sm text-ink-3">Try one of these:</p>
      <div className="flex flex-wrap gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            disabled={disabled}
            onClick={() => onPick(s)}
            className="rounded-full border border-line-strong px-3 py-1.5 text-xs hover:bg-surface-2 disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function DegradationNotice({ degradations }: { degradations: string[] }) {
  return (
    <details className="rounded-lg border border-warn/50 bg-warn-soft px-3 py-2 text-sm text-warn">
      <summary className="cursor-pointer font-medium">
        Running in degraded mode — {degradations.length} subsystem
        {degradations.length === 1 ? "" : "s"} using a fallback
      </summary>
      <ul className="mt-2 list-disc pl-5 text-xs">
        {degradations.map((d) => (
          <li key={d}>{d}</li>
        ))}
      </ul>
    </details>
  );
}
