"use client";

import Link from "next/link";

import { useCallback, useEffect, useState } from "react";
import { getAgentName, setAgentName } from "@/lib/agent";
import {
  assignTicket,
  fetchQueue,
  type QueueEntry,
  resumeConversation,
  type SlaStatus,
  type SlaSummary,
} from "@/lib/api";

/** How often the queue re-polls. A handoff is time-critical; stale is useless. */
const POLL_INTERVAL_MS = 10_000;

export function HandoffQueue() {
  const [entries, setEntries] = useState<QueueEntry[]>([]);
  const [summary, setSummary] = useState<SlaSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyTicket, setBusyTicket] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [reloads, setReloads] = useState(0);

  // The clock lives in state, ticking once a second. Reading Date.now() during
  // render would make rendering impure; countdowns still move because this
  // re-renders, while the data itself refreshes on the slower poll below.
  const [now, setNow] = useState<number>(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const load = useCallback(() => setReloads((n) => n + 1), []);

  useEffect(() => {
    let active = true;
    // A poll still in flight when the next tick fires must not start a second
    // one: against a stalled API that is how you get dozens of pending
    // requests and an unresponsive tab.
    let inFlight = false;

    const poll = () => {
      if (inFlight) return;
      inFlight = true;
      fetchQueue()
        .then((data) => {
          if (!active) return;
          setEntries(data.queue);
          setSummary(data.sla_summary);
          setLastUpdated(new Date());
          setError(null);
        })
        .catch(() => {
          if (active) setError("Could not load the handoff queue. Is the API running on port 8000?");
        })
        .finally(() => {
          inFlight = false;
          if (active) setLoading(false);
        });
    };

    poll();
    // Without this the queue only ever showed what existed when the page was
    // opened, so an escalation raised in another tab never appeared.
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, [reloads]);

  const claim = useCallback(
    async (entry: QueueEntry) => {
      setBusyTicket(entry.ticket_id);
      try {
        const agent =
          getAgentName() ?? window.prompt("Your name, so the claim is attributable:");
        if (!agent) return;
        setAgentName(agent);
        await assignTicket(entry.ticket_id, agent);
        load();
      } catch {
        setError(`Could not claim ${entry.ticket_id}.`);
      } finally {
        setBusyTicket(null);
      }
    },
    [load],
  );

  const resolve = useCallback(
    async (entry: QueueEntry) => {
      const conversationId = entry.handoff_packet?.conversation_id;
      if (!conversationId) {
        setError(`Ticket ${entry.ticket_id} has no conversation id, so it cannot be resumed here.`);
        return;
      }
      setBusyTicket(entry.ticket_id);
      try {
        await resumeConversation(conversationId, "Resolved by a human agent from the console");
        load();
      } catch {
        setError(`Could not resume ${entry.ticket_id}.`);
      } finally {
        setBusyTicket(null);
      }
    },
    [load],
  );

  if (loading) return <p className="text-sm text-ink-3">Loading queue…</p>;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold tracking-tight">Handoff queue</h2>
        {summary && <SlaCounts summary={summary} />}
        <span className="ml-auto text-xs text-ink-3">
          {lastUpdated ? `updated ${lastUpdated.toLocaleTimeString()}` : ""} · auto-refresh{" "}
          {POLL_INTERVAL_MS / 1000}s
        </span>
        <button
          type="button"
          onClick={load}
          className="rounded border border-line-strong px-2 py-1 text-xs hover:bg-surface-2"
        >
          Refresh
        </button>
      </div>

      {error && (
        <p
          role="alert"
          className="rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger"
        >
          {error}
        </p>
      )}

      {entries.length === 0 ? (
        <p className="rounded-xl border border-dashed border-line-strong px-4 py-8 text-center text-sm text-ink-3">
          Nothing waiting. Escalate a conversation from the chat page — try asking to speak to a
          human.
        </p>
      ) : (
        <ul className="flex flex-col gap-3">
          {entries.map((entry) => (
            <li key={entry.ticket_id}>
              <TicketCard
                entry={entry}
                now={now}
                busy={busyTicket === entry.ticket_id}
                onClaim={() => void claim(entry)}
                onResolve={() => void resolve(entry)}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SlaCounts({ summary }: { summary: SlaSummary }) {
  return (
    <div className="flex items-center gap-1.5 text-xs">
      {summary.breached > 0 && (
        <Badge tone="danger">{summary.breached} breached</Badge>
      )}
      {summary.due_soon > 0 && <Badge tone="warn">{summary.due_soon} due soon</Badge>}
      <Badge>{summary.on_track} on track</Badge>
    </div>
  );
}

function TicketCard({
  entry,
  now,
  busy,
  onClaim,
  onResolve,
}: {
  entry: QueueEntry;
  now: number;
  busy: boolean;
  onClaim: () => void;
  onResolve: () => void;
}) {
  const packet = entry.handoff_packet;
  const sla = entry.sla;

  return (
    <article
      className={[
        "rounded-xl border bg-surface p-4 shadow-sm",
        sla?.state === "breached"
          ? "border-danger/60"
          : sla?.state === "due_soon"
            ? "border-warn/60"
            : "border-line",
      ].join(" ")}
    >
      <header className="flex flex-wrap items-center gap-2">
        <Link
          href={`/tickets/${encodeURIComponent(entry.ticket_id)}`}
          className="text-xs text-ink-3 underline decoration-dotted hover:decoration-solid"
        >
          {entry.ticket_id}
        </Link>
        <Badge tone={priorityTone(entry.priority)}>{entry.priority}</Badge>
        <Badge>{entry.intent}</Badge>
        <ChannelTrail channels={entry.channels} fallback={entry.channel} />
        {packet?.churn_risk && (
          <Badge tone={packet.churn_risk === "high" ? "danger" : "neutral"}>
            churn: {packet.churn_risk}
          </Badge>
        )}
        {entry.assigned_to && <Badge tone="info">claimed by {entry.assigned_to}</Badge>}

        <div className="ml-auto flex items-center gap-2">
          {!entry.assigned_to && (
            <button
              type="button"
              onClick={onClaim}
              disabled={busy}
              className="rounded border border-line-strong px-2.5 py-1 text-xs hover:bg-surface-2 disabled:opacity-40"
            >
              Claim
            </button>
          )}
          <button
            type="button"
            onClick={onResolve}
            disabled={busy}
            className="rounded bg-ink px-3 py-1 text-xs font-medium text-white disabled:opacity-40 dark:bg-white dark:text-canvas"
          >
            {busy ? "Working…" : "Mark handled"}
          </button>
        </div>
      </header>

      {sla && <SlaBar sla={sla} now={now} />}

      <dl className="mt-2 grid gap-x-4 gap-y-1 text-xs sm:grid-cols-3">
        <TimeField label="Escalated" iso={entry.escalated_at_iso} now={now} />
        <TimeField label="Respond by" iso={sla?.due_at ?? null} now={now} />
        <TimeField label="Ticket opened" iso={entry.created_at_iso} now={now} />
      </dl>

      <p className="mt-3 text-sm font-medium">{entry.subject}</p>
      {packet?.executive_summary && (
        <p className="mt-1 text-sm text-ink-2">
          {packet.executive_summary}
        </p>
      )}

      <dl className="mt-3 grid gap-3 sm:grid-cols-2">
        {entry.escalation_reason && (
          <Block title="Why it escalated">
            <code className="text-xs">{entry.escalation_reason}</code>
          </Block>
        )}
        {packet?.sentiment_trajectory && packet.sentiment_trajectory.length > 0 && (
          <Block title="Sentiment trajectory">
            <Sparkline values={packet.sentiment_trajectory} />
          </Block>
        )}
        {packet?.actions_attempted && packet.actions_attempted.length > 0 && (
          <Block title="Already attempted">
            <ul className="list-disc pl-4 text-xs">
              {packet.actions_attempted.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </Block>
        )}
        {packet?.recommended_actions && packet.recommended_actions.length > 0 && (
          <Block title="Recommended next">
            <ul className="list-disc pl-4 text-xs">
              {packet.recommended_actions.map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
          </Block>
        )}
      </dl>

      {packet?.transcript && packet.transcript.length > 0 && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-ink-3">
            Full transcript ({packet.transcript.length} turns, PII masked)
          </summary>
          <ul className="mt-2 flex flex-col gap-1.5 text-xs">
            {packet.transcript.map((row, index) => (
              <li key={`${row.role}-${index}`}>
                <span className="font-medium">{row.role}: </span>
                <span className="text-ink-2">{row.text}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </article>
  );
}

/** Countdown recomputed from the deadline, so it stays right between polls. */
function SlaBar({ sla, now }: { sla: SlaStatus; now: number }) {
  const remaining = Math.round((new Date(sla.due_at).getTime() - now) / 1000);
  const total = sla.target_minutes * 60;
  const used = Math.min(1, Math.max(0, 1 - remaining / total));

  const tone =
    remaining < 0
      ? { bar: "bg-danger", text: "text-danger" }
      : remaining <= total * 0.25
        ? { bar: "bg-warn", text: "text-amber-700 dark:text-amber-400" }
        : { bar: "bg-ok", text: "text-ok" };

  return (
    <div className="mt-2 flex items-center gap-2">
      <span className="h-1.5 flex-1 overflow-hidden rounded bg-surface-2">
        <span className={`block h-full rounded ${tone.bar}`} style={{ width: `${used * 100}%` }} />
      </span>
      <span className={`w-40 shrink-0 text-right text-xs font-medium tabular-nums ${tone.text}`}>
        {remaining < 0 ? `overdue by ${formatDuration(-remaining)}` : `${formatDuration(remaining)} left`}
      </span>
    </div>
  );
}

function TimeField({ label, iso, now }: { label: string; iso: string | null; now: number }) {
  if (!iso) return null;
  const date = new Date(iso);
  return (
    <div>
      <dt className="text-ink-3">{label}</dt>
      <dd title={date.toISOString()}>
        {date.toLocaleString()}{" "}
        <span className="text-ink-3">({relative(date, now)})</span>
      </dd>
    </div>
  );
}

function ChannelTrail({ channels, fallback }: { channels: string[]; fallback: string }) {
  const list = channels?.length ? channels : [fallback];
  return (
    <span className="flex items-center gap-1">
      {list.map((c, i) => (
        <span key={`${c}-${i}`} className="flex items-center gap-1">
          {i > 0 && <span className="text-ink-3">→</span>}
          <Badge>{c}</Badge>
        </span>
      ))}
    </span>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  if (hours < 24) return mins ? `${hours}h ${mins}m` : `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

function relative(date: Date, now: number): string {
  const seconds = Math.round((now - date.getTime()) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 0) return `in ${formatDuration(-seconds)}`;
  return `${formatDuration(seconds)} ago`;
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-medium text-ink-3">{title}</dt>
      <dd className="mt-1">{children}</dd>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const width = 96;
  const height = 24;
  const step = values.length > 1 ? width / (values.length - 1) : 0;
  const points = values
    .map((v, i) => {
      const x = i * step;
      const y = height / 2 - (Math.max(-1, Math.min(1, v)) * height) / 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const last = values[values.length - 1] ?? 0;

  return (
    <span className="flex items-center gap-2">
      <svg width={width} height={height} role="img" aria-label="sentiment over time">
        <line x1="0" y1={height / 2} x2={width} y2={height / 2} stroke="currentColor" strokeOpacity="0.15" />
        <polyline
          points={points}
          fill="none"
          stroke={last < -0.3 ? "#dc2626" : last > 0.3 ? "#16a34a" : "#6b7280"}
          strokeWidth="1.5"
        />
      </svg>
      <span className="text-xs tabular-nums text-ink-3">{last.toFixed(2)}</span>
    </span>
  );
}

function Badge({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "warn" | "danger" | "info";
}) {
  const tones = {
    neutral: "bg-surface-2 text-ink-2",
    warn: "bg-warn-soft text-warn",
    danger: "bg-danger-soft text-danger",
    info: "bg-info-soft text-info",
  } as const;
  return <span className={`rounded px-1.5 py-0.5 text-[11px] ${tones[tone]}`}>{children}</span>;
}

function priorityTone(priority: string): "neutral" | "warn" | "danger" {
  if (priority === "urgent") return "danger";
  if (priority === "high") return "warn";
  return "neutral";
}
