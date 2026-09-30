"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { fetchTickets, type TicketSummary } from "@/lib/api";
import { relativeTime } from "@/lib/time";

/**
 * Every ticket, not just the ones waiting on a human.
 *
 * `GET /api/tickets` has existed since the beginning and nothing in the UI
 * called it -- `fetchTickets` was dead code. A support console that can only
 * show the escalation queue cannot answer "what happened to that customer",
 * which is most of the job.
 */

const STATUSES = ["all", "open", "pending_human", "resolved"] as const;
type StatusFilter = (typeof STATUSES)[number];

const STATUS_LABELS: Record<string, string> = {
  open: "Open",
  pending_human: "Awaiting human",
  resolved: "Resolved",
};

export function TicketList() {
  const [tickets, setTickets] = useState<TicketSummary[]>([]);
  const [status, setStatus] = useState<StatusFilter>("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    // No setLoading(true) here: keeping the current rows on screen while the
    // next filter loads avoids a blank flash on every tab click.
    fetchTickets(status === "all" ? undefined : status)
      .then((data) => {
        if (!active) return;
        setTickets(data.tickets);
        setError(null);
      })
      .catch(() => active && setError("Could not load tickets."))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [status]);

  // Filtering client-side: the endpoint returns at most 200 rows, so a round
  // trip per keystroke would cost more than it saves.
  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return tickets;
    return tickets.filter((t) =>
      [t.ticket_id, t.customer_id, t.subject, t.summary, t.intent]
        .join(" ")
        .toLowerCase()
        .includes(needle),
    );
  }, [tickets, query]);

  return (
    <div className="flex flex-col gap-4">
      <header className="flex flex-wrap items-center gap-3">
        <h1 className="text-lg font-semibold tracking-tight">Tickets</h1>
        <div className="flex gap-1" role="group" aria-label="Filter by status">
          {STATUSES.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setStatus(s)}
              aria-pressed={s === status}
              className={[
                "rounded px-2 py-0.5 text-xs",
                s === status
                  ? "bg-ink text-canvas"
                  : "border border-line-strong hover:bg-surface-2",
              ].join(" ")}
            >
              {s === "all" ? "All" : (STATUS_LABELS[s] ?? s)}
            </button>
          ))}
        </div>
        <label className="ml-auto">
          <span className="sr-only">Search tickets</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search id, customer, subject…"
            className="w-56 rounded border border-line-strong bg-white px-2 py-1 text-xs"
          />
        </label>
      </header>

      {error && (
        <p
          role="alert"
          className="rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger"
        >
          {error}
        </p>
      )}

      {loading ? (
        <p className="text-sm text-ink-3">Loading tickets…</p>
      ) : visible.length === 0 ? (
        <p className="rounded-xl border border-dashed border-line-strong px-4 py-8 text-center text-sm text-ink-3">
          {tickets.length === 0
            ? "No tickets yet. Start a conversation on the chat page."
            : "No tickets match that search."}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {visible.map((ticket) => (
            <li key={ticket.ticket_id}>
              <Row ticket={ticket} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Row({ ticket }: { ticket: TicketSummary }) {
  return (
    <Link
      href={`/tickets/${encodeURIComponent(ticket.ticket_id)}`}
      className="block rounded-xl border border-line bg-white p-3 shadow-sm transition hover:border-line-strong dark:hover:border-line-strong"
    >
      <div className="flex flex-wrap items-center gap-2">
        <code className="text-xs text-ink-3">{ticket.ticket_id}</code>
        <Pill tone={ticket.status === "pending_human" ? "warn" : "neutral"}>
          {STATUS_LABELS[ticket.status] ?? ticket.status}
        </Pill>
        <Pill>{ticket.intent}</Pill>
        {ticket.channels?.length > 1 && <Pill>{ticket.channels.join(" → ")}</Pill>}
        {ticket.assigned_to && <Pill tone="info">{ticket.assigned_to}</Pill>}
        <span className="ml-auto text-xs text-ink-3">
          {relativeTime(ticket.updated_at * 1000)}
        </span>
      </div>
      <p className="mt-1 truncate text-sm">{ticket.subject}</p>
    </Link>
  );
}

function Pill({
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
