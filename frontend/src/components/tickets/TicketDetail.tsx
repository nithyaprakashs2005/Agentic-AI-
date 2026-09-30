"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { assignTicket, fetchTicket, type TicketDetail as Detail } from "@/lib/api";
import { getAgentName } from "@/lib/agent";
import { absoluteTime, relativeTime } from "@/lib/time";

/**
 * One ticket, end to end: what was said, what the agent did, and where it went.
 *
 * `GET /api/tickets/{id}` returns the full transcript and handoff packet and
 * nothing in the UI ever rendered them, so the conversation behind an
 * escalation was only visible in the database.
 */

export function TicketDetail({ ticketId }: { ticketId: string }) {
  const [ticket, setTicket] = useState<Detail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    let active = true;
    fetchTicket(ticketId)
      .then((data) => {
        if (!active) return;
        setTicket(data);
        setError(null);
      })
      .catch((caught) =>
        active &&
        setError(
          caught instanceof Error && caught.message.includes("not found")
            ? "That ticket does not exist."
            : "Could not load this ticket.",
        ),
      )
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [ticketId]);

  useEffect(() => load(), [load]);

  const claim = async () => {
    const agent = getAgentName() ?? window.prompt("Your name, so the claim is attributable:");
    if (!agent) return;
    setBusy(true);
    try {
      await assignTicket(ticketId, agent);
      load();
    } catch {
      setError("Could not claim this ticket.");
    } finally {
      setBusy(false);
    }
  };

  if (loading) return <p className="text-sm text-ink-3">Loading ticket…</p>;
  if (error) {
    return (
      <div className="flex flex-col gap-3">
        <p role="alert" className="rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger">
          {error}
        </p>
        <Link href="/tickets" className="text-sm underline">
          Back to tickets
        </Link>
      </div>
    );
  }
  if (!ticket) return null;

  const packet = ticket.handoff_packet;

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-col gap-2">
        <Link href="/tickets" className="text-xs text-ink-3 underline">
          ← All tickets
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-lg font-semibold tracking-tight">{ticket.subject}</h1>
          <Pill tone={ticket.status === "pending_human" ? "warn" : "neutral"}>
            {ticket.status}
          </Pill>
          <Pill tone={ticket.priority === "urgent" ? "danger" : "neutral"}>
            {ticket.priority}
          </Pill>
          {ticket.escalated && <Pill tone="warn">escalated</Pill>}
        </div>
        <code className="text-xs text-ink-3">{ticket.ticket_id}</code>
      </header>

      <section className="grid gap-3 sm:grid-cols-2">
        <Facts
          rows={[
            ["Customer", ticket.customer_id],
            ["Intent", ticket.intent],
            ["Channels", (ticket.channels ?? [ticket.channel]).join(" → ")],
            ["Assigned to", ticket.assigned_to ?? "unclaimed"],
          ]}
        />
        <Facts
          rows={[
            ["Opened", `${absoluteTime(ticket.created_at_iso)} (${relativeTime(ticket.created_at * 1000)})`],
            ["Escalated", ticket.escalated_at_iso ? absoluteTime(ticket.escalated_at_iso) : "—"],
            ["First response", ticket.first_responded_at_iso ? absoluteTime(ticket.first_responded_at_iso) : "not yet"],
            ["Respond by", ticket.sla ? `${absoluteTime(ticket.sla.due_at)} — ${ticket.sla.label}` : "—"],
          ]}
        />
      </section>

      {ticket.status === "pending_human" && !ticket.assigned_to && (
        <div>
          <button
            type="button"
            onClick={() => void claim()}
            disabled={busy}
            className="rounded-lg bg-ink px-3 py-1.5 text-sm font-medium text-canvas disabled:opacity-40"
          >
            {busy ? "Claiming…" : "Claim this ticket"}
          </button>
        </div>
      )}

      {ticket.summary && (
        <Panel title="Summary">
          <p className="text-sm whitespace-pre-wrap">{ticket.summary}</p>
        </Panel>
      )}

      {packet && (
        <Panel title="Handoff packet">
          <dl className="flex flex-col gap-2 text-sm">
            {packet.executive_summary && (
              <Row label="Executive summary">{packet.executive_summary}</Row>
            )}
            {packet.escalation_reason && (
              <Row label="Reason">
                <code className="text-xs">{packet.escalation_reason}</code>
              </Row>
            )}
            {packet.churn_risk && <Row label="Churn risk">{packet.churn_risk}</Row>}
            {packet.actions_attempted?.length ? (
              <Row label="Already attempted">
                <List items={packet.actions_attempted} />
              </Row>
            ) : null}
            {packet.recommended_actions?.length ? (
              <Row label="Recommended next">
                <List items={packet.recommended_actions} />
              </Row>
            ) : null}
            {packet.deep_agent_todos?.length ? (
              <Row label="Investigation plan">
                <List items={packet.deep_agent_todos} />
              </Row>
            ) : null}
            {packet.sentiment_trajectory?.length ? (
              <Row label="Sentiment trajectory">
                <Trajectory points={packet.sentiment_trajectory} />
              </Row>
            ) : null}
          </dl>
        </Panel>
      )}

      <Panel title={`Transcript (${ticket.transcript?.length ?? 0} turns)`}>
        {ticket.transcript?.length ? (
          <ul className="flex flex-col gap-2">
            {ticket.transcript.map((turn, index) => (
              <li
                key={index}
                className={turn.role === "customer" ? "flex justify-end" : "flex justify-start"}
              >
                <div
                  className={[
                    "max-w-[85%] rounded-2xl px-3 py-2 text-sm whitespace-pre-wrap",
                    turn.role === "customer"
                      ? "bg-ink text-canvas"
                      : "bg-surface-2 text-ink",
                  ].join(" ")}
                >
                  {turn.text}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-ink-3">No transcript was stored for this ticket.</p>
        )}
      </Panel>

      {ticket.citations?.length > 0 && (
        <Panel title="Sources cited">
          <ul className="flex flex-col gap-1 text-sm">
            {ticket.citations.map((c) => (
              <li key={c.id}>
                <a
                  href={c.source}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="underline decoration-dotted hover:decoration-solid"
                >
                  {c.title || c.id}
                </a>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="rounded-xl border border-line bg-white p-3 text-sm shadow-sm">
      {rows.map(([label, value]) => (
        <div key={label} className="flex gap-2 py-0.5">
          <dt className="w-32 shrink-0 text-xs text-ink-3">{label}</dt>
          <dd className="min-w-0 break-words">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-surface p-4 shadow-sm">
      <h2 className="mb-2 text-sm font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 sm:flex-row sm:gap-2">
      <dt className="w-40 shrink-0 text-xs text-ink-3">{label}</dt>
      <dd className="min-w-0">{children}</dd>
    </div>
  );
}

function List({ items }: { items: string[] }) {
  return (
    <ul className="list-disc pl-5">
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  );
}

function Trajectory({ points }: { points: number[] }) {
  return (
    <span className="flex items-end gap-0.5" title={points.join(", ")}>
      {points.map((value, index) => {
        // -1..+1 mapped to a bar height; negative reads red, positive green.
        const height = Math.max(2, Math.round((Math.abs(value) || 0.05) * 20));
        return (
          <span
            key={index}
            style={{ height: `${height}px` }}
            className={`w-1.5 rounded-sm ${value < 0 ? "bg-red-400" : "bg-emerald-400"}`}
          />
        );
      })}
      <span className="ml-1 text-xs text-ink-3">
        {points[points.length - 1]?.toFixed(2)}
      </span>
    </span>
  );
}

function Pill({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "warn" | "danger";
}) {
  const tones = {
    neutral: "bg-surface-2 text-ink-2",
    warn: "bg-warn-soft text-warn",
    danger: "bg-danger-soft text-danger",
  } as const;
  return (
    <span className={`rounded px-1.5 py-0.5 text-[11px] ${tones[tone]}`}>{children}</span>
  );
}
