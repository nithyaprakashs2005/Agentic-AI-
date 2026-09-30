"use client";

import { useEffect, useState } from "react";
import {
  fetchMetrics,
  type MetricsSummary,
  type MetricsWindow,
} from "@/lib/api";

const WINDOWS: MetricsWindow[] = ["1h", "24h", "7d", "all"];

export function MetricsPanel() {
  const [data, setData] = useState<MetricsSummary | null>(null);
  const [window, setWindow] = useState<MetricsWindow>("24h");
  const [error, setError] = useState<string | null>(null);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    let active = true;
    fetchMetrics(window)
      .then((d) => active && setData(d))
      .catch(() => active && setError("Could not load metrics."));
    return () => {
      active = false;
    };
  }, [window, reloads]);

  if (error) {
    return (
      <p
        role="alert"
        className="rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger"
      >
        {error}
      </p>
    );
  }
  if (!data) return <p className="text-sm text-ink-3">Loading metrics…</p>;

  const noData = data.totals.turns === 0;

  return (
    <section className="flex flex-col gap-3">
      <header className="flex flex-wrap items-center gap-2">
        <h2 className="text-lg font-semibold tracking-tight">Operations</h2>
        <div className="flex gap-1" role="group" aria-label="Time window">
          {WINDOWS.map((w) => (
            <button
              key={w}
              type="button"
              onClick={() => setWindow(w)}
              aria-pressed={w === window}
              className={[
                "rounded px-2 py-0.5 text-xs",
                w === window
                  ? "bg-ink text-canvas"
                  : "border border-line-strong hover:bg-surface-2",
              ].join(" ")}
            >
              {w}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setReloads((n) => n + 1)}
          className="ml-auto rounded border border-line-strong px-2 py-1 text-xs hover:bg-surface-2"
        >
          Refresh
        </button>
      </header>

      {data.data_quality.warning && (
        <p className="rounded-lg border border-warn/50 bg-warn-soft px-3 py-2 text-xs text-warn">
          {data.data_quality.warning} (
          {Math.round(data.data_quality.stubbed_share * 100)}% of turns)
        </p>
      )}

      {noData ? (
        <p className="rounded-xl border border-dashed border-line-strong px-4 py-6 text-center text-sm text-ink-3">
          No traffic in this window. Send a message from the chat page.
        </p>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Stat
              label="First-contact resolution"
              value={pct(data.fcr.rate)}
              detail={`${data.fcr.resolved_first_contact} of ${data.fcr.conversations} · target ${pct(data.fcr.target)}`}
              tone={toneFor(data.fcr.rate, data.fcr.target)}
              title={data.fcr.definition}
            />
            <Stat
              label="Escalation rate"
              value={pct(data.escalation.rate)}
              detail={`${data.escalation.escalated_conversations} conversation(s) to a human`}
            />
            <Stat
              label="CSAT"
              value={data.csat.score === null ? "—" : pct(data.csat.score)}
              detail={
                data.csat.responses === 0
                  ? "no ratings submitted"
                  : `${data.csat.responses} rating(s) · mean ${data.csat.mean_rating}/5`
              }
              title={data.csat.note}
            />
            <Stat
              label="Turn latency p95"
              value={`${Math.round(data.latency.turn_total.p95 ?? 0)} ms`}
              detail={`p50 ${Math.round(data.latency.turn_total.p50 ?? 0)} ms · ${data.totals.turns} turns`}
            />
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <Card title="Slowest nodes (p50)">
              <ul className="flex flex-col gap-1.5 text-xs">
                {Object.entries(data.latency.by_node)
                  .slice(0, 6)
                  .map(([node, stats]) => (
                    <li key={node} className="flex items-center gap-2">
                      <span className="w-36 shrink-0 truncate font-mono">{node}</span>
                      <span className="h-1.5 flex-1 overflow-hidden rounded bg-surface-2">
                        <span
                          className="block h-full rounded bg-ink-2"
                          style={{ width: `${barWidth(stats.p50, data.latency.by_node)}%` }}
                        />
                      </span>
                      <span className="w-24 shrink-0 text-right tabular-nums text-ink-3">
                        {stats.p50} / {stats.p95} ms
                      </span>
                    </li>
                  ))}
              </ul>
            </Card>

            <Card title="Escalation reasons">
              {Object.keys(data.escalation.by_reason).length === 0 ? (
                <p className="text-xs text-ink-3">Nothing escalated in this window.</p>
              ) : (
                <ul className="flex flex-col gap-1 text-xs">
                  {Object.entries(data.escalation.by_reason).map(([reason, count]) => (
                    <li key={reason} className="flex justify-between gap-2">
                      <code className="truncate">{reason}</code>
                      <span className="tabular-nums text-ink-3">{count}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>

            <Card title="By intent">
              <Bars counts={data.distribution.by_intent} />
            </Card>

            <Card title="By channel">
              <Bars counts={data.distribution.by_channel} />
            </Card>
          </div>
        </>
      )}
    </section>
  );
}

function Stat({
  label,
  value,
  detail,
  tone = "neutral",
  title,
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "neutral" | "good" | "bad";
  title?: string;
}) {
  const tones = {
    neutral: "text-ink",
    good: "text-ok",
    bad: "text-danger",
  } as const;
  return (
    <div
      title={title}
      className="rounded-xl border border-line bg-surface p-3 shadow-sm"
    >
      <p className="text-xs text-ink-3">{label}</p>
      <p className={`mt-1 text-2xl font-semibold tabular-nums ${tones[tone]}`}>{value}</p>
      <p className="mt-1 text-xs text-ink-3">{detail}</p>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-3 shadow-sm">
      <h3 className="mb-2 text-xs font-medium text-ink-3">{title}</h3>
      {children}
    </div>
  );
}

function Bars({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts);
  if (entries.length === 0) {
    return <p className="text-xs text-ink-3">No data.</p>;
  }
  const max = Math.max(...entries.map(([, n]) => n));
  return (
    <ul className="flex flex-col gap-1.5 text-xs">
      {entries.map(([key, count]) => (
        <li key={key} className="flex items-center gap-2">
          <span className="w-24 shrink-0 truncate">{key}</span>
          <span className="h-1.5 flex-1 overflow-hidden rounded bg-surface-2">
            <span
              className="block h-full rounded bg-ink-2"
              style={{ width: `${(count / max) * 100}%` }}
            />
          </span>
          <span className="w-8 shrink-0 text-right tabular-nums text-ink-3">{count}</span>
        </li>
      ))}
    </ul>
  );
}

function pct(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function toneFor(value: number | null, target: number): "neutral" | "good" | "bad" {
  if (value === null) return "neutral";
  return value >= target ? "good" : "bad";
}

function barWidth(
  value: number,
  all: Record<string, { p50: number }>,
): number {
  const max = Math.max(...Object.values(all).map((s) => s.p50), 1);
  return Math.max(2, (value / max) * 100);
}
