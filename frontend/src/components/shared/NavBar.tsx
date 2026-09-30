"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { fetchHealth, type HealthReport } from "@/lib/api";
import { getAgentName, setAgentName } from "@/lib/agent";

const LINKS = [
  { href: "/chat", label: "Chat" },
  { href: "/voice", label: "Voice" },
  { href: "/channels", label: "Channels" },
  { href: "/tickets", label: "Tickets" },
  { href: "/dashboard", label: "Dashboard" },
];

const HEALTH_POLL_MS = 30_000;

export function NavBar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const [agent, setAgent] = useState<string | null>(null);

  // localStorage is unreadable during SSR, so the operator's name can only be
  // picked up after hydration -- a lazy initialiser would desync the markup.
  // eslint-disable-next-line react-hooks/set-state-in-effect -- read after hydration
  useEffect(() => setAgent(getAgentName()), []);

  useEffect(() => {
    let active = true;
    const check = () =>
      fetchHealth()
        .then((report) => {
          if (!active) return;
          setHealth(report);
          setUnreachable(false);
        })
        .catch(() => {
          if (!active) return;
          // Distinct from "degraded": the API itself did not answer.
          setUnreachable(true);
        });
    check();
    const id = setInterval(check, HEALTH_POLL_MS);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const rename = () => {
    const next = window.prompt("Your name, used when you claim a ticket:", agent ?? "");
    if (next?.trim()) {
      setAgentName(next);
      setAgent(next.trim());
    }
  };

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/85 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-4 py-3">
        <Link href="/" className="font-semibold tracking-tight">
          Support Agent
        </Link>

        <div className="flex gap-1 text-sm">
          {LINKS.map((link) => {
            const active = pathname === link.href || pathname.startsWith(`${link.href}/`);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={[
                  "rounded px-2 py-1",
                  active ? "bg-ink text-canvas" : "text-ink-2 hover:bg-surface-2 hover:text-ink",
                ].join(" ")}
              >
                {link.label}
              </Link>
            );
          })}
        </div>

        <div className="ml-auto flex items-center gap-3 text-xs">
          <button
            type="button"
            onClick={rename}
            title="Who you are when claiming tickets"
            className="rounded border border-line-strong px-2 py-1 hover:bg-surface-2"
          >
            {agent ?? "Set your name"}
          </button>
          <StatusDot health={health} unreachable={unreachable} />
        </div>
      </nav>
    </header>
  );
}

function StatusDot({
  health,
  unreachable,
}: {
  health: HealthReport | null;
  unreachable: boolean;
}) {
  const { tone, label, detail } = describe(health, unreachable);

  return (
    <span className="group relative flex items-center gap-1.5">
      <span className={`h-2 w-2 rounded-full ${tone}`} aria-hidden />
      <span className="text-ink-3">{label}</span>
      {detail.length > 0 && (
        <span className="pointer-events-none absolute top-full right-0 z-30 mt-1 hidden w-80 rounded-lg border border-line bg-surface p-2 text-left text-[11px] shadow-lg group-hover:block">
          <ul className="flex list-disc flex-col gap-1 pl-4">
            {detail.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </span>
      )}
    </span>
  );
}

function describe(
  health: HealthReport | null,
  unreachable: boolean,
): { tone: string; label: string; detail: string[] } {
  if (unreachable) {
    return {
      tone: "bg-danger",
      label: "API unreachable",
      detail: ["The backend did not respond. Is it running on port 8000?"],
    };
  }
  if (!health) return { tone: "bg-ink-3", label: "checking…", detail: [] };

  // A store that stopped answering is worse than one that was never wired:
  // it means requests are silently falling back.
  const down = Object.entries(health.reachable ?? {})
    .filter(([, ok]) => ok === false)
    .map(([name]) => name);

  if (down.length > 0) {
    return {
      tone: "bg-danger",
      label: `${down.join(", ")} down`,
      detail: health.degradations,
    };
  }
  if (health.status === "degraded") {
    return {
      tone: "bg-warn",
      label: `degraded (${health.degradations.length})`,
      detail: health.degradations,
    };
  }
  return { tone: "bg-ok", label: "all systems live", detail: [] };
}
