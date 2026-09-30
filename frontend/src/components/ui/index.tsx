/**
 * The console's shared primitives.
 *
 * These exist because every screen previously re-derived the same decisions --
 * a card's border, a badge's padding, what an empty state looks like -- and
 * quietly disagreed with the others. One definition each, so a status colour
 * means the same thing on the queue as on the channels page.
 */

import type { ReactNode } from "react";

export type Tone = "neutral" | "ok" | "warn" | "danger" | "info";

const TONE_SOFT: Record<Tone, string> = {
  neutral: "bg-surface-2 text-ink-2",
  ok: "bg-ok-soft text-ok",
  warn: "bg-warn-soft text-warn",
  danger: "bg-danger-soft text-danger",
  info: "bg-info-soft text-info",
};

const TONE_DOT: Record<Tone, string> = {
  neutral: "bg-ink-3",
  ok: "bg-ok",
  warn: "bg-warn",
  danger: "bg-danger",
  info: "bg-info",
};

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/* --- Layout ------------------------------------------------------------- */

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-5 flex flex-wrap items-start gap-3">
      <div className="min-w-0 flex-1">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="mt-1 max-w-2xl text-sm text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Card({
  children,
  className,
  tone = "neutral",
  as: Tag = "section",
}: {
  children: ReactNode;
  className?: string;
  /** Colours the border only — the surface stays neutral so text keeps contrast. */
  tone?: Tone;
  as?: "section" | "article" | "div";
}) {
  const border =
    tone === "neutral"
      ? "border-line"
      : { ok: "border-ok", warn: "border-warn", danger: "border-danger", info: "border-info" }[
          tone
        ];
  return (
    <Tag
      className={cx(
        "rounded-xl border bg-surface shadow-sm",
        tone === "neutral" ? border : `${border}/60`,
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export function CardHeader({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
      <div className="min-w-0 flex-1">
        <h2 className="text-sm font-semibold">{title}</h2>
        {subtitle && <p className="mt-0.5 text-xs text-ink-3">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

/* --- Atoms -------------------------------------------------------------- */

export function Badge({
  children,
  tone = "neutral",
  dot = false,
}: {
  children: ReactNode;
  tone?: Tone;
  dot?: boolean;
}) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium",
        TONE_SOFT[tone],
      )}
    >
      {dot && <span className={cx("h-1.5 w-1.5 rounded-full", TONE_DOT[tone])} aria-hidden />}
      {children}
    </span>
  );
}

export function StatusDot({ tone, label }: { tone: Tone; label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cx("h-2 w-2 shrink-0 rounded-full", TONE_DOT[tone])} aria-hidden />
      {label && <span className="text-xs text-ink-2">{label}</span>}
    </span>
  );
}

type ButtonProps = {
  children: ReactNode;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md";
} & React.ButtonHTMLAttributes<HTMLButtonElement>;

export function Button({
  children,
  variant = "secondary",
  size = "md",
  className,
  ...rest
}: ButtonProps) {
  const variants = {
    primary: "bg-ink text-canvas hover:opacity-90",
    secondary: "border border-line-strong hover:bg-surface-2",
    ghost: "hover:bg-surface-2",
    danger: "bg-danger text-white hover:opacity-90",
  } as const;
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition disabled:cursor-not-allowed disabled:opacity-40",
        size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-2 text-sm",
        variants[variant],
        className,
      )}
    >
      {children}
    </button>
  );
}

/* --- States ------------------------------------------------------------- */

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-dashed border-line-strong px-6 py-12 text-center">
      <p className="text-sm font-medium">{title}</p>
      {hint && <p className="mx-auto mt-1 max-w-md text-sm text-ink-3">{hint}</p>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger"
    >
      {children}
    </p>
  );
}

/** Skeleton rows, so a slow list does not collapse the layout as it loads. */
export function Skeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="flex flex-col gap-2" aria-hidden>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-16 animate-pulse rounded-xl bg-surface-2" />
      ))}
    </div>
  );
}

/* --- Data display ------------------------------------------------------- */

export function Stat({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: Tone;
}) {
  const valueTone =
    tone === "neutral"
      ? ""
      : { ok: "text-ok", warn: "text-warn", danger: "text-danger", info: "text-info" }[tone];
  return (
    <Card className="px-4 py-3">
      <p className="text-xs text-ink-3">{label}</p>
      <p className={cx("mt-1 text-2xl font-semibold tabular-nums tracking-tight", valueTone)}>
        {value}
      </p>
      {hint && <p className="mt-0.5 text-xs text-ink-3">{hint}</p>}
    </Card>
  );
}

export function DefinitionRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5 py-1 sm:flex-row sm:gap-3">
      <dt className="w-40 shrink-0 text-xs text-ink-3">{label}</dt>
      <dd className="min-w-0 flex-1 text-sm break-words">{children}</dd>
    </div>
  );
}
