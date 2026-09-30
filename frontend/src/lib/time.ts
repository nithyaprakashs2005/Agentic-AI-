/**
 * Time formatting shared by the console.
 *
 * Both an absolute and a relative form, because they answer different
 * questions: "3m ago" is what you scan a queue with, and the timestamp is what
 * you quote in an incident review. The queue shows both for that reason.
 */

/** "3m ago" / "in 4m" — coarse, and stable enough to re-render every second. */
export function relativeTime(epochMs: number, now = Date.now()): string {
  const deltaSeconds = Math.round((epochMs - now) / 1000);
  const ahead = deltaSeconds > 0;
  const seconds = Math.abs(deltaSeconds);

  const phrase =
    seconds < 45
      ? `${seconds}s`
      : seconds < 3600
        ? `${Math.round(seconds / 60)}m`
        : seconds < 86400
          ? `${Math.round(seconds / 3600)}h`
          : `${Math.round(seconds / 86400)}d`;

  return ahead ? `in ${phrase}` : `${phrase} ago`;
}

/** Local wall-clock time, for anything someone might quote or cross-reference. */
export function absoluteTime(iso: string | null): string {
  if (!iso) return "—";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return "—";
  return parsed.toLocaleString();
}
