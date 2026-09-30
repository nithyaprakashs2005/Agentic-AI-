/**
 * Who is using the console.
 *
 * The queue previously claimed every ticket as the literal string "me", so the
 * assignment column could not answer the one question it exists to answer: who
 * is dealing with this. Until there is real auth, the operator names themselves
 * once and it persists locally.
 *
 * This is deliberately not a security boundary -- it identifies, it does not
 * authenticate. When SSO arrives this is the single place that changes.
 */

const STORAGE_KEY = "support-console.agent";

export function getAgentName(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(STORAGE_KEY);
  } catch {
    // Private mode, or storage disabled by policy.
    return null;
  }
}

export function setAgentName(name: string): void {
  if (typeof window === "undefined") return;
  try {
    const trimmed = name.trim();
    if (trimmed) window.localStorage.setItem(STORAGE_KEY, trimmed);
  } catch {
    // Non-fatal: the caller keeps the name for this session only.
  }
}
