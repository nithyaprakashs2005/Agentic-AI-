"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  DefinitionRow,
  ErrorNote,
  PageHeader,
  StatusDot,
  cx,
} from "@/components/ui";
import { fetchVoiceStatus, previewVoice, streamMessage, type VoiceStatus } from "@/lib/api";
import {
  detectSpeechSupport,
  listenOnce,
  loadVoices,
  speak,
  stopSpeaking,
  type SpeechSupport,
} from "@/lib/speech";

interface VoiceTurn {
  id: string;
  role: "customer" | "agent";
  text: string;
  interim?: boolean;
}

type Phase = "idle" | "listening" | "thinking" | "speaking";

const PHASE_LABEL: Record<Phase, string> = {
  idle: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

export function VoiceConsole() {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  const [support, setSupport] = useState<SpeechSupport | null>(null);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [voiceURI, setVoiceURI] = useState<string | null>(null);
  const [rate, setRate] = useState(1);
  const [muted, setMuted] = useState(false);

  const [phase, setPhase] = useState<Phase>("idle");
  const [turns, setTurns] = useState<VoiceTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [typed, setTyped] = useState("");

  const conversationRef = useRef<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const stopListeningRef = useRef<(() => void) | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const seq = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const nextId = () => `v-${(seq.current += 1)}`;

  /* eslint-disable react-hooks/set-state-in-effect -- browser speech APIs and
     the installed-voice list are unreadable during SSR, so capability can only
     be detected after hydration. */
  useEffect(() => {
    let active = true;
    fetchVoiceStatus()
      .then((s) => active && setStatus(s))
      .catch(() => active && setStatus(null));
    setSupport(detectSpeechSupport());
    loadVoices().then((list) => {
      if (!active) return;
      setVoices(list);
      // Prefer an English voice; the agent replies in English.
      const preferred = list.find((v) => v.lang.startsWith("en")) ?? list[0];
      if (preferred) setVoiceURI(preferred.voiceURI);
    });
    return () => {
      active = false;
      stopSpeaking();
    };
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, phase]);

  /** Run one turn: send text, stream the reply, then speak it. */
  const ask = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message) return;

      stopSpeaking();
      setError(null);
      setPhase("thinking");
      setTurns((prior) => [
        ...prior.filter((t) => !t.interim),
        { id: nextId(), role: "customer", text: message },
      ]);

      const controller = new AbortController();
      abortRef.current = controller;

      let liveId: string | null = null;
      let streamed = "";
      let finalText = "";

      try {
        for await (const event of streamMessage(
          {
            message,
            channel: "voice", // the backend strips markdown and shortens for speech
            customer_id: "cust_demo",
            conversation_id: conversationRef.current,
          },
          controller.signal,
        )) {
          if (event.type === "start") {
            conversationRef.current = event.conversation_id;
            setConversationId(event.conversation_id);
          } else if (event.type === "delta") {
            streamed += event.text;
            if (liveId === null) {
              liveId = nextId();
              const id = liveId;
              setTurns((prior) => [...prior, { id, role: "agent", text: streamed }]);
            } else {
              const id = liveId;
              const snapshot = streamed;
              setTurns((prior) =>
                prior.map((t) => (t.id === id ? { ...t, text: snapshot } : t)),
              );
            }
          } else if (event.type === "message") {
            // Voice replies come back as plain sentences; an array would be SMS
            // segmentation, which reads fine joined.
            finalText = Array.isArray(event.reply) ? event.reply.join(" ") : event.reply;
            const settled = { id: liveId ?? nextId(), role: "agent" as const, text: finalText };
            setTurns((prior) =>
              liveId !== null
                ? prior.map((t) => (t.id === liveId ? settled : t))
                : [...prior, settled],
            );
          } else if (event.type === "error") {
            setError(event.detail);
          }
        }
      } catch (caught) {
        if (!controller.signal.aborted) {
          setError("Could not reach the agent. Is the API running?");
        }
      } finally {
        abortRef.current = null;
      }

      if (finalText && !muted && support?.synthesis) {
        setPhase("speaking");
        await speak(finalText, { rate, voiceURI });
      }
      setPhase("idle");
    },
    [muted, rate, voiceURI, support?.synthesis],
  );

  const startListening = useCallback(() => {
    setError(null);
    stopSpeaking();
    setPhase("listening");

    const stop = listenOnce({
      onInterim: (text) =>
        setTurns((prior) => [
          ...prior.filter((t) => !t.interim),
          { id: "interim", role: "customer", text, interim: true },
        ]),
      onFinal: (text) => {
        setTurns((prior) => prior.filter((t) => !t.interim));
        void ask(text);
      },
      onError: (reason) => {
        setError(reason);
        setPhase("idle");
      },
      onEnd: () => {
        stopListeningRef.current = null;
        // Only drop back to idle if no turn started; `ask` owns the phase after.
        setPhase((current) => (current === "listening" ? "idle" : current));
        setTurns((prior) => prior.filter((t) => !t.interim));
      },
    });

    if (!stop) {
      setError("Speech recognition is not available in this browser.");
      setPhase("idle");
      return;
    }
    stopListeningRef.current = stop;
  }, [ask]);

  const stopEverything = useCallback(() => {
    stopListeningRef.current?.();
    stopListeningRef.current = null;
    abortRef.current?.abort();
    stopSpeaking();
    setPhase("idle");
  }, []);

  const busy = phase !== "idle";

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Voice"
        description="Speak to the agent and hear it answer. Turns run through the same graph as chat, on the voice channel, so replies come back as plain spoken sentences with no markdown."
        actions={<PipelineBadge status={status} />}
      />

      {support?.note && (
        <Card tone="warn" className="px-4 py-3">
          <p className="text-sm">{support.note}</p>
        </Card>
      )}

      <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
        <Card className="flex flex-col">
          <CardHeader
            title="Conversation"
            subtitle={conversationId ?? "Not started"}
            actions={
              <div className="flex items-center gap-2">
                <StatusDot
                  tone={phase === "idle" ? "neutral" : phase === "listening" ? "info" : "ok"}
                  label={PHASE_LABEL[phase]}
                />
              </div>
            }
          />

          <div className="h-[22rem] overflow-y-auto px-4 py-4" aria-live="polite">
            {turns.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                <p className="text-sm text-ink-2">
                  {support?.recognition
                    ? "Press and hold the microphone, or type below."
                    : "Type below — replies will be read aloud."}
                </p>
                <p className="text-xs text-ink-3">
                  Try: “Where is my order O R D eight four zero two one?”
                </p>
              </div>
            ) : (
              <ul className="flex flex-col gap-3">
                {turns.map((turn) => (
                  <li
                    key={turn.id}
                    className={turn.role === "customer" ? "flex justify-end" : "flex justify-start"}
                  >
                    <div
                      className={cx(
                        "max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm whitespace-pre-wrap",
                        turn.role === "customer"
                          ? "bg-ink text-canvas"
                          : "bg-surface-2 text-ink",
                        turn.interim && "opacity-60 italic",
                      )}
                    >
                      {turn.text}
                    </div>
                  </li>
                ))}
              </ul>
            )}
            <div ref={bottom} />
          </div>

          {error && <div className="px-4 pb-3"><ErrorNote>{error}</ErrorNote></div>}

          <div className="flex flex-wrap items-center gap-2 border-t border-line p-3">
            <button
              type="button"
              onMouseDown={startListening}
              onMouseUp={() => stopListeningRef.current?.()}
              onTouchStart={(e) => {
                e.preventDefault();
                startListening();
              }}
              onTouchEnd={() => stopListeningRef.current?.()}
              disabled={!support?.recognition || (busy && phase !== "listening")}
              aria-label="Hold to talk"
              className={cx(
                "flex h-12 w-12 shrink-0 items-center justify-center rounded-full transition disabled:opacity-40",
                phase === "listening"
                  ? "listening bg-accent text-white"
                  : "border border-line-strong hover:bg-surface-2",
              )}
            >
              <MicIcon />
            </button>

            <form
              className="flex min-w-0 flex-1 gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                const text = typed;
                setTyped("");
                void ask(text);
              }}
            >
              <label htmlFor="voice-input" className="sr-only">
                Message
              </label>
              <input
                id="voice-input"
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                placeholder={support?.recognition ? "…or type instead" : "Type your question"}
                disabled={busy}
                className="min-w-0 flex-1 rounded-lg border border-line-strong bg-surface px-3 py-2 text-sm outline-none disabled:opacity-60"
              />
              {busy ? (
                <Button type="button" onClick={stopEverything}>
                  Stop
                </Button>
              ) : (
                <Button type="submit" variant="primary" disabled={!typed.trim()}>
                  Send
                </Button>
              )}
            </form>
          </div>
        </Card>

        <div className="flex flex-col gap-5">
          <Card>
            <CardHeader title="Playback" />
            <div className="flex flex-col gap-3 px-4 py-3">
              <label className="flex items-center justify-between gap-2 text-sm">
                <span>Speak replies</span>
                <input
                  type="checkbox"
                  checked={!muted}
                  onChange={(e) => {
                    setMuted(!e.target.checked);
                    if (!e.target.checked) stopSpeaking();
                  }}
                  className="h-4 w-4"
                />
              </label>

              <label className="flex flex-col gap-1 text-sm">
                <span className="text-xs text-ink-3">Voice</span>
                <select
                  value={voiceURI ?? ""}
                  onChange={(e) => setVoiceURI(e.target.value || null)}
                  disabled={voices.length === 0}
                  className="rounded border border-line-strong bg-surface px-2 py-1 text-sm"
                >
                  {voices.length === 0 && <option>No voices installed</option>}
                  {voices.map((v) => (
                    <option key={v.voiceURI} value={v.voiceURI}>
                      {v.name} ({v.lang})
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-col gap-1 text-sm">
                <span className="text-xs text-ink-3">Rate — {rate.toFixed(1)}×</span>
                <input
                  type="range"
                  min={0.6}
                  max={1.6}
                  step={0.1}
                  value={rate}
                  onChange={(e) => setRate(Number(e.target.value))}
                />
              </label>
            </div>
          </Card>

          <VoicePreview />
        </div>
      </div>

      <PipelineStatus status={status} />
    </div>
  );
}

function PipelineBadge({ status }: { status: VoiceStatus | null }) {
  if (!status) return <Badge tone="neutral">status unknown</Badge>;
  if (status.enabled) {
    return (
      <Badge tone="ok" dot>
        LiveKit pipeline live
      </Badge>
    );
  }
  // Keys set but nothing listening is worth distinguishing from no keys at all:
  // the first is a server that needs starting, the second is unconfigured.
  const detail = status.credentials_configured
    ? "LiveKit server not running"
    : "LiveKit not configured";
  return (
    <Badge tone="info" dot>
      Browser speech · {detail}
    </Badge>
  );
}

/**
 * What is actually running versus what the design calls for.
 *
 * Stated plainly rather than hidden: the browser path is a real, working voice
 * channel, but it is not the sub-400ms Groq/Kokoro pipeline the spec budgets
 * for, and conflating the two would misrepresent what has been built.
 */
function PipelineStatus({ status }: { status: VoiceStatus | null }) {
  return (
    <Card>
      <CardHeader
        title="Voice pipeline"
        subtitle="What is running now, and what the production path still needs"
      />
      <div className="grid gap-4 px-4 py-3 sm:grid-cols-2">
        <div>
          <p className="mb-1 text-xs font-semibold text-ink-2">Running now</p>
          <dl>
            <DefinitionRow label="Transcription">
              Browser speech recognition
            </DefinitionRow>
            <DefinitionRow label="Reasoning">
              The same LangGraph agent as chat, on <code>channel: voice</code>
            </DefinitionRow>
            <DefinitionRow label="Speech">Browser speech synthesis</DefinitionRow>
          </dl>
        </div>

        <div>
          <p className="mb-1 text-xs font-semibold text-ink-2">Production path</p>
          <dl>
            <DefinitionRow label="Transport">
              LiveKit —{" "}
              {status?.transport.reachable ? (
                <Badge tone="ok">server reachable</Badge>
              ) : status?.transport.configured ? (
                <Badge tone="warn">keys set, server not running</Badge>
              ) : (
                <Badge tone="warn">not configured</Badge>
              )}
            </DefinitionRow>
            <DefinitionRow label="STT">
              {status?.stt.model ?? "whisper-large-v3"} —{" "}
              {status?.stt.configured ? (
                <Badge tone="ok">configured</Badge>
              ) : (
                <Badge tone="warn">needs GROQ_API_KEY</Badge>
              )}
            </DefinitionRow>
            <DefinitionRow label="TTS">
              Kokoro-82M ({status?.tts.voice ?? "af_heart"}) —{" "}
              {status?.tts.model_present ? (
                <Badge tone="ok">model present</Badge>
              ) : (
                <Badge tone="warn">model not downloaded</Badge>
              )}
            </DefinitionRow>
            <DefinitionRow label="Pipeline">
              Pipecat, in its own process — <Badge tone="warn">not started</Badge>
            </DefinitionRow>
          </dl>
        </div>
      </div>
      {status?.missing_configuration && status.missing_configuration.length > 0 && (
        <div className="border-t border-line px-4 py-3">
          <p className="text-xs text-ink-3">
            Set these to enable the LiveKit path:{" "}
            {status.missing_configuration.map((m) => (
              <code key={m} className="mr-2 rounded bg-surface-2 px-1 py-0.5 text-[11px]">
                {m}
              </code>
            ))}
          </p>
        </div>
      )}
    </Card>
  );
}

/**
 * The voice adapter, visible.
 *
 * This is the transform the TTS stage receives, so it is worth being able to
 * check: markdown, URLs and bare symbols read badly aloud, and this is where
 * they are removed.
 */
function VoicePreview() {
  const [text, setText] = useState(
    "**Refunds** take 3-5 days. See https://kb.example.com/refunds or email us at help@example.com for $50+ orders.",
  );
  const [result, setResult] = useState<{ spoken: string; checks: Record<string, boolean> } | null>(
    null,
  );
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      const data = await previewVoice(text);
      setResult({ spoken: data.spoken, checks: data.checks });
    } catch {
      setResult(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader
        title="Voice rendering"
        subtitle="How written text is adapted before it is spoken"
      />
      <div className="flex flex-col gap-2 px-4 py-3">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          className="w-full rounded border border-line-strong bg-surface px-2 py-1.5 text-xs"
        />
        <Button size="sm" onClick={() => void run()} disabled={busy}>
          {busy ? "Rendering…" : "Render for speech"}
        </Button>

        {result && (
          <div className="flex flex-col gap-2">
            <p className="rounded bg-surface-2 px-2 py-1.5 text-xs">{result.spoken}</p>
            <div className="flex flex-wrap gap-1">
              {Object.entries(result.checks).map(([name, failed]) => (
                <Badge key={name} tone={failed ? "danger" : "ok"}>
                  {name.replace(/_/g, " ")}: {failed ? "present" : "clean"}
                </Badge>
              ))}
            </div>
            <Button size="sm" variant="ghost" onClick={() => void speak(result.spoken)}>
              Hear it
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}

function MicIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      aria-hidden
    >
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v4" />
    </svg>
  );
}
