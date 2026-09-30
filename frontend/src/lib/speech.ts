/**
 * Browser speech, used to make the voice channel actually usable today.
 *
 * The production voice path is LiveKit + Pipecat + Groq Whisper + Kokoro, and
 * it cannot run here: Pipecat pins an `openai` major that conflicts with
 * LiteLLM, so it needs its own process, and no LiveKit credentials are set.
 * That left `/api/voice/*` with no UI at all and voice as a claim rather than
 * a feature.
 *
 * The browser already ships speech recognition and synthesis. Driving the same
 * agent over `channel: "voice"` gives a real spoken conversation -- the same
 * graph, the same voice-adapted text the TTS stage would receive -- without
 * pretending the production pipeline is attached. The page says which one is
 * running.
 *
 * Support is uneven (recognition is Chrome/Edge only), so every entry point
 * reports capability rather than throwing.
 */

// The Web Speech API is not in TypeScript's DOM lib; this is the slice used here.
interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}
interface SpeechRecognitionResult {
  0: SpeechRecognitionAlternative;
  isFinal: boolean;
  length: number;
}
interface SpeechRecognitionResultList {
  length: number;
  [index: number]: SpeechRecognitionResult;
}
interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}
interface SpeechRecognitionErrorEventLike extends Event {
  error: string;
}
interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

function recognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export interface SpeechSupport {
  recognition: boolean;
  synthesis: boolean;
  /** Plain-language reason, for the UI to show instead of a dead button. */
  note: string | null;
}

export function detectSpeechSupport(): SpeechSupport {
  if (typeof window === "undefined") {
    return { recognition: false, synthesis: false, note: null };
  }
  const recognition = recognitionCtor() !== null;
  const synthesis = "speechSynthesis" in window;

  let note: string | null = null;
  if (!recognition && !synthesis) {
    note = "This browser exposes neither speech recognition nor synthesis.";
  } else if (!recognition) {
    note =
      "This browser cannot transcribe speech. Recognition is currently Chrome and Edge only; you can still type, and replies will be spoken.";
  } else if (!synthesis) {
    note = "This browser cannot speak replies, but it can still transcribe what you say.";
  }
  return { recognition, synthesis, note };
}

export interface ListenHandlers {
  /** Fires repeatedly with the best-guess transcript so far. */
  onInterim?: (text: string) => void;
  /** Fires once with the settled transcript. */
  onFinal: (text: string) => void;
  onError?: (reason: string) => void;
  onEnd?: () => void;
}

/**
 * Listen for a single utterance.
 *
 * Returns a stop function, or null when unsupported. Deliberately
 * single-utterance: continuous mode keeps the microphone open and picks up the
 * agent's own synthesised reply, which turns into a feedback loop.
 */
export function listenOnce(handlers: ListenHandlers, lang = "en-US"): (() => void) | null {
  const Ctor = recognitionCtor();
  if (!Ctor) return null;

  const recognition = new Ctor();
  recognition.lang = lang;
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  let settled = false;

  recognition.onresult = (event) => {
    let interim = "";
    let final = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i] as SpeechRecognitionResult | undefined;
      if (!result) continue;
      if (result.isFinal) final += result[0].transcript;
      else interim += result[0].transcript;
    }
    if (interim) handlers.onInterim?.(interim);
    if (final.trim() && !settled) {
      settled = true;
      handlers.onFinal(final.trim());
    }
  };

  recognition.onerror = (event) => {
    // `no-speech` and `aborted` are ordinary outcomes of letting go of the
    // button, not failures worth showing anyone.
    if (event.error !== "no-speech" && event.error !== "aborted") {
      handlers.onError?.(describeRecognitionError(event.error));
    }
  };

  recognition.onend = () => handlers.onEnd?.();

  try {
    recognition.start();
  } catch {
    // Already running: the caller pressed twice.
    return () => recognition.abort();
  }
  return () => recognition.stop();
}

function describeRecognitionError(code: string): string {
  switch (code) {
    case "not-allowed":
    case "service-not-allowed":
      return "Microphone access was denied. Allow it in the browser's site settings.";
    case "audio-capture":
      return "No microphone was found.";
    case "network":
      return "Speech recognition needs a network connection and could not reach it.";
    default:
      return `Speech recognition failed (${code}).`;
  }
}

/** Cancel anything currently being spoken. */
export function stopSpeaking(): void {
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

/**
 * Speak text, resolving when finished.
 *
 * Chunked on sentence boundaries: several implementations truncate a single
 * long utterance, and chunking also lets playback start sooner.
 */
export function speak(
  text: string,
  options: { rate?: number; voiceURI?: string | null } = {},
): Promise<void> {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return Promise.resolve();
  }
  stopSpeaking();

  const chunks = text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (chunks.length === 0) return Promise.resolve();

  const voice = options.voiceURI
    ? window.speechSynthesis.getVoices().find((v) => v.voiceURI === options.voiceURI)
    : undefined;

  return new Promise((resolve) => {
    let index = 0;
    const next = () => {
      if (index >= chunks.length) {
        resolve();
        return;
      }
      const utterance = new SpeechSynthesisUtterance(chunks[index]);
      index += 1;
      utterance.rate = options.rate ?? 1;
      if (voice) utterance.voice = voice;
      utterance.onend = next;
      // Do not strand the caller if the engine drops an utterance.
      utterance.onerror = next;
      window.speechSynthesis.speak(utterance);
    };
    next();
  });
}

/** Installed voices, once the list has loaded (it is populated lazily). */
export function loadVoices(): Promise<SpeechSynthesisVoice[]> {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return Promise.resolve([]);
  }
  const existing = window.speechSynthesis.getVoices();
  if (existing.length > 0) return Promise.resolve(existing);

  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(window.speechSynthesis.getVoices()), 1000);
    window.speechSynthesis.onvoiceschanged = () => {
      clearTimeout(timer);
      resolve(window.speechSynthesis.getVoices());
    };
  });
}
