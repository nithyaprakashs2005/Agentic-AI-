"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Badge,
  Button,
  Card,
  CardHeader,
  ErrorNote,
  PageHeader,
  Skeleton,
  cx,
} from "@/components/ui";
import {
  fetchEmailStatus,
  fetchTwilioStatus,
  fetchVoiceStatus,
  type EmailStatus,
  type VoiceStatus,
  type WebhookStatus,
} from "@/lib/api";

/**
 * How each channel is connected, and what it still needs.
 *
 * The agent accepts five channels and the console only ever exposed one, so
 * there was no way to tell from the UI whether SMS was wired, what Twilio
 * should be pointed at, or why inbound mail was being refused. Each card
 * answers those three questions for one channel.
 */

type State = "live" | "partial" | "not-configured";

interface ChannelCard {
  id: string;
  name: string;
  blurb: string;
  state: State;
  /** Env vars that are missing, verbatim from the backend where it reports them. */
  missing: string[];
  /** Webhook paths a provider needs to be pointed at. */
  endpoints?: Record<string, string>;
  steps?: string[];
  tryHref?: string;
  tryLabel?: string;
  note?: string;
}

const STATE_META: Record<State, { tone: "ok" | "warn" | "danger"; label: string }> = {
  live: { tone: "ok", label: "Live" },
  partial: { tone: "warn", label: "Partly configured" },
  "not-configured": { tone: "danger", label: "Not configured" },
};

export function ChannelsBoard() {
  const [voice, setVoice] = useState<VoiceStatus | null>(null);
  const [twilio, setTwilio] = useState<WebhookStatus | null>(null);
  const [email, setEmail] = useState<EmailStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [publicBase, setPublicBase] = useState("https://your-tunnel.example.com");

  useEffect(() => {
    let active = true;
    Promise.allSettled([fetchVoiceStatus(), fetchTwilioStatus(), fetchEmailStatus()])
      .then(([v, t, e]) => {
        if (!active) return;
        if (v.status === "fulfilled") setVoice(v.value);
        if (t.status === "fulfilled") setTwilio(t.value);
        if (e.status === "fulfilled") setEmail(e.value);
        if (v.status === "rejected" && t.status === "rejected" && e.status === "rejected") {
          setError("Could not reach the API to read channel status.");
        }
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, []);

  const cards: ChannelCard[] = [
    {
      id: "webchat",
      name: "Web chat",
      blurb:
        "Browser chat over SSE, with answers streamed token by token. Needs no external provider.",
      state: "live",
      missing: [],
      tryHref: "/chat",
      tryLabel: "Open chat",
    },
    {
      id: "voice",
      name: "Voice",
      blurb:
        "Spoken conversation. Runs on browser speech today; the production path is LiveKit + Pipecat with Groq Whisper and Kokoro.",
      state: voice?.enabled ? "live" : "partial",
      missing: voice?.missing_configuration ?? [],
      steps: voice?.enabled
        ? undefined
        : [
            "Browser speech works now — no setup needed.",
            "For the LiveKit path: run the LiveKit server (docker compose up livekit).",
            "Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET.",
            "Start the Pipecat worker in its own virtualenv (see voice-worker/).",
          ],
      tryHref: "/voice",
      tryLabel: "Open voice",
      note: voice?.enabled
        ? undefined
        : "Usable now through the browser; the sub-400ms pipeline is not attached.",
    },
    {
      id: "sms",
      name: "SMS",
      blurb: "Inbound SMS via Twilio, answered as TwiML and segmented to 160 characters.",
      state: twilio?.inbound_enabled ? "live" : "not-configured",
      missing: twilio?.missing_configuration ?? [],
      endpoints: twilio?.endpoints ? { sms: twilio.endpoints.sms ?? "" } : undefined,
      steps: [
        "Expose this backend publicly (a tunnel is fine for testing).",
        "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER and TWILIO_WEBHOOK_BASE_URL.",
        "In the Twilio console, set the number's inbound webhook to the URL below.",
      ],
      note: twilio?.signature_validation,
    },
    {
      id: "whatsapp",
      name: "WhatsApp",
      blurb: "Inbound WhatsApp through the Twilio sandbox or an approved sender.",
      state: twilio?.inbound_enabled ? "live" : "not-configured",
      missing: twilio?.missing_configuration ?? [],
      endpoints: twilio?.endpoints ? { whatsapp: twilio.endpoints.whatsapp ?? "" } : undefined,
      steps: [
        "Complete the SMS setup above — WhatsApp uses the same Twilio credentials.",
        "Join the Twilio WhatsApp sandbox, or use an approved sender.",
        "Point the sandbox's inbound webhook at the URL below.",
      ],
    },
    {
      id: "email",
      name: "Email",
      blurb:
        "Inbound mail via a provider webhook (Mailgun-style HMAC) or raw MIME. Replies are logged, not sent.",
      state: email?.inbound_enabled ? (email.outbound_enabled ? "live" : "partial") : "not-configured",
      missing: email?.missing_configuration ?? [],
      endpoints: email?.endpoints,
      steps: [
        "Set EMAIL_WEBHOOK_SECRET to the provider's signing key.",
        "Set EMAIL_DOMAIN and EMAIL_FROM_ADDRESS.",
        "Point the provider's inbound route at the URL below.",
      ],
      note: email?.outbound_note,
    },
  ];

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="Channels"
        description="One agent answers every channel. This is what each one needs to be connected, and what it is doing right now."
      />

      {error && <ErrorNote>{error}</ErrorNote>}

      <Card className="px-4 py-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-ink-3">
            Your public base URL — used to build the webhook addresses below
          </span>
          <input
            value={publicBase}
            onChange={(e) => setPublicBase(e.target.value)}
            spellCheck={false}
            className="w-full rounded border border-line-strong bg-surface px-2 py-1.5 font-mono text-xs"
          />
        </label>
        <p className="mt-1.5 text-xs text-ink-3">
          Twilio signs the URL it calls, so this must be the address the provider actually
          reaches — not what the app sees behind a tunnel. That is why it is configuration
          (<code>TWILIO_WEBHOOK_BASE_URL</code>) rather than something the app infers.
        </p>
      </Card>

      {loading ? (
        <Skeleton rows={4} />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {cards.map((card) => (
            <ChannelPanel key={card.id} card={card} publicBase={publicBase} />
          ))}
        </div>
      )}
    </div>
  );
}

function ChannelPanel({ card, publicBase }: { card: ChannelCard; publicBase: string }) {
  const meta = STATE_META[card.state];

  return (
    <Card tone={meta.tone} className="flex flex-col">
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            {card.name}
            <Badge tone={meta.tone} dot>
              {meta.label}
            </Badge>
          </span>
        }
        subtitle={card.blurb}
        actions={
          card.tryHref && (
            <Link href={card.tryHref}>
              <Button size="sm">{card.tryLabel ?? "Open"}</Button>
            </Link>
          )
        }
      />

      <div className="flex flex-1 flex-col gap-3 px-4 py-3">
        {card.missing.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-semibold text-ink-2">Missing configuration</p>
            <ul className="flex flex-col gap-1">
              {card.missing.map((item) => (
                <li key={item} className="text-xs text-ink-2">
                  <code className="rounded bg-surface-2 px-1 py-0.5">{item}</code>
                </li>
              ))}
            </ul>
          </div>
        )}

        {card.endpoints && Object.keys(card.endpoints).length > 0 && (
          <div>
            <p className="mb-1 text-xs font-semibold text-ink-2">Point your provider here</p>
            <div className="flex flex-col gap-1.5">
              {Object.entries(card.endpoints)
                .filter(([, path]) => path)
                .map(([label, path]) => (
                  <CopyableUrl key={label} label={label} url={`${publicBase.replace(/\/$/, "")}${path}`} />
                ))}
            </div>
          </div>
        )}

        {card.steps && (
          <details className="text-xs">
            <summary className="cursor-pointer text-ink-3">Setup steps</summary>
            <ol className="mt-1.5 flex list-decimal flex-col gap-1 pl-4 text-ink-2">
              {card.steps.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
          </details>
        )}

        {card.note && <p className="mt-auto text-xs text-ink-3">{card.note}</p>}
      </div>
    </Card>
  );
}

function CopyableUrl({ label, url }: { label: string; url: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard blocked (insecure origin, or permission denied). The URL is
      // on screen and selectable, so this is a convenience, not the only route.
      setCopied(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <span className="w-20 shrink-0 text-[11px] text-ink-3">{label}</span>
      <code
        className={cx(
          "min-w-0 flex-1 truncate rounded bg-surface-2 px-2 py-1 font-mono text-[11px]",
        )}
        title={url}
      >
        {url}
      </code>
      <Button size="sm" variant="ghost" onClick={() => void copy()}>
        {copied ? "Copied" : "Copy"}
      </Button>
    </div>
  );
}
