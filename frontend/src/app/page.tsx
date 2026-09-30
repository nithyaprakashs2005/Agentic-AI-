import Link from "next/link";
import { Badge, Card } from "@/components/ui";

const DESTINATIONS = [
  {
    href: "/chat",
    title: "Chat",
    body: "Talk to the agent and watch it answer token by token. Switch channel to see the same reply adapted for voice or segmented for SMS.",
    tag: "Live",
  },
  {
    href: "/voice",
    title: "Voice",
    body: "Hold the microphone and speak. Replies come back as plain spoken sentences and are read aloud.",
    tag: "Live",
  },
  {
    href: "/channels",
    title: "Channels",
    body: "What each of the five channels needs to be connected, the webhook URLs to hand your provider, and what is wired right now.",
  },
  {
    href: "/tickets",
    title: "Tickets",
    body: "Every conversation the agent has handled, with its transcript, cited sources, and how it was resolved.",
  },
  {
    href: "/dashboard",
    title: "Dashboard",
    body: "Resolution and escalation metrics, and the handoff queue ordered by which deadline breaches first.",
  },
];

export default function Home() {
  return (
    <div className="flex flex-col gap-8">
      <section className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight">Customer Support AI Agent</h1>
        <p className="mt-2 text-sm text-ink-2">
          One LangGraph orchestrator behind every channel. Answers are grounded in cited
          knowledge base sources over hybrid retrieval; sentiment and confidence decide when a
          human is needed, and anything escalated arrives with a full context packet and a
          response deadline.
        </p>
      </section>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {DESTINATIONS.map((item) => (
          <Link key={item.href} href={item.href} className="group">
            <Card className="h-full px-4 py-3.5 transition group-hover:border-line-strong">
              <div className="flex items-center gap-2">
                <h2 className="font-medium">{item.title}</h2>
                {item.tag && <Badge tone="ok">{item.tag}</Badge>}
                <span
                  className="ml-auto text-ink-3 transition group-hover:translate-x-0.5"
                  aria-hidden
                >
                  →
                </span>
              </div>
              <p className="mt-1.5 text-sm text-ink-2">{item.body}</p>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
