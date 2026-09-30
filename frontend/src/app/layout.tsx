import type { Metadata } from "next";
import { NavBar } from "@/components/shared/NavBar";
import "@/styles/globals.css";

export const metadata: Metadata = {
  title: "Support Agent Console",
  description:
    "Multi-channel AI customer support: chat, tickets, escalation queue, and handoff packets.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-full antialiased">
        <NavBar />
        <main className="mx-auto max-w-6xl px-4 py-7">{children}</main>
      </body>
    </html>
  );
}
