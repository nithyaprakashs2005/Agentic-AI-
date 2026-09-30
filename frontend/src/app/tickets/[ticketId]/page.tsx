import { TicketDetail } from "@/components/tickets/TicketDetail";

export const metadata = { title: "Ticket | Support Agent" };

export default async function TicketPage({
  params,
}: {
  params: Promise<{ ticketId: string }>;
}) {
  // Next 16: route params are async.
  const { ticketId } = await params;
  return <TicketDetail ticketId={decodeURIComponent(ticketId)} />;
}
