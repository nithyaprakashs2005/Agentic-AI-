import { HandoffQueue } from "@/components/dashboard/HandoffQueue";
import { MetricsPanel } from "@/components/dashboard/MetricsPanel";

export const metadata = { title: "Dashboard | Support Agent" };

export default function DashboardPage() {
  return (
    <div className="flex flex-col gap-8">
      <MetricsPanel />
      <HandoffQueue />
    </div>
  );
}
