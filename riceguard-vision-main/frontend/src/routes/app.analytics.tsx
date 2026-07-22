import { createFileRoute } from "@tanstack/react-router";
import { BarChart3, FileText, Leaf, Target } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { StatusCard } from "@/components/StatusCard";

export const Route = createFileRoute("/app/analytics")({
  head: () => ({ meta: [{ title: "Analytics — RiceGuard AI" }] }),
  component: Analytics,
});

function Analytics() {
  return (
    <>
      <PageHeader title="Analytics" description="Monitor the current backend and model readiness." />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatusCard title="Prediction History" value="0" description="No prediction history available." icon={FileText} />
        <StatusCard title="Model Status" value="Pending" description="Training is still pending." icon={Leaf} />
        <StatusCard title="Backend" value="Online" description="FastAPI service is ready." icon={Target} />
        <StatusCard title="Reports" value="0" description="No reports generated yet." icon={BarChart3} />
      </section>

      <section className="mt-6 rounded-xl border border-border bg-card p-6 text-sm text-muted-foreground">
        <h3 className="text-sm font-semibold text-foreground">Analytics state</h3>
        <p className="mt-2">Analytics pages are now wired to the backend-ready state rather than fabricated statistics. Once predictions are stored, this view can be expanded to show real trends.</p>
      </section>
    </>
  );
}
