import { createFileRoute } from "@tanstack/react-router";
import { Activity, AlertTriangle, Leaf, Target } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { StatusCard } from "@/components/StatusCard";
import { useApiStatus } from "@/hooks/use-api-status";

export const Route = createFileRoute("/app/")({
  head: () => ({ meta: [{ title: "Dashboard — RiceGuard AI" }] }),
  component: Dashboard,
});

function Dashboard() {
  const { health, modelInfo, loading, error } = useApiStatus();

  return (
    <>
      <PageHeader title="Dashboard" description="Overview of the current backend and model readiness." />

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatusCard title="API Status" value={loading ? "Checking" : health?.status ?? "Offline"} description={error ?? "Backend health check"} icon={Activity} />
        <StatusCard title="Model Loaded" value={loading ? "—" : health?.model_loaded ? "Yes" : "No"} description={modelInfo?.model_name ?? "Model pending"} icon={Leaf} />
        <StatusCard title="Prediction State" value={modelInfo ? "Ready" : "Pending"} description="Upload an image to test the endpoint" icon={AlertTriangle} />
        <StatusCard title="Classes" value={modelInfo ? `${modelInfo.number_of_classes}` : "—"} description="Number of disease classes configured" icon={Target} />
      </section>

      <section className="mt-6">
        <div className="card-surface overflow-hidden">
          <div className="flex items-center justify-between border-b border-border px-6 py-4">
            <div>
              <h3 className="text-sm font-semibold text-foreground">Prediction Activity</h3>
              <p className="text-xs text-muted-foreground">The project is currently waiting for the first real prediction.</p>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-border bg-muted/40 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th className="px-6 py-3 font-medium">Date</th>
                  <th className="px-6 py-3 font-medium">Disease</th>
                  <th className="px-6 py-3 font-medium">Confidence</th>
                  <th className="px-6 py-3 font-medium">Status</th>
                  <th className="px-6 py-3 font-medium">Result</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td colSpan={5} className="px-6 py-14">
                    <EmptyState title="No Prediction History" description="Upload an image to create the first real prediction entry." />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </>
  );
}
