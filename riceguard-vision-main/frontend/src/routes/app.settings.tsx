import { createFileRoute } from "@tanstack/react-router";
import { CheckCircle2, Database, Settings2, ShieldCheck } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";

export const Route = createFileRoute("/app/settings")({
  head: () => ({ meta: [{ title: "Settings — RiceGuard AI" }] }),
  component: SettingsPage,
});

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between border-b border-border py-4 last:border-0">
      <div className="text-sm font-medium text-foreground">{label}</div>
      <div className="text-sm text-muted-foreground">{value}</div>
    </div>
  );
}

function SettingsPage() {
  return (
    <>
      <PageHeader title="Settings" description="Project configuration and deployment readiness." />

      <div className="grid gap-5 lg:grid-cols-2">
        <section className="card-surface p-6">
          <div className="flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Configuration</h3>
          </div>
          <div className="mt-4">
            <Field label="Backend API" value="FastAPI" />
            <Field label="Frontend" value="React + TanStack Router" />
            <Field label="Model Status" value="Pending training" />
            <Field label="Dataset" value="Not required yet" />
          </div>
        </section>

        <section className="card-surface p-6">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold text-foreground">Readiness</h3>
          </div>
          <div className="mt-4 space-y-3 text-sm text-muted-foreground">
            <div className="flex items-center gap-2 rounded-lg bg-muted/40 p-3"><CheckCircle2 className="h-4 w-4 text-emerald-600" /> Backend endpoints prepared.</div>
            <div className="flex items-center gap-2 rounded-lg bg-muted/40 p-3"><CheckCircle2 className="h-4 w-4 text-emerald-600" /> Frontend integrated with API layer.</div>
            <div className="flex items-center gap-2 rounded-lg bg-muted/40 p-3"><Database className="h-4 w-4 text-primary" /> Database persistence is optional and ready for later integration.</div>
          </div>
        </section>
      </div>
    </>
  );
}
