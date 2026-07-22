import { createFileRoute } from "@tanstack/react-router";
import { FileText, FileDown } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";

export const Route = createFileRoute("/app/reports")({
  head: () => ({ meta: [{ title: "Reports — RiceGuard AI" }] }),
  component: Reports,
});

function Reports() {
  return (
    <>
      <PageHeader
        title="Reports"
        description="Review prediction summaries once the backend stores results."
        actions={
          <button
            disabled
            className="inline-flex cursor-not-allowed items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground opacity-60"
          >
            <FileDown className="h-4 w-4" /> Generate Report
          </button>
        }
      />

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="card-surface overflow-hidden lg:col-span-2">
          <div className="border-b border-border px-6 py-4">
            <h3 className="text-sm font-semibold text-foreground">Recent Reports</h3>
            <p className="text-xs text-muted-foreground">No reports are available yet because the model is still pending.</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-border bg-muted/40 text-left text-xs uppercase tracking-wide text-muted-foreground">
                <tr>
                  <th className="px-6 py-3 font-medium">Name</th>
                  <th className="px-6 py-3 font-medium">Range</th>
                  <th className="px-6 py-3 font-medium">Created</th>
                  <th className="px-6 py-3 font-medium">Status</th>
                  <th className="px-6 py-3 font-medium">Action</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td colSpan={5} className="px-6 py-14">
                    <EmptyState icon={FileText} title="No Reports Generated" description="Prediction reports will appear here once results are available." />
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="card-surface p-6">
          <h3 className="text-sm font-semibold text-foreground">Report Preview</h3>
          <p className="text-xs text-muted-foreground">A live preview will appear after a prediction is stored.</p>
          <div className="mt-4 flex aspect-[3/4] flex-col items-center justify-center rounded-lg border border-dashed border-border bg-muted/30 text-center">
            <FileText className="h-8 w-8 text-muted-foreground" />
            <div className="mt-3 text-sm font-medium text-foreground">No report selected</div>
            <div className="mt-1 text-xs text-muted-foreground">Awaiting prediction history</div>
          </div>
        </div>
      </div>
    </>
  );
}
