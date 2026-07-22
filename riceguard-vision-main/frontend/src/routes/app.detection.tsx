import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { AlertTriangle, Sparkles, CheckCircle2 } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { PredictionForm } from "@/components/PredictionForm";

export const Route = createFileRoute("/app/detection")({
  head: () => ({ meta: [{ title: "Disease Detection — RiceGuard AI" }] }),
  component: Detection,
});

interface PredictionResult {
  status: string;
  disease: string;
  confidence: number;
  message: string;
}

function Detection() {
  const [result, setResult] = useState<PredictionResult | null>(null);

  return (
    <>
      <PageHeader
        title="Disease Detection"
        description="Upload a rice leaf image and send it to the backend for analysis."
      />

      <div className="grid gap-5 lg:grid-cols-[1.3fr_0.7fr]">
        <PredictionForm onResult={setResult} />

        <div className="space-y-5">
          <div className="card-surface p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-foreground">Prediction Result</h3>
              <span className="rounded-md bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                {result ? result.status : "Awaiting"}
              </span>
            </div>
            <dl className="mt-4 space-y-3">
              <div className="flex items-center justify-between border-b border-border/70 pb-2">
                <dt className="text-xs text-muted-foreground">Status</dt>
                <dd className="text-sm font-semibold text-foreground">{result?.status ?? "—"}</dd>
              </div>
              <div className="flex items-center justify-between border-b border-border/70 pb-2">
                <dt className="text-xs text-muted-foreground">Disease Name</dt>
                <dd className="text-sm font-semibold text-foreground">{result?.disease ?? "—"}</dd>
              </div>
              <div className="flex items-center justify-between border-b border-border/70 pb-2">
                <dt className="text-xs text-muted-foreground">Confidence</dt>
                <dd className="text-sm font-semibold text-foreground">{result ? `${result.confidence.toFixed(2)}%` : "—"}</dd>
              </div>
            </dl>
            <div className={`mt-4 rounded-lg p-3 text-sm ${result?.status === "success" ? "bg-emerald-50 text-emerald-700" : "bg-muted/50 text-muted-foreground"}`}>
              {result?.message ?? "Upload an image to receive the backend response."}
            </div>
          </div>

          <div className="card-surface p-6">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold text-foreground">Model Status</h3>
            </div>
            <div className="mt-4 rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2 text-foreground">
                {result?.status === "Model not trained" ? <AlertTriangle className="h-4 w-4 text-amber-600" /> : <CheckCircle2 className="h-4 w-4 text-emerald-600" />}
                <span>{result?.status === "Model not trained" ? "Model Not Available Yet" : "Backend integration is ready."}</span>
              </div>
              <p className="mt-2">The prediction endpoint currently returns a development response until the model is trained and the Keras file is available.</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
