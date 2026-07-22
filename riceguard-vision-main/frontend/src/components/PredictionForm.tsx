import { useRef, useState, type ChangeEvent } from "react";
import { AlertCircle, CheckCircle2, ImagePlus, Loader2, UploadCloud } from "lucide-react";
import { predictImage } from "@/lib/api-client";

interface PredictionFormProps {
  onResult: (result: { status: string; disease: string; confidence: number; message: string }) => void;
}

export function PredictionForm({ onResult }: PredictionFormProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handlePickFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (!selected) return;

    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setError(null);
  }

  async function handleSubmit() {
    if (!file) {
      setError("Select an image before submitting.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await predictImage(file);
      onResult(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Prediction request failed";
      setError(message);
      onResult({ status: "error", disease: "Unavailable", confidence: 0, message });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-dashed border-border bg-muted/30 p-6 text-center transition hover:border-primary/50 hover:bg-muted/50">
        <input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/jpg" className="hidden" onChange={handlePickFile} />
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary">
          <UploadCloud className="h-6 w-6" />
        </div>
        <h3 className="mt-4 text-sm font-semibold text-foreground">Upload a rice leaf image</h3>
        <p className="mt-1 text-xs text-muted-foreground">PNG, JPG, and JPEG images are supported.</p>
        <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
          <button type="button" onClick={() => fileInputRef.current?.click()} className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition hover:bg-accent">
            <ImagePlus className="h-4 w-4" /> Choose Image
          </button>
          <button type="button" onClick={handleSubmit} disabled={!file || isLoading} className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium text-foreground transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60">
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4" />}
            {isLoading ? "Sending..." : "Run Prediction"}
          </button>
        </div>
      </div>

      {error ? (
        <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      ) : null}

      <div className="rounded-xl border border-border bg-card p-4">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Image Preview</h4>
        <div className="mt-3 flex min-h-56 items-center justify-center overflow-hidden rounded-lg border border-dashed border-border bg-muted/20">
          {previewUrl ? (
            <img src={previewUrl} alt="Uploaded preview" className="h-full w-full object-contain" />
          ) : (
            <div className="flex flex-col items-center text-muted-foreground">
              <ImagePlus className="h-8 w-8" />
              <span className="mt-2 text-sm">No image selected yet</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
