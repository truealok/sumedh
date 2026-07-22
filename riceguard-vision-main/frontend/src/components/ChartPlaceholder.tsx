import { BarChart3, LineChart, PieChart } from "lucide-react";

type Variant = "bar" | "line" | "pie";

interface Props {
  title: string;
  subtitle?: string;
  variant?: Variant;
  className?: string;
}

export function ChartPlaceholder({ title, subtitle, variant = "bar", className = "" }: Props) {
  const Icon = variant === "line" ? LineChart : variant === "pie" ? PieChart : BarChart3;

  return (
    <div className={`card-surface p-6 ${className}`}>
      <div className="mb-4 flex items-start justify-between">
        <div>
          <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>}
        </div>
        <span className="rounded-md bg-muted px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Awaiting data
        </span>
      </div>

      <div className="relative flex h-56 items-end justify-center overflow-hidden rounded-lg border border-dashed border-border bg-muted/30">
        {variant === "pie" ? (
          <div className="my-auto flex flex-col items-center gap-3">
            <div className="h-28 w-28 rounded-full border-8 border-border/70" />
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Icon className="h-4 w-4" />
              No Data Available
            </div>
          </div>
        ) : (
          <>
            <div className="flex h-full w-full items-end gap-3 px-6 pb-6 opacity-40">
              {[40, 65, 30, 80, 50, 70, 45].map((h, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t-md bg-border"
                  style={{ height: `${h}%` }}
                />
              ))}
            </div>
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-xs text-muted-foreground">
              <Icon className="h-5 w-5" />
              No Data Available
            </div>
          </>
        )}
      </div>
    </div>
  );
}
