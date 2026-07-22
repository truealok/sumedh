import { Leaf } from "lucide-react";

export function SiteFooter() {
  const cols: { title: string; items: string[] }[] = [
    { title: "Team Members", items: ["—", "—", "—"] },
    { title: "Project Guide", items: ["—"] },
    { title: "College", items: ["—"] },
    { title: "Contact", items: ["—", "—"] },
  ];

  return (
    <footer className="border-t border-border bg-card">
      <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6 lg:px-8">
        <div className="grid gap-10 md:grid-cols-5">
          <div className="md:col-span-1">
            <div className="flex items-center gap-2">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Leaf className="h-5 w-5" />
              </span>
              <span className="text-base font-semibold tracking-tight text-foreground">
                RiceGuard <span className="text-primary">AI</span>
              </span>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">
              A Final Year Project on AI-powered rice disease detection.
            </p>
          </div>
          {cols.map((c) => (
            <div key={c.title}>
              <h4 className="text-sm font-semibold text-foreground">{c.title}</h4>
              <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                {c.items.map((it, i) => (
                  <li key={i}>{it}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-border pt-6 text-xs text-muted-foreground sm:flex-row sm:items-center">
          <span>© {new Date().getFullYear()} RiceGuard AI. All rights reserved.</span>
          <span>Built with React · TypeScript · TensorFlow · MobileNetV2</span>
        </div>
      </div>
    </footer>
  );
}
