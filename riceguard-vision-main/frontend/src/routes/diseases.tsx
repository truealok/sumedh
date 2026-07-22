import { createFileRoute } from "@tanstack/react-router";
import { Leaf, Bug, Droplets, Sprout } from "lucide-react";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";

export const Route = createFileRoute("/diseases")({
  head: () => ({ meta: [{ title: "Disease Information — RiceGuard AI" }] }),
  component: DiseasesPage,
});

const diseases = [
  { name: "Leaf Blast", icon: Bug, tag: "Fungal" },
  { name: "Brown Spot", icon: Droplets, tag: "Fungal" },
  { name: "Bacterial Blight", icon: Sprout, tag: "Bacterial" },
  { name: "Healthy Leaf", icon: Leaf, tag: "Healthy" },
] as const;

function DiseasesPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-16 sm:px-6 lg:px-8">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Rice Disease Information
          </h1>
          <p className="mt-3 text-muted-foreground">
            Reference cards for the disease classes the model can identify. Content will be populated
            once the knowledge base is connected.
          </p>
        </div>

        <div className="mt-10 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {diseases.map((d) => (
            <article key={d.name} className="card-surface overflow-hidden transition hover:-translate-y-0.5">
              <div className="flex aspect-[4/3] items-center justify-center border-b border-border bg-muted/40">
                <d.icon className="h-10 w-10 text-muted-foreground" />
              </div>
              <div className="p-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-semibold text-foreground">{d.name}</h3>
                  <span className="rounded-md bg-primary/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary">
                    {d.tag}
                  </span>
                </div>
                <dl className="mt-4 space-y-3 text-sm">
                  {["Description", "Symptoms", "Treatment"].map((k) => (
                    <div key={k}>
                      <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {k}
                      </dt>
                      <dd className="mt-1 rounded-md bg-muted/60 px-2.5 py-2 text-xs text-muted-foreground">
                        No Data Available
                      </dd>
                    </div>
                  ))}
                </dl>
              </div>
            </article>
          ))}
        </div>
      </main>

      <SiteFooter />
    </div>
  );
}
