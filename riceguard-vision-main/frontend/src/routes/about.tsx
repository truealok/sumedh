import { createFileRoute } from "@tanstack/react-router";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { Brain, Layers, Cpu, GraduationCap } from "lucide-react";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — RiceGuard AI" },
      {
        name: "description",
        content:
          "About the RiceGuard AI Final Year Project: objectives, research gap, expected outcomes, and technology stack.",
      },
    ],
  }),
  component: AboutPage,
});

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="card-surface p-6">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-primary">{title}</h3>
      <div className="mt-3 text-sm leading-relaxed text-muted-foreground">{children}</div>
    </section>
  );
}

function AboutPage() {
  const stack = [
    { g: "Frontend", items: ["React", "TypeScript", "Vite", "Tailwind CSS"] },
    { g: "AI / Deep Learning", items: ["TensorFlow", "Keras", "OpenCV", "CNN", "MobileNetV2"] },
    { g: "Backend", items: ["To be integrated"] },
    { g: "Database", items: ["To be integrated"] },
  ];

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-16 sm:px-6 lg:px-8">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            About the Project
          </h1>
          <p className="mt-3 text-muted-foreground">
            RiceGuard AI is a Final Year Project exploring computer vision for early rice disease detection.
          </p>
        </div>

        <div className="mt-10 grid gap-5 lg:grid-cols-2">
          <Section title="Project Overview">
            An end-to-end system that classifies rice leaf images into disease categories using a
            convolutional neural network. The goal is to support farmers with rapid, low-cost diagnostics.
          </Section>
          <Section title="Objectives">
            Build a reliable image classification pipeline, deploy an accessible interface, and provide
            treatment recommendations tailored to each detected class.
          </Section>
          <Section title="Research Gap">
            Existing tools often assume ideal lab conditions. RiceGuard AI targets robust performance on
            real field photos taken with smartphones.
          </Section>
          <Section title="Expected Outcomes">
            A working prototype with a trained CNN / MobileNetV2 model, an intuitive dashboard, and
            comprehensive documentation suitable for academic evaluation.
          </Section>
        </div>

        <div className="mt-10 card-surface p-6">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold uppercase tracking-wide text-primary">
              Technology Stack
            </h3>
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {stack.map((s) => (
              <div key={s.g}>
                <div className="text-xs font-semibold text-foreground">{s.g}</div>
                <ul className="mt-2 space-y-1">
                  {s.items.map((i) => (
                    <li key={i} className="text-sm text-muted-foreground">
                      · {i}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-10 card-surface p-6">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold uppercase tracking-wide text-primary">
              Deep Learning Workflow
            </h3>
          </div>
          <div className="mt-6 grid gap-4 sm:grid-cols-5">
            {[
              "Data Collection",
              "Preprocessing",
              "Model Training (CNN)",
              "Evaluation",
              "Deployment",
            ].map((step, i) => (
              <div key={step} className="rounded-lg border border-dashed border-border bg-muted/30 p-4 text-center">
                <div className="text-xs font-semibold text-primary">Step {i + 1}</div>
                <div className="mt-1 text-sm font-medium text-foreground">{step}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-10 grid gap-5 lg:grid-cols-2">
          <div className="card-surface p-6">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold uppercase tracking-wide text-primary">Team</h3>
            </div>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {[1, 2, 3, 4].map((n) => (
                <div key={n} className="flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
                    <span className="text-xs font-semibold">M{n}</span>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-foreground">Team Member —</div>
                    <div className="text-xs text-muted-foreground">Role placeholder</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="card-surface p-6">
            <div className="flex items-center gap-2">
              <GraduationCap className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold uppercase tracking-wide text-primary">
                Faculty Guide
              </h3>
            </div>
            <div className="mt-4 flex items-center gap-4 rounded-lg border border-border bg-muted/30 p-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                <GraduationCap className="h-5 w-5" />
              </div>
              <div>
                <div className="text-sm font-medium text-foreground">Guide Name —</div>
                <div className="text-xs text-muted-foreground">Department placeholder</div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <SiteFooter />
    </div>
  );
}
