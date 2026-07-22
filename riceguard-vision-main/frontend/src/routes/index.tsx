import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowRight,
  Upload,
  ScanLine,
  Brain,
  Zap,
  Image as ImageIcon,
  Target,
  Sprout,
  Clock,
  DollarSign,
  ShieldCheck,
  ChevronRight,
} from "lucide-react";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import heroImg from "@/assets/hero-rice.jpg";

export const Route = createFileRoute("/")({
  component: Landing,
});

const features = [
  { icon: ScanLine, title: "AI Disease Detection", desc: "Identify rice leaf diseases from a single image." },
  { icon: Brain, title: "Deep Learning", desc: "Powered by CNN and MobileNetV2 architectures." },
  { icon: Zap, title: "Fast Prediction", desc: "Get results in seconds, right on the field." },
  { icon: ImageIcon, title: "Image Classification", desc: "Robust preprocessing pipeline for real photos." },
  { icon: Target, title: "High Accuracy", desc: "Trained on curated rice leaf datasets." },
  { icon: ShieldCheck, title: "Reliable", desc: "Consistent, reproducible predictions." },
];

const steps = [
  { n: "01", title: "Upload Leaf Image", desc: "Snap or drop a rice leaf photo." },
  { n: "02", title: "Image Preprocessing", desc: "Resize, normalize, and enhance." },
  { n: "03", title: "Deep Learning Model", desc: "CNN / MobileNetV2 inference." },
  { n: "04", title: "Disease Prediction", desc: "Class, confidence, and severity." },
  { n: "05", title: "Treatment Recommendation", desc: "Actionable next steps." },
];

const benefits = [
  { icon: Sprout, title: "Farmer Friendly", desc: "Simple interface built for the field." },
  { icon: Clock, title: "Real-time Detection", desc: "Seconds from photo to diagnosis." },
  { icon: DollarSign, title: "Low Cost", desc: "No specialized hardware required." },
  { icon: Target, title: "High Accuracy", desc: "Deep learning trained on real data." },
];

function Landing() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="mx-auto grid max-w-7xl gap-12 px-4 py-20 sm:px-6 lg:grid-cols-2 lg:gap-16 lg:px-8 lg:py-28">
          <div className="flex flex-col justify-center animate-fade-in">
            <span className="inline-flex w-fit items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full bg-primary" />
              Final Year Project · AI + Agriculture
            </span>
            <h1 className="mt-5 text-4xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-6xl">
              Detect rice diseases in seconds with{" "}
              <span className="text-primary">deep learning</span>.
            </h1>
            <p className="mt-5 max-w-xl text-base text-muted-foreground sm:text-lg">
              RiceGuard AI analyzes rice leaf images using computer vision and CNN / MobileNetV2
              models to help farmers identify diseases early and act with confidence.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                to="/app/detection"
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground transition hover:bg-accent"
              >
                <Upload className="h-4 w-4" />
                Upload Image
              </Link>
              <Link
                to="/app"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-5 py-3 text-sm font-semibold text-foreground transition hover:bg-muted"
              >
                View Dashboard <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            <dl className="mt-12 grid max-w-md grid-cols-3 gap-6">
              {[
                { k: "Diseases", v: "4+" },
                { k: "Model", v: "CNN" },
                { k: "Framework", v: "TF" },
              ].map((s) => (
                <div key={s.k}>
                  <dt className="text-xs uppercase tracking-wide text-muted-foreground">{s.k}</dt>
                  <dd className="mt-1 text-2xl font-bold tracking-tight text-foreground">{s.v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="relative">
            <div className="card-surface overflow-hidden p-2">
              <img
                src={heroImg}
                alt="Healthy rice paddy field illustration"
                className="h-full w-full rounded-lg object-cover"
              />
            </div>
            <div className="absolute -bottom-6 -left-6 hidden w-56 card-surface p-4 sm:block">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <ScanLine className="h-5 w-5" />
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Live status</div>
                  <div className="text-sm font-semibold text-foreground">Awaiting Prediction</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-border bg-card/40">
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <div className="max-w-2xl">
            <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
              Built for accurate, real-world rice diagnostics
            </h2>
            <p className="mt-3 text-muted-foreground">
              A focused feature set covering the entire prediction pipeline.
            </p>
          </div>
          <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {features.map((f) => (
              <div
                key={f.title}
                className="card-surface p-6 transition hover:-translate-y-0.5 hover:shadow-lg"
              >
                <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <f.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-5 text-base font-semibold text-foreground">{f.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section>
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <div className="max-w-2xl">
            <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">How it works</h2>
            <p className="mt-3 text-muted-foreground">A clear 5-step pipeline from photo to recommendation.</p>
          </div>

          <ol className="mt-12 grid gap-5 lg:grid-cols-5">
            {steps.map((s, i) => (
              <li key={s.n} className="relative">
                <div className="card-surface h-full p-5">
                  <div className="text-xs font-semibold text-primary">Step {s.n}</div>
                  <h3 className="mt-2 text-base font-semibold text-foreground">{s.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{s.desc}</p>
                </div>
                {i < steps.length - 1 && (
                  <ChevronRight className="absolute -right-3 top-1/2 hidden h-5 w-5 -translate-y-1/2 text-border lg:block" />
                )}
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* Benefits */}
      <section className="border-t border-border bg-card/40">
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">Project benefits</h2>
          <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {benefits.map((b) => (
              <div key={b.title} className="card-surface p-6">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <b.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-4 text-base font-semibold text-foreground">{b.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground">{b.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section>
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <div className="card-surface flex flex-col items-start justify-between gap-6 p-10 sm:flex-row sm:items-center">
            <div>
              <h3 className="text-2xl font-bold tracking-tight text-foreground">
                Ready to analyze your first rice leaf?
              </h3>
              <p className="mt-1 text-sm text-muted-foreground">
                Upload an image and open the dashboard to see the results workflow.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link
                to="/app/detection"
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground transition hover:bg-accent"
              >
                <Upload className="h-4 w-4" /> Upload Image
              </Link>
              <Link
                to="/about"
                className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-5 py-3 text-sm font-semibold text-foreground transition hover:bg-muted"
              >
                Learn more
              </Link>
            </div>
          </div>
        </div>
      </section>

      <SiteFooter />
    </div>
  );
}
