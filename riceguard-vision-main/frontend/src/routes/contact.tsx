import { createFileRoute } from "@tanstack/react-router";
import { SiteHeader } from "@/components/SiteHeader";
import { SiteFooter } from "@/components/SiteFooter";
import { Mail, MapPin, Phone, Send } from "lucide-react";

export const Route = createFileRoute("/contact")({
  head: () => ({ meta: [{ title: "Contact — RiceGuard AI" }] }),
  component: ContactPage,
});

function ContactPage() {
  const input =
    "h-11 w-full rounded-lg border border-border bg-background px-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20";

  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-16 sm:px-6 lg:px-8">
        <div className="max-w-2xl">
          <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">Get in touch</h1>
          <p className="mt-3 text-muted-foreground">
            Reach out about the project, collaboration, or feedback.
          </p>
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-3">
          <div className="space-y-4">
            {[
              { icon: Mail, label: "Email", value: "—" },
              { icon: Phone, label: "Phone", value: "—" },
              { icon: MapPin, label: "Address", value: "—" },
            ].map((c) => (
              <div key={c.label} className="card-surface flex items-center gap-4 p-5">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <c.icon className="h-5 w-5" />
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">{c.label}</div>
                  <div className="text-sm font-medium text-foreground">{c.value}</div>
                </div>
              </div>
            ))}

            <div className="card-surface aspect-video overflow-hidden">
              <div className="flex h-full w-full items-center justify-center bg-muted/40">
                <div className="text-center">
                  <MapPin className="mx-auto h-6 w-6 text-muted-foreground" />
                  <div className="mt-2 text-sm font-medium text-foreground">Map placeholder</div>
                  <div className="text-xs text-muted-foreground">Google Map coming soon</div>
                </div>
              </div>
            </div>
          </div>

          <form
            onSubmit={(e) => e.preventDefault()}
            className="card-surface space-y-4 p-6 lg:col-span-2"
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-foreground">Name</label>
                <input className={input} placeholder="Your name" />
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-foreground">Email</label>
                <input type="email" className={input} placeholder="you@example.com" />
              </div>
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-foreground">Subject</label>
              <input className={input} placeholder="How can we help?" />
            </div>
            <div>
              <label className="mb-1.5 block text-xs font-medium text-foreground">Message</label>
              <textarea
                rows={6}
                className="w-full rounded-lg border border-border bg-background p-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20"
                placeholder="Write your message here…"
              />
            </div>
            <button
              type="submit"
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-5 py-2.5 text-sm font-semibold text-primary-foreground transition hover:bg-accent"
            >
              <Send className="h-4 w-4" /> Send message
            </button>
          </form>
        </div>
      </main>

      <SiteFooter />
    </div>
  );
}
