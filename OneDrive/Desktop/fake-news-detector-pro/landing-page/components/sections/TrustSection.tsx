"use client";

import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Cpu, FileText, ShieldCheck, Sparkles, Clock } from "lucide-react";
import GlassCard from "@/components/ui/glass-card";
import Container from "@/components/ui/container";
import SectionHeading from "@/components/ui/section-heading";
import { fadeInUp } from "@/components/ui/motion";

function useCountUp(target: number, durationMs: number) {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let raf = 0;
    const start = performance.now();

    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(target * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);

  return value;
}

type Metric = {
  label: string;
  icon: React.ReactNode;
  value: number;
  suffix?: string;
  prefix?: string;
};

function formatMetric(m: Metric, v: number) {
  const digits = m.label.includes("Accuracy") ? 2 : m.label.includes("Precision") ? 0 : 0;
  if (m.value >= 1000) return `${Math.round(v).toLocaleString()}${m.suffix ?? ""}`;
  if (digits === 2) return `${v.toFixed(2)}${m.suffix ?? ""}`;
  if (digits === 0 && m.label.includes("Precision")) return `${Math.round(v)}${m.suffix ?? "%"}`;
  return `${Math.round(v)}${m.suffix ?? ""}`;
}

export default function TrustSection() {
  const metrics: Metric[] = useMemo(
    () => [
      { label: "Accuracy", icon: <Sparkles className="h-4 w-4 text-cyan-300" />, value: 98.92, suffix: "%" },
      { label: "Test Samples", icon: <FileText className="h-4 w-4 text-indigo-300" />, value: 8980 },
      { label: "Precision", icon: <ShieldCheck className="h-4 w-4 text-fuchsia-300" />, value: 99, suffix: "%" },
      { label: "OCR Enabled", icon: <Cpu className="h-4 w-4 text-cyan-300" />, value: 100, suffix: "" },
      { label: "Fast Prediction (<2 sec)", icon: <Clock className="h-4 w-4 text-indigo-300" />, value: 2, suffix: "s" },
    ],
    []
  );

  const shouldReduceMotion =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  return (
    <section className="py-12 sm:py-16">
      <Container>
        <div className="flex flex-col gap-8">
          <SectionHeading
            eyebrow="TRUST"
            title="Performance you can verify"
            description="Measured offline and designed for practical workflows: fast predictions, OCR support, and confidence scores you can interpret."
          />

          <GlassCard className="p-5 sm:p-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
              {metrics.map((m, idx) => (
                <motion.div
                  key={m.label}
                  variants={fadeInUp}
                  initial="hidden"
                  whileInView="show"
                  viewport={{ once: true, margin: "-80px" }}
                  transition={{ delay: idx * 0.05 }}
                  className="rounded-xl border border-white/10 bg-black/10 p-4"
                >
                  <div className="flex items-center gap-2">
                    {m.icon}
                    <p className="text-xs font-semibold text-white/65">{m.label}</p>
                  </div>
                  <div className="mt-2">
                    {shouldReduceMotion ? (
                      <p className="text-2xl font-semibold text-white">
                        {formatMetric(m, m.value)}
                      </p>
                    ) : (
                      <CountValue metric={m} />
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </GlassCard>
        </div>
      </Container>
    </section>
  );
}

function CountValue({ metric }: { metric: Metric }) {
  const v = useCountUp(metric.value, 900);
  if (metric.label === "OCR Enabled") return <p className="text-2xl font-semibold text-white">OCR</p>;
  if (metric.label.startsWith("Fast Prediction")) return <p className="text-2xl font-semibold text-white">{v.toFixed(0)}s</p>;
  if (metric.label === "Test Samples")
    return <p className="text-2xl font-semibold text-white">{Math.round(v).toLocaleString()}</p>;
  if (metric.label === "Accuracy")
    return <p className="text-2xl font-semibold text-white">{v.toFixed(2)}%</p>;
  if (metric.label === "Precision")
    return <p className="text-2xl font-semibold text-white">{Math.round(v)}%</p>;

  return <p className="text-2xl font-semibold text-white">{formatMetric(metric, v)}</p>;
}

