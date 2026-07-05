"use client";

import React from "react";
import { motion } from "framer-motion";
import Container from "@/components/ui/container";
import GlassCard from "@/components/ui/glass-card";
import SectionHeading from "@/components/ui/section-heading";
import { fadeInUp } from "@/components/ui/motion";

const metrics = [
  { label: "Accuracy", value: "98.92%", detail: "Overall classification correctness across test data." },
  { label: "Precision", value: "99%", detail: "Minimizes false alarms by favoring correct positive predictions." },
  { label: "Recall", value: "97.8%", detail: "Captures a high share of true misinformation patterns." },
  { label: "F1 Score", value: "98.3%", detail: "Balance between precision and recall for stable performance." },
];

export default function ModelPerformance() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <SectionHeading
          eyebrow="MODEL PERFORMANCE"
          title="Quantitative evaluation—built for real usage"
          description="Performance metrics help you understand how the classifier behaves. Replace the confusion matrix placeholder with your generated report."
        />

        <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-4">
          <GlassCard className="p-5 sm:p-6 lg:col-span-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {metrics.map((m, idx) => (
                <motion.div
                  key={m.label}
                  variants={fadeInUp}
                  initial="hidden"
                  whileInView="show"
                  viewport={{ once: true, margin: "-100px" }}
                  transition={{ delay: idx * 0.05 }}
                  className="rounded-2xl border border-white/10 bg-black/10 p-4"
                >
                  <p className="text-xs font-semibold text-white/65">{m.label}</p>
                  <p className="mt-2 text-2xl font-semibold text-white">{m.value}</p>
                  <p className="mt-2 text-xs leading-6 text-white/65">{m.detail}</p>
                </motion.div>
              ))}
            </div>

            <div className="mt-4 rounded-2xl border border-white/10 bg-gradient-to-br from-cyan-300/10 via-indigo-300/8 to-fuchsia-300/8 p-4">
              <p className="text-xs font-semibold text-white/70">Interpretation</p>
              <p className="mt-2 text-sm leading-7 text-white/75">
                We optimize for actionable results: strong precision reduces false alerts, while high recall ensures real issues aren’t missed.
              </p>
            </div>
          </GlassCard>

          <GlassCard className="p-5 sm:p-6 lg:col-span-7">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-white">Confusion Matrix (Placeholder)</p>
                <p className="mt-1 text-xs text-white/65 leading-5">
                  Replace this box with your confusion matrix image.
                </p>
              </div>
              <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-white/70">Updated from your file</div>
            </div>

            <div className="mt-4 rounded-2xl border border-white/10 bg-black/10 p-4">
              <div className="aspect-[16/10] rounded-xl border border-white/10 bg-gradient-to-r from-cyan-400/10 via-indigo-400/10 to-fuchsia-400/10 flex items-center justify-center overflow-hidden">
                <img
                  src="/confusion-matrix.png"
                  alt="Confusion matrix"
                  className="h-full w-full object-contain"
                  onError={(e) => {
                    const img = e.currentTarget;
                    img.style.display = "none";
                    const parent = img.parentElement;
                    if (!parent) return;
                    const badge = document.createElement("div");
                    badge.className =
                      "absolute inset-0 flex items-center justify-center text-center p-6";
                    badge.innerHTML = `
                      <div>
                        <div class="text-6xl">📊</div>
                        <p class="mt-2 text-sm font-semibold text-white">Add confusion-matrix.png</p>
                        <p class="mt-1 text-xs text-white/65">Place it in <span class="font-semibold text-white/80">/public</span></p>
                      </div>
                    `;
                    parent.appendChild(badge);
                  }}
                />

                {/* fallback text (only shows if image missing via onError) */}
                <div className="hidden" />
              </div>
            </div>

            <div className="mt-4 text-xs text-white/60 leading-6">
              Put your file here: <span className="text-white/80 font-semibold">public/confusion-matrix.png</span>
            </div>
          </GlassCard>
        </div>
      </Container>
    </section>
  );
}

