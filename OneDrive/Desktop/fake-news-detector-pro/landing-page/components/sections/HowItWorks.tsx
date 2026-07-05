"use client";

import React from "react";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import Container from "@/components/ui/container";
import SectionHeading from "@/components/ui/section-heading";
import GlassCard from "@/components/ui/glass-card";
import { fadeInUp } from "@/components/ui/motion";

const steps = [
  "User Input",
  "Text Cleaning",
  "TF-IDF Vectorization",
  "Machine Learning Prediction",
  "Confidence Score",
  "Final Result",
];

export default function HowItWorks() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <SectionHeading
          eyebrow="HOW IT WORKS"
          title="A transparent pipeline from text to verdict"
          description="From cleaning and vectorization to model inference and confidence scoring—each stage is designed for accuracy and interpretability."
        />

        <GlassCard className="p-5 sm:p-6 mt-8">
          <div className="hidden md:block">
            <div className="flex items-center gap-4">
              {steps.map((s, idx) => (
                <React.Fragment key={s}>
                  <motion.div
                    variants={fadeInUp}
                    initial="hidden"
                    whileInView="show"
                    viewport={{ once: true, margin: "-120px" }}
                    transition={{ delay: idx * 0.05 }}
                    className="relative flex-1 min-w-0"
                  >
                    <div className="rounded-2xl border border-white/10 bg-black/10 p-4">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-300/25 via-indigo-300/20 to-fuchsia-300/20 border border-white/10 flex items-center justify-center">
                          <span className="text-sm font-semibold text-white">{idx + 1}</span>
                        </div>
                        <p className="text-sm font-semibold text-white">{s}</p>
                      </div>
                      <div className="mt-3 h-2 rounded-full bg-gradient-to-r from-cyan-300/50 via-indigo-300/40 to-fuchsia-300/45" />
                    </div>
                  </motion.div>
                  {idx < steps.length - 1 ? (
                    <div className="flex items-center justify-center w-10">
                      <ArrowRight className="h-5 w-5 text-white/25" />
                    </div>
                  ) : null}
                </React.Fragment>
              ))}
            </div>
          </div>

          <div className="md:hidden">
            <div className="flex flex-col gap-4">
              {steps.map((s, idx) => (
                <motion.div key={s} variants={fadeInUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-120px" }} transition={{ delay: idx * 0.05 }}>
                  <div className="rounded-2xl border border-white/10 bg-black/10 p-4">
                    <div className="flex items-start gap-3">
                      <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-cyan-300/25 via-indigo-300/20 to-fuchsia-300/20 border border-white/10 flex items-center justify-center">
                        <span className="text-sm font-semibold text-white">{idx + 1}</span>
                      </div>
                      <div>
                        <p className="text-sm font-semibold text-white">{s}</p>
                        <p className="mt-1 text-sm text-white/65">{idx === 0 ? "Paste text or upload OCR screenshots." : ""}</p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </GlassCard>
      </Container>
    </section>
  );
}

