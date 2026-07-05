"use client";

import React from "react";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import Container from "@/components/ui/container";
import SectionHeading from "@/components/ui/section-heading";
import GlassCard from "@/components/ui/glass-card";
import { fadeInUp } from "@/components/ui/motion";

const timeline = [
  "Internship Project",
  "Research",
  "Model Training",
  "98.92% Accuracy",
  "Production Version",
  "Deployment",
];

export default function ProjectJourney() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <SectionHeading
          eyebrow="PROJECT JOURNEY"
          title="From prototype to production-grade pipeline"
          description="Built through iteration: research rigor, model training, and production hardening to deliver consistent results."
        />

        <GlassCard className="p-5 sm:p-6 mt-8">
          <div className="hidden lg:block">
            <div className="flex items-center gap-4">
              {timeline.map((t, idx) => (
                <React.Fragment key={t}>
                  <motion.div variants={fadeInUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-140px" }} transition={{ delay: idx * 0.05 }} className="flex-1 min-w-0">
                    <div className="rounded-2xl border border-white/10 bg-black/10 p-4">
                      <p className="text-sm font-semibold text-white">{t}</p>
                      <p className="mt-2 text-xs leading-6 text-white/65">
                        {idx === 0
                          ? "A real-world misinformation challenge."
                          : idx === 1
                            ? "Feature engineering + interpretability focus."
                            : idx === 2
                              ? "Training with structured pipelines and evaluation."
                              : idx === 3
                                ? "Measured performance across test sets."
                                : idx === 4
                                  ? "Clean architecture, better UX, reliability."
                                  : "Shipping the model to users."}
                      </p>
                    </div>
                  </motion.div>
                  {idx < timeline.length - 1 ? (
                    <div className="w-10 flex justify-center">
                      <ArrowRight className="h-5 w-5 text-white/25" />
                    </div>
                  ) : null}
                </React.Fragment>
              ))}
            </div>
          </div>

          <div className="lg:hidden">
            <div className="flex flex-col gap-4">
              {timeline.map((t, idx) => (
                <motion.div key={t} variants={fadeInUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-120px" }} transition={{ delay: idx * 0.05 }}>
                  <div className="rounded-2xl border border-white/10 bg-black/10 p-4">
                    <p className="text-sm font-semibold text-white">{t}</p>
                    <p className="mt-2 text-xs leading-6 text-white/65">
                      {idx === 0
                        ? "A real-world misinformation challenge."
                        : idx === 1
                          ? "Feature engineering + interpretability focus."
                          : idx === 2
                            ? "Training with structured pipelines and evaluation."
                            : idx === 3
                              ? "Measured performance across test sets."
                              : idx === 4
                                ? "Clean architecture, better UX, reliability."
                                : "Shipping the model to users."}
                    </p>
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

