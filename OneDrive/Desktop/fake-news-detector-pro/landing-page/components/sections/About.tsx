"use client";

import React from "react";
import Container from "@/components/ui/container";
import SectionHeading from "@/components/ui/section-heading";
import GlassCard from "@/components/ui/glass-card";
import { fadeInUp } from "@/components/ui/motion";
import { motion } from "framer-motion";

export default function About() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
          <div className="lg:col-span-5">
            <SectionHeading
              eyebrow="ABOUT"
              title="A detection engine with explainable outputs"
              description="Fake News Detector Pro combines OCR and NLP to interpret both screenshots and text. The result is not just a verdict—it's an evidence-driven confidence score and keyword-based signals."
            />
          </div>

          <div className="lg:col-span-7">
            <GlassCard className="p-5 sm:p-6">
              <div className="space-y-4">
                {[
                  {
                    title: "Natural Language Processing that prioritizes meaning",
                    body: "The pipeline cleans input text, vectorizes terms with TF‑IDF, and uses a machine learning classifier to detect misinformation patterns.",
                  },
                  {
                    title: "OCR support for real-world content",
                    body: "Screenshots and image-based text are processed using OCR so the same detection logic can be applied consistently.",
                  },
                  {
                    title: "Confidence-driven decisions",
                    body: "Each prediction includes a confidence score to help you decide whether to trust the result or wait for additional verification.",
                  },
                ].map((item, idx) => (
                  <motion.div key={item.title} variants={fadeInUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-120px" }} transition={{ delay: idx * 0.06 }}>
                    <div className="rounded-2xl border border-white/10 bg-black/10 p-4">
                      <h3 className="text-sm font-semibold text-white">{item.title}</h3>
                      <p className="mt-2 text-sm leading-7 text-white/70">{item.body}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </GlassCard>
          </div>
        </div>
      </Container>
    </section>
  );
}

