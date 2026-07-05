"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  Bot,
  FileText,
  Upload,
  Percent,
  Search,
  Sparkles,
  Timer,
  Blocks,
  Shield,
} from "lucide-react";
import GlassCard from "@/components/ui/glass-card";
import Container from "@/components/ui/container";
import SectionHeading from "@/components/ui/section-heading";
import { stagger, fadeInUp } from "@/components/ui/motion";

const features = [
  { title: "AI News Detection", desc: "NLP-powered classification to detect likely misinformation signals.", icon: Bot },
  { title: "OCR Screenshot Analysis", desc: "Extracts text from images using OCR for reliable downstream analysis.", icon: FileText },
  { title: "TXT Upload Support", desc: "Upload text files to run detection in seconds—no manual copy-paste needed.", icon: Upload },
  { title: "Confidence Score", desc: "A calibrated confidence score helps you judge result reliability.", icon: Percent },
  { title: "Keyword Analysis", desc: "Highlights indicative keywords and patterns to support explainability.", icon: Search },
  { title: "Explainable Predictions", desc: "Transparent reasoning—so the model is understandable, not a black box.", icon: Sparkles },
  { title: "Real-Time Verification (Coming Soon)", desc: "Cross-check claims against real-time sources for stronger verification.", icon: Timer },
  { title: "Production Ready Architecture", desc: "A clean pipeline designed to scale from experiments to deployment.", icon: Blocks },
];

export default function Features() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <SectionHeading
          eyebrow="FEATURES"
          title="Designed for trust, built for speed"
          description="Everything you need for a premium detection experience—fast, explainable, and OCR-ready."
        />

        <motion.div variants={stagger} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {features.map((f, idx) => (
            <motion.div key={f.title} variants={fadeInUp} transition={{ delay: idx * 0.03 }} className="group">
              <GlassCard className="p-5 h-full">
                <div className="flex items-start gap-3">
                  <div className="rounded-xl border border-white/10 bg-gradient-to-br from-cyan-300/15 via-indigo-300/10 to-fuchsia-300/15 p-3">
                    <f.icon className="h-5 w-5 text-white/85" />
                  </div>
                  <div className="min-w-0">
                    <h3 className="text-base font-semibold text-white">{f.title}</h3>
                    <p className="mt-2 text-sm leading-7 text-white/70">{f.desc}</p>
                  </div>
                </div>
                <div className="mt-4 h-1 rounded-full bg-gradient-to-r from-cyan-300/60 via-indigo-300/50 to-fuchsia-300/55 opacity-0 group-hover:opacity-100 transition-opacity" />
              </GlassCard>
            </motion.div>
          ))}
        </motion.div>
      </Container>
    </section>
  );
}

