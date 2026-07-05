"use client";

import React from "react";
import { motion } from "framer-motion";
import Container from "@/components/ui/container";
import SectionHeading from "@/components/ui/section-heading";
import GlassCard from "@/components/ui/glass-card";
import { fadeInUp } from "@/components/ui/motion";
import { Rocket, Layers, Search, Languages, Cloud } from "lucide-react";

const items = [
  { title: "Real-Time News Verification", desc: "Cross-check claims with up-to-date sources for stronger validation.", icon: Rocket },
  { title: "RAG", desc: "Retrieval-Augmented Generation to provide grounded explanations.", icon: Layers },
  { title: "Latest News Cross Checking", desc: "Reduce stale results by comparing with current reporting.", icon: Search },
  { title: "Multi-language Support", desc: "Expand OCR + NLP coverage for multiple languages.", icon: Languages },
  { title: "API Version", desc: "Bring the detector into your apps with clean REST endpoints.", icon: Cloud },
  { title: "Cloud Deployment", desc: "Scalable hosting with reliable latency and monitoring.", icon: Cloud },
];

export default function Roadmap() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <SectionHeading
          eyebrow="ROADMAP"
          title="Next steps toward a more reliable verification layer"
          description="These are the upcoming capabilities that will turn the detector into a broader real-time verification product."
        />

        <GlassCard className="p-5 sm:p-6 mt-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {items.map((it, idx) => (
              <motion.div
                key={it.title}
                variants={fadeInUp}
                initial="hidden"
                whileInView="show"
                viewport={{ once: true, margin: "-120px" }}
                transition={{ delay: idx * 0.05 }}
                className="group"
              >
                <div className="rounded-2xl border border-white/10 bg-black/10 p-4 h-full">
                  <div className="flex items-start justify-between gap-3">
                    <div className="rounded-xl border border-white/10 bg-white/5 p-2">
                      <it.icon className="h-5 w-5 text-white/85" />
                    </div>
                    <span className="text-xs font-semibold text-white/55">Upcoming</span>
                  </div>
                  <h3 className="mt-3 text-sm font-semibold text-white">{it.title}</h3>
                  <p className="mt-2 text-sm leading-7 text-white/70">{it.desc}</p>
                  <div className="mt-4 h-1 rounded-full bg-gradient-to-r from-cyan-300/60 via-indigo-300/50 to-fuchsia-300/55 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </motion.div>
            ))}
          </div>
        </GlassCard>
      </Container>
    </section>
  );
}

