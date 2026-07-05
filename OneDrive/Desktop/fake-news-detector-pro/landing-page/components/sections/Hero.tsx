"use client";

import React from "react";
import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";

import Container from "@/components/ui/container";
import GlassCard from "@/components/ui/glass-card";
import GradientButton from "@/components/ui/gradient-button";
import SecondaryButton from "@/components/ui/secondary-button";
import { fadeIn, fadeInUp } from "@/components/ui/motion";

export default function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="absolute inset-0 -z-10">
        <div className="absolute -top-48 left-1/2 h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-gradient-to-r from-cyan-400/30 via-indigo-400/20 to-fuchsia-400/25 blur-3xl" />
        <div className="absolute top-24 -left-28 h-64 w-64 rounded-full bg-gradient-to-r from-cyan-300/20 to-indigo-300/20 blur-2xl" />
        <div className="absolute bottom-0 right-[-6rem] h-80 w-80 rounded-full bg-gradient-to-r from-fuchsia-300/15 via-indigo-300/15 to-cyan-300/15 blur-3xl" />

        <motion.div
          aria-hidden
          className="absolute left-[10%] top-[15%] h-24 w-24 rounded-full bg-cyan-300/30 blur-2xl"
          animate={{ y: [0, 28, 0], x: [0, 18, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          aria-hidden
          className="absolute right-[12%] top-[22%] h-28 w-28 rounded-full bg-indigo-300/25 blur-2xl"
          animate={{ y: [0, 22, 0], x: [0, -16, 0] }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          aria-hidden
          className="absolute right-[24%] bottom-[-3rem] h-32 w-32 rounded-full bg-fuchsia-300/20 blur-2xl"
          animate={{ y: [0, -20, 0], x: [0, 12, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      <Container className="pt-16 sm:pt-20 pb-10 sm:pb-16">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-14 items-center">
          <div className="lg:col-span-7">
            <motion.div variants={fadeInUp} initial="hidden" animate="show">
              <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-white/80 backdrop-blur">
                <Sparkles className="h-4 w-4 text-cyan-300" />
                Explainable AI with OCR + NLP
              </div>

              <h1 className="mt-5 text-4xl sm:text-5xl lg:text-6xl font-semibold leading-[1.05] text-white">
                Fake News Detector Pro
              </h1>

              <p className="mt-4 max-w-2xl text-base sm:text-lg leading-8 text-white/72">
                AI-powered fake news detection using Natural Language Processing, Machine Learning and OCR.
                Get clarity with confidence scores, keyword signals, and explainable reasoning.
              </p>

              <div className="mt-7 flex flex-col sm:flex-row gap-3 sm:items-center">
                <GradientButton
                  href="https://fake-news-detector-pro-mwxyyxzvclvkyrjc3iw8re.streamlit.app/"
                  className="group"
                >
                  Try Live Demo
                  <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                </GradientButton>

                <SecondaryButton
                  href="https://github.com/samirhegadmal7262/fake-news-detector-pro"
                  className="group"
                >
                  GitHub Repository
                </SecondaryButton>
              </div>

              <motion.div variants={fadeIn} initial="hidden" animate="show" className="mt-6">
                <div className="flex flex-wrap gap-2">
                  {[
                    "OCR + Text NLP",
                    "Confidence scoring",
                    "Keyword analysis",
                    "Secure by design",
                  ].map((t) => (
                    <span
                      key={t}
                      className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-white/70"
                    >
                      {t}
                    </span>
                  ))}
                </div>
              </motion.div>
            </motion.div>
          </div>

          <div className="lg:col-span-5">
            <motion.div
              variants={fadeInUp}
              initial="hidden"
              animate="show"
              transition={{ delay: 0.1 }}
            >
              <GlassCard className="p-5 sm:p-6">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-sm font-semibold text-white">Live detection preview</p>
                    <p className="mt-1 text-xs text-white/60 leading-5">
                      Swap this panel with your real demo screenshots when ready.
                    </p>
                  </div>

                  <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-white/80">
                    <span className="text-cyan-300">Less than 2 seconds</span> predictions
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {[
                    "OCR upload",
                    "Text analysis",
                    "Confidence score",
                    "Explainable result",
                  ].map((label) => (
                    <div
                      key={label}
                      className="rounded-xl border border-white/10 bg-black/10 p-3"
                    >
                      <p className="text-xs font-semibold text-white/75">{label}</p>
                      <div className="mt-2 h-10 rounded-lg bg-gradient-to-r from-cyan-400/20 via-indigo-400/15 to-fuchsia-400/20" />
                    </div>
                  ))}
                </div>

                <div className="mt-4 rounded-2xl border border-white/10 bg-black/10 p-4">
                  <p className="text-xs font-semibold text-white/70">Result placeholder</p>
                  <div className="mt-2 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <div className="h-10 w-10 rounded-full bg-gradient-to-r from-cyan-300/40 to-fuchsia-300/30" />
                      <div>
                        <p className="text-sm font-semibold text-white">Likely Fake</p>
                        <p className="text-xs text-white/60">Replace with your output logic</p>
                      </div>
                    </div>

                    <div className="text-right">
                      <p className="text-xs text-white/60">Confidence</p>
                      <p className="text-lg font-semibold text-cyan-300">0.73</p>
                    </div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </div>
      </Container>
    </section>
  );
}

