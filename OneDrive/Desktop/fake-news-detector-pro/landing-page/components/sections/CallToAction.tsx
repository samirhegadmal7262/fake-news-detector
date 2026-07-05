"use client";

import React from "react";
import Container from "@/components/ui/container";
import GlassCard from "@/components/ui/glass-card";
import GradientButton from "@/components/ui/gradient-button";
import SecondaryButton from "@/components/ui/secondary-button";
import { ArrowRight } from "lucide-react";
import { fadeInUp } from "@/components/ui/motion";
import { motion } from "framer-motion";

export default function CallToAction() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <GlassCard className="p-5 sm:p-6">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
            <motion.div variants={fadeInUp} initial="hidden" whileInView="show" viewport={{ once: true, margin: "-120px" }}>
              <h2 className="text-2xl sm:text-3xl font-semibold text-white leading-tight">
                Ready to experience AI-powered fake news detection?
              </h2>
              <p className="mt-3 text-sm sm:text-base leading-7 text-white/70 max-w-2xl">
                Try the live workflow: OCR and NLP analysis with confidence scoring and explainable predictions.
              </p>
            </motion.div>

            <div className="flex flex-col sm:flex-row gap-3">
              <GradientButton href="https://fake-news-detector-pro-mwxyyxzvclvkyrjc3iw8re.streamlit.app/" className="group">
                Launch Live Demo
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </GradientButton>

              <SecondaryButton href="#roadmap" className="group">
                View Roadmap
              </SecondaryButton>
            </div>
          </div>
        </GlassCard>
      </Container>
    </section>
  );
}

