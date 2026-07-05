"use client";

import React from "react";
import { motion } from "framer-motion";
import {
  Code,
  GitFork as GithubIcon,
  GitBranch,
  Landmark,
  GraduationCap,
  Cpu,
  Workflow,
  Shield,
} from "lucide-react";
import Container from "@/components/ui/container";
import SectionHeading from "@/components/ui/section-heading";
import GlassCard from "@/components/ui/glass-card";
import { fadeInUp } from "@/components/ui/motion";

const items = [
  { label: "Python", icon: Code },
  { label: "Scikit Learn", icon: Landmark },
  { label: "Streamlit", icon: Workflow },
  { label: "Pandas", icon: Shield },
  { label: "NumPy", icon: Cpu },
  { label: "OpenCV", icon: Workflow },
  { label: "Tesseract OCR", icon: GraduationCap },
  { label: "Git", icon: GitBranch },
  { label: "GitHub", icon: GithubIcon },
  { label: "Next.js", icon: Code },
  { label: "Tailwind CSS", icon: Code },
];

export default function TechStack() {
  return (
    <section className="py-12 sm:py-16">
      <Container>
        <SectionHeading
          eyebrow="TECH STACK"
          title="Built with tools that make accuracy achievable"
          description="A practical stack for OCR, NLP, model training, and a clean UI experience."
        />

        <motion.div
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="mt-8 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4"
        >
          {items.map((it, idx) => (
            <motion.div
              key={it.label}
              variants={fadeInUp}
              transition={{ delay: idx * 0.03 }}
              className="group"
            >
              <GlassCard className="p-4 h-full">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-xl border border-white/10 bg-gradient-to-br from-cyan-300/15 via-indigo-300/10 to-fuchsia-300/15 flex items-center justify-center">
                    <it.icon className="h-5 w-5 text-white/85" />
                  </div>
                  <p className="text-sm font-semibold text-white group-hover:text-white transition">{it.label}</p>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </motion.div>
      </Container>
    </section>
  );
}

