"use client";

import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight, Image as ImageIcon } from "lucide-react";

import Container from "@/components/ui/container";
import GlassCard from "@/components/ui/glass-card";
import SectionHeading from "@/components/ui/section-heading";
import { fadeInUp } from "@/components/ui/motion";

type Slide = {
  title: string;
  subtitle: string;
  tag: string;
};

const slides: Slide[] = [
  { title: "OCR Input Screenshot", subtitle: "Replace with your OCR input example.", tag: "OCR" },
  { title: "Keyword Signals", subtitle: "Replace with your keyword analysis output.", tag: "Explainable" },
  { title: "Confidence Score Output", subtitle: "Replace with your verdict + confidence view.", tag: "Model" },
];

export default function ScreenshotsCarousel() {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setIndex((i) => (i + 1) % slides.length), 6500);
    return () => clearInterval(t);
  }, []);

  const current = slides[index];
  const src = `/screenshot-${index + 1}.png`;

  return (
    <section className="py-12 sm:py-16">
      <Container>
        <SectionHeading
          eyebrow="SCREENSHOTS"
          title="A premium workflow you can validate"
          description="Use these placeholders to showcase your actual UI: OCR input, explainable signals, and final verdict outputs."
        />

        <GlassCard className="mt-8 p-5 sm:p-6">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-white">Carousel (Replaceable)</p>
              <p className="mt-1 text-xs text-white/65">Swap these screenshots once you have your UI ready.</p>
            </div>
            <div className="flex items-center gap-2 text-xs font-semibold text-white/70">
              <ImageIcon className="h-4 w-4" />
              {index + 1}/{slides.length}
            </div>
          </div>

          <div className="mt-4 relative">
            <div className="rounded-2xl border border-white/10 bg-black/10 overflow-hidden">
              <div className="p-4 flex items-center justify-between gap-3 border-b border-white/10">
                <div className="min-w-0">
                  <span className="inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-white/70">
                    {current.tag}
                  </span>
                  <h3 className="mt-2 text-base font-semibold text-white">{current.title}</h3>
                  <p className="mt-1 text-sm text-white/65">{current.subtitle}</p>
                </div>

                <div className="hidden sm:flex items-center gap-2">
                  {slides.map((s, i) => (
                    <button
                      key={s.title}
                      aria-label={`Go to slide ${i + 1}`}
                      onClick={() => setIndex(i)}
                      className={
                        i === index
                          ? "h-2.5 w-7 rounded-full bg-gradient-to-r from-cyan-300 via-indigo-300 to-fuchsia-300"
                          : "h-2.5 w-2.5 rounded-full bg-white/15 hover:bg-white/25 transition"
                      }
                    />
                  ))}
                </div>
              </div>

              <div className="p-4">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={current.title}
                    variants={fadeInUp}
                    initial="hidden"
                    animate="show"
                    exit="hidden"
                    transition={{ duration: 0.45 }}
                  >
                    <div className="aspect-[16/9] rounded-xl border border-white/10 bg-gradient-to-br from-cyan-400/10 via-indigo-400/10 to-fuchsia-400/10 overflow-hidden">
                          <div className="h-full w-full relative">
                        <img
                          alt={current.title}
                          src={src}
                          className="h-full w-full object-contain bg-gradient-to-br from-cyan-400/10 via-indigo-400/10 to-fuchsia-400/10"

                          onError={(e) => {
                            const img = e.currentTarget;
                            img.style.display = "none";

                            const parent = img.parentElement;
                            if (!parent) return;

                            if (parent.querySelector("[data-missing='1']")) return;

                            const badge = document.createElement("div");
                            badge.setAttribute("data-missing", "1");
                            badge.className =
                              "absolute inset-0 flex items-center justify-center p-6 bg-gradient-to-br from-cyan-400/10 via-indigo-400/10 to-fuchsia-400/10";
                            badge.innerHTML = `
                              <div class="text-center">
                                <div class="text-4xl mb-2">🖼️</div>
                                <p class="text-sm font-semibold text-white">Missing screenshot</p>
                                <p class="mt-1 text-xs text-white/65">Add <span class="font-semibold">${src}</span> to <span class="font-semibold">/public</span></p>
                              </div>
                            `;
                            parent.appendChild(badge);
                          }}
                        />

                        <div className="absolute inset-0 pointer-events-none" />

                        {/* Info panel */}
                        <div className="absolute left-0 right-0 bottom-0 p-5">
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2 text-xs font-semibold text-white/75">
                              <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-black/15">
                                IMG
                              </span>
                              <span>Replaceable placeholder</span>
                            </div>

                            <span className="text-[11px] font-semibold rounded-full border border-white/10 bg-white/5 px-3 py-1 text-white/65">
                              {src}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <p className="text-sm font-semibold text-white">{current.title}</p>
                      <p className="mt-2 text-xs leading-6 text-white/65">
                        Add your screenshot here later. This carousel expects images in <span className="text-white/80">/public</span>.
                      </p>

                      <div className="mt-5 rounded-xl border border-white/10 bg-black/10 p-4">
                        <div className="text-xs font-semibold text-white/60">Suggested mapping</div>
                        <div className="mt-2 grid grid-cols-2 gap-2">
                          <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                            <div className="text-[11px] text-white/55 font-semibold">Slot 1</div>
                            <div className="mt-1 text-sm font-semibold text-white">OCR input</div>
                          </div>
                          <div className="rounded-lg border border-white/10 bg-white/5 p-3">
                            <div className="text-[11px] text-white/55 font-semibold">Slot 2</div>
                            <div className="mt-1 text-sm font-semibold text-white">Verdict & confidence</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>

            <button
              aria-label="Previous screenshot"
              onClick={() => setIndex((i) => (i - 1 + slides.length) % slides.length)}
              className="absolute left-2 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full border border-white/10 bg-black/30 hover:bg-black/50 transition flex items-center justify-center"
            >
              <ChevronLeft className="h-5 w-5 text-white/85" />
            </button>

            <button
              aria-label="Next screenshot"
              onClick={() => setIndex((i) => (i + 1) % slides.length)}
              className="absolute right-2 top-1/2 -translate-y-1/2 h-10 w-10 rounded-full border border-white/10 bg-black/30 hover:bg-black/50 transition flex items-center justify-center"
            >
              <ChevronRight className="h-5 w-5 text-white/85" />
            </button>
          </div>
        </GlassCard>
      </Container>
    </section>
  );
}

