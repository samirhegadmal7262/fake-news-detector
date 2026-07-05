"use client";

import React from "react";
import Container from "@/components/ui/container";
import { Mail, Link as LinkIcon, GitFork } from "lucide-react";

export default function Footer() {
  return (
    <footer className="py-10 border-t border-white/10">
      <Container>
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold text-white">Fake News Detector Pro</p>
            <p className="mt-1 text-xs text-white/60">AI-powered detection with OCR + NLP explainability.</p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <a
              href="https://github.com/samirhegadmal7262/fake-news-detector-pro"
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white/75 hover:bg-white/10 transition"
              target="_blank"
              rel="noopener noreferrer"
            >
              <GitFork className="h-4 w-4" />
              GitHub
            </a>
            <a
              href="https://www.linkedin.com/"
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white/75 hover:bg-white/10 transition"
              target="_blank"
              rel="noopener noreferrer"
            >
              <LinkIcon className="h-4 w-4" />
              LinkedIn
            </a>

            <a
              href="mailto:hello@example.com"
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-white/75 hover:bg-white/10 transition"
            >
              <Mail className="h-4 w-4" />
              Email
            </a>
          </div>
        </div>

        <div className="mt-8 text-xs text-white/50">
          © {new Date().getFullYear()} Fake News Detector Pro. All rights reserved.
        </div>
      </Container>
    </footer>
  );
}

