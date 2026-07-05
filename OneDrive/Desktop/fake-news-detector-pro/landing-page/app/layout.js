import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "Fake News Detector Pro",
  description:
    "AI-powered fake news detection using Natural Language Processing, Machine Learning and OCR. Get explainable confidence scores and keyword signals.",
  openGraph: {
    title: "Fake News Detector Pro",
    description:
      "AI-powered fake news detection using Natural Language Processing, Machine Learning and OCR.",
    type: "website",
    url: "https://example.com",
  },
  twitter: {
    card: "summary_large_image",
    title: "Fake News Detector Pro",
    description:
      "AI-powered fake news detection using Natural Language Processing, Machine Learning and OCR.",
  },
};

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
