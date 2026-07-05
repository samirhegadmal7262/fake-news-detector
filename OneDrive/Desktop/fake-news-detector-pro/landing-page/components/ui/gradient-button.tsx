import React from "react";

export default function GradientButton({
  children,
  className = "",
  href,
}: {
  children: React.ReactNode;
  className?: string;
  href?: string;
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-full px-5 py-3 text-sm font-semibold tracking-wide " +
    "text-black shadow-sm transition-transform focus:outline-none focus:ring-2 focus:ring-white/40 " +
    "hover:scale-[1.02] active:scale-[0.99] disabled:opacity-60 disabled:hover:scale-100 " +
    "bg-gradient-to-r from-cyan-300 via-indigo-300 to-fuchsia-300";

  if (href) {
    const isExternal = href.startsWith("http");
    return (
      <a
        href={href}
        target={isExternal ? "_blank" : undefined}
        rel={isExternal ? "noopener noreferrer" : undefined}
        className={`${base} ${className}`}
      >
        {children}
      </a>
    );
  }

  return (
    <button type="button" className={`${base} ${className}`}>
      {children}
    </button>
  );
}

