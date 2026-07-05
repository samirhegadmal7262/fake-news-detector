import React from "react";

export default function SecondaryButton({
  children,
  className = "",
  href,
  disabled,
}: {
  children: React.ReactNode;
  className?: string;
  href?: string;
  disabled?: boolean;
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-full border border-white/15 px-5 py-3 text-sm font-semibold tracking-wide " +
    "text-white/90 transition-transform focus:outline-none focus:ring-2 focus:ring-white/35 " +
    "hover:scale-[1.02] active:scale-[0.99] disabled:opacity-60 disabled:hover:scale-100 " +
    "bg-white/5 backdrop-blur";

  const content = <>{children}</>;

  if (href) {
    const isExternal = href.startsWith("http");
    return (
      <a
        href={disabled ? undefined : href}
        target={isExternal ? "_blank" : undefined}
        rel={isExternal ? "noopener noreferrer" : undefined}
        aria-disabled={disabled ? "true" : undefined}
        className={`${base} ${className}`}
        onClick={(e) => {
          if (disabled) e.preventDefault();
        }}
      >
        {content}
      </a>
    );
  }

  return (
    <button type="button" className={`${base} ${className}`} disabled={disabled}>
      {content}
    </button>
  );
}

