import React from "react";

export default function SectionHeading({
  eyebrow,
  title,
  description,
  align = "left",
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  align?: "left" | "center";
}) {
  const alignClass = align === "center" ? "text-center items-center" : "text-left";

  return (
    <div className={`flex flex-col gap-3 ${alignClass}`}>
      {eyebrow ? (
        <p className={`text-xs font-semibold tracking-[0.2em] uppercase text-white/60 ${align === "center" ? "text-center" : "text-left"}`}>
          {eyebrow}
        </p>
      ) : null}
      <h2 className={`text-2xl sm:text-3xl font-semibold leading-tight text-white ${align === "center" ? "mx-auto" : ""}`}>
        {title}
      </h2>
      {description ? (
        <p className={`max-w-2xl text-sm sm:text-base leading-7 text-white/70 ${align === "center" ? "mx-auto" : ""}`}>
          {description}
        </p>
      ) : null}
    </div>
  );
}

