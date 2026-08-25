import type { ReactNode } from "react";

interface Props {
  content: ReactNode;
  children: ReactNode;
  className?: string;
}

/** CSS-only hover/focus tooltip -- no JS state, works with keyboard focus
 * too (tabIndex + :focus-within), used for the jargon glossary and
 * decision explanations. */
export default function Tooltip({ content, children, className }: Props) {
  return (
    <span className={`tooltip-wrap ${className ?? ""}`} tabIndex={0}>
      {children}
      <span className="tooltip-bubble">{content}</span>
    </span>
  );
}
