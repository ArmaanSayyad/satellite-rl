import Tooltip from "./Tooltip";

/** A small "ⓘ" glossary icon -- hover/focus to see what a jargon term
 * (Pc, Δv, TCA, ...) actually means. */
export default function Info({ text }: { text: string }) {
  return (
    <Tooltip content={text}>
      <span className="info-icon" aria-label="more info">
        ⓘ
      </span>
    </Tooltip>
  );
}
