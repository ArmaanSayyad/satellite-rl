import type { Decision, Keyframe } from "./types";

const FIRE_THRESHOLD_MS = 1e-3;

/** A short, honest explanation of one policy decision, using the real Pc
 * estimate the policy actually observed (the keyframe recorded just
 * before this decision -- see simulation_runner.run_simulation, the obs
 * that produced decisions[i] is exactly the obs stored as
 * keyframes[i].pc_estimate) versus the real pc_threshold. Doesn't claim
 * the policy follows a simple threshold rule -- it's a learned policy,
 * and sometimes burns under threshold or coasts over it, which this
 * reports honestly rather than papering over.
 */
export function explainDecision(decision: Decision, precedingKeyframe: Keyframe | null, pcThreshold: number): string {
  const pcBefore = precedingKeyframe?.pc_estimate;
  const fired = decision.action_magnitude_ms > FIRE_THRESHOLD_MS;
  const pcText = pcBefore != null ? pcBefore.toExponential(2) : "unknown";
  const overThreshold = pcBefore != null && pcBefore > pcThreshold;

  if (fired && overThreshold) {
    return `Estimated Pc (${pcText}) exceeded the ${pcThreshold.toExponential(0)} threshold → burned ${decision.action_magnitude_ms.toFixed(2)} m/s.`;
  }
  if (fired) {
    return `Burned ${decision.action_magnitude_ms.toFixed(2)} m/s even though estimated Pc (${pcText}) was under threshold — the learned policy judged it worth the fuel anyway.`;
  }
  if (overThreshold) {
    return `Estimated Pc (${pcText}) was over threshold, but the policy chose not to burn here — it may be waiting for a later, cheaper opportunity.`;
  }
  return `Estimated Pc (${pcText}) was under the ${pcThreshold.toExponential(0)} threshold → coasted, no burn.`;
}
