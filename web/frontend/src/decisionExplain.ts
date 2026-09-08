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
export function explainDecision(
  decision: Decision,
  precedingKeyframe: Keyframe | null,
  pcThreshold: number,
): string {
  const pcBefore = precedingKeyframe?.pc_estimate;
  const fired = decision.action_magnitude_ms > FIRE_THRESHOLD_MS;
  const pcText = pcBefore != null ? pcBefore.toExponential(2) : "unknown";
  const overThreshold = pcBefore != null && pcBefore > pcThreshold;

  if (pcBefore == null) {
    return `Observed Pc was not recorded. ${fired ? `A ${decision.action_magnitude_ms.toFixed(3)} m/s maneuver` : "No measurable maneuver"} was recorded; no risk association or internal rationale is inferred.`;
  }

  if (fired && overThreshold) {
    return `Observed Pc ${pcText} was above threshold. The recorded action was ${decision.action_magnitude_ms.toFixed(3)} m/s. This association does not establish the policy's reasoning.`;
  }
  if (fired) {
    return `Burned ${decision.action_magnitude_ms.toFixed(3)} m/s with observed Pc ${pcText} below threshold. Its internal rationale is unknown.`;
  }
  if (overThreshold) {
    return `Observed Pc ${pcText} was above threshold. No measurable maneuver was recorded at this update; no reason is inferred.`;
  }
  return `Observed Pc ${pcText} was below threshold. No measurable maneuver was recorded.`;
}
