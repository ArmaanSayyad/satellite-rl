import type { Encounter } from "./types";
export const policyName = (id: string) =>
  ({
    v1: "PPO · v1",
    ppo_v1: "PPO · v1",
    never_maneuver: "No maneuver",
    no_maneuver: "No maneuver",
    threshold: "Threshold",
    geometry_threshold: "Geometry controller",
    planner: "Numerical planner",
    v2: "Experimental · v2",
    ppo_v2: "Experimental · v2",
  })[id] ?? id.replaceAll("_", " ");
export const probability = (value: number | null | undefined) =>
  value == null
    ? "Not recorded"
    : value === 0
      ? "Below floor"
      : value.toExponential(2);
export function ellipseGeometry(encounter: Encounter) {
  const [[a, b], [, d]] = encounter.covariance_m2;
  const root = Math.hypot(a - d, 2 * b);
  return {
    major: Math.sqrt(Math.max(0, (a + d + root) / 2)),
    minor: Math.sqrt(Math.max(0, (a + d - root) / 2)),
    angle: Math.atan2(2 * b, a - d) / 2,
  };
}
