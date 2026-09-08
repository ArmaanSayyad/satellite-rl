import type { Frame } from "./types";

// Real Newtonian point-mass gravity, g = mu * r / |r|^3 -- same formula
// (and same mu constant, passed from the backend) used throughout this
// project's own physics code (e.g. satellite_rl.scenario.targeting).
export function gravityAccelMs2(
  r: [number, number, number],
  muM3S2: number,
): number {
  const rMag = Math.hypot(r[0], r[1], r[2]);
  return muM3S2 / (rMag * rMag);
}

export function speedMs(v: [number, number, number]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

export function separationM(
  a: [number, number, number],
  b: [number, number, number],
): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function subtract(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** Linear interpolation between two real simulated frames, for smooth
 * animation between the ~60s-resolution samples the backend returns.
 * Positions/velocities move smoothly and predictably over such short
 * (relative to orbital period) gaps, so this is a faithful visual
 * representation of the real trajectory, not a fabrication of it.
 */
export function interpolateFrame(a: Frame, b: Frame, t: number): Frame {
  const span = b.t_s - a.t_s;
  const frac = span > 0 ? (t - a.t_s) / span : 0;
  const lerp3 = (
    p: [number, number, number],
    q: [number, number, number],
  ): [number, number, number] => [
    p[0] + (q[0] - p[0]) * frac,
    p[1] + (q[1] - p[1]) * frac,
    p[2] + (q[2] - p[2]) * frac,
  ];
  return {
    t_s: t,
    ego_r: lerp3(a.ego_r, b.ego_r),
    ego_v: lerp3(a.ego_v, b.ego_v),
    sec_r: lerp3(a.sec_r, b.sec_r),
    sec_v: lerp3(a.sec_v, b.sec_v),
  };
}

/** Find the frame at mission time t via binary search over the
 * (t_s-sorted) dense_frames array, and interpolate between the two
 * bracketing real samples.
 */
export function frameAtTime(frames: Frame[], t: number): Frame {
  if (frames.length === 0) throw new Error("frameAtTime: empty frames array");
  if (t <= frames[0].t_s) return frames[0];
  if (t >= frames[frames.length - 1].t_s) return frames[frames.length - 1];

  let lo = 0;
  let hi = frames.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t_s <= t) lo = mid;
    else hi = mid;
  }
  return interpolateFrame(frames[lo], frames[hi], t);
}

/** Split a dense-frame trajectory into what's already been "flown" as of
 * missionTimeS (drawn at full opacity, ending exactly at the current
 * interpolated position so the trail tip matches the marker) and a short
 * faded preview beyond it (drawn at low opacity). Used so trajectories
 * reveal progressively during playback rather than showing the whole
 * (already-decided) outcome from frame one.
 */
export function splitFramesAtTime(
  frames: Frame[],
  missionTimeS: number,
  previewWindowS: number,
): { traveled: Frame[]; preview: Frame[] } {
  if (frames.length === 0) return { traveled: [], preview: [] };

  const traveled: Frame[] = [];
  let i = 0;
  for (; i < frames.length && frames[i].t_s <= missionTimeS; i++) {
    traveled.push(frames[i]);
  }
  const current = frameAtTime(frames, missionTimeS);
  if (
    traveled.length === 0 ||
    traveled[traveled.length - 1].t_s < missionTimeS
  ) {
    traveled.push(current);
  }

  const previewEnd = missionTimeS + previewWindowS;
  const preview: Frame[] = [current];
  for (; i < frames.length && frames[i].t_s <= previewEnd; i++) {
    preview.push(frames[i]);
  }
  return { traveled, preview };
}
