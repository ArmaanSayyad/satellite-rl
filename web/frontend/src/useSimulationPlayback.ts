import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Frame, Keyframe, SimulationResult } from "./types";
import { frameAtTime } from "./physics";

// Wall-clock seconds to play the whole mission at speedFactor=1. Real
// missions run up to ~4.8h; this compresses that into something
// watchable while still being real, continuously-sampled physics, not a
// summary.
const NATURAL_PLAYBACK_DURATION_S = 26;

export const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4, 8] as const;

export interface PlaybackState {
  playing: boolean;
  finished: boolean;
  missionTimeS: number;
  missionDurationS: number;
  speedFactor: number;
  frame: Frame | null;
  currentKeyframe: Keyframe | null;
  maneuversSoFar: number;
  progress: number; // 0..1
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  restart: () => void;
  seek: (missionTimeS: number) => void;
  setSpeedFactor: (factor: number) => void;
  stepToKeyframe: (direction: 1 | -1) => void;
}

export function useSimulationPlayback(
  result: SimulationResult | null,
): PlaybackState {
  const [missionTimeS, setMissionTimeS] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedFactor, setSpeedFactor] = useState(1);

  const rafRef = useRef<number | null>(null);
  const lastWallMsRef = useRef<number | null>(null);
  const missionTimeRef = useRef(0);
  const previousScenarioRef = useRef<string | null>(null);
  missionTimeRef.current = missionTimeS;

  const missionDurationS =
    result?.dense_frames[result.dense_frames.length - 1]?.t_s ?? 0;

  const stopLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    lastWallMsRef.current = null;
  }, []);

  // Policy comparisons share mission time. A different scenario resets the clock.
  useEffect(() => {
    stopLoop();
    const scenarioKey = result
      ? `${result.provenance?.source_event_id ?? result.scenario.miss_distance_m}:${result.seed}:${result.provenance?.kind}:${result.scenario.combined_radius_m}`
      : null;
    const sameScenario =
      scenarioKey != null && scenarioKey === previousScenarioRef.current;
    setMissionTimeS((current) =>
      sameScenario ? Math.min(current, missionDurationS) : 0,
    );
    previousScenarioRef.current = scenarioKey;
    setSpeedFactor(1);
    setPlaying(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  useEffect(() => stopLoop, [stopLoop]);

  // The animation loop itself: only runs while playing, restarts
  // (cleanly, no time-jump) whenever speed or the result changes.
  useEffect(() => {
    if (!playing || !result || missionDurationS <= 0) return;
    const baseRate = missionDurationS / NATURAL_PLAYBACK_DURATION_S;

    const tick = (wallMs: number) => {
      if (lastWallMsRef.current === null) lastWallMsRef.current = wallMs;
      const deltaWallS = (wallMs - lastWallMsRef.current) / 1000;
      lastWallMsRef.current = wallMs;
      const next = missionTimeRef.current + deltaWallS * baseRate * speedFactor;
      if (next >= missionDurationS) {
        missionTimeRef.current = missionDurationS;
        setMissionTimeS(missionDurationS);
        setPlaying(false);
        return;
      }
      missionTimeRef.current = next;
      setMissionTimeS(next);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return stopLoop;
  }, [playing, result, missionDurationS, speedFactor, stopLoop]);

  const finished = missionDurationS > 0 && missionTimeS >= missionDurationS;

  const play = useCallback(() => setPlaying(true), []);
  const pause = useCallback(() => setPlaying(false), []);
  const togglePlay = useCallback(() => setPlaying((p) => !p), []);
  const restart = useCallback(() => {
    setMissionTimeS(0);
    setPlaying(true);
  }, []);
  const seek = useCallback(
    (t: number) => {
      const clamped = Math.max(0, Math.min(t, missionDurationS));
      setMissionTimeS(clamped);
    },
    [missionDurationS],
  );
  const stepToKeyframe = useCallback(
    (direction: 1 | -1) => {
      if (!result) return;
      setPlaying(false);
      const times = result.keyframes.map((k) => k.t_s);
      const current = missionTimeRef.current;
      if (direction > 0) {
        const next = times.find((t) => t > current + 1e-6);
        setMissionTimeS(next ?? missionDurationS);
      } else {
        const prior = times.filter((t) => t < current - 1e-6);
        setMissionTimeS(prior.length ? prior[prior.length - 1] : 0);
      }
    },
    [result, missionDurationS],
  );

  const frame = result ? frameAtTime(result.dense_frames, missionTimeS) : null;

  const currentKeyframe = useMemo(() => {
    if (!result) return null;
    let kf: Keyframe | null = result.keyframes[0] ?? null;
    for (const k of result.keyframes) {
      if (k.t_s <= missionTimeS) kf = k;
    }
    return kf;
  }, [result, missionTimeS]);

  const maneuversSoFar = useMemo(() => {
    if (!result) return 0;
    return result.decisions.filter(
      (d) => d.t_s <= missionTimeS && d.action_magnitude_ms > 1e-3,
    ).length;
  }, [result, missionTimeS]);

  return {
    playing,
    finished,
    missionTimeS,
    missionDurationS,
    speedFactor,
    frame,
    currentKeyframe,
    maneuversSoFar,
    progress: missionDurationS > 0 ? missionTimeS / missionDurationS : 0,
    play,
    pause,
    togglePlay,
    restart,
    seek,
    setSpeedFactor,
    stepToKeyframe,
  };
}
