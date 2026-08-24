import { useCallback, useEffect, useRef, useState } from "react";
import type { Frame, Keyframe, SimulationResult } from "./types";
import { frameAtTime } from "./physics";

const PLAYBACK_DURATION_S = 26; // wall-clock seconds to play the whole mission

export interface PlaybackState {
  playing: boolean;
  finished: boolean;
  missionTimeS: number;
  missionDurationS: number;
  frame: Frame | null;
  currentKeyframe: Keyframe | null;
  maneuversSoFar: number;
  progress: number; // 0..1
}

export function useSimulationPlayback(result: SimulationResult | null) {
  const [state, setState] = useState<PlaybackState>({
    playing: false,
    finished: false,
    missionTimeS: 0,
    missionDurationS: 0,
    frame: null,
    currentKeyframe: null,
    maneuversSoFar: 0,
    progress: 0,
  });

  const rafRef = useRef<number | null>(null);
  const startedAtRef = useRef<number | null>(null);

  const stop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  useEffect(() => stop, [stop]);

  useEffect(() => {
    stop();
    if (!result) {
      setState((s) => ({ ...s, playing: false, finished: false, missionTimeS: 0, frame: null }));
      return;
    }
    const missionDurationS = result.dense_frames[result.dense_frames.length - 1]?.t_s ?? 0;
    const speedMultiplier = missionDurationS / PLAYBACK_DURATION_S;
    startedAtRef.current = null;

    const tick = (wallMs: number) => {
      if (startedAtRef.current === null) startedAtRef.current = wallMs;
      const wallElapsedS = (wallMs - startedAtRef.current) / 1000;
      const missionTimeS = Math.min(wallElapsedS * speedMultiplier, missionDurationS);
      const frame = frameAtTime(result.dense_frames, missionTimeS);
      let currentKeyframe: Keyframe | null = result.keyframes[0] ?? null;
      let maneuversSoFar = 0;
      for (const kf of result.keyframes) {
        if (kf.t_s <= missionTimeS) currentKeyframe = kf;
      }
      for (const d of result.decisions) {
        if (d.t_s <= missionTimeS) maneuversSoFar += 1;
      }
      const finished = missionTimeS >= missionDurationS;
      setState({
        playing: !finished,
        finished,
        missionTimeS,
        missionDurationS,
        frame,
        currentKeyframe,
        maneuversSoFar,
        progress: missionDurationS > 0 ? missionTimeS / missionDurationS : 1,
      });
      if (!finished) rafRef.current = requestAnimationFrame(tick);
    };

    setState((s) => ({ ...s, playing: true, finished: false, missionDurationS }));
    rafRef.current = requestAnimationFrame(tick);
    return stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  return state;
}
