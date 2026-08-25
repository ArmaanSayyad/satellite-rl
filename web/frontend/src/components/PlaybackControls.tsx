import type { PlaybackState } from "../useSimulationPlayback";
import { SPEED_OPTIONS } from "../useSimulationPlayback";

interface Props {
  playback: PlaybackState;
}

export default function PlaybackControls({ playback }: Props) {
  const { playing, finished, speedFactor, togglePlay, restart, stepToKeyframe, setSpeedFactor } = playback;

  return (
    <div className="playback-controls">
      <button
        className="playback-btn"
        onClick={() => stepToKeyframe(-1)}
        title="Jump to previous decision point"
        aria-label="Previous decision"
      >
        ⏮
      </button>

      {finished ? (
        <button className="playback-btn playback-btn-main" onClick={restart} title="Replay from the start">
          ↻ Replay
        </button>
      ) : (
        <button className="playback-btn playback-btn-main" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
      )}

      <button
        className="playback-btn"
        onClick={() => stepToKeyframe(1)}
        title="Jump to next decision point"
        aria-label="Next decision"
      >
        ⏭
      </button>

      <div className="speed-group" role="group" aria-label="Playback speed">
        {SPEED_OPTIONS.map((s) => (
          <button
            key={s}
            className={`speed-btn ${speedFactor === s ? "active" : ""}`}
            onClick={() => setSpeedFactor(s)}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
