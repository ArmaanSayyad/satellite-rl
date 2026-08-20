"""Tests for satellite_rl.training.

Fast, synthetic-data checks only -- a real PPO training run takes many
minutes (Phase 6/docs/21-training-results.md found ~2 steps/sec effective
throughput including PPO overhead) and needs bsk_rl/Basilisk, so it's
exercised manually via `python -m satellite_rl.training.train_ppo`, not
in the automated suite. These tests cover the pieces that don't need a
real environment or a real trained model.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from satellite_rl.training.evaluate import run_episodes
from satellite_rl.training.plot_training_curve import load_monitor_csv


class _FakeEnv:
    """A minimal stand-in for CollisionAvoidanceEnv, fast and dependency-
    free, to test run_episodes()'s bookkeeping logic without needing a
    real Basilisk-backed environment.
    """

    def __init__(self, episode_length: int = 3):
        self.episode_length = episode_length
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        terminated = self._step_count >= self.episode_length
        info = {
            "cumulative_fuel_used_ms": float(self._step_count),
            "maneuver_count": self._step_count,
        }
        if terminated:
            info["pc_final"] = 0.001
        return np.zeros(4, dtype=np.float32), -0.1, terminated, False, info


def test_run_episodes_aggregates_correctly():
    env = _FakeEnv(episode_length=3)
    stats = run_episodes(env, action_fn=lambda obs: np.zeros(3), n_episodes=4)
    assert stats["n_episodes"] == 4
    assert stats["mean_reward"] == pytest.approx(-0.3, abs=1e-6)  # 3 steps * -0.1
    assert stats["mean_fuel_used_ms"] == pytest.approx(3.0)
    assert stats["mean_maneuver_count"] == pytest.approx(3.0)
    assert stats["mean_pc_final"] == pytest.approx(0.001)


def test_load_monitor_csv_skips_sb3_header(tmp_path):
    # SB3's Monitor writes a JSON comment header line, then a normal CSV
    # -- replicate that format exactly rather than assume.
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text(
        '#{"t_start": 123.0, "env_id": null}\n' "r,l,t\n" "-0.5,4,1.2\n" "-0.3,4,2.5\n"
    )
    df = load_monitor_csv(monitor_path)
    assert list(df.columns) == ["r", "l", "t"]
    assert len(df) == 2
    assert df["r"].tolist() == [-0.5, -0.3]


def test_load_monitor_csv_matches_pandas_direct_read(tmp_path):
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text('#{"t_start": 0.0}\n' "r,l,t\n" "1.0,2,0.1\n")
    df = load_monitor_csv(monitor_path)
    expected = pd.DataFrame({"r": [1.0], "l": [2], "t": [0.1]})
    pd.testing.assert_frame_equal(df, expected)
