"""Tests for satellite_rl.training.baselines -- pure logic (threshold
heuristic) and the bisection search (hindsight oracle), both exercised
against fake data/envs so no bsk_rl/Basilisk install is needed.
"""

import numpy as np
import pytest

from satellite_rl.training.baselines import (
    RADIAL_DV_DIRECTION,
    hindsight_oracle_fuel,
    threshold_heuristic_action,
)


def test_threshold_heuristic_no_action_below_threshold():
    obs = np.array([1e-6, 1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action = threshold_heuristic_action(obs, max_dv_ms=10.0, pc_threshold=1e-4)
    assert np.allclose(action, 0.0)


def test_threshold_heuristic_acts_above_threshold():
    obs = np.array([1e-2, 1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action = threshold_heuristic_action(obs, max_dv_ms=10.0, pc_threshold=1e-4)
    assert np.allclose(action, RADIAL_DV_DIRECTION * 10.0)


class _FakeOracleEnv:
    """pc_final decreases linearly with the magnitude of the FIRST
    action's radial (index 0) component, clipped at zero -- deterministic
    given (base_pc, slope), to test the bisection search's convergence
    and edge cases without a real simulator.
    """

    def __init__(self, base_pc: float, slope: float):
        self.base_pc = base_pc
        self.slope = slope
        self._first_action = None
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        self._first_action = None
        return np.zeros(9, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        if self._first_action is None:
            self._first_action = np.asarray(action, dtype=float)
        terminated = self._step_count >= 2
        info = {"cumulative_fuel_used_ms": 0.0, "maneuver_count": 0}
        if terminated:
            magnitude = abs(self._first_action[0])
            info["pc_final"] = max(0.0, self.base_pc - self.slope * magnitude)
        return np.zeros(9, dtype=np.float32), 0.0, terminated, False, info


def test_hindsight_oracle_converges_to_known_minimum():
    # base_pc=1e-3, slope=1e-4, threshold=1e-4 -> need magnitude >= 9.
    env = _FakeOracleEnv(base_pc=1e-3, slope=1e-4)
    fuel, achieved = hindsight_oracle_fuel(
        env, seed=0, max_dv_ms=10.0, pc_threshold=1e-4, tol_ms=0.01, max_iters=20
    )
    assert achieved is True
    assert fuel == pytest.approx(9.0, abs=0.05)


def test_hindsight_oracle_zero_fuel_when_already_below_threshold():
    env = _FakeOracleEnv(base_pc=1e-5, slope=1e-4)
    fuel, achieved = hindsight_oracle_fuel(env, seed=0, max_dv_ms=10.0, pc_threshold=1e-4)
    assert achieved is True
    assert fuel == 0.0


def test_hindsight_oracle_reports_infeasible_honestly():
    # Needs magnitude ~90 to reach threshold, but max_dv_ms=10 -- the
    # oracle must report achieved=False, not silently extrapolate.
    env = _FakeOracleEnv(base_pc=1e-3, slope=1e-5)
    fuel, achieved = hindsight_oracle_fuel(env, seed=0, max_dv_ms=10.0, pc_threshold=1e-4)
    assert achieved is False
    assert fuel == 10.0
