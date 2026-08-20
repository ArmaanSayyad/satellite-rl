"""Tests for the Phase 4 CollisionAvoidanceEnv.

Requires the full bsk_rl/Basilisk stack (not installed in the lightweight
CI environment, see docs/12-architecture.md and docs/17-env-
implementation-notes.md) -- skipped automatically if unavailable via
`pytest.importorskip`, rather than failing CI or silently not running
these when a contributor does have the stack installed locally.

Uses a low-relative-speed scenario throughout: docs/17's TCA-timing-
sensitivity finding means high-relative-speed scenarios need a proper
TCA-refinement step (not yet implemented, Phase 5 follow-up) to produce
realized encounters close to the targeted geometry -- these tests
validate pipeline correctness, not that specific arbitrary parameters
produce a specific miss distance.
"""

import numpy as np
import pytest

pytest.importorskip("bsk_rl")

from satellite_rl.env import CollisionAvoidanceEnv

# Parameters chosen in the regime where docs/17's timing-sensitivity
# finding is small relative to the miss distance (see that doc for why
# low relative speed matters here).
ENV_KWARGS = {
    "sim_rate": 5.0,
    "miss_distance_m": 300.0,
    "relative_speed_ms": 20.0,
    "schedule_days_before_tca": (0.2, 0.1, 0.05, 0.01, 0.0),
}


def test_env_checker_compliance():
    from gymnasium.utils.env_checker import check_env

    env = CollisionAvoidanceEnv(**ENV_KWARGS)
    check_env(env, skip_render_check=True)


def test_episode_runs_full_schedule_with_no_action():
    env = CollisionAvoidanceEnv(**ENV_KWARGS)
    _obs, _info = env.reset()
    n_steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _obs, _reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        n_steps += 1
    assert n_steps == len(ENV_KWARGS["schedule_days_before_tca"]) - 1
    assert terminated
    assert not truncated
    assert "pc_final" in info
    assert 0.0 <= info["pc_final"] <= 1.0


def test_never_maneuver_uses_no_fuel():
    env = CollisionAvoidanceEnv(**ENV_KWARGS)
    env.reset()
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["cumulative_fuel_used_ms"] == pytest.approx(0.0, abs=1e-9)
    assert info["maneuver_count"] == 0


def test_max_thrust_uses_fuel_and_reduces_final_risk():
    env = CollisionAvoidanceEnv(**ENV_KWARGS)
    env.reset()
    terminated = truncated = False
    info = {}
    while not (terminated or truncated):
        action = np.full(3, env.max_dv_ms / np.sqrt(3), dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
    assert info["cumulative_fuel_used_ms"] > 0.0
    assert info["maneuver_count"] > 0
    # A satellite thrusting away every step should end up far from the
    # secondary, well below any realistic collision-risk threshold.
    assert info["pc_final"] < 1e-6


def test_action_space_matches_max_dv():
    env = CollisionAvoidanceEnv(**ENV_KWARGS, max_dv_ms=5.0)
    assert env.action_space.shape == (3,)
    assert np.all(env.action_space.low == -5.0)
    assert np.all(env.action_space.high == 5.0)


def test_invalid_schedule_rejected():
    with pytest.raises(ValueError):
        CollisionAvoidanceEnv(schedule_days_before_tca=(1.0, 2.0, 0.0))  # not descending
    with pytest.raises(ValueError):
        CollisionAvoidanceEnv(schedule_days_before_tca=(2.0, 1.0, 0.5))  # doesn't end at 0.0


# Curriculum stage 2 (docs/19-curriculum-stage-2.md): sampled geometry.
SAMPLING_ENV_KWARGS = {
    "sample_geometry": True,
    "sim_rate": 5.0,
    "schedule_days_before_tca": (0.2, 0.1, 0.05, 0.01, 0.0),
    "targeting_seed": 0,
}


def test_sampling_env_checker_compliance():
    from gymnasium.utils.env_checker import check_env

    env = CollisionAvoidanceEnv(**SAMPLING_ENV_KWARGS)
    check_env(env, skip_render_check=True)


def test_sampling_env_draws_different_scenarios_across_resets():
    """Each reset must sample a genuinely different real event -- the
    whole point of curriculum stage 2 versus stage 1's fixed scenario.
    """
    env = CollisionAvoidanceEnv(**SAMPLING_ENV_KWARGS)
    samples = []
    for _ in range(5):
        env.reset()
        samples.append(env._sampler.current_sample["miss_distance"])
    assert len(set(samples)) > 1


def test_sampling_env_runs_full_episodes():
    env = CollisionAvoidanceEnv(**SAMPLING_ENV_KWARGS)
    for _ in range(3):
        env.reset()
        terminated = truncated = False
        n_steps = 0
        info = {}
        while not (terminated or truncated):
            _obs, _reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
            n_steps += 1
        assert n_steps == len(SAMPLING_ENV_KWARGS["schedule_days_before_tca"]) - 1
        assert terminated
        assert 0.0 <= info["pc_final"] <= 1.0


def test_sampling_env_pc_sigma_and_radius_vary_with_sample():
    """The Pc observation's sigma/combined_radius must actually track the
    sampled event, not silently stay at some stale/default value --
    directly exercises the fix in observations.make_collision_pc_fn that
    made these mutable per-episode state instead of a baked-in constant.
    """
    env = CollisionAvoidanceEnv(**SAMPLING_ENV_KWARGS)
    seen_sigmas = set()
    for _ in range(5):
        env.reset()
        seen_sigmas.add(env.satellites[0]._pc_sigma)
    assert len(seen_sigmas) > 1
