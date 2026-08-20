"""Tests for satellite_rl.scenario.tca_refinement.

Requires the full bsk_rl/Basilisk stack -- skipped automatically in the
lightweight CI environment (see docs/12-architecture.md), same pattern as
tests/test_env.py.
"""

import numpy as np
import pytest

pytest.importorskip("Basilisk")

from satellite_rl.scenario.targeting import (
    example_leo_orbit,
    solve_secondary_initial_state_robust,
)
from satellite_rl.scenario.tca_refinement import refine_tca


def test_refine_tca_finds_a_minimum_at_or_better_than_nominal():
    """The refined separation must never be worse than the separation at
    the originally-assumed nominal instant -- refinement searches for the
    true local minimum near the nominal point, so it can only match or
    improve on it, never regress.
    """
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(1)
    nominal_tca = 0.2 * 86400.0
    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, nominal_tca, miss_distance_m=300.0, relative_speed_ms=200.0,
        orientation_angle_rad=1.0, rng=rng,
    )
    result = refine_tca(
        ego_r0, ego_v0, scenario.r_sec_t0, scenario.v_sec_t0, nominal_tca, sim_rate_s=5.0
    )
    assert result["min_separation_m"] <= result["nominal_separation_m"] + 1e-6


def test_refine_tca_offset_is_small_relative_to_total_duration():
    """The refined TCA should be close to the nominal one -- large
    corrections would indicate the targeting solver's assumed TCA is
    badly wrong, not just imprecise (a much bigger problem than this
    refinement step is meant to handle). Empirically (docs/18) observed
    offsets are at most tens of seconds against multi-hour-to-day
    durations.
    """
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(2)
    nominal_tca = 86400.0
    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, nominal_tca, miss_distance_m=1000.0, relative_speed_ms=2000.0,
        orientation_angle_rad=0.5, rng=rng,
    )
    result = refine_tca(
        ego_r0, ego_v0, scenario.r_sec_t0, scenario.v_sec_t0, nominal_tca, sim_rate_s=5.0
    )
    assert abs(result["refined_tca_s"] - nominal_tca) < 300.0


def test_refine_tca_returns_expected_keys_and_types():
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(3)
    nominal_tca = 0.1 * 86400.0
    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, nominal_tca, miss_distance_m=500.0, relative_speed_ms=1000.0,
        orientation_angle_rad=2.0, rng=rng,
    )
    result = refine_tca(
        ego_r0, ego_v0, scenario.r_sec_t0, scenario.v_sec_t0, nominal_tca, sim_rate_s=5.0
    )
    assert set(result.keys()) == {
        "refined_tca_s",
        "min_separation_m",
        "nominal_separation_m",
        "sample_resolution_s",
    }
    assert result["min_separation_m"] >= 0.0
    assert result["nominal_separation_m"] >= 0.0
