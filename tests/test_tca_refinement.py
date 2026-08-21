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
from satellite_rl.scenario.tca_refinement import correct_targeting_geometry, refine_tca


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


# docs/26-precise-targeting.md: J2-only targeting is off by ~100-200m at
# these short (0.2-day) lead times, regardless of target size -- invisible
# at km-scale miss distances (everything above), but it swamps the
# tens-of-meters targets real high-risk events need. Test cases below
# span the validated range (2-642m, 0.2-3 day lead times, docs/26).
# max_expected_iters: 0.2-day (the actual curriculum-stage-2 production
# regime, per env/scenario_sampling.py) converges in 1-3; longer lead
# times converge more slowly (Sun's third-body perturbation makes the
# achieved-vs-requested relationship less linear over a longer arc) -- a
# 3-day case needed 7 (monotonically improving: 97m/15m/2.3m/0.36m at
# iterations 4-7, not diverging, just slow).
@pytest.mark.parametrize(
    "miss_m,angle_rad,speed_ms,tca_days,seed,max_expected_iters",
    [
        (38.0, 1.0, 8000.0, 0.2, 1, 3),
        (10.0, 2.5, 7500.0, 0.2, 2, 3),
        (642.0, 5.5, 6000.0, 0.2, 5, 3),
        (98.45, 0.433, 6260.23, 0.2, 1010, 3),  # oscillation edge case found during development
        (38.0, 1.0, 8000.0, 3.0, 6, 8),  # longer lead time, converges slower
    ],
)
def test_correct_targeting_geometry_converges_to_sub_meter_accuracy(
    miss_m, angle_rad, speed_ms, tca_days, seed, max_expected_iters
):
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(seed)
    nominal_tca_s = tca_days * 86400.0
    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, nominal_tca_s, miss_m, speed_ms, angle_rad, rng
    )
    _corrected, diagnostics = correct_targeting_geometry(ego_r0, ego_v0, scenario, nominal_tca_s)
    assert diagnostics["final_error_m"] < 1.0
    assert diagnostics["n_basilisk_calls"] <= max_expected_iters


def test_correct_targeting_geometry_only_changes_t0_state():
    """r_sec_tca_target/v_sec_tca_target/r_ego_tca/v_ego_tca/
    miss_distance_target/relative_speed_target define the TARGET and must
    stay exactly as the J2 solver produced them -- only r_sec_t0/v_sec_t0
    (the actual initial state used to achieve that target) may change.
    """
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(4)
    nominal_tca_s = 0.2 * 86400.0
    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, nominal_tca_s, 50.0, 8000.0, 1.5, rng
    )
    corrected, _diagnostics = correct_targeting_geometry(ego_r0, ego_v0, scenario, nominal_tca_s)
    assert np.array_equal(corrected.r_ego_tca, scenario.r_ego_tca)
    assert np.array_equal(corrected.v_ego_tca, scenario.v_ego_tca)
    assert np.array_equal(corrected.r_sec_tca_target, scenario.r_sec_tca_target)
    assert np.array_equal(corrected.v_sec_tca_target, scenario.v_sec_tca_target)
    assert corrected.miss_distance_target == scenario.miss_distance_target
    assert corrected.relative_speed_target == scenario.relative_speed_target


def test_correct_targeting_geometry_improves_on_uncorrected_j2_solve():
    """The whole point: the corrected state's ACTUAL Basilisk-simulated
    miss distance must land much closer to target than the raw J2 solve
    did (docs/26 measured ~100-200m raw error at this lead time for
    small targets).
    """
    ego_r0, ego_v0 = example_leo_orbit()
    rng = np.random.default_rng(1)
    nominal_tca_s = 0.2 * 86400.0
    scenario = solve_secondary_initial_state_robust(
        ego_r0, ego_v0, nominal_tca_s, 38.0, 8000.0, 1.0, rng
    )
    from satellite_rl.pc.geometry import encounter_plane_basis
    from satellite_rl.scenario.tca_refinement import _fly_passive_pair

    v_rel_target = scenario.v_sec_tca_target - scenario.v_ego_tca
    basis = encounter_plane_basis(v_rel_target)
    target_2d = basis.T @ (scenario.r_sec_tca_target - scenario.r_ego_tca)

    times_s, r1, r2 = _fly_passive_pair(
        ego_r0, ego_v0, scenario.r_sec_t0, scenario.v_sec_t0, nominal_tca_s, sim_rate_s=2.0
    )
    idx = int(np.argmin(np.abs(times_s - nominal_tca_s)))
    uncorrected_error_m = np.linalg.norm(basis.T @ (r2[idx] - r1[idx]) - target_2d)

    _corrected, diagnostics = correct_targeting_geometry(ego_r0, ego_v0, scenario, nominal_tca_s)
    assert diagnostics["final_error_m"] < uncorrected_error_m / 10.0
