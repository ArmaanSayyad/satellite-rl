"""Tests of scientific bookkeeping and search behavior, independent of Basilisk."""

import numpy as np
import pytest

from satellite_rl.lab.evaluate import event_bootstrap, observation_sensitivity, summarize
from satellite_rl.lab.planner import plane_pc, search_plan


def estimate(miss=(0, 0)):
    return {"miss_plane_m": miss, "covariance_plane_m2": np.eye(2) * 100,
            "radius_m": 1.0, "fuel_remaining_ms": 1.0}


def test_planner_waits_when_safe_without_propagating():
    def fail(*args):
        raise AssertionError("safe event need not propagate candidates")
    result = search_plan(estimate((1000, 1000)), response_fn=fail)
    assert result.feasible
    assert result.action_rtn_ms == [0.0, 0.0, 0.0]


def test_planner_searches_nonradial_direction_and_timing():
    def response(_, wait):
        return np.array([[0, 1000 * (wait + 1), 0], [0, 0, 0]])
    result = search_plan(estimate(), waits_s=(0, 1), magnitudes=(.01, .1, 1),
                         response_fn=response)
    assert result.feasible
    assert result.wait_s == 1
    assert abs(result.action_rtn_ms[1]) > 0
    assert plane_pc(np.asarray(result.action_rtn_ms) @ response(None, result.wait_s).T,
                    np.eye(2) * 100, 1) <= 1e-4


def test_planner_reports_infeasible_and_respects_fuel():
    result = search_plan(estimate(), response_fn=lambda *_: np.zeros((2, 3)))
    assert not result.feasible
    assert np.linalg.norm(result.action_rtn_ms) <= 1


def test_metrics_do_not_dilute_high_risk_or_hide_false_positives():
    rows = [{"event_id": 1, "native_pc": .01, "pc_final": 1e-6, "fuel_ms": .1},
            {"event_id": 2, "native_pc": 1e-9, "pc_final": 1e-9, "fuel_ms": .2},
            {"event_id": 3, "native_pc": 1e-8, "pc_final": 1e-8, "fuel_ms": 0}]
    stats = summarize(rows)
    assert stats["mitigation_rate"] == 1
    assert stats["mitigation_ci95"] is None  # one risky family is not enough for an interval
    assert stats["false_positive_rate"] == .5
    assert stats["safe_mean_wasted_fuel_ms"] == .1
    assert stats["mean_fuel_regret_vs_feasible_online_planner_ms"] is None
    assert summarize([])["mitigation_rate"] is None


def test_bootstrap_groups_variants_and_declines_one_family():
    assert event_bootstrap([{"event_id": 1}] * 100, len) is None
    rows = [{"event_id": 1, "value": 0}] * 10 + [{"event_id": 2, "value": 1}]
    interval = event_bootstrap(rows, lambda rs: np.mean([r["value"] for r in rs]))
    assert interval == pytest.approx([0, 1])


def test_sensitivity_distinguishes_constant_policy():
    class Constant:
        def predict(self, obs, deterministic):
            return np.ones(3), None
    assert all(r["delta_action_norm_ms"] == 0 for r in
               observation_sensitivity(Constant(), np.zeros(4), {"pc": 0}))


def test_failed_simulations_do_not_become_measured_risk():
    rows = [{"event_id": 1, "native_pc": .01, "pc_final": None, "fuel_ms": 0},
            {"event_id": 2, "native_pc": None, "pc_final": .001, "fuel_ms": .1}]
    stats = summarize(rows)
    assert stats["mitigation_rate"] == 0
    assert stats["n_high_risk"] == 1
    assert stats["n_failed"] == stats["n_invalid_baseline"] == 1
    assert stats["pc_final_quantiles"]["max"] == .001
    assert stats["mean_risk_reduction_per_ms"] is None


def test_categorical_wait_and_rtn_magnitudes():
    pytest.importorskip("gymnasium")
    from satellite_rl.lab.train import ACTION_TABLE, decode_action
    assert np.array_equal(decode_action(0, "discrete19"), [0, 0, 0])
    assert ACTION_TABLE.shape == (19, 3)
    assert np.allclose(np.unique(np.round(np.linalg.norm(ACTION_TABLE, axis=1), 3)),
                       [0, .01, .1, 1])
    with pytest.raises(ValueError):
        decode_action(19, "discrete19")


def test_hindsight_grid_finds_tangential_solution_and_labels_budget():
    from satellite_rl.lab.planner import hindsight_grid_reference
    class Env:
        schedule_s = (1, 0)
        @property
        def unwrapped(self):
            return self
        def reset(self, **kwargs):
            return None, {}
        def step(self, action):
            pc = max(0., .001 - .001 * abs(action[1]))
            return None, 0., True, False, {"pc_final": pc,
                "cumulative_fuel_used_ms": float(np.linalg.norm(action))}
    result = hindsight_grid_reference(Env(), 1, magnitudes=(.1, 1.))
    assert result["minimum_grid_fuel_established"]
    assert result["best"]["fuel_ms"] == pytest.approx(1)
    assert abs(result["best"]["action_rtn_ms"][1]) == 1
    assert not result["globally_optimal"]
    limited = hindsight_grid_reference(Env(), 1, magnitudes=(.1, 1.), max_evaluations=2)
    assert not limited["completed_search"]
    assert limited["best"] is None
