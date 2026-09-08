"""Scientific invariants: source grouping, seeded sensing and truth isolation."""

from itertools import pairwise

import numpy as np
import pytest

from satellite_rl.lab.data import grouped_split, load_keyed_schedules


def test_partitions_exclude_every_variant_of_a_source():
    split = grouped_split(list(range(100)) * 3)
    assert split == grouped_split(reversed(list(range(100)) * 3))
    assert sum(map(len, split.values())) == 100
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["validation"]) & set(split["test"])


def test_keyed_schedules_preserve_source_ids():
    schedules = load_keyed_schedules()
    assert "170" in schedules
    assert all(np.isfinite(schedules["170"]))


def test_schedule_keeps_minimum_interval_at_tca():
    pytest.importorskip("Basilisk")
    from satellite_rl.env.scenario_sampling import _clean_schedule

    schedule = _clean_schedule([1, 0.0001])
    assert all(a - b >= 60 for a, b in pairwise(schedule))


def test_v2_noise_is_reproducible_and_does_not_change_truth():
    pytest.importorskip("Basilisk")
    from satellite_rl.lab.environment import make_v2_env

    env = make_v2_env(
        sample_geometry=False,
        evolve_uncertainty=False,
        relative_speed_ms=20,
        schedule_days_before_tca=(0.002, 0.001, 0),
        sim_rate=5,
    )
    try:
        obs1, info1 = env.reset(seed=23)
        obs2, info2 = env.reset(seed=23)
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(
            info1["truth"]["predicted_miss_plane_m"], info2["truth"]["predicted_miss_plane_m"]
        )
        assert obs1.shape == (72,)
        assert np.all(obs1[:48] == 0)
        assert not np.allclose(
            info1["observation"]["miss_plane_m"], info1["truth"]["predicted_miss_plane_m"]
        )
        _, info3 = env.reset(seed=23, options={"observation_seed": 24})
        np.testing.assert_allclose(
            info1["truth"]["predicted_miss_plane_m"], info3["truth"]["predicted_miss_plane_m"]
        )
        assert info1["observation"]["miss_plane_m"] != info3["observation"]["miss_plane_m"]
    finally:
        env.close()


def test_explicit_source_and_covariance_are_joined():
    pytest.importorskip("Basilisk")
    from satellite_rl.lab.environment import make_v2_env

    env = make_v2_env(event_ids=[170], max_lead_days=0.002, sim_rate=5)
    try:
        _, info = env.reset(seed=3, options={"event_id": 170})
        assert info["event_id"] == 170
        sampler = env.unwrapped._sampler
        first_state = sampler._cached_scenario.r_sec_t0.copy()
        first_rng = repr(sampler.rng.bit_generator.state)
        env.reset(seed=3, options={"event_id": 170})
        assert sampler.target_cache_hits == 1
        np.testing.assert_array_equal(first_state, sampler._cached_scenario.r_sec_t0)
        assert repr(sampler.rng.bit_generator.state) == first_rng
        row = sampler.evolution_df[sampler.evolution_df.event_id == 170].iloc[0]
        assert sampler.sigma_xz_at_fraction(1) == pytest.approx(
            (row.sigma_x_last, row.sigma_z_last)
        )
        with pytest.raises(ValueError, match="partition"):
            env.reset(seed=3, options={"event_id": 171})
    finally:
        env.close()


def test_basilisk_probe_returns_exact_fractional_epoch():
    pytest.importorskip("Basilisk")
    from satellite_rl.scenario.targeting import example_leo_orbit
    from satellite_rl.scenario.tca_refinement import _fly_passive_pair

    r, v = example_leo_orbit()
    times, r1, v1, r2, v2 = _fly_passive_pair(r, v, r, v, 123.4567, 5.0)
    assert times[-1] == pytest.approx(123.4567, abs=1e-9)
    np.testing.assert_allclose(r1[-1], r2[-1], atol=1e-8)
    np.testing.assert_allclose(v1[-1], v2[-1], atol=1e-8)
    _, refined_r, refined_v, _, _ = _fly_passive_pair(r, v, r, v, 123.4567, 1.0)
    assert np.linalg.norm(r1[-1] - refined_r[-1]) < 0.01
    assert np.linalg.norm(v1[-1] - refined_v[-1]) < 0.001


def test_forced_dangerous_source_gets_precision_without_sampling_lottery(monkeypatch):
    pytest.importorskip("Basilisk")
    import pandas as pd

    from satellite_rl.env import scenario_sampling
    from satellite_rl.scenario.targeting import example_leo_orbit

    calls = []

    def correct(r, v, scenario, duration):
        calls.append(duration)
        return scenario, {"final_error_m": 0.1}

    monkeypatch.setattr(scenario_sampling, "correct_targeting_geometry", correct)
    r, v = example_leo_orbit()
    data = pd.DataFrame(
        [
            {
                "event_id": 4,
                "miss_distance": 200.0,
                "relative_speed": 20.0,
                "sigma_x": 50.0,
                "sigma_z": 100.0,
                "combined_radius": 2.0,
                "alignment_angle_rad": 1.0,
                "native_pc": 0.0,
                "esa_reported_pc": 0.001,
            }
        ]
    )
    sampler = scenario_sampling.SecondaryScenarioSampler(
        data, r, v, np.random.default_rng(4), nominal_tca_s=100, high_risk_fraction=0
    )
    sampler.requested_event_id = 4
    assert sampler.current_sample["precise_targeting_error_m"] == 0.1
    assert calls == [100]
