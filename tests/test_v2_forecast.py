"""Full-dynamics sensor forecasts must match subsequent physical no-action coasts."""
import json

import numpy as np
import pytest

pytest.importorskip("Basilisk")

from satellite_rl.lab.environment import make_v2_env


def test_full_forecast_matches_initial_and_intermediate_epoch_terminal_truth():
    env = make_v2_env(
        event_ids=[8767], evolve_uncertainty=False,
        schedule_days_before_tca=(0.2, 0.1, 0.0),
        observation_noise_scale=1.0, forecast_fidelity="basilisk",
    )
    try:
        _, initial = env.reset(seed=100, options={"event_id": 8767, "observation_seed": 123})
        _, _, _, _, intermediate = env.step(np.zeros(3))
        _, _, terminated, truncated, final = env.step(np.zeros(3))
        assert terminated and not truncated
        assert final["pc_final_valid"]
        ego, sec = env.unwrapped.satellites
        actual = np.array(sec.dynamics.r_BN_N) - np.array(ego.dynamics.r_BN_N)
        initial_error = np.linalg.norm(np.array(initial["truth"]["predicted_relative_r_eci_m"]) - actual)
        intermediate_error = np.linalg.norm(np.array(intermediate["truth"]["predicted_relative_r_eci_m"]) - actual)
        assert initial_error < 1.0
        assert intermediate_error < 1.0
        packet = initial["observation"]
        assert packet["version"] == "encounter-history-v2.1-basilisk"
        assert packet["forecast_fidelity"] == "basilisk"
        sigma = np.sqrt(np.diag(packet["covariance_plane_m2"]))
        actual_noise = (np.array(packet["miss_plane_m"]) - initial["truth"]["predicted_miss_plane_m"]) / sigma
        expected_noise = np.random.default_rng(np.random.SeedSequence([123, 2026])).normal(size=2)
        np.testing.assert_allclose(actual_noise, expected_noise, atol=1e-12)
        print(json.dumps({
            "source_event_id": 8767, "geometry_seed": 100, "observation_seed": 123,
            "forecast_initial_error_m": float(initial_error),
            "forecast_intermediate_error_m": float(intermediate_error),
            "actual_terminal_miss_m": float(np.linalg.norm(actual)),
            "final_pc_chan": final["pc_final"],
            "actual_terminal_time_s": env.unwrapped.simulator.sim_time,
        }))
    finally:
        env.close()
