"""Noisy encounter-estimate history over the unchanged Basilisk v1 simulator.

Noise is generated in the predicted TCA encounter plane, where the fitted
covariance is defined. It is an explicit IID CDM-error approximation, not a
reconstruction of historical orbit determination. Truth never enters features.
"""

from collections import deque

import gymnasium as gym
import numpy as np

from satellite_rl.pc import compute_pc
from satellite_rl.pc.geometry import encounter_plane_basis
from satellite_rl.scenario.targeting import propagate_state

from .data import load_keyed_schedules
from .observations import FEATURE_NAMES, OBSERVATION_VERSION, encode_estimate


class EncounterHistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_length=3, observation_noise_scale=1.0, forecast_fidelity="j2"):
        super().__init__(env)
        if (
            history_length < 1
            or not np.isfinite(observation_noise_scale)
            or observation_noise_scale < 0
        ):
            raise ValueError("History must be positive and noise scale nonnegative")
        self.history_length = history_length
        self.observation_noise_scale = observation_noise_scale
        if forecast_fidelity not in ("j2", "basilisk"):
            raise ValueError("forecast_fidelity must be j2 or basilisk")
        self.forecast_fidelity = forecast_fidelity
        self.observation_version = (
            "encounter-history-v2.1-basilisk"
            if forecast_fidelity == "basilisk"
            else OBSERVATION_VERSION
        )
        self.observation_space = gym.spaces.Box(
            -10, 10, (len(FEATURE_NAMES) * history_length,), dtype=np.float32
        )
        self.history = deque(maxlen=history_length)
        self.noise_rng = np.random.default_rng()
        self.latest_estimate = {}
        self.previous_action = np.zeros(3)

    def _observe(self, info):
        base = self.env.unwrapped
        ego, secondary = base.satellites
        r = np.array(ego.dynamics.r_BN_N)
        v = np.array(ego.dynamics.v_BN_N)
        dt = float(base.schedule_s[base.schedule_index])
        if self.forecast_fidelity == "basilisk":
            from .forecast import forecast_states

            r1, v1, r2, v2 = forecast_states(
                r,
                v,
                secondary.dynamics.r_BN_N,
                secondary.dynamics.v_BN_N,
                dt,
                base.simulator.sim_time,
                base.simulator.sim_rate,
            )
        else:
            r1, v1 = propagate_state(r, v, dt)
            r2, v2 = propagate_state(
                np.array(secondary.dynamics.r_BN_N), np.array(secondary.dynamics.v_BN_N), dt
            )
        basis = encounter_plane_basis(v2 - v1)
        sigma = np.array([ego._pc_sigma_x, ego._pc_sigma_z])
        cov = np.diag(sigma**2)
        true_miss = basis.T @ (r2 - r1)
        observed_miss = (
            true_miss + self.noise_rng.normal(size=2) * sigma * self.observation_noise_scale
        )
        radius = float(ego._pc_combined_radius)
        pc = compute_pc(
            basis @ observed_miss, v2 - v1, basis @ cov @ basis.T, radius, method="chan"
        )
        radial = r / np.linalg.norm(r)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        rtn = np.column_stack([radial, np.cross(normal, radial), normal])
        index = base.schedule_index
        next_dt = dt - base.schedule_s[index + 1] if index + 1 < len(base.schedule_s) else 0.0
        remaining = float(ego.fsw.dv_available)
        packet = {
            "version": self.observation_version,
            "forecast_fidelity": self.forecast_fidelity,
            "kind": "generated_measurement",
            "miss_plane_m": observed_miss.tolist(),
            "covariance_plane_m2": cov.tolist(),
            "basis_eci": basis.tolist(),
            "basis_rtn": (rtn.T @ basis).tolist(),
            "estimated_pc": float(pc),
            "ego_r_eci_m": r.tolist(),
            "ego_v_eci_ms": v.tolist(),
            "time_to_tca_s": dt,
            "next_update_s": float(next_dt),
            "radius_m": radius,
            "fuel_remaining_ms": remaining,
            "fuel_fraction": remaining / self.initial_fuel,
            "noise_scale": self.observation_noise_scale,
            "noise_model": "iid_encounter_plane_gaussian",
            "relative_velocity_assumption": "known_direction",
        }
        self.latest_estimate = packet
        features = encode_estimate(
            packet, self.previous_action, index / max(len(base.schedule_s) - 1, 1)
        )
        self.history.append(features)
        info = dict(info)
        info["observation"] = packet
        info["truth"] = {
            "predicted_miss_plane_m": true_miss.tolist(),
            "predicted_relative_r_eci_m": (r2 - r1).tolist(),
            "prediction_fidelity": self.forecast_fidelity,
        }
        return np.concatenate(self.history), info

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        noise_seed = (options or {}).get("observation_seed", seed)
        if noise_seed is not None:
            self.noise_rng = np.random.default_rng(np.random.SeedSequence([noise_seed, 2026]))
        self.initial_fuel = max(float(self.env.unwrapped.satellites[0].fsw.dv_available), 1e-9)
        self.previous_action = np.zeros(3)
        self.history.clear()
        for _ in range(self.history_length - 1):
            self.history.append(np.zeros(len(FEATURE_NAMES), dtype=np.float32))
        return self._observe(info)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.previous_action = np.asarray(action).copy()
        obs, info = self._observe(info)
        return obs, reward, terminated, truncated, info


def make_v2_env(
    event_ids=None,
    history_length=3,
    observation_noise_scale=1.0,
    max_lead_days=None,
    forecast_fidelity="j2",
    **env_kwargs,
):
    from satellite_rl.env import CollisionAvoidanceEnv

    if max_lead_days is not None and (not np.isfinite(max_lead_days) or max_lead_days * 86400 < 60):
        raise ValueError("Bounded lead windows must be finite and at least 60 seconds")
    if not 0 < env_kwargs.get("sim_rate", 1.0) <= 30:
        raise ValueError("V2 sim_rate must be in (0, 30] seconds for the 60-second minimum gap")

    env_kwargs.setdefault("sample_geometry", True)
    env_kwargs.setdefault("evolve_uncertainty", True)
    if env_kwargs["evolve_uncertainty"]:
        env_kwargs.setdefault("keyed_schedules", load_keyed_schedules())
    env = CollisionAvoidanceEnv(event_ids=event_ids, max_lead_days=max_lead_days, **env_kwargs)
    return EncounterHistoryWrapper(env, history_length, observation_noise_scale, forecast_fidelity)
