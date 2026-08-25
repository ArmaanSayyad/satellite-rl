"""Runs one full collision-avoidance episode with the trained policy and
returns everything the frontend needs to visualize it: real orbital
trajectories (dense-sampled via the same Basilisk physics used
throughout this project, not interpolated), the policy's actual
decisions, and the resulting outcome.

Scenario selection: draws one of the 7 real historical events confirmed
(docs/27-riskaware-training-results.md) to actually cross
pc_threshold=1e-4 -- the same targeted evaluation setup used to validate
the trained policy, not a random real event (most real events are safe,
per docs/23; a random pick would usually show "nothing happens").
`high_risk_augment=False` -- the real event's own exact geometry, not a
resampled variant, so what's shown is a genuine historical near-miss.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from satellite_rl.env import CollisionAvoidanceEnv
from satellite_rl.scenario.targeting import propagate_state
from satellite_rl.scenario.tca_refinement import _fly_passive_pair

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(REPO_ROOT / "runs" / "ppo_stage2_riskaware_run1")

SCHEDULE_DAYS_BEFORE_TCA = (0.2, 0.1, 0.05, 0.01, 0.0)
# The exact pool_fraction found (docs/27) to isolate precisely the 7 real
# events ESA-anchored-ranked above pc_threshold=1e-4, out of 8,672.
HIGH_RISK_POOL_FRACTION = 0.00081
TARGETING_SEED = 999  # held-out from training's targeting_seed=0

ENV_KWARGS = {
    "sample_geometry": True,
    "schedule_days_before_tca": SCHEDULE_DAYS_BEFORE_TCA,
    "targeting_seed": TARGETING_SEED,
    "high_risk_fraction": 1.0,
    "high_risk_pool_fraction": HIGH_RISK_POOL_FRACTION,
    "high_risk_augment": False,
    "high_risk_precise_targeting": True,
}

# Dense-resample every ~60s of mission time between decision points, for
# smooth animation -- capped so a long (0.2-day) interval doesn't produce
# an excessive frame count. This is a real re-propagation of the same
# Basilisk physics (see tca_refinement._fly_passive_pair), not a fake
# interpolation between the two real endpoint states.
TARGET_SAMPLE_INTERVAL_S = 60.0
MAX_SAMPLES_PER_INTERVAL = 300
MIN_SAMPLES_PER_INTERVAL = 8

_model: PPO | None = None
_env: CollisionAvoidanceEnv | None = None
_scenario_seeds: dict[float, int] | None = None


def _get_model() -> PPO:
    global _model
    if _model is None:
        _model = PPO.load(MODEL_PATH)
    return _model


def _get_env() -> CollisionAvoidanceEnv:
    global _env
    if _env is None:
        _env = CollisionAvoidanceEnv(**ENV_KWARGS)
    return _env


def discover_scenario_seeds(max_seed: int = 300) -> dict[float, int]:
    """Find one seed producing each of the 7 real events in the
    ESA-anchored-ranked pool, by trying seeds in order and keying on
    each event's (unique, per docs/27) real miss_distance. Cached after
    the first call -- the pool is fixed (built from the static
    geometry_events.csv), so this mapping never changes at runtime.
    """
    global _scenario_seeds
    if _scenario_seeds is not None:
        return _scenario_seeds

    env = _get_env()
    pool_size = len(env._sampler.high_risk_df)
    found: dict[float, int] = {}
    seed = 0
    while len(found) < pool_size and seed < max_seed:
        env.reset(seed=seed)
        miss_distance = round(env._sampler.current_sample["miss_distance"], 1)
        if miss_distance not in found:
            found[miss_distance] = seed
        seed += 1

    if len(found) < pool_size:
        raise RuntimeError(
            f"Only found {len(found)}/{pool_size} distinct real events within "
            f"{max_seed} seed trials -- raise max_seed."
        )
    _scenario_seeds = found
    return found


def list_scenarios() -> list[dict]:
    """The 7 known real events, with their seeds, for a frontend scenario
    picker. Sorted by miss distance (closest/most dangerous first).
    """
    seeds = discover_scenario_seeds()
    env = _get_env()
    out = []
    for _, row in env._sampler.high_risk_df.iterrows():
        miss_distance = round(float(row["miss_distance"]), 1)
        seed = seeds.get(miss_distance)
        if seed is None:
            continue
        out.append(
            {
                "seed": seed,
                "miss_distance_m": float(row["miss_distance"]),
                "relative_speed_ms": float(row["relative_speed"]),
                "native_pc": float(row["native_pc"]),
                "combined_radius_m": float(row["combined_radius"]),
            }
        )
    out.sort(key=lambda r: r["miss_distance_m"])
    return out


def _run_never_maneuver_baseline(env: CollisionAvoidanceEnv, seed: int) -> dict:
    """Re-run the SAME real scenario (same seed -> same exact, non-
    augmented geometry, per discover_scenario_seeds' own reliance on this
    determinism) with a policy that never burns fuel, so the demo can
    show what the encounter's real risk looked like unmitigated -- the
    actual baseline docs/27 evaluates the trained policy against, not
    currently visible anywhere in the UI.
    """
    zero_action = np.zeros(3, dtype=np.float32)
    expected_decisions = len(env.schedule_s) - 1
    _obs, _info = env.reset(seed=seed)
    n_steps = 0
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        _obs, _reward, terminated, truncated, info = env.step(zero_action)
        n_steps += 1
    return {
        "policy": "never_maneuver",
        "pc_final": info.get("pc_final"),
        "collision_occurred": n_steps < expected_decisions,
    }


def _dense_resample(
    ego_r0, ego_v0, sec_r0, sec_v0, duration_s: float
) -> list[dict]:
    """Real Basilisk propagation (not interpolation) between two known
    real states, at fine time resolution, for smooth animation.
    """
    if duration_s <= 0:
        return []
    n_samples = int(np.clip(duration_s / TARGET_SAMPLE_INTERVAL_S, MIN_SAMPLES_PER_INTERVAL, MAX_SAMPLES_PER_INTERVAL))
    sim_rate_s = duration_s / n_samples
    times_s, r1, v1, r2, v2 = _fly_passive_pair(ego_r0, ego_v0, sec_r0, sec_v0, duration_s, sim_rate_s)
    return [
        {
            "t_s": float(times_s[i]),
            "ego_r": r1[i].tolist(),
            "ego_v": v1[i].tolist(),
            "sec_r": r2[i].tolist(),
            "sec_v": v2[i].tolist(),
        }
        for i in range(len(times_s))
    ]


def run_simulation(seed: int | None = None) -> dict:
    """Run one full episode with the trained policy on a real dangerous
    scenario, and return the full trajectory + outcome. `seed` selects
    one of the 7 known scenarios (see list_scenarios()); a random one is
    drawn if omitted.
    """
    seeds = discover_scenario_seeds()
    if seed is None:
        seed = random.choice(list(seeds.values()))
    elif seed not in seeds.values():
        raise ValueError(f"seed {seed} is not one of the known scenario seeds: {sorted(seeds.values())}")

    env = _get_env()
    model = _get_model()

    obs, _info = env.reset(seed=seed)
    sample = dict(env._sampler.current_sample)

    def _ego_state():
        return (
            np.array(env.satellites[0].dynamics.r_BN_N),
            np.array(env.satellites[0].dynamics.v_BN_N),
        )

    def _sec_state():
        return (
            np.array(env.satellites[1].dynamics.r_BN_N),
            np.array(env.satellites[1].dynamics.v_BN_N),
        )

    ego_r, ego_v = _ego_state()
    sec_r, sec_v = _sec_state()
    fuel0 = float(env.satellites[0].fsw.dv_available)

    keyframes = [
        {
            "t_s": 0.0,
            "ego_r": ego_r.tolist(),
            "ego_v": ego_v.tolist(),
            "sec_r": sec_r.tolist(),
            "sec_v": sec_v.tolist(),
            "fuel_ms": fuel0,
            "pc_estimate": float(obs[0]),
        }
    ]
    decisions = []
    dense_frames = []
    t_elapsed = 0.0
    terminated = truncated = False
    info: dict = {}

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        fuel_before = float(env.satellites[0].fsw.dv_available)
        current_idx = env.schedule_index
        duration_s = env.schedule_s[current_idx] - env.schedule_s[current_idx + 1]
        pre_ego_r = ego_r  # pre-maneuver position; impulsive dv doesn't move it
        pre_sec_r, pre_sec_v = sec_r, sec_v

        obs, _reward, terminated, truncated, info = env.step(action)

        fuel_after = float(env.satellites[0].fsw.dv_available)
        t_elapsed += duration_s
        ego_r, ego_v = _ego_state()
        sec_r, sec_v = _sec_state()

        decisions.append(
            {
                "t_s": t_elapsed,
                "action_dv_ms": [float(x) for x in action],
                "action_magnitude_ms": float(np.linalg.norm(action)),
                "fuel_before_ms": fuel_before,
                "fuel_after_ms": fuel_after,
            }
        )
        keyframes.append(
            {
                "t_s": t_elapsed,
                "ego_r": ego_r.tolist(),
                "ego_v": ego_v.tolist(),
                "sec_r": sec_r.tolist(),
                "sec_v": sec_v.tolist(),
                "fuel_ms": fuel_after,
                "pc_estimate": float(obs[0]) if not terminated else info.get("pc_final"),
            }
        )

        # Dense in-between animation frames need the ego's POST-maneuver
        # (pre-drift) velocity, which env.step() doesn't expose directly
        # -- the impulsive burn happens inside it. Recovered by backward-
        # propagating the TRUE end-of-interval state (which we do have,
        # from Basilisk itself) by -duration_s via the project's existing
        # J2 propagator: an impulsive burn changes velocity only, not
        # position, so pre_ego_r is already exactly the post-maneuver
        # position -- only the velocity needs recovering. This can differ
        # very slightly from the true post-maneuver velocity (J2 vs full
        # Basilisk fidelity, the same gap docs/26 investigated at length)
        # but at animation-smoothness precision, not decision-relevant
        # precision, that's immaterial -- the recovered trajectory is
        # still real physics, not an interpolation, and reconnects almost
        # exactly with the true endpoint keyframe either way.
        _recovered_r, recovered_v = propagate_state(ego_r, ego_v, -duration_s)
        dense = _dense_resample(pre_ego_r, recovered_v, pre_sec_r, pre_sec_v, duration_s)
        for frame in dense:
            frame["t_s"] += t_elapsed - duration_s
        dense_frames.extend(dense)

    expected_decisions = len(env.schedule_s) - 1
    collision_occurred = len(decisions) < expected_decisions

    # Re-run the identical real scenario with no maneuvers at all, so the
    # UI can show the trained policy's outcome against the actual
    # unmitigated risk -- the comparison docs/27 evaluates against.
    baseline = _run_never_maneuver_baseline(env, seed)

    return {
        "seed": seed,
        "scenario": {
            "miss_distance_m": sample["miss_distance"],
            "relative_speed_ms": sample["relative_speed"],
            "native_pc": sample["native_pc"],
            "combined_radius_m": sample["combined_radius"],
        },
        "constants": {
            "earth_mu_m3s2": 3.986004418e14,
            "earth_radius_m": 6378136.6,
            "max_dv_ms": env.max_dv_ms,
            "pc_threshold": env.pc_threshold,
        },
        "keyframes": keyframes,
        "decisions": decisions,
        "dense_frames": dense_frames,
        "result": {
            "pc_final": info.get("pc_final"),
            "total_fuel_used_ms": info.get("cumulative_fuel_used_ms"),
            "maneuver_count": info.get("maneuver_count"),
            "collision_occurred": collision_occurred,
        },
        "baseline": baseline,
    }
