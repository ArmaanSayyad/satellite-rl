"""Phase 7: the full baseline suite + metric suite from docs/11-
evaluation.md, run against the Phase 6 checkpoint (`runs/ppo_stage2_
run1`) on a held-out (`targeting_seed=999`) set of real Kelvins-derived
scenarios. Not part of the automated test suite -- needs bsk_rl/Basilisk
and a real trained model, like training/evaluate.py; run manually via
`python -m satellite_rl.training.full_evaluation`. See docs/22-
evaluation-results.md for the real results and their interpretation.

`run_scenario`/`percentile_stats` don't import bsk_rl or stable-
baselines3 at module level, so tests/test_training.py can exercise their
bookkeeping logic against a fake env without either dependency installed
-- same pattern as training/evaluate.py's run_episodes().
"""

import numpy as np

from .baselines import hindsight_oracle_fuel, threshold_heuristic_action

DEADZONE_MS = 1e-3  # matches env.collision_avoidance_env.DEADZONE_MS


def run_scenario(env, seed: int, action_fn) -> dict:
    """Run one episode, tracking not just the terminal outcome but which
    step (if any) the policy first maneuvered on -- for the "timing
    behavior" metric in docs/11, which the base env's info dict doesn't
    expose directly (only the running total), so it's derived here from
    consecutive fuel readings instead of touching env code.
    """
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    total_reward = 0.0
    step_idx = 0
    first_maneuver_step = None
    fuel_before = 0.0
    while not (terminated or truncated):
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_idx += 1
        fuel_now = info["cumulative_fuel_used_ms"]
        if first_maneuver_step is None and (fuel_now - fuel_before) > DEADZONE_MS:
            first_maneuver_step = step_idx
        fuel_before = fuel_now
    return {
        "reward": total_reward,
        "fuel_ms": info["cumulative_fuel_used_ms"],
        "maneuver_count": info["maneuver_count"],
        "pc_final": info["pc_final"],
        "first_maneuver_step": first_maneuver_step,
        "schedule_length": info.get("schedule_length"),
    }


def percentile_stats(values) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def run_full_evaluation(
    model_path: str,
    n_episodes: int = 60,
    seed_start: int = 100,
    targeting_seed: int = 999,
    max_dv_ms: float = 10.0,
    pc_threshold: float = 1e-4,
    **env_kwargs,
) -> dict:
    """Run baselines 1-4 (never-maneuver, always-max-thrust, threshold
    heuristic, hindsight oracle) plus the trained policy and a random
    policy, over the same `n_episodes` held-out real-event scenarios
    (matched seeds across all policies, for a fair paired comparison),
    and report the full docs/11 metric suite -- including native-risk
    (never-maneuver Pc) stratification, since the real Kelvins
    distribution is dominated by low-risk events (docs/14) and an
    unstratified mean would dilute exactly the high-risk cases where a
    good policy should look most different from doing nothing.
    """
    from stable_baselines3 import PPO

    from ..env import CollisionAvoidanceEnv

    env = CollisionAvoidanceEnv(
        targeting_seed=targeting_seed, max_dv_ms=max_dv_ms, pc_threshold=pc_threshold, **env_kwargs
    )
    model = PPO.load(model_path)
    seeds = [seed_start + i for i in range(n_episodes)]

    policies = {
        "never_maneuver": lambda obs: np.zeros(3, dtype=np.float32),
        "always_max_thrust": lambda obs: np.full(3, max_dv_ms / np.sqrt(3), dtype=np.float32),
        "threshold_heuristic": lambda obs: threshold_heuristic_action(obs, max_dv_ms, pc_threshold),
        "trained_policy": lambda obs: model.predict(obs, deterministic=True)[0],
    }
    env.action_space.seed(42)
    policies["random_policy"] = lambda obs: env.action_space.sample()

    per_policy_episodes = {name: [run_scenario(env, seed, fn) for seed in seeds] for name, fn in policies.items()}

    native_pc = dict(zip(seeds, [e["pc_final"] for e in per_policy_episodes["never_maneuver"]]))
    high_risk_seeds = {s for s, pc in native_pc.items() if pc > pc_threshold}

    oracle_results = {seed: hindsight_oracle_fuel(env, seed, max_dv_ms, pc_threshold) for seed in seeds}

    summary = {}
    for name, episodes in per_policy_episodes.items():
        rewards = [e["reward"] for e in episodes]
        fuels = [e["fuel_ms"] for e in episodes]
        maneuvers = [e["maneuver_count"] for e in episodes]
        pcs = [e["pc_final"] for e in episodes]
        first_steps = [e["first_maneuver_step"] for e in episodes if e["first_maneuver_step"] is not None]

        regrets = []
        for seed, e in zip(seeds, episodes):
            oracle_fuel, achieved = oracle_results[seed]
            if achieved and e["pc_final"] <= pc_threshold:
                regrets.append(e["fuel_ms"] - oracle_fuel)

        high_risk_pcs = [e["pc_final"] for seed, e in zip(seeds, episodes) if seed in high_risk_seeds]
        low_risk_pcs = [e["pc_final"] for seed, e in zip(seeds, episodes) if seed not in high_risk_seeds]

        summary[name] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "fuel_mean": float(np.mean(fuels)),
            "maneuver_mean": float(np.mean(maneuvers)),
            "pc_final": percentile_stats(pcs),
            "pc_final_high_risk_subset": percentile_stats(high_risk_pcs) if high_risk_pcs else None,
            "pc_final_low_risk_subset": percentile_stats(low_risk_pcs) if low_risk_pcs else None,
            "n_high_risk": len(high_risk_pcs),
            "n_low_risk": len(low_risk_pcs),
            "mean_first_maneuver_step": float(np.mean(first_steps)) if first_steps else None,
            "n_acted_episodes": len(first_steps),
            "regret_vs_oracle_mean": float(np.mean(regrets)) if regrets else None,
            "n_regret_valid": len(regrets),
        }

    summary["_oracle"] = {
        "mean_fuel": float(np.mean([f for f, _ in oracle_results.values()])),
        "n_infeasible_within_max_dv": sum(1 for _, achieved in oracle_results.values() if not achieved),
        "n_scenarios": n_episodes,
        "n_high_risk_scenarios": len(high_risk_seeds),
    }
    return summary


def main() -> None:
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    model_path = str(repo_root / "runs" / "ppo_stage2_run1")
    summary = run_full_evaluation(
        model_path,
        n_episodes=60,
        sample_geometry=True,
        schedule_days_before_tca=(0.2, 0.1, 0.05, 0.01, 0.0),
        targeting_seed=999,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
