"""Light post-training sanity check: does the trained policy actually
behave differently from simple baselines? The full baseline suite
(threshold heuristic, hindsight-optimal oracle) is Phase 7's job per
docs/11-evaluation.md -- this only checks that Phase 6's training run had
a real, measurable effect, not a proper evaluation.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from stable_baselines3 import PPO


def run_episodes(env: Any, action_fn: Callable, n_episodes: int, seed: int = 100) -> dict:
    """Run n_episodes with the given action_fn(obs) -> action, return
    summary statistics.

    `env` is duck-typed (any Gymnasium-like env with the same
    reset()/step() info-dict keys CollisionAvoidanceEnv provides) --
    deliberately not type-hinted to that class specifically, so this
    function (and anything testing it) doesn't need bsk_rl/Basilisk
    importable, only whatever `env` actually is at call time.
    """
    rewards, fuels, maneuvers, pcs = [], [], [], []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = action_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        fuels.append(info["cumulative_fuel_used_ms"])
        maneuvers.append(info["maneuver_count"])
        pcs.append(info.get("pc_final", np.nan))
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_fuel_used_ms": float(np.mean(fuels)),
        "mean_maneuver_count": float(np.mean(maneuvers)),
        "mean_pc_final": float(np.nanmean(pcs)),
        "n_episodes": n_episodes,
    }


def compare_policy_to_baselines(
    model_path: str, n_episodes: int = 20, **env_kwargs
) -> dict[str, dict]:
    """Compare a trained PPO model against never-maneuver, always-max-
    thrust, and random baselines, on held-out (seeded, but not seen
    during training) episodes.
    """
    from ..env import CollisionAvoidanceEnv  # local: needs bsk_rl/Basilisk, see run_episodes

    env = CollisionAvoidanceEnv(**env_kwargs)
    model = PPO.load(model_path)

    results = {}
    results["trained_policy"] = run_episodes(
        env, lambda obs: model.predict(obs, deterministic=True)[0], n_episodes
    )
    results["never_maneuver"] = run_episodes(
        env, lambda _obs: np.zeros(3, dtype=np.float32), n_episodes
    )
    results["always_max_thrust"] = run_episodes(
        env, lambda _obs: np.full(3, env.max_dv_ms / np.sqrt(3), dtype=np.float32), n_episodes
    )
    env.action_space.seed(42)
    results["random_policy"] = run_episodes(
        env, lambda _obs: env.action_space.sample(), n_episodes
    )
    return results


def main() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    model_path = str(repo_root / "runs" / "ppo_stage2_run1")
    results = compare_policy_to_baselines(
        model_path,
        n_episodes=20,
        sample_geometry=True,
        schedule_days_before_tca=(0.2, 0.1, 0.05, 0.01, 0.0),
        targeting_seed=999,  # different from training's targeting_seed=0 -- held-out scenarios
    )
    for name, stats in results.items():
        print(
            f"{name:20s}: reward={stats['mean_reward']:+.4f}±{stats['std_reward']:.4f}  "
            f"fuel={stats['mean_fuel_used_ms']:.3f}  maneuvers={stats['mean_maneuver_count']:.2f}  "
            f"pc_final={stats['mean_pc_final']:.2e}"
        )


if __name__ == "__main__":
    main()
