"""PPO training for CollisionAvoidanceEnv, via Stable-Baselines3.

See docs/10-rl-algorithm.md for the design rationale (why PPO, why SB3,
starting hyperparameters) and docs/21-training-results.md for what a real
run actually produced -- Phase 5 found stage 3 (evolving uncertainty,
real schedules) runs at only ~1.3 steps/sec versus ~8-9 steps/sec for
stages 1/2 (short fixed schedules), so this defaults to stage 2 (sampled
geometry, short fixed schedule) to keep training tractable, not stage 3.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ..env import CollisionAvoidanceEnv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS_DIR = REPO_ROOT / "runs"


def make_env(
    sample_geometry: bool = True,
    evolve_uncertainty: bool = False,
    schedule_days_before_tca: tuple = (0.2, 0.1, 0.05, 0.01, 0.0),
    monitor_path: str | None = None,
    **env_kwargs,
) -> Monitor:
    """Build a Monitor-wrapped CollisionAvoidanceEnv. Monitor logs
    per-episode reward/length to `monitor_path` (if given) for later
    training-curve plotting -- SB3's own mechanism for this, not a custom
    one.
    """
    env = CollisionAvoidanceEnv(
        sample_geometry=sample_geometry,
        evolve_uncertainty=evolve_uncertainty,
        schedule_days_before_tca=schedule_days_before_tca,
        **env_kwargs,
    )
    return Monitor(env, filename=monitor_path)


def train(
    total_timesteps: int = 10_000,
    n_steps: int = 64,
    batch_size: int = 32,
    gamma: float = 0.95,
    gae_lambda: float = 0.9,
    ent_coef: float = 0.01,
    learning_rate: float = 3e-4,
    seed: int = 0,
    out_name: str = "ppo_stage2",
    **env_kwargs,
) -> tuple[PPO, Monitor]:
    """Train PPO on CollisionAvoidanceEnv and save a checkpoint + monitor log.

    Hyperparameter choices vs. SB3/docs/10's generic defaults, and why:
    - n_steps=64, batch_size=32: docs/10 default (n_steps=2048) assumes
      long episodes; stage 2's episodes are only 4-6 steps, so 2048 would
      need 340-500+ episodes collected before a single PPO update.
    - gamma=0.95 (not 0.99): short episode horizon (4-6 decision steps),
      per docs/10's note that 0.99 may be higher than needed here.
    - ent_coef=0.01 (SB3 default is 0.0): docs/10 flags that a policy
      could collapse to always-near-zero-action ("wait") early in
      training without explicit exploration pressure, since that's a
      locally low-fuel-cost, low-effort optimum even though it's not
      actually optimal once real risk is present.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    monitor_path = str(RUNS_DIR / f"{out_name}_monitor.csv")
    env = make_env(monitor_path=monitor_path, **env_kwargs)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(str(RUNS_DIR / out_name))
    return model, env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--out-name", type=str, default="ppo_stage2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(total_timesteps=args.total_timesteps, out_name=args.out_name, seed=args.seed)


if __name__ == "__main__":
    main()
