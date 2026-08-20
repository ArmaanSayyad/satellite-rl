#!/usr/bin/env python3
"""Plot episode reward over training from an SB3 Monitor CSV log."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_monitor_csv(path: Path) -> pd.DataFrame:
    """SB3's Monitor writes a one-line JSON header (starts with '#') then
    a normal CSV -- skip that header row.
    """
    return pd.read_csv(path, skiprows=1)


def plot_training_curve(monitor_path: Path, out_path: Path, rolling_window: int = 20) -> None:
    df = load_monitor_csv(monitor_path)
    df["cumulative_timesteps"] = df["l"].cumsum()
    df["reward_rolling_mean"] = df["r"].rolling(rolling_window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["cumulative_timesteps"], df["r"], alpha=0.25, label="episode reward")
    ax.plot(
        df["cumulative_timesteps"],
        df["reward_rolling_mean"],
        linewidth=2,
        label=f"rolling mean ({rolling_window} episodes)",
    )
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("episode reward")
    ax.set_title(f"PPO training curve ({len(df)} episodes)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"First 10 episodes mean reward: {df['r'][:10].mean():.4f}")
    print(f"Last 10 episodes mean reward:  {df['r'][-10:].mean():.4f}")
    print(f"Total episodes: {len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("monitor_csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out_path = args.out or args.monitor_csv.with_suffix(".png")
    plot_training_curve(args.monitor_csv, out_path)


if __name__ == "__main__":
    main()
