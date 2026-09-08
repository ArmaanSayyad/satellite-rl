"""Process-isolated stage-3 PPO training with recorded source partitions.

Example: python -m satellite_rl.lab.train --steps 100000 --workers 4 --out runs/v2
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

from .protocol import provenance, sha256

ACTION_TABLE = np.array([[0., 0., 0.]] + [
    (np.eye(3)[axis] * sign * magnitude).tolist()
    for magnitude in (.01, .1, 1.) for axis in range(3) for sign in (-1., 1.)
], dtype=np.float32)


def decode_action(action, mode="continuous"):
    if mode == "discrete19":
        value = np.asarray(action).item()
        index = int(value)
        if value != index:
            raise ValueError("categorical maneuver must be an integer index")
        if not 0 <= index < len(ACTION_TABLE):
            raise ValueError("invalid categorical maneuver")
        return ACTION_TABLE[index].copy()
    if mode != "continuous":
        raise ValueError(f"unknown action mode: {mode}")
    return np.asarray(action, dtype=np.float32)


class DiscreteManeuverWrapper(gym.ActionWrapper):
    """Explicit wait and bounded RTN impulses; unchanged physical execution."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))

    def action(self, action):
        return decode_action(action, "discrete19")


class ProfileWrapper(gym.Wrapper):
    """Per-worker latency records expose expensive resets separately from steps."""
    def __init__(self, env, output):
        super().__init__(env)
        self.output = Path(output)

    def _record(self, record):
        with self.output.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def reset(self, **kwargs):
        start = time.perf_counter()
        self._record({"operation": "reset", "status": "started", "seed": kwargs.get("seed")})
        obs, info = self.env.reset(**kwargs)
        self._record({"operation": "reset", "status": "completed",
                      "elapsed_seconds": time.perf_counter() - start, "event_id": info.get("event_id")})
        return obs, info

    def step(self, action):
        start = time.perf_counter()
        result = self.env.step(action)
        self._record({"operation": "step", "status": "completed",
                      "elapsed_seconds": time.perf_counter() - start,
                      "terminal": bool(result[2] or result[3])})
        return result


def worker_env(event_ids, config, rank, output):
    import torch
    from stable_baselines3.common.monitor import Monitor

    from .environment import make_v2_env

    torch.set_num_threads(1)
    env = make_v2_env(
        event_ids=event_ids, history_length=config["history"],
        observation_noise_scale=config["noise"], max_lead_days=config["max_lead_days"],
        forecast_fidelity=config.get("forecast_fidelity", "j2"),
        high_risk_fraction=config["risk_fraction"],
        high_risk_pool_fraction=config.get("risk_pool_fraction", 0.01),
        high_risk_augment=config["augment"], fuel_weight=config["fuel_weight"],
        disruption_weight=config["disruption_weight"], targeting_seed=config["seed"] + rank,
    )
    if config.get("action_mode", "continuous") == "discrete19":
        env = DiscreteManeuverWrapper(env)
    env = ProfileWrapper(env, Path(output) / f"worker-{rank}.profile.jsonl")
    return Monitor(env, str(Path(output) / f"worker-{rank}.monitor.csv"))


def warmstart_from_planner(model, event_ids, config, output):
    """Distill approximate observation-based plans from source-linked CDM variants.

    These are labeled observation examples, not claimed executed simulator outcomes.
    Source families are restricted before variant generation. Final evaluation must
    still execute policies in Basilisk, including the teacher-only checkpoint.
    """
    import pandas as pd
    import torch
    from torch.nn import functional

    from ..scenario.distributions import FITTED_DIR
    from .environment import make_v2_env
    from .observations import FEATURE_NAMES, encode_estimate
    from .planner import plane_pc, response_matrix

    if config.get("action_mode") != "discrete19" or config.get("forecast_fidelity") != "basilisk":
        raise ValueError("planner warmstart requires discrete19 and Basilisk forecasts")
    table = pd.read_csv(FITTED_DIR / "geometry_events.csv")
    table = table[table.event_id.isin(event_ids)]
    source_count = config.get("imitation_sources", 12)
    if source_count < 2 or config["imitation_samples"] < source_count:
        raise ValueError("imitation requires at least two sources and at least one sample per source")
    ranked = table[["native_pc", "esa_reported_pc"]].max(axis=1).nlargest(source_count // 2).index
    risky = table.loc[ranked].event_id.astype(int).tolist()
    background = table[~table.event_id.isin(risky)].sample(min(source_count - len(risky), len(table) - len(risky)),
                                                    random_state=config["seed"])
    sources = risky + background.event_id.astype(int).tolist()
    rng = np.random.default_rng(config["seed"])
    env = make_v2_env(event_ids=sources, history_length=config["history"],
                      observation_noise_scale=0, max_lead_days=config["max_lead_days"],
                      forecast_fidelity="basilisk", high_risk_fraction=0, high_risk_augment=False)
    inputs, labels, families = [], [], []
    started = time.perf_counter()
    try:
        for source in sources:
            _, _ = env.reset(seed=config["seed"], options={"event_id": source})
            base = dict(env.latest_estimate)
            response = response_matrix(base)
            count = config["imitation_samples"] // len(sources)
            for _ in range(count):
                packet = dict(base)
                packet["noise_scale"] = config["noise"]
                packet["miss_plane_m"] = rng.multivariate_normal(
                    base["miss_plane_m"], np.asarray(base["covariance_plane_m2"]) * config["noise"]**2
                ).tolist()
                packet["estimated_pc"] = plane_pc(packet["miss_plane_m"],
                    packet["covariance_plane_m2"], packet["radius_m"])
                pcs = np.array([plane_pc(np.asarray(packet["miss_plane_m"]) + response @ action,
                                        packet["covariance_plane_m2"], packet["radius_m"])
                                for action in ACTION_TABLE])
                feasible = np.flatnonzero(pcs <= 1e-4)
                label = (min(feasible, key=lambda index: (np.linalg.norm(ACTION_TABLE[index]), pcs[index]))
                         if len(feasible) else int(np.argmin(pcs)))
                features = encode_estimate(packet, np.zeros(3), 0)
                inputs.append(np.concatenate([np.zeros(len(FEATURE_NAMES) * (config["history"] - 1)), features]))
                labels.append(label)
                families.append(source)
            print(f"teacher source={source} examples={len(inputs)}", flush=True)
    finally:
        env.close()
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if len(x) < 12 or len(np.unique(y)) < 2:
        raise ValueError("teacher corpus lacks enough samples or action diversity")
    np.savez_compressed(Path(output) / "teacher-examples.npz", observations=x, actions=y,
                        event_ids=np.asarray(families))
    order = rng.permutation(len(x))
    split = max(1, int(len(x) * .8))
    training, validation = order[:split], order[split:]
    observations, targets = torch.as_tensor(x), torch.as_tensor(y)
    weights = torch.ones(len(ACTION_TABLE))
    weights[0] = float(max(np.sum(y[training] != 0), 1) / max(np.sum(y[training] == 0), 1))
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=0.003)
    model.policy.set_training_mode(True)
    for _ in range(150):
        shuffled = rng.permutation(training)
        for offset in range(0, len(training), 128):
            indices = shuffled[offset:offset + 128]
            logits = model.policy.get_distribution(observations[indices]).distribution.logits
            loss = functional.cross_entropy(logits, targets[indices], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.policy.set_training_mode(False)
    predictions = np.asarray(model.predict(x[validation], deterministic=True)[0])
    model.save(str(Path(output) / "teacher.zip"))
    report = {
        "kind": "approximate-planner-imitation", "source_event_ids": sources,
        "samples": len(x), "label_counts": np.bincount(y, minlength=len(ACTION_TABLE)).tolist(),
        "loss_class_weights": weights.tolist(), "epochs": 150,
        "labels": "HF baseline + finite-difference J2 action response; not executed outcomes",
        "validation": "within-source held-out observation examples, not source-family generalization",
        "validation_action_accuracy": float(np.mean(predictions == y[validation])),
        "validation_gate_accuracy": float(np.mean((predictions == 0) == (y[validation] == 0))),
        "elapsed_seconds": time.perf_counter() - started,
        "dataset_sha256": sha256(Path(output) / "teacher-examples.npz"),
        "checkpoint_sha256": sha256(Path(output) / "teacher.zip"),
    }
    (Path(output) / "imitation.json").write_text(json.dumps(report, indent=2))
    return report


def train(config):
    import pandas as pd
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from ..scenario.distributions import FITTED_DIR
    from .data import grouped_split

    if config["steps"] < 1 or config["workers"] < 1 or config["rollout"] < 2:
        raise ValueError("steps/workers must be positive and rollout at least 2")
    output = Path(config["out"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("output already contains a run; resume into a new output directory")
    event_ids = pd.read_csv(FITTED_DIR / "geometry_events.csv")["event_id"].astype(int).tolist()
    splits = grouped_split(event_ids, seed=config["split_seed"])
    # Predeclared leave-one-family-out runs. Never use this to relabel a trained run.
    heldout = set(config.get("holdout_event", []))
    if not heldout.issubset(set(event_ids)):
        raise ValueError("holdout event IDs must exist in the fitted source table")
    if heldout:
        splits = {name: [event for event in ids if event not in heldout]
                  for name, ids in splits.items()}
        splits["test"].extend(sorted(heldout))
    manifest = provenance(config, FITTED_DIR)
    manifest.update(partitions=splits, status="running", fidelity="Basilisk", observation="v2")
    manifest["observation_version"] = ("encounter-history-v2.1-basilisk"
        if config.get("forecast_fidelity") == "basilisk" else "encounter-history-v2.0")
    if config.get("action_mode") == "discrete19":
        manifest["action_table_rtn_ms"] = ACTION_TABLE.tolist()
    if config["resume"]:
        source = Path(config["resume"])
        previous_manifest = source.parent / "manifest.json"
        if not previous_manifest.exists():
            previous_manifest = source.parent.parent / "manifest.json"
        if not previous_manifest.exists():
            raise ValueError("resume requires original manifest.json beside checkpoint or its parent")
        previous = json.loads(previous_manifest.read_text())
        for key in ("history", "noise", "max_lead_days", "split_seed", "risk_fraction",
                    "augment", "fuel_weight", "disruption_weight", "rollout"):
            if previous["config"][key] != config[key]:
                if key == "max_lead_days" and config.get("allow_curriculum_change"):
                    manifest["curriculum_change"] = {key: {"before": previous["config"][key],
                                                          "after": config[key]}}
                    continue
                raise ValueError(f"resume configuration mismatch: {key}")
        for key, default in (("action_mode", "continuous"), ("risk_pool_fraction", .01),
                             ("forecast_fidelity", "j2")):
            if previous["config"].get(key, default) != config.get(key, default):
                raise ValueError(f"resume configuration mismatch: {key}")
        if previous["partitions"] != splits or previous["data_sha256"] != manifest["data_sha256"]:
            raise ValueError("resume data or source partitions changed")
        manifest["parent_checkpoint_sha256"] = sha256(source)
        manifest["resume_semantics"] = "optimizer/counters restored; simulator episodes restart"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    factories = [partial(worker_env, splits["train"], config, rank, str(output))
                 for rank in range(config["workers"])]
    torch.set_num_threads(1)
    started = time.perf_counter()
    vector = None
    model = None
    try:
        vector = (SubprocVecEnv(factories, start_method="spawn") if len(factories) > 1
                  else DummyVecEnv(factories))
        vector.seed(config["seed"])
        if config["resume"]:
            model = PPO.load(config["resume"], env=vector, device="cpu")
        else:
            model = PPO("MlpPolicy", vector, n_steps=config["rollout"],
                        batch_size=min(32, config["rollout"] * config["workers"]),
                        gamma=0.99, gae_lambda=0.95, ent_coef=0.005,
                        learning_rate=3e-4, seed=config["seed"], device="cpu", verbose=1,
                        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}})
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        if config.get("imitation_samples", 0):
            if config["resume"]:
                raise ValueError("imitation initialization is only supported for a fresh run")
            manifest["imitation"] = warmstart_from_planner(model, splits["train"], config, output)
        initial_steps = model.num_timesteps
        model.learn(total_timesteps=config["steps"], reset_num_timesteps=not bool(config["resume"]),
                    callback=CheckpointCallback(
                        save_freq=max(1, config["checkpoint_every"] // config["workers"]),
                        save_path=str(output / "checkpoints"), name_prefix="ppo"))
        model.save(str(output / "model.zip"))
        manifest.update(status="completed", total_timesteps=model.num_timesteps,
                        collected_timesteps=model.num_timesteps - initial_steps,
                        checkpoint_sha256=sha256(output / "model.zip"))
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        if model is not None:
            model.save(str(output / "interrupted.zip"))
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["steps_per_second"] = (manifest.get("collected_timesteps", 0)
                                        / manifest["elapsed_seconds"])
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if vector is not None:
            vector.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/v2")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--holdout-event", type=int, action="append", default=[],
                        help="exclude a source family from all training (repeat for LOEO studies)")
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--max-lead-days", type=float, default=None,
                        help="explicit truncated-horizon curriculum; default preserves full schedules")
    parser.add_argument("--risk-fraction", type=float, default=0.5)
    parser.add_argument("--risk-pool-fraction", type=float, default=0.01)
    parser.add_argument("--action-mode", choices=("continuous", "discrete19"), default="continuous")
    parser.add_argument("--forecast-fidelity", choices=("j2", "basilisk"), default="j2")
    parser.add_argument("--imitation-samples", type=int, default=0,
                        help="optional approximate planner distillation before PPO; source-grouped")
    parser.add_argument("--imitation-sources", type=int, default=12,
                        help="distinct training families in the optional teacher corpus")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--fuel-weight", type=float, default=0.1)
    parser.add_argument("--disruption-weight", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--allow-curriculum-change", action="store_true",
                        help="permit an explicitly recorded lead-time-cap change when resuming")
    args = parser.parse_args()
    result = train(vars(args))
    print(json.dumps({key: result[key] for key in
                      ("status", "total_timesteps", "elapsed_seconds", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
