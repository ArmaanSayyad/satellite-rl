"""Paired Basilisk evaluation, event-bootstrap uncertainty, and action probes."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .planner import geometry_threshold_action, plane_pc, search_plan
from .protocol import CRITERIA, SCHEMA_VERSION, provenance, sha256


def event_bootstrap(rows, metric, seed=0, repetitions=2000):
    """Resample source families, retaining every variant/seed within each family."""
    groups = {}
    for row in rows:
        groups.setdefault(row["event_id"], []).append(row)
    if len(groups) < 2:
        return None
    families = list(groups.values())
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        sample = [row for index in rng.integers(len(families), size=len(families))
                  for row in families[index]]
        value = metric(sample)
        if value is not None and np.isfinite(value):
            estimates.append(value)
    return (np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else None)


def summarize(rows, threshold=1e-4):
    high = [r for r in rows if r.get("native_pc") is not None and r["native_pc"] > threshold]
    safe = [r for r in rows if r.get("native_pc") is not None
            and r["native_pc"] <= CRITERIA["safe_pc_threshold"]]
    below_threshold = [r for r in rows if r.get("native_pc") is not None and r["native_pc"] <= threshold]
    def mitigation(rs):
        subset = [r for r in rs if r.get("native_pc") is not None and r["native_pc"] > threshold]
        return float(np.mean([r["pc_final"] is not None and r["pc_final"] <= threshold
                              for r in subset])) if subset else None
    def false_positive(rs):
        subset = [r for r in rs if r.get("native_pc") is not None
                  and r["native_pc"] <= CRITERIA["safe_pc_threshold"]]
        return float(np.mean([r["fuel_ms"] > 1e-3 for r in subset])) if subset else None
    pcs = [r["pc_final"] for r in rows if r["pc_final"] is not None]
    efficiency = [(r["native_pc"] - r["pc_final"]) / r["fuel_ms"] for r in rows
                  if r["fuel_ms"] > 1e-3 and r.get("native_pc") is not None
                  and r["pc_final"] is not None]
    regret = [r["fuel_ms"] - r["planner_fuel_ms"] for r in rows
              if r.get("planner_feasible") and r["pc_final"] is not None
              and r["pc_final"] <= threshold]
    hindsight_rows = [r for r in rows if r.get("hindsight_grid_fuel_ms") is not None
                      and r["pc_final"] is not None and r["pc_final"] <= threshold]
    hindsight_regret = [r["fuel_ms"] - r["hindsight_grid_fuel_ms"] for r in hindsight_rows]
    first_actions = [r["actions"][0]["action_rtn_ms"] for r in rows if r.get("actions")]
    probes = [probe for row in rows for probe in row.get("matched_estimate_sensitivity", [])]
    return {
        "n_episodes": len(rows), "n_source_events": len({r["event_id"] for r in rows}),
        "n_high_risk": len(high), "n_safe": len(safe),
        "n_high_risk_source_events": len({r["event_id"] for r in high}),
        "n_safe_source_events": len({r["event_id"] for r in safe}),
        "interval_method": "source-family percentile bootstrap; conditional on observed families",
        "n_failed": sum(r["pc_final"] is None for r in rows),
        "n_invalid_baseline": sum(r.get("native_pc") is None for r in rows),
        "n_chan_foster_threshold_disagreements": sum(
            bool(r.get("chan_foster_threshold_disagreement")) for r in rows),
        "mitigation_rate": mitigation(rows), "mitigation_ci95": event_bootstrap(high, mitigation),
        "false_positive_rate": false_positive(rows),
        "false_positive_ci95": event_bootstrap(safe, false_positive),
        "mean_fuel_ms": float(np.mean([r["fuel_ms"] for r in rows])) if rows else None,
        "safe_mean_wasted_fuel_ms": float(np.mean([r["fuel_ms"] for r in safe])) if safe else None,
        "below_action_threshold_maneuver_rate": float(np.mean([r["fuel_ms"] > 1e-3
            for r in below_threshold])) if below_threshold else None,
        "pc_final_quantiles": dict(zip(("median", "p95", "p99", "max"),
                                        np.quantile(pcs, [.5, .95, .99, 1]).tolist())) if pcs else None,
        "mean_risk_reduction_per_ms": float(np.mean(efficiency)) if efficiency else None,
        "mean_fuel_regret_vs_feasible_online_planner_ms": float(np.mean(regret)) if regret else None,
        "regret_n": len(regret),
        "mean_fuel_regret_vs_restricted_hindsight_ms": float(np.mean(hindsight_regret))
        if hindsight_regret else None,
        "hindsight_regret_valid_n": len(hindsight_rows),
        "hindsight_regret_high_risk_n": sum(r["native_pc"] > threshold for r in hindsight_rows),
        "first_action_mean_rtn_ms": np.mean(first_actions, axis=0).tolist() if first_actions else None,
        "first_action_std_rtn_ms": np.std(first_actions, axis=0).tolist() if first_actions else None,
        "n_distinct_first_actions_1e6_precision": len({tuple(np.round(a, 6)) for a in first_actions}),
        "matched_cdm_probe_count": len(probes),
        "matched_cdm_max_action_change_ms": max((p["delta_action_norm_ms"] for p in probes), default=None),
    }


def observation_sensitivity(model, observation, feature_indices, values=(-1.0, 0.0, 1.0)):
    """Feature-only intervention: diagnostic sensitivity, not physical counterfactual."""
    base = np.asarray(observation, dtype=np.float32)
    original = np.asarray(model.predict(base, deterministic=True)[0])
    records = []
    for feature, index in feature_indices.items():
        for value in values:
            changed = base.copy()
            changed[index] = value
            action = np.asarray(model.predict(changed, deterministic=True)[0])
            records.append({"feature": feature, "index": index, "normalized_value": value,
                            "action_rtn_ms": action.tolist(),
                            "delta_action_norm_ms": float(np.linalg.norm(action - original)),
                            "kind": "feature intervention; not physically coherent scenario"})
    return records


def matched_estimate_sensitivity(model, observation, estimate, action_mode="continuous"):
    """Coherent single-CDM counterfactuals, explicitly not re-simulated outcomes.

    Alter miss/covariance/radius, recompute Pc and every dependent current feature,
    retain prior history. This resembles a changed latest warning, not changing
    the underlying historical event. A norm change alone is not evidence of skill.
    """
    from .observations import FEATURE_NAMES, encode_estimate
    from .train import decode_action

    baseline = decode_action(model.predict(observation, deterministic=True)[0], action_mode)
    records = []
    for field in ("miss_plane_m", "covariance_plane_m2", "radius_m"):
        for multiplier in (0.5, 2.0):
            packet = dict(estimate)
            packet[field] = (np.asarray(packet[field]) * multiplier).tolist()
            packet["estimated_pc"] = plane_pc(packet["miss_plane_m"],
                                               packet["covariance_plane_m2"], packet["radius_m"])
            changed = np.asarray(observation).copy()
            current = changed[-len(FEATURE_NAMES):]
            previous = current[11:14] * 10
            changed[-len(FEATURE_NAMES):] = encode_estimate(packet, previous, current[10])
            action = decode_action(model.predict(changed, deterministic=True)[0], action_mode)
            records.append({"intervention": field, "multiplier": multiplier,
                            "estimated_pc": packet["estimated_pc"],
                            "action_rtn_ms": action.tolist(),
                            "delta_action_norm_ms": float(np.linalg.norm(action - baseline)),
                            "kind": "matched latest-CDM intervention; outcome not re-simulated"})
    return records


def run_episode(env, event_id, seed, policy, model=None, action_mode="continuous",
                observation_seed=None):
    from .train import decode_action
    options = {"event_id": event_id}
    if observation_seed is not None:
        options["observation_seed"] = observation_seed
    obs, info = env.reset(seed=seed, options=options)
    initial = env.latest_estimate.copy()
    sensitivity = (matched_estimate_sensitivity(model, obs, initial, action_mode)
                   if policy == "v2" else [])
    actions, total_reward = [], 0.0
    while True:
        estimate = env.latest_estimate
        prediction = None
        if policy == "never_maneuver":
            action = np.zeros(3, dtype=np.float32)
        elif policy == "threshold":
            action = np.array([10.0, 0.0, 0.0] if estimate["estimated_pc"] > 1e-4 else [0.] * 3)
        elif policy == "geometry_threshold":
            action = geometry_threshold_action(estimate)
        elif policy in ("planner", "timed_planner"):
            waits = [0.0]
            if policy == "timed_planner" and 0 < estimate["next_update_s"] < estimate["time_to_tca_s"]:
                waits.append(estimate["next_update_s"])
            prediction = search_plan(estimate, waits_s=waits).to_dict()
            action = (np.zeros(3) if prediction["wait_s"] > 0
                      else np.asarray(prediction["action_rtn_ms"]))
        elif policy == "v1":
            # Deliberately privileged legacy input, visibly labeled in artifact.
            raw_obs = env.unwrapped.satellites[0].get_obs()
            action = model.predict(raw_obs, deterministic=True)[0]
        else:
            action = decode_action(model.predict(obs, deterministic=True)[0], action_mode)
        actions.append({"step": len(actions), "time_to_tca_s": estimate["time_to_tca_s"],
                        "observed_pc": float(raw_obs[0]) if policy == "v1" else estimate["estimated_pc"],
                        "action_rtn_ms": np.asarray(action).tolist(), "planner_prediction": prediction,
                        "observation": dict(estimate), "truth": info.get("truth"),
                        "policy_input": np.asarray(raw_obs if policy == "v1" else obs).tolist()})
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    valid = info.get("pc_final_valid", "pc_final" in info) and np.isfinite(info.get("pc_final", np.nan))
    foster = local_tca = final_miss_norm = None
    if valid:
        from ..pc import compute_pc
        from ..pc.geometry import encounter_plane_basis
        ego, secondary = env.unwrapped.satellites
        relative_r = np.asarray(secondary.dynamics.r_BN_N) - np.asarray(ego.dynamics.r_BN_N)
        relative_v = np.asarray(secondary.dynamics.v_BN_N) - np.asarray(ego.dynamics.v_BN_N)
        basis = encounter_plane_basis(relative_v)
        final_miss_norm = float(np.linalg.norm(basis.T @ relative_r))
        covariance = basis @ np.diag([ego._pc_sigma_x**2, ego._pc_sigma_z**2]) @ basis.T
        foster = compute_pc(relative_r, relative_v, covariance, ego._pc_combined_radius, method="foster")
        local_tca = -float(relative_r @ relative_v) / float(relative_v @ relative_v)
    return {
        "event_id": int(event_id), "seed": seed, "observation_seed": observation_seed,
        "seed_semantics": "joint geometry/sensing" if observation_seed is None else "independent geometry/sensing",
        "policy": policy,
        "information": "privileged legacy true state" if policy == "v1" else "v2 noisy estimate",
        "pc_final": float(info["pc_final"]) if valid else None,
        "pc_final_foster": foster, "local_linear_tca_offset_s": local_tca,
        "final_basilisk_encounter_miss_norm_m": final_miss_norm,
        "initial_j2_predicted_miss_norm_m": float(np.linalg.norm(
            actions[0]["truth"]["predicted_miss_plane_m"]))
        if actions[0]["truth"] and initial.get("forecast_fidelity", "j2") == "j2" else None,
        "initial_forecast_miss_norm_m": float(np.linalg.norm(
            actions[0]["truth"]["predicted_miss_plane_m"])) if actions[0]["truth"] else None,
        "forecast_fidelity": initial.get("forecast_fidelity", "j2"),
        "chan_foster_threshold_disagreement": ((float(info["pc_final"]) <= 1e-4) != (foster <= 1e-4))
        if valid else None,
        "status": info.get("status", "completed" if valid else "failed"),
        "pc_final_valid": bool(valid), "failure_reason": info.get("failure_reason"),
        "fuel_ms": float(info["cumulative_fuel_used_ms"]), "reward": total_reward,
        "maneuver_count": int(info["maneuver_count"]), "actions": actions,
        "initial_lead_s": initial["time_to_tca_s"],
        "initial_covariance_eigenvalues_m2": np.linalg.eigvalsh(initial["covariance_plane_m2"]).tolist(),
        "radius_m": initial["radius_m"],
        "matched_estimate_sensitivity": sensitivity,
    }


def evaluate(args):
    from stable_baselines3 import PPO

    from ..scenario.distributions import FITTED_DIR
    from .environment import make_v2_env

    config = vars(args).copy()
    if args.seeds < 1 or args.observation_seeds < 1:
        raise ValueError("geometry and observation seed counts must be positive")
    run_provenance = provenance(config, FITTED_DIR)
    models = {}
    if args.checkpoint:
        models["v2"] = PPO.load(args.checkpoint, device="cpu")
    if args.v1:
        models["v1"] = PPO.load(args.v1, device="cpu")
    events = [int(e) for e in args.events.split(",")]
    policies = [p.strip() for p in args.policies.split(",")]
    if "never_maneuver" not in policies:
        policies.insert(0, "never_maneuver")
    for policy in policies:
        if policy not in {"never_maneuver", "threshold", "geometry_threshold", "planner",
                          "timed_planner", "v1", "v2"}:
            raise ValueError(f"unknown policy: {policy}")
        if policy in ("v1", "v2") and policy not in models:
            raise ValueError(f"{policy} requires its checkpoint argument")
    partition = "explicit challenge selection; not population estimate"
    action_mode = "continuous"
    distribution_shifts = {}
    event_partitions = {}
    imitation_sources = set()
    if args.checkpoint:
        manifest_path = Path(args.checkpoint).parent / "manifest.json"
        if not manifest_path.exists():
            manifest_path = Path(args.checkpoint).parent.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        event_partitions = {event: name for name, ids in manifest["partitions"].items() for event in ids}
        imitation = manifest.get("imitation")
        if imitation is None and (manifest_path.parent / "imitation.json").exists():
            imitation = json.loads((manifest_path.parent / "imitation.json").read_text())
        imitation_sources = set((imitation or {}).get("source_event_ids", []))
        if args.history != manifest["config"]["history"]:
            raise ValueError("evaluation history length must match checkpoint observation schema")
        for key in ("noise", "max_lead_days"):
            if config[key] != manifest["config"][key]:
                distribution_shifts[key] = {"training": manifest["config"][key], "evaluation": config[key]}
        if args.forecast_fidelity != manifest["config"].get("forecast_fidelity", "j2"):
            distribution_shifts["forecast_fidelity"] = {
                "training": manifest["config"].get("forecast_fidelity", "j2"),
                "evaluation": args.forecast_fidelity}
        action_mode = manifest["config"].get("action_mode", "continuous")
        overlap = set(events) & set(manifest["partitions"]["train"])
        if overlap and not args.allow_training_events:
            raise ValueError(f"evaluation includes training families: {sorted(overlap)}")
        partition = "training-family diagnostic" if overlap else "source-held-out challenge"
    env = make_v2_env(event_ids=events, history_length=args.history,
                      observation_noise_scale=args.noise, max_lead_days=args.max_lead_days,
                      forecast_fidelity=args.forecast_fidelity,
                      high_risk_fraction=0, high_risk_augment=False)
    started = time.perf_counter()
    rows = []
    noise_seeds = [None] if args.observation_seeds == 1 else list(range(1000, 1000 + args.observation_seeds))
    try:
        for event in events:
            for seed in range(args.seed, args.seed + args.seeds):
                for noise_seed in noise_seeds:
                    paired = [run_episode(env, event, seed, policy, models.get(policy), action_mode,
                                          noise_seed) for policy in policies]
                    native = next(r["pc_final"] for r in paired if r["policy"] == "never_maneuver")
                    planner = next((r for r in paired if r["policy"] == "planner"), None)
                    for row in paired:
                        row["source_partition"] = event_partitions.get(event, "explicit challenge")
                        row["seen_by_imitation"] = event in imitation_sources if imitation_sources else None
                        row["native_pc"] = native
                        row["planner_feasible"] = bool(planner and planner["pc_final"] is not None
                                                       and planner["pc_final"] <= 1e-4)
                        row["planner_fuel_ms"] = planner["fuel_ms"] if planner else None
                    rows.extend(paired)
                    print(f"event={event} seed={seed} noise_seed={noise_seed} complete", flush=True)
    finally:
        env.close()
    artifact = {
        "schema_version": SCHEMA_VERSION, "kind": "paired-evaluation", "fidelity": "Basilisk",
        "config": config, "partition": partition, "elapsed_seconds": time.perf_counter() - started,
        "partition_reference": "v2 run manifest when supplied; not a legacy-v1 source-holdout claim",
        "distribution_shifts": distribution_shifts,
        "provenance": run_provenance,
        "checkpoint_sha256": {name: sha256(path) for name, path in
                               (("v2", args.checkpoint), ("v1", args.v1)) if path},
        "checkpoint_role": "teacher-only initialization" if args.checkpoint
        and Path(args.checkpoint).name == "teacher.zip" else "policy checkpoint",
        "acceptance": "not established; inspect sample size and registered protocol",
        "summary": {policy: summarize([r for r in rows if r["policy"] == policy]) for policy in policies},
        "episodes": rows,
    }
    strata = {}
    for policy in policies:
        selected = [r for r in rows if r["policy"] == policy]
        for label, predicate in {
            "lead_under_1h": lambda r: r["initial_lead_s"] < 3600,
            "lead_1h_to_1day": lambda r: 3600 <= r["initial_lead_s"] < 86400,
            "lead_over_1day": lambda r: r["initial_lead_s"] >= 86400,
            "anisotropy_under_10": lambda r: np.sqrt(r["initial_covariance_eigenvalues_m2"][1]
                / r["initial_covariance_eigenvalues_m2"][0]) < 10,
            "anisotropy_over_10": lambda r: np.sqrt(r["initial_covariance_eigenvalues_m2"][1]
                / r["initial_covariance_eigenvalues_m2"][0]) >= 10,
        }.items():
            strata[f"{policy}/{label}"] = summarize([r for r in selected if predicate(r)])
    artifact["strata"] = strata
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(artifact, indent=2, allow_nan=False))
    return artifact


def export_evidence(run_directories, destination):
    """Publish measured artifacts only; recompute summaries with current null semantics."""
    run_directories = list(run_directories)
    runs = []
    for directory in map(Path, run_directories):
        manifest = json.loads((directory / "manifest.json").read_text())
        curve = []
        progress = directory / "progress.csv"
        if progress.exists() and progress.stat().st_size:
            with progress.open() as stream:
                for row in csv.DictReader(stream):
                    curve.append({"step": int(float(row["time/total_timesteps"])),
                                  "reward": float(row["rollout/ep_rew_mean"])
                                  if row.get("rollout/ep_rew_mean") else None})
        evaluations = []
        documents = [(path, json.loads(path.read_text())) for path in sorted(directory.glob("*.json"))
                     if path.name != "manifest.json"]
        references = [document for _, document in documents if document.get("kind") == "restricted-hindsight-grid"]
        for path, artifact in documents:
            if artifact.get("kind") != "paired-evaluation":
                continue
            episodes = artifact["episodes"]
            for reference in references:
                if reference["config"].get("max_lead_days") != artifact["config"].get("max_lead_days"):
                    continue
                for episode in episodes:
                    native = episode.get("native_pc")
                    if native is not None and native <= 1e-4:
                        episode["hindsight_grid_fuel_ms"] = 0.0  # paired coast proves zero is feasible
                    elif (episode["event_id"] == reference["event_id"]
                          and episode["seed"] == reference["seed"] and reference.get("best")
                          and native is not None and np.isclose(native, reference["evaluations"][0]["pc_final"],
                                                               rtol=1e-6, atol=1e-12)):
                        episode["hindsight_grid_fuel_ms"] = reference["best"]["fuel_ms"]
                        episode["hindsight_grid_scope"] = "single burn; six RTN axes; registered magnitude/timing grid"
            if artifact["config"].get("checkpoint"):
                membership = {event: name for name, ids in manifest["partitions"].items() for event in ids}
                teacher_sources = set((manifest.get("imitation") or {}).get("source_event_ids", []))
                for episode in episodes:
                    episode.setdefault("source_partition", membership.get(episode["event_id"], "unknown"))
                    if teacher_sources:
                        episode.setdefault("seen_by_imitation", episode["event_id"] in teacher_sources)
            policies = sorted({row["policy"] for row in episodes})
            artifact["summary"] = {policy: summarize([r for r in episodes if r["policy"] == policy])
                                   for policy in policies}
            # Keep complete measured decisions for public inspection; no invented rationale.
            evaluations.append({"label": path.stem, **artifact})
        runs.append({
            "id": directory.name, "label": directory.name.replace("-", " "),
            "steps": manifest.get("total_timesteps"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "steps_per_second": manifest.get("steps_per_second"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "manifest_sha256": sha256(directory / "manifest.json"),
            "config": manifest["config"], "data_sha256": manifest["data_sha256"],
            "source_sha256": manifest.get("source_sha256"),
            "versions": manifest["versions"], "learning_curve": curve,
            "observation_version": manifest.get("observation_version", "encounter-history-v2.0"),
            "imitation": manifest.get("imitation"),
            "references": references,
            "parent_checkpoint_sha256": manifest.get("parent_checkpoint_sha256"),
            "curriculum_change": manifest.get("curriculum_change"),
            "evaluations": evaluations, "status": manifest["status"],
        })
    result = {
        "schema_version": SCHEMA_VERSION, "kind": "research-evidence",
        "status": "acceptance-not-met", "runs": runs,
        "interpretation": "State-conditional imitation prototype observed; mitigation and generalization criteria not met.",
        "acceptance_criteria": CRITERIA,
        "limitations": [
            "Small pipeline proofs, not converged models or flight-qualified controllers.",
            "Seed-42 held-out hash test partition has no known ESA>1e-4 source family.",
            "Seen-family challenges are not held-out generalization evidence.",
            "The 0.2-day cap leaves many one-decision episodes; current covariance remains source-time-anchored.",
            "One dangerous source family cannot support a mitigation confidence interval.",
            "v1 receives privileged truth state; v2 and online baselines receive generated noisy estimates.",
            "J2 forecasts can disagree materially with Basilisk; noisy estimates and uncertainty evolution also affect warning risk.",
            "RCS-derived radius is an uncertain proxy; no real historical maneuver outcome is reconstructed.",
            "Early continuous proof predates per-decision packet and independent Foster diagnostics.",
        ],
    }
    validation = Path(run_directories[0]).parent / "teacher-response-validation.json"
    if validation.exists():
        result["validation_artifacts"] = [json.loads(validation.read_text())]
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", help="comma-separated source event IDs")
    parser.add_argument("--export-runs", nargs="+", help="export completed run directories as public evidence")
    parser.add_argument("--policies", default="never_maneuver,threshold,geometry_threshold,planner")
    parser.add_argument("--checkpoint")
    parser.add_argument("--v1")
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--forecast-fidelity", choices=("j2", "basilisk"), default="j2")
    parser.add_argument("--max-lead-days", type=float, default=None)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--observation-seeds", type=int, default=1,
                        help="independent sensor-noise repeats per fixed generated geometry")
    parser.add_argument("--allow-training-events", action="store_true")
    parser.add_argument("--out", default="runs/evaluation.json")
    args = parser.parse_args()
    if args.export_runs:
        export_evidence(args.export_runs, args.out)
        print(f"Published measured evidence: {args.out}")
    elif args.events:
        print(json.dumps(evaluate(args)["summary"], indent=2))
    else:
        parser.error("provide --events or --export-runs")


if __name__ == "__main__":
    main()
