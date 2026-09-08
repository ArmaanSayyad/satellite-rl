"""Generate artifacts directly from live Basilisk recorders, never reconstructed arcs."""
import json
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

from satellite_rl.env import CollisionAvoidanceEnv
from satellite_rl.lab.data import load_keyed_schedules
from satellite_rl.lab.environment import EncounterHistoryWrapper
from satellite_rl.lab.planner import search_plan
from satellite_rl.lab.train import decode_action
from satellite_rl.pc import compute_pc
from satellite_rl.pc.geometry import encounter_plane_basis

from .catalog import GEOMETRY, ROOT, model_path, scenarios, sha256
from .contracts import require_scored_episode

SCHEDULE = (0.2, 0.1, 0.05, 0.01, 0.0)


def _frame(base):
    ego, sec = base.satellites
    return {"t_s": float(base.simulator.sim_time),
            "ego_r": list(ego.dynamics.r_BN_N), "ego_v": list(ego.dynamics.v_BN_N),
            "sec_r": list(sec.dynamics.r_BN_N), "sec_v": list(sec.dynamics.v_BN_N)}


def _encounter(packet):
    return {"miss_vector_m": packet["miss_plane_m"],
            "covariance_m2": packet["covariance_plane_m2"],
            "time_to_tca_s": packet["time_to_tca_s"], "basis_eci": packet["basis_eci"],
            "kind": packet["kind"] if packet["noise_scale"] else "truth_derived_estimate",
            "prediction_fidelity": packet.get("forecast_fidelity", "J2"),
            "noise_scale": packet["noise_scale"]}


def _episode(request, policy_id, progress):
    event_id = int(request["scenario_id"].split("-")[-1])
    source_schedule = request.get("schedule_mode") == "source"
    base = CollisionAvoidanceEnv(sample_geometry=True, event_ids=[event_id],
                                evolve_uncertainty=source_schedule,
                                keyed_schedules=load_keyed_schedules() if source_schedule else None,
                                max_lead_days=request.get("max_lead_days"),
                                schedule_days_before_tca=SCHEDULE,
                                high_risk_fraction=1.0, high_risk_pool_fraction=1.0,
                                high_risk_augment=request.get("variant") == "posterior_variant",
                                high_risk_precise_targeting=True)
    # Radius scaling changes the risk model, not the sampled historical geometry.
    scale = request.get("radius_scale", 1.0)
    for table in (base._sampler.geometry_df, base._sampler.high_risk_df):
        table["combined_radius"] = table["combined_radius"] * scale
    history = 3
    forecast_fidelity = "j2"
    action_mode = "continuous"
    model = None
    if policy_id in ("v1", "v2"):
        from stable_baselines3 import PPO
        model = PPO.load(model_path(policy_id), device="cpu")
        if policy_id == "v2":
            history = model.observation_space.shape[0] // 24
            manifest_path = model_path(policy_id).parent / "manifest.json"
            if not manifest_path.exists():
                raise ValueError("V2 checkpoint requires its training manifest.json alongside it")
            manifest = json.loads(manifest_path.read_text())
            action_mode = manifest["config"].get("action_mode", "continuous")
            forecast_fidelity = manifest["config"].get("forecast_fidelity", "j2")
    env = EncounterHistoryWrapper(base, history_length=history,
                                  observation_noise_scale=1.0 if policy_id in ("v2", "planner") else 0.0,
                                  forecast_fidelity=forecast_fidelity)
    try:
        options = {"event_id": event_id}
        if request.get("observation_seed") is not None:
            options["observation_seed"] = request["observation_seed"]
        obs, reset_info = env.reset(seed=request.get("seed", 0), options=options)
        sample = dict(base._sampler.current_sample)
        recorders = []
        for satellite in base.satellites:
            recorder = satellite.dynamics.scObject.scStateOutMsg.recorder()
            base.simulator.AddModelToTask(satellite.dynamics.task_name, recorder)
            recorder.Reset(0)
            recorders.append(recorder)

        def keyframe():
            raw = base.satellites[0].get_obs()
            pc = float(raw[0]) if policy_id in ("v1", "threshold", "never_maneuver") else env.latest_estimate["estimated_pc"]
            return {**_frame(base), "fuel_ms": float(base.satellites[0].fsw.dv_available),
                    "pc_estimate": pc, "encounter": _encounter(env.latest_estimate)}

        keyframes = [keyframe()]
        decisions = []
        info = {}
        done = False
        while not done:
            policy_observation = obs.copy()
            packet = dict(env.latest_estimate)
            raw = base.satellites[0].get_obs()
            planner = None
            if policy_id in ("v1", "v2"):
                action, _ = model.predict(raw if policy_id == "v1" else obs, deterministic=True)
                action = decode_action(action, action_mode)
            elif policy_id == "planner":
                planner = search_plan(packet, threshold=base.pc_threshold, max_dv_ms=base.max_dv_ms)
                action = np.array(planner.action_rtn_ms, dtype=np.float32)
            elif policy_id == "threshold":
                action = np.array([base.max_dv_ms if raw[0] > base.pc_threshold else 0, 0, 0], dtype=np.float32)
            else:
                action = np.zeros(3, dtype=np.float32)
            time_s = float(base.simulator.sim_time)
            before = float(base.satellites[0].fsw.dv_available)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            after = float(base.satellites[0].fsw.dv_available)
            decisions.append({"t_s": time_s, "action_dv_ms": np.asarray(action).tolist(),
                              "action_magnitude_ms": float(np.linalg.norm(action)),
                              "fuel_before_ms": before, "fuel_after_ms": after,
                              "observation": {"pc_estimate": float(raw[0]) if policy_id in ("v1", "threshold", "never_maneuver") else packet["estimated_pc"],
                                              "vector": raw.tolist() if policy_id == "v1" else policy_observation.tolist() if policy_id == "v2" else None,
                                              "version": "v1-truth-9" if policy_id == "v1" else packet["version"],
                                              "packet": packet},
                              "encounter": _encounter(packet), "planner_prediction": planner.to_dict() if planner else None,
                              "reward": float(reward)})
            keyframes.append(keyframe())
            progress(base.schedule_index / (len(base.schedule_s) - 1))

        require_scored_episode(info)
        times = np.asarray(recorders[0].times()) * 1e-9
        r1, v1 = np.asarray(recorders[0].r_BN_N), np.asarray(recorders[0].v_BN_N)
        r2, v2 = np.asarray(recorders[1].r_BN_N), np.asarray(recorders[1].v_BN_N)
        if len(times) < 2:
            raise RuntimeError("Basilisk recorder did not produce a trajectory")
        # Verify every decision endpoint against the independent message recorder.
        errors = []
        for frame in keyframes[1:]:
            i = int(np.argmin(np.abs(times - frame["t_s"])))
            if abs(times[i] - frame["t_s"]) > 1e-6:
                raise RuntimeError("Recorder does not bracket the exact decision endpoint")
            errors.extend([float(np.linalg.norm(r1[i] - frame["ego_r"])),
                           float(np.linalg.norm(r2[i] - frame["sec_r"]))])
        if max(errors) > 1e-5:
            raise RuntimeError(f"Recorder endpoint mismatch: {max(errors)} m")
        stride = max(1, len(times) // 480)
        indices = sorted(set(range(0, len(times), stride)) | {len(times)-1})
        dense = [{"t_s": float(times[i]), "ego_r": r1[i].tolist(), "ego_v": v1[i].tolist(),
                  "sec_r": r2[i].tolist(), "sec_v": v2[i].tolist()} for i in indices]
        # Include exact decision knots (and the reset state) for synchronized seeking.
        dense_by_time = {frame["t_s"]: frame for frame in dense}
        for frame in keyframes:
            dense_by_time[frame["t_s"]] = {k: frame[k] for k in ("t_s", "ego_r", "ego_v", "sec_r", "sec_v")}
        final = keyframes[-1]
        relative = np.array(final["sec_r"]) - final["ego_r"]
        velocity = np.array(final["sec_v"]) - final["ego_v"]
        basis = encounter_plane_basis(velocity)
        ego = base.satellites[0]
        covariance = np.diag([ego._pc_sigma_x**2, ego._pc_sigma_z**2])
        radius = ego._pc_combined_radius
        foster = compute_pc(relative, velocity, basis @ covariance @ basis.T, radius, method="foster")
        sensitivity = {str(factor): compute_pc(relative, velocity, basis @ covariance @ basis.T, radius * factor, method="foster") for factor in (0.5, 1, 2)}
        local_offset = -float(relative @ velocity) / float(velocity @ velocity)
        # Truth geometry is separate from the last generated measurement.
        final["truth_encounter"] = {"miss_vector_m": (basis.T @ relative).tolist(),
                                    "covariance_m2": covariance.tolist(), "basis_eci": basis.tolist(),
                                    "time_to_tca_s": 0, "prediction_fidelity": "Basilisk terminal state"}
        return {"sample": sample, "keyframes": keyframes, "decisions": decisions,
                "dense_frames": [dense_by_time[t] for t in sorted(dense_by_time)],
                "verification": {"max_position_error_m": max(errors), "checked_endpoints": len(keyframes)-1,
                                 "method": "live spacecraft message recorder vs environment state at every decision endpoint",
                                 "recorded_samples": len(times)},
                "result": {"pc_final": float(info["pc_final"]), "pc_final_foster": float(foster),
                           "total_fuel_used_ms": float(info["cumulative_fuel_used_ms"]),
                           "maneuver_count": int(info["maneuver_count"]), "status": info.get("status", "completed"),
                           "collision_occurred": None, "radius_sensitivity_foster": sensitivity,
                           "local_linear_tca_offset_s": local_offset,
                           "pc_final_valid": bool(info.get("pc_final_valid", info.get("status") == "completed")),
                           "failure_reason": info.get("failure_reason")},
                "scenario_info": reset_info.get("scenario", {}),
                "schedule_days": [float(t / 86400) for t in base.schedule_s],
                "forecast_fidelity": forecast_fidelity, "action_mode": action_mode}
    finally:
        env.close()


def generate_replay(request: dict, progress=lambda _: None) -> dict:
    policy = request["policy_id"]
    episode = _episode(request, policy, lambda p: progress(p * 0.8))
    baseline = episode if policy == "never_maneuver" else _episode(request, "never_maneuver", lambda p: progress(0.8 + p * 0.2))
    sample = episode.pop("sample")
    source = next(row for row in scenarios() if row["id"] == request["scenario_id"])
    identifier = f"{request['scenario_id']}-{policy}-s{request.get('seed', 0)}"
    if request.get("variant", "historical_geometry") != "historical_geometry" or request.get("radius_scale", 1) != 1:
        identifier += f"-{request.get('variant', 'historical_geometry')}-r{request.get('radius_scale', 1):g}"
    return {"schema_version": "2.0", "id": identifier, "policy_id": policy, "seed": request.get("seed", 0),
            "provenance": {"source_event_id": int(request["scenario_id"].split("-")[-1]),
                           "kind": request.get("variant", "historical_geometry"), "simulator": "Basilisk",
                           "generated_at": datetime.now(timezone.utc).isoformat(), "seed": request.get("seed", 0),
                           "bsk_rl_version": version("bsk-rl"), "sb3_version": version("stable-baselines3"),
                           "geometry_sha256": sha256(GEOMETRY),
                           "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src/satellite_rl").rglob("*.py"))},
                           "forecast_fidelity": episode["forecast_fidelity"],
                           "action_mode": episode["action_mode"],
                           "checkpoint_sha256": sha256(model_path(policy)) if policy in ("v1", "v2") else None,
                           "verification": episode.pop("verification"),
                           "schedule_kind": "controlled_fixed_comparison" if request.get("schedule_mode", "controlled") == "controlled" else "bounded_source_window" if request.get("max_lead_days") else "source_linked_irregular",
                           "schedule_days": episode["schedule_days"],
                           "observation_seed": request.get("observation_seed"),
                           "radius_scale": request.get("radius_scale", 1.0),
                           "fidelity_note": f"Actual Basilisk spacecraft recordings; display interpolates between samples. Predicted CDM geometry uses {episode['forecast_fidelity']}. Historical encounter geometry is embedded in a generated reference orbit, not a historical orbit reconstruction.",
                           "observation_note": "v1/legacy baselines receive truth-derived J2 estimates; planner/v2 receive noisy generated estimates. Relative-velocity direction is assumed known."},
            "scenario": {"miss_distance_m": sample["miss_distance"], "relative_speed_ms": sample["relative_speed"],
                         "native_pc": sample["native_pc"], "combined_radius_m": sample["combined_radius"],
                         "source_native_pc": source["native_pc"], "esa_reported_pc": source["esa_reported_pc"],
                         "event_id": int(request["scenario_id"].split("-")[-1]), "sigma_x_m": sample["sigma_x"], "sigma_z_m": sample["sigma_z"]},
            "constants": {"earth_mu_m3s2": 3.986004418e14, "earth_radius_m": 6378136.6,
                          "max_dv_ms": 10, "pc_threshold": 1e-4},
            **episode,
            "baseline": {"policy": "never_maneuver", "pc_final": baseline["result"]["pc_final"],
                         "collision_occurred": None, "status": baseline["result"]["status"]}}
