"""Bounded full-physics check of the local J2 burn response around an HF forecast."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from satellite_rl.lab.environment import make_v2_env
from satellite_rl.lab.planner import plane_pc, response_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", type=int, default=8767)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--magnitude", type=float, default=0.01)
    parser.add_argument("--max-lead-days", type=float, default=0.2)
    parser.add_argument("--out", default="runs/teacher-response-validation.json")
    args = parser.parse_args()
    started = time.perf_counter()
    env = make_v2_env(
        event_ids=[args.event_id],
        max_lead_days=args.max_lead_days,
        observation_noise_scale=0,
        forecast_fidelity="basilisk",
    )
    rows = []
    try:
        _, _ = env.reset(seed=args.seed, options={"event_id": args.event_id})
        packet = dict(env.latest_estimate)
        basis = np.asarray(packet["basis_eci"])
        initial = np.asarray(packet["miss_plane_m"])
        response = response_matrix(packet)
        final_cov = np.diag(np.asarray(env.unwrapped._sampler.sigma_xz_at_fraction(1)) ** 2)
        actions = [("none", np.zeros(3))] + [
            (f"{axis}{'+' if sign > 0 else '-'}", np.eye(3)[index] * sign * args.magnitude)
            for index, axis in enumerate("RTN")
            for sign in (-1, 1)
        ]
        for name, action in actions:
            env.reset(seed=args.seed, options={"event_id": args.event_id})
            first = True
            while True:
                _, _, terminated, truncated, info = env.step(action if first else np.zeros(3))
                first = False
                if terminated or truncated:
                    break
            if not info.get("pc_final_valid"):
                raise RuntimeError(f"Invalid physical episode: {name}")
            ego, sec = env.unwrapped.satellites
            actual = basis.T @ (np.asarray(sec.dynamics.r_BN_N) - np.asarray(ego.dynamics.r_BN_N))
            predicted = initial + response @ action
            row = {
                "action": name,
                "action_rtn_ms": action.tolist(),
                "predicted_miss_fixed_plane_m": predicted.tolist(),
                "actual_miss_fixed_plane_m": actual.tolist(),
                "miss_vector_error_m": float(np.linalg.norm(predicted - actual)),
                "predicted_miss_norm_m": float(np.linalg.norm(predicted)),
                "actual_miss_norm_m": float(np.linalg.norm(actual)),
                "predicted_pc_current_covariance": plane_pc(
                    predicted, packet["covariance_plane_m2"], packet["radius_m"]
                ),
                "predicted_pc_final_covariance": plane_pc(predicted, final_cov, packet["radius_m"]),
                "actual_pc_final": float(info["pc_final"]),
                "actual_fuel_ms": info["cumulative_fuel_used_ms"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    artifact = {
        "kind": "local-j2-response-vs-basilisk",
        "config": vars(args),
        "elapsed_seconds": time.perf_counter() - started,
        "initial_estimate": packet,
        "response_m_per_ms": response.tolist(),
        "final_covariance_plane_m2": final_cov.tolist(),
        "rows": rows,
        "scope": "One source, one generated orbit, six small first-step impulses. Not a global surrogate validation. Predictions freeze encounter basis; final covariance is used only for diagnostic isolation, not supplied to the online teacher.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
