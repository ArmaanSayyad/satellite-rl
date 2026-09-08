"""Observation-based, restricted J2 maneuver search; Basilisk judges outcomes.

No true secondary state is consumed. Current covariance and encounter plane are
frozen; finite-difference orbital response is linearized. Predictions are not
claims of fidelity and must be compared with final simulator outcomes.
"""

from dataclasses import asdict, dataclass

import numpy as np

from ..pc import EncounterGeometry2D, pc_chan


def plane_pc(miss, covariance, radius):
    values, vectors = np.linalg.eigh(np.asarray(covariance))
    if np.any(values <= 0):
        raise ValueError("covariance must be positive definite")
    offset = vectors.T @ np.asarray(miss)
    return pc_chan(EncounterGeometry2D(*offset, *np.sqrt(values)), float(radius))


def rtn_basis(r, v):
    radial = np.asarray(r) / np.linalg.norm(r)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal)
    return np.column_stack((radial, np.cross(normal, radial), normal))


def response_matrix(estimate, wait_s=0.0, epsilon_ms=0.01):
    from ..scenario.targeting import propagate_state

    lead = float(estimate["time_to_tca_s"])
    if not 0 <= wait_s <= lead:
        raise ValueError("burn timing must lie before TCA")
    r, v = propagate_state(np.asarray(estimate["ego_r_eci_m"]),
                           np.asarray(estimate["ego_v_eci_ms"]), wait_s)
    frame = rtn_basis(r, v)
    basis = np.asarray(estimate["basis_eci"])
    columns = []
    for axis in frame.T:
        plus, _ = propagate_state(r, v + epsilon_ms * axis, lead - wait_s)
        minus, _ = propagate_state(r, v - epsilon_ms * axis, lead - wait_s)
        columns.append(-basis.T @ (plus - minus) / (2 * epsilon_ms))
    return np.column_stack(columns)


@dataclass
class ManeuverPlan:
    action_rtn_ms: list
    wait_s: float
    predicted_pc: float
    baseline_predicted_pc: float
    feasible: bool
    candidates: int
    model: str = "finite-difference J2, frozen covariance/plane; not global optimum"

    def to_dict(self):
        return asdict(self)


def search_plan(estimate, threshold=1e-4, max_dv_ms=10.0, waits_s=(0.0,),
                magnitudes=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
                response_fn=response_matrix):
    """Search 26 RTN directions plus dominant response directions at each time."""
    miss = np.asarray(estimate["miss_plane_m"])
    cov = np.asarray(estimate["covariance_plane_m2"])
    radius = estimate["radius_m"]
    initial = plane_pc(miss, cov, radius)
    best = ManeuverPlan([0.0] * 3, 0.0, initial, initial, initial <= threshold, 1)
    if best.feasible:
        return best
    bound = min(max_dv_ms, estimate.get("fuel_remaining_ms", max_dv_ms))
    count = 1
    for wait in waits_s:
        response = response_fn(estimate, wait)
        directions = np.array([[r, t, n] for r in (-1, 0, 1) for t in (-1, 0, 1)
                               for n in (-1, 0, 1) if (r, t, n) != (0, 0, 0)], dtype=float)
        _, _, right = np.linalg.svd(response)
        directions = np.vstack((directions, right, -right))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        for magnitude in sorted({float(m) for m in magnitudes if 0 < m <= bound}):
            for direction in directions:
                action = direction * magnitude
                pc = plane_pc(miss + response @ action, cov, radius)
                count += 1
                feasible = pc <= threshold
                candidate = ManeuverPlan(action.tolist(), float(wait), pc, initial, feasible, count)
                better_feasible = feasible and (not best.feasible or magnitude < np.linalg.norm(best.action_rtn_ms)
                                 or (np.isclose(magnitude, np.linalg.norm(best.action_rtn_ms))
                                     and pc < best.predicted_pc))
                if better_feasible or (not best.feasible and pc < best.predicted_pc):
                    best = candidate
    best.candidates = count
    return best


def geometry_threshold_action(estimate, threshold=1e-4, magnitude_ms=0.1):
    """A risk-gated direction search with one fixed magnitude, no future knowledge."""
    return np.asarray(search_plan(estimate, threshold, magnitude_ms,
                                 magnitudes=(magnitude_ms,)).action_rtn_ms, dtype=np.float32)


def hindsight_grid_reference(env, event_id, seed=100, threshold=1e-4,
                             magnitudes=(.001, .003, .01, .03, .1), max_evaluations=100):
    """Privileged repeated Basilisk outcomes; restricted single-burn grid, not global optimum.

    Search each real decision time and six signed RTN axes, in increasing fuel
    order. Once a magnitude succeeds, finish its directions/times and stop: all
    lower grid magnitudes have been checked. An evaluation budget makes unfinished
    grids explicit rather than returning an unsupported optimality claim.
    """
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be positive")
    records = []

    def execute(step, action):
        _, _ = env.reset(seed=seed, options={"event_id": event_id})
        index = 0
        while True:
            _, _, terminated, truncated, info = env.step(action if index == step else np.zeros(3))
            index += 1
            if terminated or truncated:
                break
        valid = info.get("pc_final_valid", "pc_final" in info) and np.isfinite(info.get("pc_final", np.nan))
        pc = float(info["pc_final"]) if valid else None
        record = {"step": step, "action_rtn_ms": np.asarray(action).tolist(),
                  "fuel_ms": float(info["cumulative_fuel_used_ms"]), "pc_final": pc,
                  "status": info.get("status", "completed" if valid else "failed"),
                  "failure_reason": info.get("failure_reason"),
                  "feasible": bool(pc is not None and pc <= threshold)}
        records.append(record)
        return record

    coast = execute(-1, np.zeros(3))
    times = list(range(len(env.unwrapped.schedule_s) - 1))
    best = coast if coast["feasible"] else None
    complete = bool(best)
    if not best:
        for magnitude in sorted(magnitudes):
            if magnitude <= 0:
                raise ValueError("grid magnitudes must be positive")
            for timing in times:
                for direction in np.vstack((np.eye(3), -np.eye(3))):
                    if len(records) >= max_evaluations:
                        break
                    candidate = execute(timing, direction * magnitude)
                    if candidate["feasible"] and (best is None or
                        (candidate["fuel_ms"], candidate["pc_final"]) < (best["fuel_ms"], best["pc_final"])):
                        best = candidate
                if len(records) >= max_evaluations:
                    break
            if best is not None:
                complete = len(records) < max_evaluations
                break
            if len(records) >= max_evaluations:
                break
        else:
            complete = True
    return {"schema_version": "satellite-lab/2.0", "kind": "restricted-hindsight-grid",
            "fidelity": "Basilisk", "event_id": int(event_id), "seed": seed,
            "information": "privileged repeated future outcomes; not an online policy",
            "globally_optimal": False, "minimum_grid_fuel_established": complete and best is not None,
            "native_pc": coast["pc_final"],
            "completed_search": complete, "best": best, "evaluations": records,
            "magnitudes_ms": list(magnitudes), "decision_steps": times,
            "directions": "six signed RTN axes", "threshold": threshold}


def main():
    import argparse
    import json
    import time
    from pathlib import Path

    from ..scenario.distributions import FITTED_DIR
    from .environment import make_v2_env
    from .protocol import provenance

    parser = argparse.ArgumentParser(description="Restricted Basilisk hindsight fuel reference")
    parser.add_argument("--event", type=int, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-lead-days", type=float, default=None)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--out", default="runs/hindsight-reference.json")
    args = parser.parse_args()
    run_provenance = provenance(vars(args), FITTED_DIR)
    env = make_v2_env(event_ids=[args.event], observation_noise_scale=0,
                      max_lead_days=args.max_lead_days, high_risk_fraction=0,
                      high_risk_augment=False).unwrapped
    started = time.perf_counter()
    try:
        result = hindsight_grid_reference(env, args.event, args.seed, max_evaluations=args.max_evaluations)
    finally:
        env.close()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["config"] = vars(args)
    result["provenance"] = run_provenance
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in ("best", "completed_search", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
