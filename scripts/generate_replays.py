"""Produce and index recorder-verified Basilisk examples. Run from repository root."""
import argparse
import json
from pathlib import Path

from web.backend.catalog import REPLAYS, sha256
from web.backend.contracts import ReplayArtifact, SimulationRequest
from web.backend.jobs import atomic_json
from web.backend.replay import generate_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, nargs="+", required=True)
    parser.add_argument("--policies", nargs="+", default=["v1", "never_maneuver", "threshold", "planner"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule-mode", choices=("controlled", "source"), default="controlled")
    parser.add_argument("--max-lead-days", type=float)
    parser.add_argument("--output", type=Path, default=REPLAYS)
    parser.add_argument("--verify", type=Path, help="Rerun this artifact request and compare physical results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for event_id in args.events:
        for policy in args.policies:
            request = SimulationRequest(scenario_id=f"kelvins-{event_id}", policy_id=policy, seed=args.seed,
                                        schedule_mode=args.schedule_mode, max_lead_days=args.max_lead_days)
            artifact = generate_replay(request.model_dump())
            ReplayArtifact.model_validate(artifact)
            path = args.output / f"{artifact['id']}.json"
            if args.verify:
                previous = json.loads(args.verify.read_text())
                for key in ("keyframes", "decisions", "dense_frames", "result"):
                    if previous[key] != artifact[key]:
                        raise RuntimeError(f"Replay verification differs in {key}")
                print(f"Deterministic re-execution matches {args.verify}", flush=True)
            else:
                atomic_json(path, artifact)
            print(artifact["id"], artifact["result"], flush=True)
    rows = []
    for path in sorted(args.output.glob("kelvins-*.json")):
        artifact = json.loads(path.read_text())
        rows.append({"id": artifact["id"], "title": f"Kelvins {artifact['scenario']['event_id']} · {artifact['scenario']['miss_distance_m']:g} m",
                     "event_id": artifact["scenario"]["event_id"], "policy_id": artifact["policy_id"],
                     "path": f"/replays/{path.name}", "verified": True, "sha256": sha256(path),
                     "summary": artifact["result"]})
    rows.sort(key=lambda row: (row["event_id"], row["policy_id"] != "v1", row["policy_id"]))
    atomic_json(args.output / "index.json", {"schema_version": "2.0", "replays": rows})


if __name__ == "__main__":
    main()
