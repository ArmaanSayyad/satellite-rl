"""Stable source IDs and policy metadata, available without Basilisk."""

import csv
import hashlib
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = ROOT / "data/fitted/geometry_events.csv"
REPLAYS = ROOT / "web/frontend/public/replays"
V1_SHA256 = "31f52cb9f640cd6f58eedb8c3afb0102ab288d8c9e2679e1d1c7f619d8d3dd64"


def model_path(policy_id: str) -> Path:
    if policy_id == "v1":
        return Path(os.getenv("APSIS_V1_MODEL", str(ROOT / "runs/ppo_stage2_riskaware_run1.zip")))
    return Path(os.getenv("APSIS_V2_MODEL", str(ROOT / "assets/models/v2-imitation-hf/model.zip")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scenarios() -> list[dict]:
    with GEOMETRY.open() as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: max(float(r["native_pc"]), float(r["esa_reported_pc"])), reverse=True)
    return [
        {
            "id": f"kelvins-{int(row['event_id'])}",
            "event_id": int(row["event_id"]),
            "title": f"Kelvins {row['event_id']}",
            "miss_distance_m": float(row["miss_distance"]),
            "relative_speed_ms": float(row["relative_speed"]),
            "native_pc": float(row["native_pc"]),
            "esa_reported_pc": float(row["esa_reported_pc"]),
            "combined_radius_m": float(row["combined_radius"]),
            "sigma_x_m": float(row["sigma_x"]),
            "sigma_z_m": float(row["sigma_z"]),
            "source_kind": "historical_geometry",
            "radius_note": "RCS-derived approximation, not measured physical radius",
        }
        for row in rows
    ]


def policies() -> list[dict]:
    return [
        {"id": "never_maneuver", "name": "Coast", "kind": "baseline", "available": True},
        {"id": "threshold", "name": "Radial threshold", "kind": "baseline", "available": True},
        {"id": "planner", "name": "Geometry planner", "kind": "numerical_controller", "available": True,
         "limitation": "Bounded J2 prediction search; final outcomes judged in Basilisk"},
        {"id": "v1", "name": "PPO · v1", "kind": "checkpoint", "available": model_path("v1").exists(),
         "observation_version": "v1-truth-9", "status": "insurance strategy; not risk-gating"},
        {"id": "v2", "name": "PPO · v2 experimental", "kind": "checkpoint", "available": model_path("v2").exists(),
         "observation_version": "v2", "status": "experimental; see evaluation artifact for measured evidence"},
    ]


def cache_key(request: dict) -> str:
    import json

    sources = sorted((ROOT / "src/satellite_rl").rglob("*.py"))
    sources += sorted((ROOT / "web/backend").glob("*.py"))
    sources += sorted((ROOT / "data/fitted").glob("*"))
    manifest = {str(p.relative_to(ROOT)): sha256(p) for p in sources if p.is_file()}
    policy = request["policy_id"]
    if policy in ("v1", "v2") and model_path(policy).exists():
        manifest["model"] = sha256(model_path(policy))
        model_manifest = model_path(policy).parent / "manifest.json"
        if model_manifest.exists():
            manifest["model_manifest"] = sha256(model_manifest)
    manifest["python"] = platform.python_version()
    for package in ("bsk", "bsk-rl", "numpy", "scipy", "stable-baselines3"):
        try:
            manifest[package] = version(package)
        except PackageNotFoundError:
            manifest[package] = "not-installed"
    return hashlib.sha256(json.dumps({"request": request, "sources": manifest}, sort_keys=True).encode()).hexdigest()
