"""Run provenance and declared acceptance criteria; no simulator dependency."""

import hashlib
import importlib.metadata
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "satellite-lab/2.0"
CRITERIA = {
    "pc_threshold": 1e-4,
    "safe_pc_threshold": 1e-6,
    "maneuver_threshold_ms": 1e-3,
    "mitigation_rate_min": 0.9,
    "mitigation_ci_lower_min": 0.8,
    "false_positive_rate_max": 0.1,
    "false_positive_ci_upper_max": 0.2,
    "training_seeds_required": 3,
    "observation_seeds_per_event_required": 3,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(config, data_dir):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unavailable"

    versions = {}
    for package in ("numpy", "gymnasium", "stable-baselines3", "bsk-rl", "Basilisk", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    if versions["Basilisk"] == "unavailable":
        try:
            import Basilisk
            versions["Basilisk"] = getattr(Basilisk, "__version__", "unknown")
        except ImportError:
            pass
    source_root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "git_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
        "versions": versions,
        "source_sha256": {str(p.relative_to(source_root)): sha256(p)
                          for p in sorted(source_root.rglob("*.py"))},
        "data_sha256": {p.name: sha256(p) for p in sorted(Path(data_dir).glob("*")) if p.is_file()},
        "acceptance_criteria": CRITERIA,
        "scientific_status": "pipeline proof; reactive-policy acceptance not demonstrated",
    }
