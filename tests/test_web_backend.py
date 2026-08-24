"""Smoke test for the web demo's FastAPI backend (web/backend/).

Needs bsk_rl (like tests/test_env.py) AND a trained model checkpoint
(runs/ppo_stage2_riskaware_run1.zip, gitignored -- see web/README.md
for how to produce one), so this skips automatically wherever either
is unavailable -- including CI, which has neither. Where it does run,
it exercises the real endpoint end-to-end: real policy, real Basilisk
physics, no mocking.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("bsk_rl")
pytest.importorskip("fastapi")

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "web" / "backend"
MODEL_PATH = REPO_ROOT / "runs" / "ppo_stage2_riskaware_run1.zip"

if not MODEL_PATH.exists():
    pytest.skip(
        f"no trained model at {MODEL_PATH} -- see web/README.md",
        allow_module_level=True,
    )

sys.path.insert(0, str(BACKEND_DIR))
from main import app


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_simulate_returns_real_episode(client):
    resp = client.post("/api/simulate")
    assert resp.status_code == 200
    body = resp.json()

    assert set(body.keys()) == {
        "scenario",
        "constants",
        "keyframes",
        "decisions",
        "dense_frames",
        "result",
    }

    # This must be one of the 7 known real ESA-anchored events, not an
    # arbitrary draw -- see simulation_runner.HIGH_RISK_POOL_FRACTION.
    known_miss_distances = {38.0, 119.0, 237.0, 352.0, 473.0, 642.0, 1102.0}
    assert body["scenario"]["miss_distance_m"] in known_miss_distances

    assert len(body["keyframes"]) >= 2
    assert len(body["dense_frames"]) >= len(body["keyframes"])
    assert len(body["decisions"]) == len(body["keyframes"]) - 1

    for frame in (body["keyframes"][0], body["dense_frames"][0]):
        for key in ("ego_r", "ego_v", "sec_r", "sec_v"):
            assert len(frame[key]) == 3

    result = body["result"]
    assert result["total_fuel_used_ms"] >= 0.0
    assert result["maneuver_count"] >= 0
    assert isinstance(result["collision_occurred"], bool)
