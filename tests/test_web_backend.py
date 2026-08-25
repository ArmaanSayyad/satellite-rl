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


# The 7 known real ESA-anchored events -- see
# simulation_runner.HIGH_RISK_POOL_FRACTION.
KNOWN_MISS_DISTANCES = {38.0, 119.0, 237.0, 352.0, 473.0, 642.0, 1102.0}


def test_list_scenarios(client):
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()

    assert len(scenarios) == len(KNOWN_MISS_DISTANCES)
    assert {s["miss_distance_m"] for s in scenarios} == KNOWN_MISS_DISTANCES
    for s in scenarios:
        assert isinstance(s["seed"], int)
        assert s["relative_speed_ms"] > 0
        assert s["native_pc"] >= 0.0


def test_simulate_rejects_unknown_seed(client):
    resp = client.post("/api/simulate", params={"seed": -999})
    assert resp.status_code == 400


def test_simulate_returns_real_episode_for_chosen_seed(client):
    scenarios = client.get("/api/scenarios").json()
    target = next(s for s in scenarios if s["miss_distance_m"] == 38.0)

    resp = client.post("/api/simulate", params={"seed": target["seed"]})
    assert resp.status_code == 200
    body = resp.json()

    assert set(body.keys()) == {
        "seed",
        "scenario",
        "constants",
        "keyframes",
        "decisions",
        "dense_frames",
        "result",
        "baseline",
    }

    assert body["seed"] == target["seed"]
    assert body["scenario"]["miss_distance_m"] == 38.0

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

    baseline = body["baseline"]
    assert baseline["policy"] == "never_maneuver"
    assert isinstance(baseline["collision_occurred"], bool)
    # The 38m event is a genuinely dangerous one (native_pc > 1e-4, per
    # TECHNICAL.md §7) -- with literally no maneuver, the live
    # re-simulation should reproduce close to that same real risk, not
    # something wildly different (a loose bound, not exact, since
    # native_pc and pc_final come from different estimators -- see
    # TECHNICAL.md §6).
    assert baseline["pc_final"] > 1e-5


def test_simulate_random_pick_omits_seed_param(client):
    resp = client.post("/api/simulate")
    assert resp.status_code == 200
    assert resp.json()["scenario"]["miss_distance_m"] in KNOWN_MISS_DISTANCES
