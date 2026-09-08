"""Service contract tests run without physics or a checkpoint."""
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from web.backend.catalog import scenarios
from web.backend.contracts import SimulationRequest
from web.backend.jobs import JobManager, atomic_json
from web.backend.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("APSIS_LIVE", "0")
    with TestClient(create_app(JobManager(tmp_path))) as client:
        yield client


def test_replay_service_available_without_live_physics(client):
    health = client.get("/api/v2/health")
    assert health.json()["live_available"] is False
    catalog = client.get("/api/v2/scenarios").json()["scenarios"]
    assert len({row["id"] for row in catalog}) == len(catalog)
    event = next(row for row in catalog if row["id"] == "kelvins-8767")
    assert event["miss_distance_m"] == 38
    assert event["native_pc"] != event["esa_reported_pc"]
    assert client.get("/api/v2/replays").status_code == 200
    assert client.get("/api/v2/policies").status_code == 200


def test_disabled_live_has_actionable_error(client):
    response = client.post("/api/v2/jobs", json={"scenario_id": "kelvins-8767"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "live_unavailable"


@pytest.mark.parametrize("body", [
    {"scenario_id": "../../etc/passwd"},
    {"scenario_id": "kelvins-8767", "radius_scale": 200},
    {"scenario_id": "kelvins-8767", "seed": -1},
    {"scenario_id": "kelvins-8767", "policy_id": "arbitrary-code"},
    {"scenario_id": "kelvins-8767", "unexpected": True},
])
def test_request_bounds(client, body):
    assert client.post("/api/v2/jobs", json=body).status_code == 422


def test_job_not_found_and_path_safety(client):
    assert client.get("/api/v2/jobs/not-a-cache-key").status_code == 404
    assert client.get("/api/v2/jobs/" + "a" * 64).status_code == 404


def test_interrupted_worker_is_failure_and_never_an_artifact(tmp_path):
    manager = JobManager(tmp_path)
    identifier = "a" * 64
    folder = tmp_path / identifier
    folder.mkdir()
    atomic_json(folder / "request.json", {"scenario_id": "kelvins-8767"})
    assert manager.get(identifier).status == "failed"
    assert manager.get(identifier).artifact_url is None
    assert manager.cancel(identifier).status == "cancelled"


def test_cached_completed_job_returns_same_artifact(tmp_path):
    from web.backend.catalog import cache_key
    request = SimulationRequest(scenario_id="kelvins-8767", policy_id="never_maneuver")
    identifier = cache_key(request.model_dump())
    folder = tmp_path / identifier
    folder.mkdir()
    atomic_json(folder / "artifact.json", {"result": "already complete"})
    manager = JobManager(tmp_path)
    response = manager.submit(request)
    assert response.cached and response.status == "completed"
    assert not manager.processes


def test_shipped_replays_match_catalog_and_endpoint_verification():
    from web.backend.catalog import REPLAYS, sha256
    from web.backend.contracts import ReplayArtifact
    index = REPLAYS / "index.json"
    if not index.exists():
        pytest.skip("Generate recorder-verified examples first")
    available = {s["event_id"] for s in scenarios()}
    for entry in json.loads(index.read_text())["replays"]:
        path = REPLAYS / Path(entry["path"]).name
        assert sha256(path) == entry["sha256"]
        artifact = ReplayArtifact.model_validate_json(path.read_text())
        assert artifact.provenance["source_event_id"] in available
        assert artifact.provenance["verification"]["max_position_error_m"] < 1e-5
        assert artifact.decisions[0]["t_s"] == 0
        times = [frame.t_s for frame in artifact.dense_frames]
        assert times == sorted(set(times))


@pytest.mark.parametrize("info", [
    {},
    {"status": "failed", "pc_final_valid": False, "pc_final": 1.0},
    {"status": "completed", "pc_final_valid": False, "pc_final": 1.0},
    {"status": "completed", "pc_final_valid": True, "pc_final": float("nan")},
    {"status": "completed", "pc_final_valid": True, "pc_final": 2.0},
])
def test_failed_or_invalid_scores_cannot_be_verified_replays(info):
    from web.backend.contracts import require_scored_episode
    with pytest.raises(ValueError):
        require_scored_episode(info)


def test_valid_zero_probability_is_a_publishable_score():
    from web.backend.contracts import require_scored_episode
    require_scored_episode({"status": "completed", "pc_final_valid": True, "pc_final": 0.0})
