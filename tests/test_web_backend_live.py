"""Opt-in process-isolation integration test, using real physics without a TCP listener.

APSIS_TEST_LIVE=1 python -m pytest tests/test_web_backend_live.py -q
"""
import os
import time

import pytest

if os.getenv("APSIS_TEST_LIVE") != "1":
    pytest.skip("Opt in with APSIS_TEST_LIVE=1", allow_module_level=True)
pytest.importorskip("bsk_rl")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from web.backend.jobs import JobManager
from web.backend.main import create_app


def test_live_job_cache_cancellation_and_checkpoint_decoding(tmp_path, monkeypatch):
    monkeypatch.setenv("APSIS_LIVE", "1")
    manager = JobManager(tmp_path, max_workers=1)
    with TestClient(create_app(manager)) as client:
        request = {"scenario_id": "kelvins-8767", "policy_id": "v2", "seed": 0}
        first = client.post("/api/v2/jobs", json=request)
        assert first.status_code == 202
        identifier = first.json()["id"]
        busy = client.post("/api/v2/jobs", json={**request, "seed": 1})
        assert busy.status_code == 429
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            job = client.get(f"/api/v2/jobs/{identifier}").json()
            if job["status"] != "running":
                break
            time.sleep(0.5)
        assert job["status"] == "completed", job
        artifact = client.get(job["artifact_url"]).json()
        assert artifact["provenance"]["forecast_fidelity"] == "basilisk"
        assert artifact["provenance"]["action_mode"] == "discrete19"
        assert len(artifact["decisions"][0]["observation"]["vector"]) == 72
        assert artifact["result"]["pc_final_valid"] is True
        repeated = client.post("/api/v2/jobs", json=request).json()
        assert repeated["cached"] and repeated["id"] == identifier
        cancelled = client.post("/api/v2/jobs", json={**request, "seed": 2}).json()
        stopped = client.delete(f"/api/v2/jobs/{cancelled['id']}").json()
        assert stopped["status"] == "cancelled"
        assert client.get(f"/api/v2/jobs/{cancelled['id']}/artifact").status_code == 409
