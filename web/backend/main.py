"""Apsis service. Replay browsing never imports Basilisk."""
import importlib.util
import json
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .catalog import REPLAYS, ROOT, model_path, policies, scenarios
from .contracts import JobResponse, SimulationRequest
from .jobs import JobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    app.state.jobs.close()


def create_app(manager: JobManager | None = None) -> FastAPI:
    app = FastAPI(title="Apsis · Conjunction Research Instrument", version="2.0", lifespan=lifespan)
    app.state.jobs = manager or JobManager(max_workers=int(os.getenv("APSIS_WORKERS", "2")))
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(CORSMiddleware,
                       allow_origins=os.getenv("APSIS_API_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])

    def job_id(identifier: str):
        if not re.fullmatch(r"[a-f0-9]{64}", identifier):
            raise HTTPException(404, {"code": "unknown_job", "message": "Job not found"})

    @app.exception_handler(KeyError)
    async def missing(_request, _error):
        return JSONResponse(status_code=404, content={"detail": {"code": "not_found", "message": "Requested artifact or job does not exist"}})

    @app.get("/api/health")
    @app.get("/api/v2/health")
    def health():
        return {"status": "ok", "schema_version": "2.0",
                "live_available": os.getenv("APSIS_LIVE", "0") == "1" and importlib.util.find_spec("Basilisk") is not None}

    @app.get("/api/v2/scenarios")
    def list_scenarios():
        return {"schema_version": "2.0", "scenarios": scenarios()}

    @app.get("/api/v2/policies")
    def list_policies():
        return {"schema_version": "2.0", "policies": policies()}

    @app.get("/api/v2/replays")
    def list_replays():
        path = REPLAYS / "index.json"
        return json.loads(path.read_text()) if path.exists() else {"schema_version": "2.0", "replays": []}

    @app.post("/api/v2/jobs", response_model=JobResponse, status_code=202)
    def submit(request: SimulationRequest):
        if not health()["live_available"]:
            raise HTTPException(503, {"code": "live_unavailable", "message": "Live simulation is disabled. Browse verified replays, or install Basilisk and set APSIS_LIVE=1."})
        if request.scenario_id not in {s["id"] for s in scenarios()}:
            raise HTTPException(404, {"code": "unknown_scenario", "message": "Source event is not in the geometry catalog"})
        if request.policy_id in ("v1", "v2") and not model_path(request.policy_id).exists():
            raise HTTPException(409, {"code": "model_unavailable", "message": "Checkpoint is not installed; inspect /api/v2/policies"})
        try:
            return app.state.jobs.submit(request)
        except OverflowError as error:
            raise HTTPException(429, {"code": "capacity", "message": str(error)}, headers={"Retry-After": "5"}) from error

    @app.get("/api/v2/jobs/{identifier}", response_model=JobResponse)
    def get_job(identifier: str):
        job_id(identifier)
        return app.state.jobs.get(identifier)

    @app.delete("/api/v2/jobs/{identifier}", response_model=JobResponse)
    def cancel_job(identifier: str):
        job_id(identifier)
        return app.state.jobs.cancel(identifier)

    @app.get("/api/v2/jobs/{identifier}/artifact")
    def artifact(identifier: str):
        job_id(identifier)
        if app.state.jobs.get(identifier).status != "completed":
            raise HTTPException(409, {"code": "not_ready", "message": "Artifact has not completed"})
        return FileResponse(app.state.jobs.directory / identifier / "artifact.json", media_type="application/json")

    if REPLAYS.exists():
        app.mount("/replays", StaticFiles(directory=REPLAYS), name="replays")
    distribution = ROOT / "web/frontend/dist"
    if distribution.exists():
        app.mount("/", StaticFiles(directory=distribution, html=True), name="frontend")
    return app


app = create_app()
