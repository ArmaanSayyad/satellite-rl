"""Bounded isolated jobs. Each spawned process owns its simulator and SPICE state."""
import json
import multiprocessing as mp
import os
import signal
import threading
from pathlib import Path

from .catalog import ROOT, cache_key
from .contracts import JobResponse, ReplayArtifact, SimulationRequest


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))
    temporary.replace(path)


def worker(request: dict, directory: str) -> None:
    if os.name == "posix":
        os.setsid()  # include forecast subprocesses in this job's cancellation boundary
    folder = Path(directory)
    try:
        from .replay import generate_replay

        def progress(value: float):
            atomic_json(folder / "progress.json", {"progress": value})

        artifact = generate_replay(request, progress=progress)
        ReplayArtifact.model_validate(artifact)
        atomic_json(folder / "artifact.json", artifact)
    except Exception as error:  # noqa: BLE001 -- process boundary must publish structured failures
        import traceback
        (folder / "worker.log").write_text(traceback.format_exc())
        atomic_json(folder / "error.json", {"code": type(error).__name__, "message": str(error)[:600]})


class JobManager:
    def __init__(self, directory: Path | None = None, max_workers: int = 2):
        self.directory = directory or Path(os.getenv("APSIS_CACHE_DIR", str(ROOT / ".cache/apsis")))
        self.max_workers = max_workers
        self.processes: dict[str, mp.Process] = {}
        self.cancelled: set[str] = set()
        self.lock = threading.Lock()

    def submit(self, request: SimulationRequest) -> JobResponse:
        identifier = cache_key(request.model_dump())
        with self.lock:
            folder = self.directory / identifier
            if (folder / "artifact.json").exists():
                response = self.get(identifier)
                response.cached = response.status == "completed"
                return response
            if identifier in self.processes and self.processes[identifier].is_alive():
                return self.get(identifier)
            if sum(p.is_alive() for p in self.processes.values()) >= self.max_workers:
                raise OverflowError("All simulation workers are busy. Retry after an active job completes.")
            folder.mkdir(parents=True, exist_ok=True)
            for filename in ("error.json", "progress.json"):
                (folder / filename).unlink(missing_ok=True)
            atomic_json(folder / "request.json", request.model_dump())
            self.cancelled.discard(identifier)
            process = mp.get_context("spawn").Process(target=worker, args=(request.model_dump(), str(folder)))
            process.start()
            self.processes[identifier] = process
            return self.get(identifier)

    def get(self, identifier: str) -> JobResponse:
        folder = self.directory / identifier
        if identifier in self.cancelled:
            return JobResponse(id=identifier, status="cancelled", progress=0)
        if (folder / "artifact.json").exists():
            process = self.processes.get(identifier)
            if process is not None:
                if process.is_alive():
                    return JobResponse(id=identifier, status="running", progress=0.99)
                process.join(timeout=0)
            return JobResponse(id=identifier, status="completed", progress=1,
                               artifact_url=f"/api/v2/jobs/{identifier}/artifact")
        if not (folder / "request.json").exists():
            raise KeyError(identifier)
        if (folder / "error.json").exists():
            return JobResponse(id=identifier, status="failed", progress=0,
                               error=json.loads((folder / "error.json").read_text()))
        process = self.processes.get(identifier)
        if process is None or not process.is_alive():
            if process is not None:
                process.join(timeout=0)
            return JobResponse(id=identifier, status="failed", progress=0,
                               error={"code": "worker_stopped", "message": "Worker stopped before publishing an artifact; retry the request."})
        path = folder / "progress.json"
        progress = json.loads(path.read_text())["progress"] if path.exists() else 0
        return JobResponse(id=identifier, status="running", progress=progress)

    def cancel(self, identifier: str) -> JobResponse:
        with self.lock:
            current = self.get(identifier)
            if current.status == "completed":
                return current
            process = self.processes.get(identifier)
            if process is not None and process.is_alive():
                if os.name == "posix" and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                process.join(timeout=2)
            self.cancelled.add(identifier)
            return self.get(identifier)

    def close(self):
        for identifier in list(self.processes):
            if self.processes[identifier].is_alive():
                self.cancel(identifier)
