"""FastAPI backend for the collision-avoidance demo UI.

GET /api/scenarios lists the 7 known real dangerous events. POST
/api/simulate runs one full episode with the trained policy on one of
them (a specific ?seed=, or a random one if omitted) and returns the
real simulated trajectory, outcome, and a never-maneuver baseline run
on the identical scenario. See simulation_runner.py for what "real"
means here -- every number returned comes from an actual Basilisk
physics simulation, not a mock or a pre-recorded fixture.
"""

from contextlib import asynccontextmanager

import simulation_runner
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading the model + discovering the 7 real scenarios' seeds takes
    # real time (~30s, mostly Basilisk simulation calls) -- doing this
    # once at startup means the first click from the UI doesn't have to
    # eat that cost.
    print("Loading trained policy and discovering real scenarios...")
    simulation_runner._get_model()
    seeds = simulation_runner.discover_scenario_seeds()
    print(f"Ready -- {len(seeds)} real scenarios available: {sorted(seeds.keys())}")
    yield


app = FastAPI(title="Satellite Collision Avoidance Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Local dev only -- Vite's default port, plus 127.0.0.1 variants.
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/scenarios")
def scenarios():
    return simulation_runner.list_scenarios()


@app.post("/api/simulate")
def simulate(seed: int | None = None):
    try:
        return simulation_runner.run_simulation(seed=seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
