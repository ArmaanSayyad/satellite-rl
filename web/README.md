# Web demo

An interactive demo of the trained collision-avoidance policy: click
**Begin Simulation**, watch it fly one of the 7 real, historically
dangerous ESA close-approach events (the ones validated in
[docs/27](../docs/27-riskaware-training-results.md)), and see whether
it burns fuel to dodge.

Two pieces:

- `backend/` — FastAPI service that lists the 7 known real events
  (`GET /api/scenarios`) and runs one real episode on a chosen or
  random one (`POST /api/simulate?seed=...`), same physics/policy as
  the rest of the repo — no mocking. Each run also re-simulates the
  identical scenario with a never-maneuver baseline, for comparison.
- `frontend/` — React + Three.js UI: two 3D panels (full orbital
  context, and a close-up of the actual encounter geometry, since the
  two are 4-5 orders of magnitude apart in scale, each with real Earth
  imagery, progressive trajectory reveal, and a live separation line),
  a scenario picker, playback controls (play/pause/speed/seek/step),
  a live stats dashboard with an inline glossary, a decision timeline
  with per-decision explanations, and a trained-policy-vs-no-maneuver
  comparison.

## Prerequisite: a trained model checkpoint

The backend loads `runs/ppo_stage2_riskaware_run1.zip`. `runs/` is
gitignored (training artifacts aren't committed — see the repo's
`.gitignore`), so a fresh clone won't have it. Produce it yourself
with the exact settings from docs/27's training run:

```bash
python -c "
from satellite_rl.training.train_ppo import train
train(
    total_timesteps=8000,
    out_name='ppo_stage2_riskaware_run1',
    targeting_seed=0,
    high_risk_fraction=0.5,
    high_risk_pool_fraction=0.01,
    high_risk_augment=True,
    high_risk_precise_targeting=True,
)
"
```

This takes roughly 90 minutes. If you use a different `out_name` or
train a different checkpoint entirely, update `MODEL_PATH` in
`backend/simulation_runner.py` to match.

## Running it

**Backend** (from the repo root, with the project's normal `.venv`
active — it needs `satellite_rl` importable, plus `fastapi`/`uvicorn`):

```bash
pip install -r web/backend/requirements.txt
cd web/backend
uvicorn main:app --reload --port 8000
```

First request after startup takes ~30s (loading the policy and
probing which random seeds reproduce each of the 7 real events).
Once you see `Ready -- 7 real scenarios available: [...]` it's warm.

**Frontend** (separate terminal):

```bash
cd web/frontend
npm install
npm run dev
```

Then open **http://localhost:5173**. The frontend expects the backend
at `http://127.0.0.1:8000` (see `frontend/src/api.ts`); CORS is
pre-configured for Vite's default dev port.

Node 20.19+ or 22.12+ is recommended (Vite 8 warns below that); it
runs fine on 20.13 in practice.

## What each simulation shows

Every click of **Begin Simulation** picks one of the 7 real events at
random, runs the full episode through the actual trained PPO policy
and Basilisk-based scenario pipeline (nothing scripted or faked), then
the backend re-samples dense (~60s-resolution) trajectory segments
between each decision point using the same physics probe used
elsewhere in this project (`tca_refinement._fly_passive_pair`) so the
frontend has something smooth to animate — the decisions and outcome
themselves are exactly what the policy produced, not adjusted for
presentation.
