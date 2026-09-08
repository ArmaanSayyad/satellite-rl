# Running Apsis

## Verified-replay mode

Requires Node 24.15+ (24.20 tested). From a fresh clone:

```sh
npm run dev
```

The root command installs the frontend lockfile and starts Vite. Open its printed
URL, normally http://localhost:5173. The committed examples and evaluation JSON load
without Python, checkpoint loading, ESA raw data or a physics service.

For the built UI and lightweight API together:

```sh
docker compose up --build
```

Open http://localhost:8000. The container runs as an unprivileged user. It includes
the built UI, source catalog and verified artifacts; `APSIS_LIVE=0` disables simulation.
If another service uses port 8000, change the host-side mapping in `compose.yaml`.
Stop this project with `docker compose stop`; unrelated containers are unaffected.
The image build and replay-only service were exercised locally on Docker Desktop.

To serve the built frontend alone, `npm run build`, then deploy `web/frontend/dist`
to any static host. Replay URLs use query parameters, so no SPA path rewrite is needed.
The Earth/Three.js view is loaded separately; charts and evidence do not wait for it.

## Full Basilisk installation

Use Python 3.11 and the [Basilisk installation documentation](https://avslab.github.io/basilisk/).
Platform-specific support files may download on first use.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install 'bsk[all]'
pip install -e '.[dev]' -r web/backend/requirements.txt
python -m scripts.download_checkpoint
APSIS_LIVE=1 uvicorn web.backend.main:app --host 127.0.0.1 --port 8100
```

In a second terminal, run the UI against the service:

```sh
VITE_API_ORIGIN=http://127.0.0.1:8100 npm --prefix web/frontend run dev
```

Do not mix an unrecorded model with a v2 observation/action schema. The bundled
experimental checkpoint has its matching manifest under `assets/models/v2-imitation-hf`.
The manifest controls history length, forecast fidelity and categorical action decoding.
The published v1 archive is an SB3 checkpoint itself: **do not extract its contents**.

Configuration:

| Variable | Default / purpose |
|---|---|
| `APSIS_LIVE` | `0`; set `1` only on installations with the physics stack |
| `APSIS_WORKERS` | `2`; maximum simultaneous isolated jobs |
| `APSIS_CACHE_DIR` | `.cache/apsis`; persisted artifacts/progress/error logs |
| `APSIS_V1_MODEL` | `runs/ppo_stage2_riskaware_run1.zip` |
| `APSIS_V2_MODEL` | `assets/models/v2-imitation-hf/model.zip` |
| `APSIS_API_ORIGINS` | comma-separated localhost Vite origins |
| `VITE_API_ORIGIN` | optional frontend API origin; set when building for another host |

Use one Uvicorn process for the built-in job manager. Multiple Uvicorn workers would
have separate capacity/cancellation state. This is a local/research service, not a
distributed scheduler. Public live compute needs an authenticated, rate-limited gateway
and external queue before exposure; the static replay deployment needs neither.

## API and artifacts

Open `/docs` on the service for the generated OpenAPI contract. Discovery endpoints:
`/api/v2/health`, `/api/v2/scenarios`, `/api/v2/policies`, `/api/v2/replays`.

```sh
curl -X POST http://127.0.0.1:8100/api/v2/jobs \
  -H 'Content-Type: application/json' \
  -d '{"scenario_id":"kelvins-8767","policy_id":"v2","seed":0}'
```

Poll `GET /api/v2/jobs/{id}`. Completion supplies `artifact_url`. `DELETE` cancels
an active isolated worker; completed artifacts are retained. A repeated deterministic
request can complete immediately with `cached=true`. A cached result is not a new run.
Capacity exhaustion returns 429 with `Retry-After`; disabled physics returns an
actionable 503. Unknown IDs and missing models are distinct from simulator failures.
Request bounds allow seeds, historical/posterior geometry and a 0.5–2 radius multiplier.
`schedule_mode=source` enables event-linked irregular warnings and evolving covariance;
`controlled` uses the shared short schedule for direct comparisons. An optional
`observation_seed` changes sensing without changing the generated orbit, and an optional
`max_lead_days` limits a source window. Full source runs can take minutes or longer.

Artifacts retain source ID, generation time, geometry/checkpoint hashes, forecast and
observation regimes, exact decision inputs/actions, dense spacecraft records and
independent terminal Foster calculations. A recorder verification is a consistency
check, not an independent validation of Basilisk's physical model. Whole-episode
policy comparisons do not identify the isolated causal effect of each burn.

## Reproduction and checks

```sh
python -m scripts.download_checkpoint
python scripts/download_kelvins.py
python scripts/build_keyed_schedules.py
python -m scripts.generate_replays --events 8767 --policies v1 never_maneuver planner v2
python -m pytest -q
ruff check src tests scripts web/backend
cd web/frontend
npm run lint
npm run test
npm run build
npx playwright install chromium
npm run test:e2e
```

The experiment protocol supplies exact training/resume/evaluation commands. Routine CI
tests math, grouped data, metrics, service contracts and browser interactions. Full
Basilisk CI is a manually selected workflow job because installation/ephemeris and
long-horizon tests are expensive. Local physics verification is documented in the
technical report. Cross-platform floating-point trajectories may differ; use tolerances
for cross-platform validation rather than expecting byte-identical JSON.
