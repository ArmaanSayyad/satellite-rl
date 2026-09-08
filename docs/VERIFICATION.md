# Local verification record

Measured on 2026-09-07 on macOS, Python 3.11 and Node 24.20. These are local
results, not a claim that the newly expanded remote CI has already run.

| Check | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | 156 passed, one opt-in live test skipped; 676.79 seconds |
| `APSIS_TEST_LIVE=1 .venv/bin/python -m pytest tests/test_web_backend_live.py -q` | One passed; 69.16 seconds |
| Final API/replay rejection regression, `pytest tests/test_web_backend.py -q` | 17 passed, including six additional invalid-score/zero-score cases |
| `.venv/bin/ruff check src tests scripts web/backend` | Passed |
| Frontend `npm run build` | TypeScript and production build passed |
| Frontend `npm run lint` / `npm run test` | Passed / eight tests passed |
| `PLAYWRIGHT_CHANNEL=chrome npm run test:e2e` | Desktop and mobile passed; two optional live cases skipped |
| `PLAYWRIGHT_CHANNEL=chrome APSIS_E2E_LIVE_ORIGIN=http://127.0.0.1:8100 npm run test:e2e -- --project=desktop` | Two passed, including a newly executed Basilisk job |
| `docker compose up -d --build` | Replay-only production image built and launched |

The opt-in API test exercises the packaged v2 checkpoint's actual 72-feature input,
isolated live workers, capacity rejection, completed-cache reuse and cancellation.
Browser tests cover deterministic replay selection, keyboard controls, downloads,
responsive layout without horizontal overflow, research navigation and live comparison
restoration. Desktop, mobile and research views were inspected as rendered screenshots.
The lazy-loaded Three.js bundle still produces a build size warning (~926 kB raw).
FastAPI's test client emits an upstream deprecation warning; Gymnasium recommends
normalizing the preserved legacy continuous action space.

## Physics and replay checks

All 17 shipped policy artifacts contain actual Basilisk recorder trajectories and a
recorder-to-environment endpoint check below 1e-5 m. The controlled event-8767 coast
artifact was independently regenerated with `scripts.generate_replays --verify`;
keyframes, decisions, dense recorder frames and results matched exactly. API tests
verify every indexed artifact's checksum and monotonically increasing timeline.
Failed or invalid terminal scores are rejected before publishing a verified replay.

The final independent re-execution used:

```sh
python -m scripts.generate_replays --events 8767 --policies never_maneuver --seed 0 --verify web/frontend/public/replays/kelvins-8767-never_maneuver-s0.json
```

The additional source-schedule integration run was:

```sh
python -m scripts.generate_replays --events 7717 --policies never_maneuver v2 --schedule-mode source --seed 100
```

Both policies coasted through the full 4.6993-day source window, with 14 decisions;
terminal Foster Pc was 6.562614e-7. This is one safe source family, not evidence
of dangerous-event mitigation or broad uncertainty robustness.

Training commands, wall times, measured negative findings and the resumable larger
training command are in the [experiment protocol](EXPERIMENT_PROTOCOL.md).
The [model card](MODEL_CARD.md) records the unmet acceptance criteria, teacher/PPO
equivalence and the dangerous controlled replay in which v2 never maneuvered.
