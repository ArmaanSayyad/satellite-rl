# V2 laboratory architecture

The existing Basilisk/bsk_rl environment remains the execution and scoring
engine. A Gymnasium wrapper introduces a separately versioned observation
contract. This preserves the published v1 checkpoint's nine inputs while making
the v2 sensing assumptions explicit.

```text
Source event partition → same-event geometry / schedule / covariance
                      → targeted initial state → Basilisk episode
                                                  ├─ scoring truth
                                                  └─ J2 encounter prediction
                                                       → seeded CDM error
                                                       → normalized history
                                                       → policy → RTN impulse
```

## Environment interface

```python
from satellite_rl.lab.environment import make_v2_env

env = make_v2_env(
    event_ids=[170], history_length=3, observation_noise_scale=1.0,
    max_lead_days=0.2, sim_rate=5,
)
observation, info = env.reset(seed=42, options={"event_id": 170})
```

The factory defaults to source-aligned stage 3; omit `max_lead_days` for the full
source schedule. Bounded windows carry `schedule_kind=bounded_source_window`.
They discard earlier decisions and start covariance interpolation partway along
the original source span. If no positive CDM remains in the bounded window, an
explicit start-at-window-limit decision is generated. Such windows are curriculum
approximations, not literal historical CDM replays.

`latest_estimate` and `info["observation"]` expose the observed encounter packet:
`miss_plane_m`, `covariance_plane_m2`, `basis_eci`, `basis_rtn`, `estimated_pc`,
ego state, remaining fuel, time to TCA and time until the next update. The
separate `info["truth"]` diagnostic is never encoded into policy features.
The packet labels IID Gaussian encounter-plane noise and the known-relative-
velocity-direction assumption. Error draws are stable for a reset seed and CDM
index; reading the latest packet does not draw new noise.
Pass `options={"event_id": 170, "observation_seed": 24}` while retaining the
same reset seed to vary sensing noise without changing targeted geometry.
Without this override, changing the reset seed varies both geometry generation
and sensing, so that ensemble is not a pure observation-noise experiment.

`encounter-history-v2.0` contains 24 features per update: logarithmic risk,
covariance scales, covariance correlation, normalized miss components, radius,
lead/next-update time, fuel, update fraction, previous RTN impulse, encounter
basis relative to RTN, noise setting and validity. Three updates produce 72
inputs. Unavailable history is zero padded with its validity channel unset.
`FEATURE_NAMES` is the ordered source of truth. Noise-free operation is an
ablation, not a claim that real sensors observe simulator truth.

Explicit source resets preserve family identity and raise for IDs outside the
configured partition. Requested elevated-risk sources receive precision
correction when enabled, independently of the random elevated-risk lottery.
Terminal failures carry `status=failed`, `failure_reason`, and
`pc_final_valid=false`. Their legacy `pc_final=1` is a conservative failure
sentinel, **not a measured probability**; scientific aggregators must retain
failure counts and exclude the sentinel from physical Pc distributions.

## Timing and fidelity

Cleaned schedules have at least 60 seconds between decisions. V2 rounds source
timestamps to the configured Basilisk integration resolution before targeting,
recorded as `schedule_resolution_s`. This prevents accumulated drift-completion
overshoot; bsk_rl checks completion at integration knots. V2 accepts integration
steps up to 30 seconds so its minimum decision gap remains executable.

The raw Basilisk precision probe propagates through the enclosing integration
knot and uses cubic Hermite position/velocity interpolation to return the exact
requested final epoch. This addresses fractional real-CDM timestamps; it does
not assert continuous-time numerical exactness. On a 123.4567-second reference
arc, 5-second versus 1-second resolution differed by 0.00139005 m and
0.0000314753 m/s at the final epoch. The convergence test bounds these differences.
A separate source-170 bounded episode ended at exactly 1730.0 seconds, matching
its quantized nominal epoch.

J2 estimates and a planner's linearized response remain approximations. Replay
artifacts use live spacecraft message recorders and compare recorded positions
with environment decision endpoints. Terminal Foster/Chan comparisons are
additional probability-method evidence; a nominal-TCA score must not be called
an independently refined closest-approach score. Do not start a separate raw
Basilisk probe while another environment is live: SPICE kernel unloading can
invalidate the other simulation.

The source-8767 forecast audit found a substantial differential model bias:
initial J2 predicted miss norm approximately 305.13 m versus Basilisk terminal
37.80 m, despite terminal Chan/Foster agreement near `1.622e-4`. Precision
targeting corrects initial states to achieve the full-dynamics encounter;
reforecasting those states with J2 does not reproduce the full-dynamics mean.
The Gaussian sensor covariance does not cover that discrepancy in J2 mode.
The optional `forecast_fidelity="basilisk"` mode addresses it by generating
no-further-burn forecasts from current simulator state under the same full
dynamics. It advances the world epoch by actual elapsed simulator time and runs
an external Python subprocess per forecast, isolating SPICE and working inside
SB3's daemonic vector workers. Subprocess calls have a 120-second timeout and
fail explicitly rather than silently falling back. This mode costs more compute.

Its packet version is `encounter-history-v2.1-basilisk`; feature ordering and
dimensions remain unchanged. J2 is still the default for existing models.
Gaussian observation error is added after propagation using the same independent
noise RNG. This is a consistent synthetic sensor generator, not a claim that
real tracking knows latent spacecraft states or predicts with perfect fidelity.
The online maneuver planner's burn-response approximation remains J2 even when
its input warning comes from the full-dynamics sensor.

`pytest tests/test_v2_forecast.py -q -s` passed in 21.10 seconds. For source 8767,
geometry seed 100, observation seed 123 and a controlled 0.2/0.1/0-day schedule,
the initial forecast differed from the eventual no-action Basilisk relative
position by `2.58568e-8 m`; a new forecast at the intermediate epoch differed by
`1.86977e-7 m`. Terminal separation was 37.812590 m, Chan Pc `1.6219909e-4`,
at exactly 17280 seconds. The test also verifies the independent Gaussian noise
draw analytically. These are simulator consistency checks on one geometry;
they do not establish general forecast calibration or risk-conditioned control.

A bounded planner-response check is reproducible with
`python scripts/validate_planner_response.py`. On source 8767, seed 100,
0.2-day bounded window and noise-free full-dynamics sensing, baseline plus six
first-step impulses (±0.01 m/s in R/T/N) took 114.51 seconds. The J2 finite-
difference response anchored to the full-dynamics mean differed from actual
Basilisk encounter-plane miss vectors by at most 0.040144 m. Holding final
covariance equal for diagnostic comparison, maximum Pc disagreement was
`8.81629e-9` (0.04645% relative). Predictions using the online current covariance
classified all seven outcomes correctly relative to `1e-4` in this small check.
The final covariance was used only for diagnostic isolation, not supplied to
the online teacher. This supports local response accuracy for these small burns;
it does not validate other events, large maneuvers or multi-day horizons.

## Prepared targeting cache and process isolation

Each sampler retains at most 32 prepared initial-state snapshots in its process.
The key includes ego state, target miss/speed/alignment, TCA, precision mode and
settings, world epoch, dependency versions, targeting source-file hashes, and
pre-solve RNG state. Schedule effects enter through TCA; covariance and radius do
not affect the orbital targeting solve. Copy on storage and retrieval protects
cache entries from mutation. A hit restores the exact post-solve RNG state, so
subsequent draws are unchanged. Failed solves are not cached. This primarily
accelerates repeated deterministic evaluation and planning; it does not claim
large speedups for novel random training scenarios.

Measured locally for source 8767, seed 42, `max_lead_days=.02`, `sim_rate=5`,
precision enabled: three identical resets took **5.217445 s, 0.080287 s,
0.097810 s** (one miss, two hits; identical sample metadata). The first time
includes cold/JIT overhead. Reproduce by timing three calls to
`env.reset(seed=42, options={"event_id": 8767})` on one factory-created env.
These are local observations, not a portable throughput promise.

Training uses `SubprocVecEnv(start_method="spawn")` for multiple workers.
Construct each simulator inside its worker; avoid threaded simulation and
shared live Basilisk environments. Run manifests retain configuration,
partitions, source/data hashes, dependency versions, checkpoints and measured
throughput. Resume restores policy/optimizer counters while restarting
simulator episodes, not bit-identical mid-episode execution.

## Verification record

- `pytest tests/test_env.py tests/test_scenario_sampling.py tests/test_tca_refinement.py -q`:
  38 passed in 510.13 s; three Gym action-space scaling warnings.
- `pytest tests/test_v2_environment.py -q`: seven passed in 8.05 s after the
  cache change, including source joins, seeded sensor/truth separation,
  partition disjointness, exact epochs and RNG preservation.
- The added finer-timestep endpoint comparison passed separately in 5.45 s.
- Ruff passed for the environment, observation/data wrapper, refinement module,
  keyed-schedule generator and new tests.

These results validate the implementation paths tested, not policy effectiveness.
See [data assumptions and exact input hashes](DATA_CARD.md) and the experiment
protocol for the independent requirements for a risk-conditioned policy claim.
