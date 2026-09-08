# Apsis technical report

This report describes v2 implementation and measured evidence. Historical intent is
recoverable with `git show e245d67^:docs/00-plan.md`; the original state/action/reward,
evaluation and roadmap documents live at that same revision. Plans are not results.

## 1. Problem statement

One ego satellite receives successive conjunction warnings and chooses a bounded
impulsive RTN maneuver, or waits. The objective balances terminal collision probability
against finite fuel and maneuver overhead. Irregular warnings and evolving uncertainty
make the timing decision consequential. Constellation scheduling, background debris
populations, attitude dynamics and flight operations are outside this implementation.

**Current scientific status: a limited state-conditional prototype, not an accepted risk-aware policy.** V2 is a
post-training and evaluation laboratory, with reproducible negative findings and an
interactive interface to its evidence. Passing unit tests or increasing reward does
not promote a checkpoint.

## 2. Data

The ESA Kelvins Collision Avoidance Challenge dataset (Zenodo
[10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683), CC-BY-4.0)
contains anonymized CDMs, not complete historical inertial trajectories. The committed
8,672-event geometry table preserves each usable source's miss distance, speed,
anisotropic encounter covariance, alignment, RCS-derived radius and reported risk.
V2 restores `event_id` throughout sampling and joins schedules and covariance evolution
to that same source. The event-linked schedule export contains 13,154 source schedules.

Previously, stage 3 independently sampled geometry, anonymous warning timing and
absolute first/final covariance from different events. That broke joint provenance
and prevented a geometry-only partition from isolating all source information.
V2 uses grouped ID partitions **before** risk-pool construction or augmentation.
Changing a seed alone is not an event-family holdout.

Posterior-resampled offsets are generated variants grounded in a historical covariance;
they are not additional observed historical conjunctions. Risk-tail oversampling is an
explicit changed training distribution. Uniform and challenge evaluations stay separate.
The RCS radius proxy is uncertain, not physical ground truth. Replay artifacts retain
Foster outcomes at 0.5×, 1× and 2× assumed radius.

See the [data card](docs/DATA_CARD.md) for hashes, filtering, attribution and assumptions.

## 3. System architecture

`pc/` retains the validated Chan approximation and Foster numerical reference.
`scenario/` targets real-event geometry in a generated LEO reference orbit using J2,
with Basilisk correction for precision-sensitive events. `env/` remains a Gymnasium
interface around bsk_rl/Basilisk. `lab/` adds event partitions, normalized observation
history, bounded maneuver planning, training manifests and paired evaluations.

Basilisk remains the execution and scoring dynamics: degree-10 Earth gravity, Sun
third-body perturbations and SPICE orientation/ephemerides. There is no learned
surrogate presented as final physics. The numerical planner's J2 finite-difference
response is an explicitly approximate prediction model.

Each vector worker uses its own spawned process. A bounded per-process prepared-target
cache preserves the sampler's post-solve RNG state as well as target state. Cache keys
include geometry, epoch, settings and source/dependency fingerprints. It accelerates
repeated identical resets, not arbitrary unseen training examples. One measured
8767 repeated-reset sequence was 5.217 / 0.080 / 0.098 seconds; cold overhead is included.

The service imports no physics at startup. Stable `kelvins-{event_id}` scenarios and
policy discovery are available independently of model installation. Live jobs run in
separate processes, publish progress and atomic JSON, support cancellation, and reuse
deterministic request/source/model cache keys. The default is replay-only.

## 4. Environment design

The legacy nine-scalar observation remains available for the published v1 checkpoint.
It contains raw Pc, fuel fraction, time and true relative position/velocity; the raw
time and small Pc have very different scales. V2 adds 24 normalized features per
warning, with a default three-warning stack (72 scalars): logarithmic Pc, covariance
scales/correlation, normalized estimated miss vector, radius, time/next update, fuel,
previous RTN action, encounter-basis orientation, sensing-noise scale and history mask.

Position-estimate error is sampled in the **predicted encounter plane**, where the
covariance is defined. It is not incorrectly added to a current orbital position using
a TCA covariance. The current noise process is IID between CDMs; velocity direction
and ego navigation are assumed known. This is a disclosed sensing approximation, not
a reconstructed orbit-determination posterior. Noise uses an independent reproducible
stream, while simulator truth remains separate for scoring.

Real source schedules are quantized to the integration resolution; near-duplicate
updates are merged and the final interval respects the minimum drift. An optional
lead-time cap truncates a source window, while covariance evolution keeps its original
source-time interpolation. Such capped proofs often have only one decision and cannot
demonstrate benefits from history.

Continuous actions retain v1 execution. An optional 19-action policy uses exact wait
plus positive/negative R, T and N impulses at 0.01, 0.1 and 1 m/s. This changes the
policy's action parameterization, not Basilisk's impulse model. Candidate restrictions
and their table are recorded in manifests.

Reward is fuel cost plus maneuver overhead and terminal risk penalty:
`Pc/threshold` below threshold, `1 + log(Pc/threshold)` above it. This function is
continuous and increasing but grows more slowly above threshold than its linear
extension; the old description saying “steeper above” was wrong. A terminal failure
has explicit invalid-score status; a conservative penalty sentinel is not reported as
a measured probability.

## 5. Training and evaluation

SB3 PPO is the reproducible baseline. Maintained
[RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
exists; history stacking provides a simpler comparison before recurrence is justified.
[Spawned vector workers](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
avoid sharing simulator state. Runs record seeds, exact source partitions, hashes,
arguments, source state, package versions, wall time, progress and checkpoint hashes.
Resume restores the model and optimizer, restarting simulator episodes; it does not
promise a bitwise-identical uninterrupted trajectory.

The [registered protocol](docs/EXPERIMENT_PROTOCOL.md) defines promotion gates before
compute: dangerous-case mitigation, safe-case false positives, successful-case fuel,
paired counterfactual behavior, multiple seeds and event-family uncertainty. Evaluation
retains failed episodes, per-decision observations/actions, strata, numerical planner
predictions, independent Foster terminal calculations and bounded local TCA diagnostics.
Confidence intervals are null when the relevant subset lacks independent families.

Baselines include coast, legacy radial threshold, geometry-aware threshold, numerical
direction/magnitude search at the current decision, published v1 and experimental v2.
The search helper supports candidate delays, but the evaluated online controllers use
zero delay; these results do not demonstrate optimized maneuver timing. The planner is a
restricted feasible search, **not a global oracle**. Negative fuel regret is possible.
The v1 baseline receives privileged truth-derived observations; comparisons disclose
that difference. Counterfactual feature interventions establish action sensitivity,
not inferred intent or correct decisions by themselves.

## 6. Corrections and negative findings

The historical project validated Pc against circular closed-form cases, Monte Carlo
and ESA risk rankings. Its reported 3,000-event correlation was 0.927 overall, with an
RCS approximation; this is a historical recorded run, not remeasured by the v2 change.
The original independent lognormal fits failed goodness-of-fit checks, motivating
joint bootstrap geometry. Isotropic covariance hid risk, so anisotropy and alignment
were restored. Only one retained source exceeds `1e-4` by the native radius/Pc model;
seven sources exceed that threshold when ranked using ESA's reported risk as well.
Those are different risk definitions, not seven independent native dangerous families.

Earlier targeting corrections addressed J2 precession, missing Sun gravity, unrealistic
orbits, a wrong simulation epoch, a projection error hiding 3D residuals, and bsk_rl
deep-copying the sampler. V2 preserves those fixes and corrects fractional endpoint
sampling and real-schedule integration timing. A 123.4567-second probe's 5-second
versus 1-second endpoint calculation differed by 1.39 mm in the measured test.

The v1 result was five dangerous **episodes** within a 30-episode, seven-source pool,
not seven independent dangerous events, and not a source-family-held-out study.
V1 averaged about 1.50 m/s versus 0.046 m/s for a restricted radial hindsight search.
Its near-constant small burn was an insurance strategy. Numerical Pc of zero must not
be described as physically certain safety or “fully neutralized” real-world risk.

V2's 64-step continuous proof and 256-step explicit-wait categorical pilot also fail
the risk-gating criterion. The latter repeats `[0, -0.1, 0] m/s` on the tested safe and
dangerous examples and matched interventions. An uncapped 4.699-day source-7717
evaluation exercised 14 decisions per policy; continuous v2 still spent 1.334 m/s
on a safe event. Detailed measured rows and settings are downloadable in the product.

**Forecast calibration is an additional blocker.** For one recorded source-8767
initial state, the J2 predicted separation was 305.127 m while Basilisk reached
37.800 m. The terminal local TCA offset was about −0.0001 s, ruling out a large timing
offset in that example. The precision-corrected initial state is accurate under full
dynamics; forecasting it with J2 reintroduces model bias. Measurement covariance does
not include that bias. Noise, covariance evolution and forecasting must be investigated
separately; attributing the failure to uncertainty contraction alone would be wrong.

An optional `forecast_fidelity=basilisk` path now addresses this boundary with an
isolated full-dynamics forecast at the advanced current epoch. On one source-8767
test, start and intermediate predictions agreed with terminal relative position
within 2.59e-8 and 1.87e-7 m. This is simulator consistency, not real-world forecasting
validation; sensing noise and uncertain covariance assumptions remain.

Planner imitation using 2,040 generated examples from 12 training families, followed
by 64 full-forecast PPO steps, produced a limited state-conditional network. It
mitigated 2/3 dangerous sensing repeats with 0.01 m/s and waited on 3/3 safe repeats.
The teacher-only checkpoint produced the same decisions: no PPO improvement was
demonstrated. Only one dangerous (seen in training) and one safe family support this
result. It fails the 90% mitigation gate and does not demonstrate multi-update timing.
The [model card](docs/MODEL_CARD.md) records the packaged checkpoint and limitations.

## 7. Measured product evidence

The committed fixed-schedule comparisons are real simulator recorder samples. For
event 8767/seed 0: coast final Foster Pc is 1.622616e-4; v1 uses 1.361372 m/s;
legacy threshold uses 10 m/s; the noisy geometry planner uses 0.020 m/s and reaches
2.393082e-6. This single replay does not override the planner's failed stage-3 examples.
Safe-under-native-model events 1148 and 3555 illustrate wasted insurance fuel.
The experimental v2 checkpoint coasts through all four warnings in the source-8767
seed-0 controlled replay and fails mitigation. Its 2/3 sensing-repeat result belongs
to the separately recorded capped stage-3 evaluation, not this fixed-schedule replay.

Every recorder-verified artifact compares spacecraft recorder positions against the
environment's positions at each decision endpoint. The initial coast replay was
independently re-executed and its keyframes, decisions, dense samples and results
matched exactly. The browser interpolates these samples for playback; interpolation
is not a new physics result. Burn timestamps now record the start of a coast, not its end.

## 8. Product and deployment

Apsis provides briefing, scenario/policy exploration and experimental evidence in one
interface. The encounter plane shows the covariance/error ellipse, miss vector and
hard-body radius at a distinct scale from the orbital view. Charts show logarithmic
Pc and uncertainty over warnings; decision cards expose recorded inputs and actions,
without inventing neural-policy reasoning. Static verified replays and new live jobs
use the same artifact envelope.

See [deployment](docs/DEPLOYMENT.md) for origins, model paths, static and full-stack
setup, and [architecture](docs/ARCHITECTURE.md) for boundaries and cache semantics.

## 9. Limitations and next experiments

The retained native-danger population is extremely small. Augmentation does not create
new historical families. Observation errors lack calibrated temporal correlation and
velocity uncertainty. The radius proxy is uncertain. Forecast model bias can dominate
small conjunction geometry. Nominal-TCA scoring and local linear diagnostics do not
establish a global minimum separation after arbitrary large burns. Short training
proofs cannot establish generalization, and process isolation alone does not solve
expensive multi-day targeting. No current checkpoint is promoted as risk-aware.

Run the registered history/no-history, sensing, curriculum and reward ablations only
with fixed provenance and explicit training budgets. Constellation scheduling remains
deferred until the single-satellite decision problem is convincingly solved.
