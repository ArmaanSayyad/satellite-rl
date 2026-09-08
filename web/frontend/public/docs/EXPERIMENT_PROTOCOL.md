# V2 experiment protocol

This protocol is specified before v2 compute. A runnable training pipeline is not
evidence that a reactive policy has been learned. Reward improvement is not a pass.

## Registered decision criteria

On a frozen, source-event-held-out test set, compare the same simulator scenarios,
observation seeds, and radius assumptions for every policy. Require all of:

- At least 90% mitigation of scenarios whose paired no-action final Pc exceeds
  `1e-4`, with the event-bootstrap 95% lower confidence bound at least 80%.
- At most 10% maneuvering on scenarios whose no-action final Pc is at most `1e-6`,
  with the 95% upper confidence bound at most 20%. Maneuvering means total fuel
  greater than `0.001 m/s`; also report actual fuel, including smaller burns.
- Lower mean fuel than v1 at matched successful mitigation, and no worse mitigation
  than the observation-based geometry planner. Fuel regret is against a *restricted*
  feasible planner, never a claimed global optimum.
- Measurable action dependence under matched risk/covariance interventions, followed
  by physical scenario counterfactuals showing useful behavior. Feature intervention
  alone demonstrates sensitivity, not correctness or causal understanding.
- Repeat for three training seeds and at least three observation seeds per event;
  report family counts and leave-one-dangerous-event-out results. With only a few
  dangerous source families, report uncertainty and do not declare broad success.

These are research acceptance criteria, not spacecraft flight qualification.
Risk reduction per fuel may be negative. Missing terminal Pc is a failure, never zero.

## Sampling and information boundaries

Split unique source `event_id` before sampling or augmentation. All variants of a
source inherit its partition. Record the exact lists and hashes. Uniform evaluation
and a separately labeled dangerous-event challenge set answer different questions;
do not aggregate the challenge mixture as real-population performance. A training
mixture explicitly oversamples the elevated-risk pool within the training partition.

The v2 observation wrapper normalizes a short history of noisy encounter estimates,
covariance and orientation, time, fuel, and previous actions. Simulator truth is used
for scoring, never substituted for the policy's noisy estimate. Source covariance
evolution and CDM schedule are linked to the selected source. The noise model and
any lead-time cap are modeling assumptions, not recovered historical telemetry.
Radius sweeps at 0.5, 1, and 2 times the RCS-derived proxy are sensitivity experiments.

## Algorithm and baselines

Use SB3 PPO with a three-update history as a reproducible first baseline. Maintained
[RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
exists, but recurrence adds state/reset complexity; compare history length 1 versus
3 before paying that complexity. The continuous action retains the simulator's
physical deadzone; no hard-coded risk gate is added to the learned policy. The optional
`discrete19` action model directly addresses Gaussian exploration almost never
selecting exact wait: its categorical support is wait and ±R/T/N at 0.01, 0.1, and
1 m/s. This restricts direction/magnitude resolution and must be labeled as an
action-space ablation, not an improvement demonstrated before evaluation. The exact
table is saved in each run manifest; inference uses `decode_action`.

The planner searches RTN directions, logarithmic magnitudes, and candidate decision
times using finite-difference J2 response matrices. It freezes the current covariance
and encounter plane for prediction. This is a locally linear *approximation*, not
a validated dynamics surrogate or global oracle. Every reported planner outcome
comes from execution in Basilisk. Its predicted Pc must be retained beside actual
final Pc. A future-timing optimum is a restricted hindsight reference only when
the entire candidate is explicitly re-simulated; it must not be labeled an online
controller that knew future measurements.
The optional evaluator policy `timed_planner` compares a burn now with a burn at
the *observed next update*, waits if that candidate is selected, and replans after
the next measurement. It freezes today's covariance for this approximation; it
does not read future measurements or claim an optimal waiting strategy.

Keep no maneuver, legacy threshold, v1, observation-based geometry threshold, and
the numerical planner in paired evaluation. The v1 checkpoint receives its original
truth observation and is explicitly marked as privileged-information legacy; it is
not an equal-information competitor to v2.

## Runs and ablations

The training CLI records arguments, git state, package versions, data SHA256,
partition membership, elapsed time, and checkpoint SHA256. It uses spawned
processes (never threads sharing Basilisk) and periodic checkpoints. Resume restores
PPO parameters/optimizer and counters, but restarts simulator episodes and RNG streams;
it is reproducible as a recorded continuation, not bitwise uninterrupted training.

Run ablations independently: history 1/3, observation noise 0/1, high-risk fraction
0/0.5, fuel weight 0.01/0.1, and disruption weight 0/0.05. Keep all other settings
and partitions fixed. Report completed episodes, actual high-risk exposure, wall
time and simulator steps/sec. A short bounded proof run checks the pipeline; it
does not meet the acceptance criteria or replace these experiments.

The evaluation artifact contains per-episode RTN actions and timing, initial
uncertainty/lead-time strata, terminal truth Pc, paired no-action risk and event-level
bootstrap intervals. Null means insufficient evidence. No confidence interval is
reported from a single source family. No inference is drawn from unexecuted runs.

## Commands and a split limitation discovered before evaluation

The seed-42 hash test partition contains no source with ESA-reported Pc over
`1e-4`; its riskiest source is event 7717. The split was not changed after starting
the proof run. A held-out test sweep therefore cannot establish dangerous-family
generalization. Report a separately labeled seen-family challenge for debugging,
and use predeclared leave-one-event-out runs for the scientific claim:

```bash
python -m satellite_rl.lab.train --steps 100000 --workers 4 --out runs/v2-full
python -m satellite_rl.lab.train --steps 100000 --workers 4 --holdout-event 1148 --out runs/v2-loeo-1148
python -m satellite_rl.lab.train --steps 100000 --workers 4 --resume runs/v2-full/model.zip --out runs/v2-continued
python -m satellite_rl.lab.evaluate --events 7717,12638 --checkpoint runs/v2-full/model.zip --policies never_maneuver,geometry_threshold,planner,v2 --seeds 3 --out runs/heldout.json
```

For a bounded pipeline check, explicitly add `--max-lead-days 0.2 --no-augment`
to training and matching evaluation. This cap retains irregular source updates
inside the final 0.2 days, but many sources have only one decision in that window;
such a proof cannot establish the benefit of history or full-horizon waiting.
`--seeds` varies geometry and sensing jointly by default; add
`--observation-seeds 3` to hold each generated geometry fixed while independently
varying the sensor-error seed. The saved matched-CDM sensitivity recomputes dependent
features consistently but does not execute a physical outcome and cannot establish
causal policy skill.

## Measured proof runs (2026-09-07)

These commands executed in the local `.venv` with Basilisk 2.11.1, bsk-rl 1.3.4,
SB3 2.9.0 and CPU-only PPO. The first manifest's Basilisk distribution-version
lookup returned `unavailable`; direct module inspection verified 2.11.1 afterward.

```bash
python -m satellite_rl.lab.train --steps 64 --workers 2 --rollout 16 --max-lead-days 0.2 --no-augment --checkpoint-every 32 --out runs/v2-proof
python -m satellite_rl.lab.evaluate --events 7717,12638 --checkpoint runs/v2-proof/model.zip --v1 runs/ppo_stage2_riskaware_run1.zip --policies never_maneuver,threshold,geometry_threshold,planner,v1,v2 --seeds 1 --max-lead-days 0.2 --out runs/v2-proof/heldout.json
python -m satellite_rl.lab.evaluate --events 8767,1148 --checkpoint runs/v2-proof/model.zip --v1 runs/ppo_stage2_riskaware_run1.zip --policies never_maneuver,threshold,geometry_threshold,planner,v1,v2 --seeds 1 --max-lead-days 0.2 --allow-training-events --out runs/v2-proof/challenge.json
python -m satellite_rl.lab.evaluate --events 7717 --checkpoint runs/v2-proof/model.zip --policies never_maneuver,v2 --seeds 1 --out runs/v2-proof/full-horizon.json
python -m satellite_rl.lab.train --steps 256 --workers 2 --rollout 32 --max-lead-days 0.2 --no-augment --risk-pool-fraction 0.001 --action-mode discrete19 --checkpoint-every 64 --out runs/v2-discrete-proof
python -m satellite_rl.lab.evaluate --events 7717,8767,1148 --checkpoint runs/v2-discrete-proof/model.zip --policies never_maneuver,v2 --seeds 1 --max-lead-days 0.2 --allow-training-events --out runs/v2-discrete-proof/challenge.json
```

The continuous run collected 64 steps in 138.67 seconds (0.462 steps/s). On two
held-out safe sources it maneuvered on both, averaging 0.11956 m/s versus legacy
v1's 0.50730 m/s. Lower fuel does **not** meet risk-gating criteria. On the seen-family
8767 challenge, no-action final Pc was `1.62190e-4`; v2 used 0.10285 m/s and
ended at `4.89944e-5`. Its first noisy warning Pc was only `9.84625e-8`:
the myopic observation-based baselines waited and failed to mitigate the later risk.
Packet inspection exposed a major forecast discrepancy: the initial J2 truth
prediction's encounter-plane miss norm was 305.1270 m, versus 37.7998 m at final
Basilisk TCA: a 267.3272 m discrepancy in miss norm. Independent final Foster Pc
was `1.62188095e-4` versus Chan `1.62190477e-4`, with local TCA offset `-0.0001013 s`.
Initial covariance sigmas were already near final values
(37/70 m versus 36.6/65.7 m). Therefore attributing this case solely to shrinking
covariance would be wrong. Forecast mismatch, sensor noise, and evolving uncertainty
are not isolated by this run. This is evidence of a limitation of the prediction and
baseline pipeline, not evidence that v2 understood covariance evolution.

An uncapped event-7717 evaluation exercised 14 real schedule decisions over 4.699
days; the paired coast/v2 evaluation took 176.51 seconds. The continuous policy
spent 1.33410 m/s on this safe case. This verifies full-horizon execution, not
beneficial history use.

The categorical run collected 256 steps in 534.75 seconds (0.479 steps/s).
It emitted the identical `[0, -0.1, 0] m/s` action on all three tested sources,
including safe 7717. Every tested matched miss/covariance/radius intervention
retained that action. It mitigated the one dangerous source but failed the
false-positive criterion. **Explicit wait support alone did not fix the insurance
burn in this short run.** On event 7717, independent terminal Foster Pc
`6.518998e-8` agreed with Chan `6.518957e-8`; local linear TCA offset was
`-0.39665 s`. A single example is not broad planner/propagator validation.

The first continuous artifacts predate complete per-decision packet and independent
Foster recording. The later categorical artifacts include those diagnostics.
Public evidence is generated by `python -m satellite_rl.lab.evaluate --export-runs
runs/v2-proof runs/v2-discrete-proof --out replays/evaluation.json` and explicitly
labels seen-family challenges. Neither run establishes a reactive policy.

A subsequent independent-noise check held generated geometry seed 100 fixed for
sources 7717 and 8767 while using sensing seeds 1000, 1001, and 1002. The categorical
policy retained its identical 0.1 m/s along-track burn in all six cases. Its 3/3
mitigations and 3/3 safe false-positive maneuvers represent only one risky and one
safe source family, so both rate confidence intervals remain null.

```bash
python -m satellite_rl.lab.evaluate --events 7717,8767 --checkpoint runs/v2-discrete-proof/model.zip --policies never_maneuver,v2 --seeds 1 --observation-seeds 3 --max-lead-days 0.2 --allow-training-events --out runs/v2-discrete-proof/noise-robustness.json
python -m satellite_rl.lab.evaluate --events 8767 --policies never_maneuver --seeds 1 --max-lead-days 0.2 --out runs/v2-discrete-proof/forecast-audit.json
```

The uncapped continuation completed 64 additional steps (320 cumulative) in
2507.75 seconds, only 0.02552 collected steps/s. Process isolation works, but this
measurement does not demonstrate practical full-horizon PPO throughput. Long
source targeting and propagation dominate; later runs record per-reset and
per-step profiles to separate those costs. The resulting checkpoint was not
evaluated before switching research focus to the forecast discrepancy.

```bash
python -m satellite_rl.lab.train --steps 64 --workers 2 --rollout 32 --no-augment --risk-pool-fraction 0.001 --action-mode discrete19 --checkpoint-every 64 --resume runs/v2-discrete-proof/model.zip --allow-curriculum-change --out runs/v2-full-history-proof
```

## Corrected forecast and imitation experiment

`--forecast-fidelity basilisk` selects the independently tested full-dynamics
forecast (`encounter-history-v2.1-basilisk`), including the advanced epoch after
each maneuver/update. It preserves the 24-feature layout and applies measurement
noise after forecasting. Old checkpoints remain explicitly J2-input checkpoints.
Evaluation labels a forecast-mode distribution shift; strict training resume
rejects an unrecorded mode change. This addresses the forecasting defect without
changing the validated terminal physics or claiming that observation noise vanished.

Optional `--imitation-samples N` requires this corrected forecast and `discrete19`.
By default it selects six elevated-risk and six uniformly selected background
**training** source families (`--imitation-sources` controls corpus diversity),
obtains full-dynamics snapshots, and draws generated noisy CDM estimates
within each source covariance. An approximate local J2 action-response matrix
anchored to the full-dynamics baseline labels the cheapest feasible categorical
action. Labels are approximate planner decisions, not executed simulator outcomes.

The corpus retains source IDs and a checksum. Cross-entropy initialization uses
150 epochs and recorded class weights to balance wait versus burn; validation
is explicitly within-source observation examples. The saved `teacher.zip` must
be evaluated separately from PPO's final `model.zip` so imitation benefits cannot
be attributed to RL. These snapshots have zero-padded initial history and do not
teach long-horizon waiting by themselves. Any improved gating remains a limited
distillation result until source-held-out, noisy, multi-update evaluation passes
the registered protocol.

The local teacher response was checked against seven real Basilisk outcomes
(coast and ±0.01 m/s R/T/N), source 8767, seed 100, 0.2-day window. In 114.51
seconds, maximum encounter-plane vector error was 0.04014 m; maximum Pc absolute
error with a shared final covariance was `8.8163e-9` (maximum relative error
0.04645%). Current-covariance predicted threshold classification matched actual
outcomes in all seven cases. Only the two along-track burns mitigated this example.
This validates local, small-impulse response for one event/seed/window; it does not
validate the teacher for arbitrary magnitudes, horizons, or source families.
Reproduce with `python scripts/validate_planner_response.py`; measured results are
in `runs/teacher-response-validation.json` and exported public evidence.

The fresh HF-only PPO proof collected 64 steps in 368.44 seconds. The imitation
run generated 2040 examples in 120.05 seconds: 1909 wait, 61 negative along-track,
and 70 positive along-track labels, with both burn magnitudes 0.01 m/s. Within-source
held-out action agreement was 98.7745%; this is a distillation-fit statistic, not
physical-policy performance or source-family generalization. The imitation plus
64-step PPO run took 607.33 seconds in total.

On the same fixed source geometry (seed 100), the teacher-only checkpoint and the
PPO checkpoint both waited on all three sensor repeats of held-out safe source
7717. On seen-family dangerous source 8767, both emitted +0.01 m/s along-track
when observed Pc was `1.3371e-4` and `1.0378e-4`, resulting in final Pc
`1.89815e-5`. Both waited when observed Pc was `1.9656e-5`, leaving final
truth Pc at `1.62190e-4`. Thus both mitigated 2/3 dangerous repeats and wasted
zero fuel on 3/3 safe repeats; mean fuel across six cases was 0.003333 m/s.

This is observed state-conditional behavior in a learned network, with no hand-coded
inference risk gate. Its source is supervised initialization: the short PPO phase
did not change these six decisions or improve their outcomes. It fails the
registered mitigation requirement, has only one risky and one safe source family,
and was trained on mostly one-decision capped episodes. It is a reactive prototype,
not an accepted risk-aware policy. Counterfactual feature probes also changed
some actions by 0.01 m/s; those probes alone are not physical performance evidence.

```bash
python -m satellite_rl.lab.train --steps 64 --workers 2 --rollout 16 --max-lead-days 0.2 --no-augment --risk-pool-fraction 0.001 --action-mode discrete19 --forecast-fidelity basilisk --checkpoint-every 32 --out runs/v2-hf-proof
python -m satellite_rl.lab.train --steps 64 --workers 2 --rollout 16 --max-lead-days 0.2 --no-augment --risk-pool-fraction 0.001 --action-mode discrete19 --forecast-fidelity basilisk --imitation-samples 2048 --checkpoint-every 32 --out runs/v2-imitation-hf-proof
python -m satellite_rl.lab.evaluate --events 7717,8767 --checkpoint runs/v2-imitation-hf-proof/teacher.zip --policies never_maneuver,v2 --seeds 1 --observation-seeds 3 --max-lead-days 0.2 --forecast-fidelity basilisk --allow-training-events --out runs/v2-imitation-hf-proof/teacher-noise-robustness.json
python -m satellite_rl.lab.evaluate --events 7717,8767 --checkpoint runs/v2-imitation-hf-proof/model.zip --policies never_maneuver,v2 --seeds 1 --observation-seeds 3 --max-lead-days 0.2 --forecast-fidelity basilisk --allow-training-events --out runs/v2-imitation-hf-proof/ppo-noise-robustness.json
```

A larger **unexecuted** continuation retains the explicit forecast/action schema:

```bash
python -m satellite_rl.lab.train --steps 10000 --workers 4 --rollout 16 --max-lead-days 0.2 --no-augment --risk-pool-fraction 0.001 --action-mode discrete19 --forecast-fidelity basilisk --resume runs/v2-imitation-hf-proof/model.zip --out runs/v2-imitation-continued
```

Removing the cap requires `--allow-curriculum-change`, restarts simulator episodes,
and may be very expensive. Predeclare leave-one-family-out experiments before
training (`--holdout-event 8767`), retain source splits across all variants, and
compare full-horizon waiting before making any general risk-aware-policy claim.

On the same six sensor-seeded cases, the fresh 64-step HF PPO control mitigated
0/3 dangerous repeats and maneuvered on all 3/3 safe repeats (mean fuel 0.505 m/s).
The online fine-grid planner also mitigated 0/3, using mean fuel 0.000667 m/s;
its point-estimate, threshold-boundary solutions under-maneuvered under noisy
observations. The distilled network's coarser 0.01 m/s actions provide more margin.
This comparison does not establish an RL advantage over planning or a robust
planner benchmark win.

A separate restricted hindsight grid executed 19 Basilisk outcomes for source
8767, seed 100, in 70.33 seconds: coast, six RTN directions at 0.001, 0.003,
and 0.01 m/s. No lower tested magnitude succeeded; ±0.01 m/s along-track did.
The minimum feasible **tested grid** fuel was 0.01 m/s. This is a privileged
single-burn reference with discrete magnitudes and six directions, not a global
continuous optimum. The capped case has only one decision time; the reusable
search supports all decision times but does not establish multi-time optimality
from this one case. The prototype matches this grid fuel on its two mitigated
risky repeats and fails on the third; spending less on that failure is not
superiority to the reference. Exported regret is conditioned on successful
mitigation and separately reports the two valid high-risk comparisons.

```bash
python -m satellite_rl.lab.evaluate --events 7717,8767 --checkpoint runs/v2-hf-proof/model.zip --policies never_maneuver,v2 --seeds 1 --observation-seeds 3 --max-lead-days 0.2 --forecast-fidelity basilisk --allow-training-events --out runs/v2-hf-proof/noise-robustness.json
python -m satellite_rl.lab.evaluate --events 7717,8767 --checkpoint runs/v2-imitation-hf-proof/model.zip --policies never_maneuver,planner,v2 --seeds 1 --observation-seeds 3 --max-lead-days 0.2 --forecast-fidelity basilisk --allow-training-events --out runs/v2-imitation-hf-proof/ppo-paired-baselines.json
python -m satellite_rl.lab.planner --event 8767 --max-lead-days 0.2 --max-evaluations 31 --out runs/v2-imitation-hf-proof/hindsight-reference.json
```
