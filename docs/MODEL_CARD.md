# Experimental v2 model card

**Status: state-conditional prototype; acceptance criteria not met.**

Checkpoint: `assets/models/v2-imitation-hf/model.zip`.
SHA-256: `300275751a842cf063f7fc43f3d1fbeec3b0cef7242efae8c54c3706975f8c7f`.
The adjacent manifest is part of the model interface, not optional documentation.

## Model and training

SB3 PPO categorical MLP with three 24-feature observation frames (72 scalars), an
explicit wait action and 18 signed RTN impulses. Forecasts use isolated full Basilisk
propagation at the current epoch. Noisy CDM means are generated in the encounter
plane; ego state and relative velocity direction are treated as known.

Planner imitation initializes the policy using 2,040 generated examples from 12
training source families. These labels come from a restricted numerical planner,
not historical operator actions. Within-source imitation agreement is not held-out
family generalization. A subsequent 64-step PPO proof runs in the actual environment.
The lead-time cap is 0.2 days; many episodes have only one decision. This does not
establish learned waiting across successive CDMs or benefits from history.

The local J2 burn response anchored to a Basilisk forecast was validated for baseline
and six ±0.01 m/s RTN actions on one source/seed: maximum encounter error 0.040144 m.
This limited check does not validate larger burns, other events or full lead times.

## Actual evaluation

Same underlying scenarios, three independent sensing seeds each:

| Test | Teacher only | Teacher + PPO |
|---|---:|---:|
| Dangerous source 8767, mitigated | 2 / 3 | 2 / 3 |
| Safe source 7717, maneuvered | 0 / 3 | 0 / 3 |
| Mean fuel over six episodes | 0.003333 m/s | 0.003333 m/s |

The policy uses 0.01 m/s on successful danger cases and waits on the safe examples.
Matched observation interventions change some actions. **There is no demonstrated
PPO improvement over imitation.** A fresh full-forecast PPO control failed all three
danger repeats and burned on all three safe repeats.

The two successful prototype burns match the minimum feasible tested magnitude in a
19-run Basilisk hindsight grid. That is a restricted reference, not a global optimum;
zero regret is reported only among successful cases, never hiding the missed case.

Only one dangerous and one safe family underlie the sensor repeats. The dangerous
family was not held out from training. Relevant family confidence intervals are null.
The observed 2/3 mitigation fails the protocol's 90% gate. The result is evidence of
limited state dependence, not broad reliability, family generalization or qualification.

An additional installed fixed-schedule replay (source 8767, geometry/sensing seed 0,
four decisions) records **no maneuver** and final Foster Pc `1.622616e-4`.
The prototype fails mitigation there. This out-of-protocol replay is retained,
not replaced by a selected successful example, and reinforces the limited claim.

## Use and limitations

Use for reproducible research and interactive exploration. Do not use for operational
maneuver decisions. Unknown sensing correlations, uncertain RCS radius, scarce dangerous
families, generated reference orbits, limited training and single-decision evaluation
restrict interpretation. Standard Gaussian covariance is a modeling assumption.

Older v2 continuous/categorical J2 proofs remain in the evidence because their constant
insurance burns and forecast mismatch motivated the changes. They are not silently
replaced by the latest model. V1 retains its original truth-based observation contract.
See [protocol](EXPERIMENT_PROTOCOL.md), [data card](DATA_CARD.md), and the application's
downloadable `evaluation.json` for exact commands, per-case observations and provenance.
