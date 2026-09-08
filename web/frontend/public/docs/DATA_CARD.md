# Data card

The laboratory uses ESA Kelvins CDM-derived encounter geometry. It does not
reconstruct historical spacecraft orbits, recover historical operator actions,
or observe counterfactual outcomes from real conjunctions.

## Attribution and licenses

Source: [ESA Collision Avoidance Challenge dataset, Zenodo](https://doi.org/10.5281/zenodo.4463683).
Credit T. Uriot, D. Izzo, L. F. Simoes, R. Abay, N. Einecke, S. Rebhan,
J. Martinez-Heras, F. Letizia, J. Siminski, K. Merz; ESA's Advanced Concepts
Team and Space Debris Office; and the US Space Surveillance Network as the
underlying source. Data and derived data artifacts retain **CC BY 4.0**
attribution requirements. The repository's **MIT code license does not replace
the data license**. See [original source record](../data/kelvins_cdm/SOURCE.md).

## What the artifacts contain

| Artifact | Meaning | Limitation |
| --- | --- | --- |
| `geometry_events.csv` | 8,672 final-CDM geometry tuples with source `event_id`, principal covariance axes, miss/covariance alignment, RCS radius proxy and native/ESA Pc | Filtered subset; geometry embedded in a generated reference orbit |
| `covariance_evolution_events.csv` | Same-event first/final principal standard deviations | Intermediate updates are log-linear interpolation, not historical covariance measurements |
| `schedules_by_event.json` | 13,154 keyed source schedules from the downloaded training CSV | Negative timestamps and short gaps are cleaned at simulation time |
| `schedule_library.json` | Original anonymous bootstrap schedule library | Legacy only: cannot prove event-family separation |

V2 joins geometry, timing and covariance evolution on the same `event_id`.
When no evolution pair exists, it retains that event's final covariance throughout
the episode. It never borrows an unrelated event's covariance as a fallback.
V1's independently sampled schedule/evolution behavior remains available for
backward compatibility and must not be described as source-aligned stage 3.

Rebuild keyed schedules with `python scripts/build_keyed_schedules.py` after
downloading the source data. The generated file records the input CSV hash.

| File | SHA-256 |
| --- | --- |
| Source `train_data.csv` | `ba47ce80580d5d6ff523ddc1d724901dbdfb3a5afdc5e755f0ca2bcefe6e4eb6` |
| `geometry_events.csv` | `bf297cdb06a8a450be53c49de6541482a08877c0de0c98a635fba65200464ac8` |
| `schedules_by_event.json` | `65715ff07dc5eb3bc8c214bbed8b7e2321951ddb6ea3a8b1f47d639a45e1760b` |

## Generated information and scientific assumptions

- Posterior-resampled variants draw miss vectors from a historical event's
  Gaussian covariance. They are generated, measurement-grounded scenarios,
  **not additional historical events**. Their calibration depends on the
  Gaussian and reported-covariance assumptions.
- The v2 sensor draws IID Gaussian errors in the predicted TCA encounter plane.
  Its direction of relative velocity and ego orbit are assumed known. This is
  a controlled partially observed problem, not realistic orbit determination:
  temporally correlated tracking errors and velocity uncertainty remain absent.
- Simulator truth is separate from observations. J2 propagation supplies
  predicted encounter estimates; Basilisk propagates the physical episode.
  Terminal Pc remains a modeled probability with covariance, not an observed
  collision frequency or a claim of zero physical risk.
- Measurement covariance does not account for the J2/full-Basilisk propagation
  discrepancy. A measured source-8767 no-action audit (seed 100, 0.2-day window)
  predicted an initial J2 miss norm of about 305.13 m while Basilisk reached
  about 37.80 m at terminal time. A low predicted Pc therefore need not mean
  the modeled physical encounter is safe. The optional, separately versioned
  Basilisk forecast mode now avoids that discrepancy in a verified source-8767
  simulator consistency check; existing J2-trained checkpoints retain their
  original observation assumptions. Full dynamics does not remove the Gaussian,
  IID-error or known-relative-velocity assumptions.
- Combined hard-body radius is `sqrt(RCS_target/pi) + sqrt(RCS_secondary/pi)`.
  Radar cross section is not geometric area ground truth. Radius multipliers
  are sensitivity analyses, not corrections known to be physically accurate.
- A source row's `native_pc` uses its original radius proxy. Changing a run's
  radius does not retrospectively change the historical source value.
- The generator rejects or redraws geometries outside its bound-LEO orbit
  constraints. Uniform accepted simulation draws therefore differ from the
  unfiltered ESA distribution. Explicitly requested infeasible event IDs fail
  instead of being silently replaced.

## Splits and claims

`lab.data.grouped_split` assigns unique source IDs to deterministic train,
validation and test partitions before any augmentation or elevated-risk pool
construction. All generated variants inherit their source family. Hash seed 42
does not guarantee dangerous events in every partition; no-dangerous-event
evaluation cannot establish mitigation skill. Predeclared leave-one-event-out
runs provide a controlled route for scarce dangerous families. Repeated seeds
and variants do not increase the number of independent historical events.

Report population sampling, elevated-risk curricula, deliberately selected
challenge events, bounded lead windows and training-family diagnostics as
different distributions. Baseline comparisons must disclose information access:
the v1 checkpoint receives privileged truth-derived observations while v2 and
the new online planner can receive noisy generated estimates. Neither reward
improvement nor small feature sensitivity establishes successful risk gating.
