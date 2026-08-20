# 15 — Distribution Fitting Results (Phase 2)

Run via `python -m satellite_rl.scenario.distributions` against the full
real Kelvins dataset (8,672 final-CDM events with usable RCS data, out of
13,154 total events). Aug 2026.

## Headline finding: independent marginal lognormal fits are statistically
## poor — pivoted to joint bootstrap resampling as the primary method

`03-scenario-design.md` originally proposed fitting independent lognormal
distributions per parameter (miss distance, relative speed, encounter-plane
σx/σz, combined radius) and sampling each independently. Running this for
real produced a clear, honest negative result:

| Parameter | μ (fit) | σ (fit) | KS statistic | KS p-value |
|---|---|---|---|---|
| miss_distance | 9.261 | 1.332 | 0.097 | 5.2e-71 |
| relative_speed | 8.987 | 0.984 | 0.251 | ~0 |
| sigma_x | 4.522 | 3.438 | 0.208 | ~0 |
| sigma_z | 6.569 | 3.541 | 0.128 | 1.4e-123 |
| combined_radius | -0.012 | 0.536 | 0.205 | ~0 |

Every parameter's lognormal fit is rejected by the Kolmogorov-Smirnov test
at an overwhelming significance level. With n=8,672 the test has enormous
power, so *some* rejection at this sample size wouldn't be surprising even
for a good practical fit — but KS statistics of 0.10–0.25 mean the fitted
CDF is off from the empirical CDF by 10–25 percentage points at its worst
point, which is a real, visible, practically-relevant deviation, not just
statistical pedantry.

There's also a structural problem independent of goodness-of-fit:
**fitting each parameter marginally throws away real cross-parameter
correlation.** Miss distance, relative speed, and covariance size are not
independent in reality (e.g. faster relative speed conjunctions and
tighter-covariance conjunctions plausibly correlate with orbit regime and
tracking quality in ways a product-of-marginals sampler can't capture).

**Decision**: pivot to **joint bootstrap resampling** as the primary
method — `sample_scenario_geometry_bootstrap()` draws an entire real
event's `(miss_distance, relative_speed, sigma_x, sigma_z,
combined_radius)` tuple with replacement from the real 8,672-event table
(`data/fitted/geometry_events.csv`), rather than sampling each parameter
independently from a fitted distribution. This is simultaneously more
honest (no forced parametric assumption that the data itself rejects) and
more correct (preserves real joint structure). The independent-marginal
lognormal sampler (`sample_scenario_geometry()`) is kept in the codebase
as a documented fallback — useful if the full event table isn't available
for some reason — but is not the recommendation.

This is exactly the kind of finding `01-problem-scope.md` and
`11-evaluation.md` said to report honestly rather than paper over, applied
one level earlier in the pipeline than expected (a data-modeling choice,
not the eventual RL-vs-baseline result) — the same principle still holds.

## Covariance shrink ratio

Ratio of (first-CDM combined-sigma magnitude) to (final-CDM combined-sigma
magnitude), across 8,482 events with more than one CDM:

- Fitted lognormal: μ=2.124, σ=3.874 — **also KS-rejected** (p≈0,
  statistic 0.252), same story as above.
- **Median shrink ratio: 8.36×** — i.e., for a typical real event, the
  reported covariance magnitude at the first CDM is about 8x larger than
  at the final CDM. This number itself (regardless of the poor parametric
  fit) is a useful, real, grounded sanity check for `03-scenario-design.md`'s
  evolving-uncertainty model: covariance shrinking by roughly an order of
  magnitude over an event's CDM sequence is a real, substantial effect,
  not a minor refinement — validates that modeling this at all (rather
  than treating covariance as roughly constant across an episode) is
  worthwhile.
- Given the same marginal-fit problem as above, Phase 3/5's scenario
  generator should sample shrink behavior from real per-event first/last
  pairs (extend the bootstrap table to include both), not from this
  fitted lognormal, for the same reasons.

## Schedule library

5,000 real event CDM-timing schedules extracted (`time_to_tca` sequences,
descending, in days) for direct bootstrap resampling per
`09-episode-design.md` — no parametric fit attempted here at all, since
`03-scenario-design.md` already anticipated irregular real schedules
wouldn't fit a clean parametric model. Empirically: 1–22 CDMs per event in
this extraction, median 13 — consistent with the full-dataset statistic
found during Phase 1 (mean 12.36, `05-datasets.md`).

## What's saved (`data/fitted/`, committed — small, derived, not raw data)

- `geometry_distributions.json` — the (statistically weak, documented as
  such) marginal lognormal fits, kept for the fallback sampler.
- `geometry_events.csv` — the real 8,672-row per-event geometry table;
  **this is the file the recommended bootstrap sampler actually uses.**
- `covariance_shrink_ratio.json` — the marginal fit (same caveat).
- `schedule_library.json` — 5,000 real event CDM-timing schedules.

## Follow-ups (not blocking Phase 3)

- Extend `geometry_events.csv` to also carry each event's first-CDM
  geometry (not just final), so the shrink-ratio sampling can also be
  bootstrap-based, consistent with the rest of this pivot.
- Consider whether a mixture-of-lognormals or a different family (e.g.
  Weibull, or a copula to explicitly model cross-parameter correlation)
  would fit meaningfully better than a single lognormal, if a parametric
  model ever becomes necessary (e.g. for generating scenarios *beyond*
  the range covered by the real 8,672-event table). Not pursued now since
  bootstrap resampling sidesteps the need entirely for Phase 3's scope.
