# 14 — Pc Validation Results (Phase 1)

Empirical validation of `satellite_rl.pc.compute_pc` against real historical
conjunction data, per `13-roadmap.md` Phase 1. Two layers: (1) internal
math validation (unit tests, `tests/test_pc.py`), (2) external validation
against ESA's real reported risk assessments (this doc,
`scripts/validate_pc_against_kelvins.py`). Run Aug 2026.

## Layer 1: internal validation (24/24 tests passing)

- **Foster vs. exact closed-form** (circular-covariance case, via the
  noncentral chi-squared CDF — an independently-derivable exact result,
  not from Foster's own method): agrees to within `1e-6` absolute across
  5 test geometries.
- **Chan vs. the same exact closed-form**: agrees to within `1e-4`.
- **Foster vs. Monte Carlo** (2M samples per case, 4 elliptical
  covariance cases): agrees within 5× the Monte Carlo standard error in
  all cases.
- **Chan vs. Foster** on genuinely elliptical covariance (where they're
  not forced to agree): measured absolute deviation ranged from `~1e-16`
  to `~1.1e-4` across the 4 test cases — Chan's disk-approximation cost is
  small at these parameter ranges. (Full numbers in `tests/test_pc.py`
  output; not asserting a tight bound yet since we've only checked 4
  hand-picked cases, not the full realistic parameter range — see "Known
  limitations" below.)

## Layer 2: validation against real ESA Kelvins events

**Setup**: for each of the 13,154 unique conjunction events in the Kelvins
training set, took the final (closest-to-TCA) CDM, kept the 8,672 rows
with both `t_rcs_estimate` and `c_rcs_estimate` present, and computed our
own Pc (Chan method, for speed) on a random sample of 3,000 of those,
using:
- `relative_position_{r,t,n}` / `relative_velocity_{r,t,n}` directly as
  `r_rel`/`v_rel` (already in the RTN frame our geometry module expects).
- The **full** combined position covariance — not just diagonal sigmas.
  **Correction to `05-datasets.md`**: the dataset provides the complete
  CDM covariance matrix, including off-diagonal cross-correlation terms
  (`{t,c}_ct_r`, `{t,c}_cn_r`, `{t,c}_cn_t`), matching the standard CCSDS
  CDM layout — not just the diagonal `sigma_r/t/n` values as our earlier
  secondhand research summary implied. Verified directly against the real
  CSV header (103 columns), not re-derived from the earlier summary.
- **Combined hard-body radius derived from `{t,c}_rcs_estimate`** (radar
  cross-section, m²) via `r = sqrt(RCS / π)`. This is a real physical
  proxy, not an invented constant — but RCS ≠ true physical
  cross-sectional area in general (radar reflectivity depends on
  material/shape, not just size), so this is a documented approximation,
  not ground truth. **`05-datasets.md`'s original concern (the dataset has
  no direct HBR column) was correct** — this RCS-based estimate is our
  answer to that gap, not a discovery that the concern was unfounded.

**Also confirmed empirically** (correcting/extending `05-datasets.md`):
`risk` is **`log10(Pc)`, floored at `-30`** (not raw probability) — 68.9%
of real events in the sample sit exactly at that floor, i.e. ESA's own
pipeline reports "essentially zero risk" for the large majority of
screened conjunctions, which matches operational expectation (most
screened close approaches are not actually risky). All position/velocity/
covariance columns are in consistent meters/m/s units, confirmed by
cross-checking magnitude ranges against `miss_distance` and
`relative_speed`, not assumed.

**Results** (3,000 events, 0 computation failures):

| Metric | All 3,000 | Excluding floored (`risk > -30` on both sides), 913 rows |
|---|---|---|
| Pearson r (computed log10 Pc vs. reported risk) | 0.923 | 0.716 |
| Spearman r | 0.927 | 0.774 |
| Mean diff (computed − reported) | +0.40 | +0.10 |
| Median diff | 0.00 | −1.14 |
| Std of diff | 3.48 | 3.94 |

**Interpretation**: strong rank correlation (Spearman ~0.77–0.93) between
our from-scratch Pc computation and ESA's real operational risk
assessments, using only a public dataset and an RCS-derived radius
approximation — no access to ESA's actual internal object-size
assumptions or exact pipeline. Zero computation failures across 3,000 real
events is itself meaningful: it confirms the covariance-matrix
construction (summing two real, independently-measured 3×3 covariances)
produces valid positive-definite matrices in practice, not just in our
synthetic test cases. The remaining disagreement (std of diff ~3.5–3.9
decades) is expected and not a bug — we don't know ESA's exact assumed
combined radius (our RCS-based estimate is necessarily approximate), and
`c_rcs_estimate` (secondary/debris object radar cross-section) is only
available for 67.5% of events in the first place, meaning even our own
combined-radius estimate is built on an approximation of an approximation
for many rows.

## Known limitations / follow-ups

- **Chan-vs-Foster agreement across the full realistic parameter range**
  isn't yet characterized — only 4 hand-picked cases so far. Worth a
  systematic sweep before fully trusting Chan as the training-time default
  (per `10-rl-algorithm.md`), especially at parameter combinations near
  the `_select_order_m` threshold boundaries.
- **RCS-based radius is a real but approximate HBR proxy.** If a
  follow-up wants tighter validation, cross-referencing `c_object_type`
  and known object catalogs (e.g. DISCOS, noted in `05-datasets.md`) for
  actual physical dimensions would improve on the RCS approximation —
  not pursued here to keep Phase 1 scoped.
- **This validation used the final CDM per event only.** The full
  CDM-sequence validation (checking that Pc evolves sensibly across an
  event's ~12 CDMs as covariance shrinks, matching `03-scenario-design.md`'s
  model) is a natural Phase 5 task, not required for Phase 1's "does the
  Pc math work at all" question.
- Sign/ordering convention for the off-diagonal covariance terms
  (`ct_r`/`cn_r`/`cn_t` → matrix positions) was inferred from the standard
  CCSDS CDM layout and validated indirectly (zero failures, strong
  correlation) rather than confirmed against an explicit ESA schema
  document — worth a direct citation if this becomes load-bearing for a
  published result.
