# 25 — Augmentation, ESA-Anchored Pool, and Threshold Findings (Phase 7d)

Follow-up to `24-risk-stratified-sampling.md`'s finding: the elevated-risk
pool mechanism works, but oversampling a pool that's still 99.8%
below-threshold doesn't manufacture more above-threshold examples, and
only one real event ever exceeds `pc_threshold=1e-4`. This phase attacks
that scarcity from three angles the user asked for explicitly: find more
real data, generate more (grounded) synthetic variety from what exists,
and check whether the threshold itself is miscalibrated.

## Finding 1: no external dataset helps

Researched Space-Track.org (CDM access is restricted to the objects an
account owns, not a public archive), CelesTrak SOCRATES (a live feed
built from lower-precision public TLEs, not the operator covariance data
a real Pc needs), Kaggle/HuggingFace mirrors (identical copies of the
same Kelvins dataset), and commercial providers (LeoLabs, Slingshot,
Kayhan — paid services, no open research datasets). Also checked the one
real historical satellite collision (Iridium-33/Cosmos-2251, 2009): its
own pre-collision risk assessment would *also* have been below any
reasonable action threshold — not because the threshold was wrong, but
because only crude public TLEs were available for that specific
conjunction, with no real covariance to compute a trustworthy Pc from at
all. This is a genuine, informative negative result, not a failed
search: actionable-risk conjunctions are this rare in the real world,
confirmed from every angle checked.

## Finding 2: posterior-resampling augmentation works, and is physically grounded

Each real event's reported covariance IS a statement of "here is the
distribution of plausible true offsets consistent with this
measurement" — not noise we're inventing. Resampling a new offset from
`N((x0, z0), diag(sigma_x^2, sigma_z^2))` (the same Gaussian the event's
own covariance defines) produces a genuinely different, still
real-measurement-consistent encounter, not a synthetic one — directly
analogous to CV data augmentation (rotate/recolor one image into many),
done with orbital uncertainty instead of pixels.

Validated directly against the real data: starting from the ~22 real
events with `native_pc > 1e-6` (the "close enough to matter" tier),
resampling generated **over 5,000 distinct actionable (`Pc > 1e-4`)
synthetic variants** from 110,000 draws, spanning 8 distinct real source
events with genuine geometric diversity (implied miss distance ranging
1.3m–552m, not tiny perturbations around one point). Two weaker
alternative techniques were also tested for comparison: independent-
marginal (decorrelated) resampling found real risk at ~11x the rate of
uniform joint sampling but discards real cross-parameter correlation;
mixup/interpolation between real event pairs found risk at a similar
order of magnitude but has no direct physical interpretation the way
posterior-resampling does. Posterior-resampling was chosen as the
implemented technique for exactly that reason — it's not just effective,
it's the one with a clean physical justification.

**Implemented** in `scenario_sampling.py`'s `SecondaryScenarioSampler`:
`high_risk_augment` (default `True`, only takes effect when
`high_risk_fraction > 0`) perturbs the drawn pool row's miss vector via
this resampling before the targeting solve, instead of using the row's
exact real `(x0, z0)` every time.

## Finding 3: our own Pc pipeline under-counts risk relative to ESA's own assessment

Cross-checked our recomputed `native_pc` against ESA's own reported
`risk` column (already present in the raw Kelvins CDM data, `risk =
log10(Pc)`) for the same events. **13 real events exceed `Pc > 1e-4` by
ESA's own assessment — not 1.** Of those, 7 are present in our geometry
table (the other 6 lack the RCS data our combined-radius estimate
needs). For several of those 7, our recomputed `native_pc` is
dramatically lower than ESA's number — most strikingly, ESA's single
riskiest real event (event 1148, `Pc=0.0207`, a ~2% collision
probability) comes out at `native_pc=6.7e-5` in our own pipeline, a
**~300x underestimate**, and below our own `pc_threshold`. This is
consistent with (not a new problem beyond) `14-pc-validation-results.md`'s
already-documented ~3.5–3.9 decades of std deviation between our
RCS-based combined-radius approximation and ESA's real assessment — this
is that same known gap surfacing concretely at the specific events that
matter most for pool selection.

**Implemented**: `geometry_events.csv` gained an `esa_reported_pc`
column (straight from the real `risk` column, no recomputation), and the
elevated-risk pool is now ranked by `max(native_pc, esa_reported_pc)`
instead of `native_pc` alone — a safety-oriented choice: either signal
suggesting real risk is enough to include a row. This alone took the
pool's count of "known-actionable" member events from 1 to 7 (at the
default `high_risk_pool_fraction=0.05`, ranked by the new combined
score).

## Finding 4: the threshold itself is not miscalibrated

Verified directly (web research, not assumed from training data): NASA
CARA's actual real-world Pc maneuver threshold **is `1e-4`** — the exact
value this project already uses. There's a separate "enhanced risk"
threshold (~1.5 decades lower, `3.1e-6`) used for a different purpose
(confirming a post-maneuver Pc no longer contributes meaningfully to
cumulative lifetime collision risk), not a lower maneuver-decision
threshold. Combined with Finding 1's Iridium-Cosmos analysis (a data-
quality failure, not a threshold-calibration one), there's no real basis
for lowering `pc_threshold` to manufacture more "actionable" events —
doing so would just be moving the goalposts, not fixing anything.

Separately checked (in service of a later scoping question, `26-
precise-targeting.md`): does a more permissive threshold shift the
qualifying event set toward *safer* (larger) miss distances, which would
be convenient if precision at small miss distances turned out to stay
hard? No fundamental tension either way: high Pc is not confined to tiny
miss distances in this dataset (achievable via tight covariance at 500m+
too), and even at the current `1e-4` threshold, 5 of 7 qualifying events
already have miss distance > 200m. This ended up being moot once
`26-precise-targeting.md`'s root-cause fix landed, but is recorded here
since it was a real, checked finding along the way.

## Net result

The elevated-risk pool now draws from 7 known-actionable real events
(up from 1) via the ESA-anchored ranking, and augmentation gives each of
those (and other near-threshold events) genuine within-event diversity
rather than replaying the exact same handful of geometries. Whether this
diversity actually reaches the simulated training data with correct
risk values depended on a separate problem — the scenario generator's
targeting precision at small miss distances — which `26-precise-
targeting.md` covers.
