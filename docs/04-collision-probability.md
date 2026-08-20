# 04 — Collision Probability (Pc) Computation

Sources: NASA CARA technical reports (NTRS 20190011726, 20190028900),
Orekit source (`org.orekit.ssa.collision.shorttermencounter.probability.twod`),
`space-track.org/documents/How_the_JSpOC_Calculates_Probability_of_Collision.pdf`,
Uriot et al., "Spacecraft Collision Avoidance Challenge: design and results"
(arXiv:2008.03069). Verified Aug 2026.

## The standard 2D Pc problem

The operational method (NASA CARA and industry) reduces a 3D encounter to
2D under the **short-term encounter** assumption: near closest approach
(TCA), relative motion is linear, relative velocity is constant, and
position uncertainty is Gaussian and doesn't evolve meaningfully across the
encounter window. Under this assumption, all collision risk collapses onto
the **encounter/conjunction plane** — the plane through the relative
position vector at TCA, normal to the relative velocity vector.

**Required inputs:**
1. Relative position vector at TCA, projected into the encounter plane
   (the "miss vector" — offset components after the along-track relative-
   velocity axis is dropped).
2. **Combined position covariance** (primary + secondary — summed, since
   independent), rotated into the encounter plane and diagonalized into
   principal-axis standard deviations σx, σz.
3. **Combined hard-body radius (HBR)** = r_primary + r_secondary, treated
   as one equivalent circle/disk.

**Core computation**: a bivariate Gaussian integrated over a disk of radius
HBR (Foster & Estes 1992 polar form, R0/φ = miss-distance polar coords,
OBJ = HBR):

```
Pc = 1/(2π·σu·σw) · ∫₀^OBJ ∫₀^2π
     exp{ −½[ ((R0·sinφ − r·sinθ)/σu)²
             + ((R0·cosφ − r·cosθ)/σw)² ] } · r dθ dr
```

**This integral has no closed form** — it is always evaluated numerically
(2D quadrature). Foster's own implementation used fixed steps (θ step 0.5°,
r step OBJ/12); modern implementations use adaptive quadrature
(`scipy.integrate.dblquad` is sufficient for our purposes).

## Named methods (what they trade off)

| Method | Approach | Notes |
|---|---|---|
| **Foster (1992)** | numerical double integration over the disk | Original NASA/JSC formulation, historical reference/ground truth, ~10–20× slower than Alfano/Patera per NASA's own benchmark |
| **Chan (1997)** | fast series expansion using error functions | Cheapest computationally, industry-standard (e.g. AGI/STK), slightly less exact for edge cases (very large HBR vs. miss distance, degenerate covariance) |
| **Alfriend (1999)** / Akella–Alfriend (2000) | time integral over the whole encounter, not a single-plane snapshot | Better when the short-encounter assumption is borderline; "Alfriend max Pc" variant reports a conservative maximum-achievable Pc under covariance uncertainty |
| **Patera (2001/2005)** | 1D line/contour integral around the HBR circle boundary | Good accuracy, handles low-Pc and short/degenerate encounters well |
| **Alfano (2005)** | 1D pdf via erf() and exponential terms, adaptive step count | Fast and accurate |

No single method is universally "the standard" — Chan/Alfano/Patera
dominate for routine speed; Foster is the historical reference; Alfriend-
type reformulations are used when the short-encounter assumption needs
checking or a conservative bound is wanted.

## Orekit's implementation (confirmed)

Since v12.0, package `org.orekit.ssa.collision.shorttermencounter.probability.twod`
(author Vincent Cucchietti) implements: `Alfano2005`, `Alfriend1999`,
`Alfriend1999Max`, `Chan1997`, `Laas2015`, `Patera2005` — **not Foster**.
All implement `ShortTermEncounter2DPOCMethod`:

```java
ProbabilityOfCollision compute(Orbit primaryAtTCA, StateCovariance primaryCovariance,
                                Orbit secondaryAtTCA, StateCovariance secondaryCovariance,
                                double combinedRadius, double zeroThreshold);
```

Python access via `orekit_jpype` (dynamic reflection — new packages like
`ssa.collision` should be available automatically without a wrapper
update) is architecturally plausible but **not independently confirmed by
any documented example** in this research. Using it means taking on a full
JVM dependency inside a pip-installable RL environment — real friction for
an open-source project aimed at being easy to `pip install` and run.

## Pure-Python landscape

No maintained, dedicated pure-Python Pc package exists. `hapsira`
(poliastro's maintained fork) has no native Pc/conjunction module. NASA's
own `nasa/CARA_Analysis_Tools` on GitHub is primarily **MATLAB**, released
as an algorithm-building-block SDK, not a pip-installable library.

## Decision: implement Foster/Chan directly in Python, no Orekit dependency

**Reasoning:**
1. The full computation is ~100–150 lines: reduce two covariances to the
   encounter-plane 2×2, diagonalize to (σx, σz), get the miss-vector
   offset, numerically integrate the bivariate Gaussian over the HBR disk.
2. Avoids a JVM dependency in an environment we want to be trivially
   `pip install`-able and CI-testable.
3. This is exactly what Orekit's own `Chan1997`/`Alfano2005` classes do
   internally — we're not skipping rigor, just not taking on the JVM.
4. We can still **validate** our implementation against Orekit's reference
   classes (via `orekit_jpype`, as a dev/test-only dependency, not a
   runtime one) and against the Kelvins dataset's precomputed `risk` column
   (see `05-datasets.md`) — real historical Pc values to check our from-
   scratch code against ground truth.

**v1 implementation plan:**
- Implement Foster's numerical double-integral form first (simplest to get
  correct, easiest to explain/audit in an open-source repo — clarity over
  speed for v1).
- Add Chan's fast series-expansion form once Foster is validated, for use
  inside the training loop (Pc gets computed at every relevant environment
  step, so we want the cheap method there; Foster stays as the accuracy
  reference in tests).
- Write unit tests comparing our Pc against: (a) known textbook/paper
  worked examples, (b) a held-out sample of Kelvins dataset events using
  their given `risk` values, (c) — optionally, dev-only — Orekit's
  `Chan1997`/`Alfano2005` via `orekit_jpype` if installed.

## Where Pc feeds into the RL environment

Computed **outside** Basilisk/bsk_rl (pure Python, from relative position +
covariance state exposed by our custom conjunction dynamics — see
`02-bsk_rl-architecture.md` §9 and `03-scenario-design.md`), then injected
into the observation via `bsk_rl.obs.SatProperties(dict(fn=compute_pc, name="collision_prob"))`.
