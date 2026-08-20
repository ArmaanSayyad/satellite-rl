# 05 — Datasets

Verified Aug 2026, live sources. Legal/redistribution terms matter for an
open-source repo, so they're called out explicitly per source — **do not
bundle anything below marked "do not redistribute."**

## 1. ESA Kelvins Collision Avoidance Challenge dataset — primary dataset

- Canonical source: **Zenodo, DOI `10.5281/zenodo.4463683`**
  ("Collision Avoidance Challenge - Dataset.zip", 221.1 MB,
  MD5 `d19dc8875229f2f6893253c38adddc87`). Also mirrored on Kaggle
  (`kaggle.com/datasets/shadmanrohan/collisionavoidancechallenge`) and
  described on `kelvins.esa.int/collision-avoidance-challenge/data/`.
- **License: CC-BY-4.0 (confirmed on the Zenodo record).** Freely
  redistributable/usable in this open-source repo, with attribution to the
  authors (Uriot, Izzo, Martinez-Heras, Letizia, Siminski, Merz), ESA's
  Advanced Concepts Team / Space Debris Office, and the US Space
  Surveillance Network as data source. **This is our one dataset that's
  unambiguously safe to commit to the repo (or a release asset) directly.**
- Content: real, anonymized Conjunction Data Messages (CDMs) ESA received
  2015–2019. Each row = one CDM; rows group by `event_id` into a time
  series (multiple CDMs as an event's prediction refines toward TCA), the
  final row's `risk` field being the eventual best risk estimate.
- Scale: train 162,634 rows / 13,154 unique events (~12 CDMs/event); test
  24,484 rows / 2,167 unique events (~11 CDMs/event). 103 columns, CSV.
- Key columns: `event_id, time_to_tca, mission_id, risk, max_risk_estimate,
  max_risk_scaling, miss_distance, relative_speed,
  relative_position_{n,r,t}, relative_velocity_{n,r,t}, c_object_type,
  geocentric_latitude, azimuth, elevation, F10, AP, F3M, SSN`, plus 39
  base fields duplicated with `c_`/`t_` (chaser/target) prefixes covering
  the RTN covariance matrix (`sigma_r/t/n/rdot/tdot/ndot`, cross-
  correlation terms `cn_r, cn_t, cndot_*`, ...), `position_covariance_det`,
  `cd_area_over_mass`, `cr_area_over_mass`, orbital elements (`h_apo,
  h_per, ecc, j2k_inc, j2k_sma`), and OD-quality metrics (`sedr, span,
  rcs_estimate, actual_od_span, obs_available, obs_used,
  recommended_od_span, residuals_accepted, time_lastob_start/end,
  weighted_rms`).
- **Includes a precomputed `risk` (Pc) value per row** — we don't need to
  compute Pc to use this dataset for grounding scenario statistics, but we
  do need our own Pc implementation (`04-collision-probability.md`) to
  validate against it and to score novel/simulated conjunctions the RL
  environment generates.
- **Corrections after actually downloading and inspecting the file**
  (Phase 1, `14-pc-validation-results.md`): `risk` is **`log10(Pc)`,
  floored at `-30`**, not raw probability. The `{t,c}_` covariance columns
  are the **full** position covariance including off-diagonal
  cross-correlation terms (`{t,c}_ct_r`, `{t,c}_cn_r`, `{t,c}_cn_t`), not
  just the diagonal sigmas — matching the standard CCSDS CDM layout. There
  is **no direct hard-body-radius column** (confirming the concern below),
  but `{t,c}_rcs_estimate` (radar cross-section, m²) is present and usable
  as an approximate radius proxy via `r = sqrt(RCS/π)` — see
  `14-pc-validation-results.md` for how this was used and its limits. The
  extracted archive also contains `raw_data/raw_data_2015-2019.txt` (raw
  CDM text) and `test_data_private.csv` alongside the expected
  `train_data.csv`/`test_data.csv` — not investigated in depth yet.

**Use in this project:** (a) fits distributions for miss distance, relative
speed, and covariance magnitude, used to sample realistic synthetic
conjunction geometries for training episodes; (b) provides real historical
events (with known outcomes/risk) held out for v1's success-criterion #2 —
sanity-checking agent behavior against real cases; (c) our Pc implementation's
validation set.

## 2. CelesTrak — background object population (TLEs)

- Modern endpoint: `https://celestrak.org/NORAD/elements/gp.php?GROUP=<group>&FORMAT=<fmt>`
  (formats: `TLE`, `2LE`, `XML`, `KVN`, `JSON`, `JSON-PRETTY`, `CSV` — CSV
  is now the default). Queryable by `CATNR`, `INTDES`, `NAME`, `SPECIAL`.
- Relevant groups: `active` (all active satellites), and named debris-cloud
  groups `cosmos-1408-debris`, `fengyun-1c-debris`, `iridium-33-debris`,
  `cosmos-2251-debris` — real fragmentation events, useful for a realistic
  debris-field background population.
- Update cadence: GP data refreshes every 2 hours — **do not poll more
  often**; violations get IPs firewalled. Cache locally in `data/`.
- **Important, current caveat**: CelesTrak crossed 100,000 cataloged
  objects in 2026 — legacy 5-digit TLE format can't represent the newer
  6-digit NORAD IDs. **Use the CSV/JSON GP format, not legacy TLE**, for
  anything built now.
- Terms of use: no formal written license found; CelesTrak states a "long
  tradition of making data freely available." Its `usage-policy.php` is a
  fair-use/rate-limit policy, not a redistribution license. Data
  ultimately originates from Space-Track/US Space Force, and CelesTrak
  operates under continuing DoD authorization to redistribute basic SSA
  data. **Treat as: freely usable in practice with attribution, but don't
  bulk-mirror it into the repo — fetch at runtime/cache locally, gitignored.**

**Use in this project:** populate a realistic background object field
(non-maneuvering objects around our ego satellite) for scenario generation,
and as an optional realistic size/orbit-regime source when we're not
directly sampling from a Kelvins event.

## 3. Space-Track.org — TLEs/OMMs only, NOT bulk CDMs

- Free registration, must accept the User Agreement (one account per
  person/entity, no credential sharing).
- Default agreement clause is **restrictive**: "The User agrees not to
  transfer any data or technical information received from this website...
  to any other entity without prior express approval" (citing 10 USC
  2274(c)(2)).
- **Carve-out**: USSPACECOM grants blanket approval to redistribute "basic
  SSA data" — explicitly TLEs/OMMs, SATCAT, and decay/reentry data —
  "conditioned on appropriate citation." This is what makes CelesTrak-style
  TLE mirroring legal.
- **CDMs are NOT in that carve-out.** No blanket redistribution language
  was found for Conjunction Data Messages (they're operator-specific, tied
  to a Primary Representative per constellation). **Do not scrape and
  redistribute bulk Space-Track CDM pulls in this repo.** Use the already-
  cleared Kelvins/Zenodo CC-BY-4.0 CDM set for historical CDM data instead.
  If we ever want live CDMs, that's a per-user, own-credentials, runtime-
  only fetch — never committed to the repo.
- Recommend re-checking the live agreement text at
  `space-track.org/documentation#/user_agree` before any release that
  touches Space-Track data, in case terms changed.

## 4. `sgp4` (Python) — TLE propagation

- PyPI package `sgp4`, current version **2.27**. `pip install sgp4`.
- Minimal usage (from package docs):
  ```python
  from sgp4.api import Satrec
  satellite = Satrec.twoline2rv(line1, line2)
  jd, fr = 2458826.5, 0.8625
  e, r, v = satellite.sgp4(jd, fr)  # r: position km (TEME), v: velocity km/s
  ```
- Wraps the official Vallado C++ SGP4 (AIAA 2006-6753) with a pure-Python
  fallback; supports OMM/JSON element loading and a batch array API.

## 5. Other reference sources (secondary, use with caution)

- **ESA DISCOS/DISCOSweb** (`sdup.esoc.esa.int`) — catalogue of 40,000+
  trackable objects (physical properties, orbital history) — useful for
  realistic hard-body-radius/mass/size reference values. Free registration
  (Space Debris User Account) required; **redistribution terms for API
  output not confirmed** — do not bundle DISCOS data in the repo without
  re-checking terms; fetch at runtime only if needed.
- **NASA ODPO** (ORDEM, DAS) — debris-environment engineering software
  tools, not flat datasets; require a NASA software usage agreement.
  Useful conceptually for background-flux modeling, not as training data.
- **UCS Satellite Database** — ~7,560 operational satellites, 28 fields.
  "Free and unrestricted use," attribution requested. **Updates paused** —
  treat as a static snapshot if used at all, low priority for this project.

## `data/` directory layout (proposed)

```
data/
  kelvins_cdm/          # Zenodo CC-BY-4.0 dataset — safe to commit or
                         # ship as a release asset; document provenance
                         # in data/kelvins_cdm/SOURCE.md
  celestrak_cache/       # gitignored — fetched at runtime, 2h+ refresh
  scratch/               # gitignored — local experiment outputs
```

`data/kelvins_cdm/SOURCE.md` will record: exact Zenodo DOI, download date,
MD5 checksum, and the CC-BY-4.0 attribution text, so provenance is
auditable independent of this doc.
