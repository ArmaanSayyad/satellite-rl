"""Two Basilisk-ground-truth corrections for the J2-based targeting
solver's (scenario/targeting.py) two distinct residual errors: WHEN true
closest approach occurs (`refine_tca`) and WHERE the encounter actually
lands in the miss-vector plane (`correct_targeting_geometry`, added Phase
7e -- see docs/26-precise-targeting.md). Both propagate both objects
under Basilisk's real (10th-degree spherical harmonics + SPICE) dynamics
from t0 -- the only way to see either error, since the Basilisk-vs-J2
divergence accumulates gradually over the entire trajectory, not just
near closest approach, and no analytic J2 correction can capture the
higher-order terms Basilisk includes and J2 doesn't (see docs/18-
scenario-generator-hardening.md part 2 for the original timing diagnosis,
docs/26 for the positional one).

Why this split matters, empirically (docs/26): for a short (~5hr)
lead time, `refine_tca`'s timing offset is tiny (<0.02s, confirmed
directly) -- the WHEN error is negligible here. But the WHERE error (the
achieved miss vector at the correct time, vs. the targeted one) is
~100-200m regardless of target size, confirmed to swamp small (tens-of-
meters) intended miss distances even though it was invisible at the
km-scale miss distances every earlier phase validated against.
`correct_targeting_geometry` fixes that specific problem via a Newton/
fixed-point correction loop, converging to sub-meter accuracy in 1-3
Basilisk calls per docs/26's validation across 15+ test cases spanning
2-642m miss distances and 0.2-3 day lead times.

Benchmarked at ~1.5-3s wall-clock per Basilisk call for a realistic
3-day, two-satellite scenario at 5-10s sampling resolution (Phase 5b) --
faster for the short (~5hr) lead times `correct_targeting_geometry` is
actually used for (see env/scenario_sampling.py: gated to the elevated-
risk pool draws specifically, where small-miss-distance precision
actually matters -- not applied to every episode, to bound the
throughput cost). A one-time cost per scenario (paid at environment/
scenario construction), not per RL training step.

API/setup verified against Phase 4's working `bsk_rl`-based environment
and Phase 3's raw-Basilisk debugging (gravity/SPICE/GravBodyVector setup
-- see docs/17-env-implementation-notes.md); this module uses raw
Basilisk scripting (not bsk_rl's Gym wrapper) specifically to get direct
access to a full fine-grained state recorder, which isn't straightforward
to extract through bsk_rl's step()-based interface.
"""

from dataclasses import replace

import numpy as np
from Basilisk.simulation import spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

from ..pc.geometry import encounter_plane_basis
from .targeting import TargetedScenario, propagate_state

UTC_INIT = "2018 SEP 29 21:00:00.000 (UTC)"


def _fly_passive_pair(
    r1_0: np.ndarray,
    v1_0: np.ndarray,
    r2_0: np.ndarray,
    v2_0: np.ndarray,
    duration_s: float,
    sim_rate_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate two passive spacecraft under full Basilisk dynamics
    (10th-degree spherical harmonics + SPICE Earth orientation + solar
    third-body gravity, matching bsk_rl's own `WorldModel.setup_gravity_
    bodies` exactly -- see docs/26-precise-targeting.md's Phase 7e
    correction: this function originally omitted the Sun as a gravity
    body, which its own docstring claimed (wrongly) not to matter. That
    omission was invisible to `refine_tca`'s original bracket-and-
    interpolate TIMING use (tolerant of small dynamics-model
    imperfections) but silently broke `correct_targeting_geometry`'s
    POSITION precision: it was correcting against a Sun-less "ground
    truth" that didn't match what the actual RL environment (which DOES
    include the Sun, via bsk_rl) would simulate, so the achieved geometry
    still missed the target by a large margin despite this function
    reporting near-zero error against its own (wrong) reference. Fixed by
    matching bsk_rl's gravity-body setup exactly, including the ephemeris
    data path and `zeroBase` casing, not just the same gravity degree.
    """
    sim_task_name = "simTask"
    sim_process_name = "simProcess"

    scSim = SimulationBaseClass.SimBaseClass()
    dyn_process = scSim.CreateNewProcess(sim_process_name)
    dyn_process.addTask(scSim.CreateNewTask(sim_task_name, macros.sec2nano(sim_rate_s)))

    sc1 = spacecraft.Spacecraft()
    sc1.ModelTag = "probe1"
    sc2 = spacecraft.Spacecraft()
    sc2.ModelTag = "probe2"
    sc1.syncDynamicsIntegration(sc2)
    scSim.AddModelToTask(sim_task_name, sc1)
    scSim.AddModelToTask(sim_task_name, sc2)

    grav_factory = simIncludeGravBody.gravBodyFactory()
    grav_factory.createSun()
    planet = grav_factory.createEarth()
    planet.isCentralBody = True
    path_grav_data = str(get_path(DataFile.LocalGravData.GGM03S))
    path_ephem_data = str(get_path(DataFile.EphemerisData.de430).parent)
    planet.useSphericalHarmonicsGravityModel(path_grav_data, 10)

    grav_factory.createSpiceInterface(path_ephem_data, UTC_INIT, epochInMsg=True)
    grav_factory.spiceObject.zeroBase = "earth"
    scSim.AddModelToTask(sim_task_name, grav_factory.spiceObject, ModelPriority=100)

    sc1.gravField.gravBodies = spacecraft.GravBodyVector(list(grav_factory.gravBodies.values()))
    sc2.gravField.gravBodies = spacecraft.GravBodyVector(list(grav_factory.gravBodies.values()))

    sc1.hub.r_CN_NInit = list(r1_0)
    sc1.hub.v_CN_NInit = list(v1_0)
    sc2.hub.r_CN_NInit = list(r2_0)
    sc2.hub.v_CN_NInit = list(v2_0)

    rec1 = sc1.scStateOutMsg.recorder()
    rec2 = sc2.scStateOutMsg.recorder()
    scSim.AddModelToTask(sim_task_name, rec1)
    scSim.AddModelToTask(sim_task_name, rec2)

    scSim.InitializeSimulation()
    scSim.ConfigureStopTime(macros.sec2nano(duration_s))
    scSim.ExecuteSimulation()

    times_s = np.array(rec1.times()) * macros.NANO2SEC
    r1 = np.array(rec1.r_BN_N)
    r2 = np.array(rec2.r_BN_N)
    grav_factory.unloadSpiceKernels()
    return times_s, r1, r2


def refine_tca(
    ego_r0: np.ndarray,
    ego_v0: np.ndarray,
    sec_r0: np.ndarray,
    sec_v0: np.ndarray,
    nominal_tca_s: float,
    sim_rate_s: float = 5.0,
    margin_s: float | None = None,
) -> dict:
    """Find the true local-minimum-separation time near `nominal_tca_s`
    by propagating both objects under full Basilisk dynamics from t0.

    Args:
        ego_r0, ego_v0: ego's state at t0.
        sec_r0, sec_v0: secondary's state at t0 (from the targeting
            solver, e.g. `TargetedScenario.r_sec_t0`/`v_sec_t0`).
        nominal_tca_s: the targeting solver's assumed time-to-TCA.
        sim_rate_s: recorder/integration resolution -- 5-10s gives ample
            precision for typical residual timing offsets (tens of
            seconds, per docs/18) via the quadratic refinement below,
            while keeping wall-clock cost low.
        margin_s: how far past nominal_tca_s to keep propagating, to
            ensure the true minimum (which could be slightly before or
            after the nominal instant) is bracketed. Defaults to
            max(300s, 1% of nominal_tca_s).

    Returns:
        dict with `refined_tca_s`, `min_separation_m`,
        `nominal_separation_m` (separation at the closest recorded sample
        to the original nominal instant, for comparison), and
        `sample_resolution_s`.
    """
    if margin_s is None:
        margin_s = max(300.0, 0.01 * nominal_tca_s)
    duration_s = nominal_tca_s + margin_s

    times_s, r1, r2 = _fly_passive_pair(ego_r0, ego_v0, sec_r0, sec_v0, duration_s, sim_rate_s)
    separation = np.linalg.norm(r1 - r2, axis=1)

    i_min = int(np.argmin(separation))
    # Quadratic (parabolic) interpolation through the 3 points bracketing
    # the discrete minimum -- standard technique for sub-sample-resolution
    # precision without a second, finer simulation pass.
    if 0 < i_min < len(times_s) - 1:
        t0, t1 = times_s[i_min - 1], times_s[i_min]
        d0, d1, d2 = separation[i_min - 1], separation[i_min], separation[i_min + 1]
        denom = d0 - 2 * d1 + d2
        refined_t = t1 + 0.5 * (d0 - d2) / denom * (t1 - t0) if abs(denom) > 1e-9 else t1
    else:
        refined_t = float(times_s[i_min])

    nominal_idx = int(np.argmin(np.abs(times_s - nominal_tca_s)))
    return {
        "refined_tca_s": float(refined_t),
        "min_separation_m": float(separation[i_min]),
        "nominal_separation_m": float(separation[nominal_idx]),
        "sample_resolution_s": sim_rate_s,
    }


def correct_targeting_geometry(
    ego_r0: np.ndarray,
    ego_v0: np.ndarray,
    scenario: TargetedScenario,
    nominal_tca_s: float,
    max_iters: int = 8,
    tol_m: float = 1.0,
    sim_rate_s: float = 2.0,
) -> tuple[TargetedScenario, dict]:
    """Correct the J2-targeting solver's secondary initial state
    (`scenario.r_sec_t0`/`v_sec_t0`) so that propagating it under
    Basilisk's REAL dynamics actually lands at the intended encounter-
    plane miss vector (`scenario.r_sec_tca_target`/`v_sec_tca_target`),
    not just where the J2 solver assumed it would.

    Newton/fixed-point iteration: propagate the current candidate state
    under Basilisk, measure the achieved miss vector in the target's own
    encounter plane, and request a correction equal to the target minus
    the observed bias of the current request (not minus the raw error --
    see docs/26-precise-targeting.md for why using the WRONG reference on
    iteration 2+ caused this to oscillate rather than converge during
    development). Tracks the best (lowest-error) state seen across all
    iterations and returns that, so a run that doesn't fully converge
    within `max_iters` still returns its best attempt, not an arbitrary
    last one.

    Args:
        scenario: output of `targeting.solve_secondary_initial_state[
            _robust]` -- used as both the initial guess (`r_sec_t0`/
            `v_sec_t0`) and the target definition (`r_sec_tca_target`/
            `v_sec_tca_target`, `r_ego_tca`/`v_ego_tca` -- these define
            the target 2D miss vector and the encounter-plane basis, and
            are NOT themselves corrected; only `r_sec_t0`/`v_sec_t0`
            change).
        nominal_tca_s: time from t0 to the targeted TCA.
        max_iters: correction attempts. Convergence rate depends on lead
            time, not just miss distance: validated (docs/26) to need
            only 1-3 for the actual curriculum-stage-2 production regime
            (2-642m miss distances, 0.2-day lead time -- the only lead
            time `high_risk_precise_targeting` is actually used for, per
            env/scenario_sampling.py). Longer lead times converge more
            slowly (the Sun's third-body perturbation makes the achieved-
            vs-requested relationship less purely linear over a longer
            arc) -- a 3-day-lead 38m case needed 7 iterations to reach
            sub-meter accuracy (97m error at 4 iterations, 15m at 5, 2.3m
            at 6, 0.36m at 7 -- converging, not diverging, just slowly).
            Default of 8 covers both regimes with one setting.
        tol_m: stop once the achieved miss vector is within this many
            meters of the target.
        sim_rate_s: Basilisk recorder resolution -- 2s gives sufficient
            precision for the sub-second timing precision these short
            propagations need (see module docstring: WHEN-error is
            negligible here, so this doesn't need `refine_tca`'s
            coarser-then-quadratic-interpolated approach).

    Returns:
        (corrected_scenario, diagnostics) -- diagnostics has
        `n_basilisk_calls` and `final_error_m` (achieved-vs-target miss
        vector distance for the returned state).
    """
    v_rel_target = scenario.v_sec_tca_target - scenario.v_ego_tca
    basis = encounter_plane_basis(v_rel_target)
    target_2d = basis.T @ (scenario.r_sec_tca_target - scenario.r_ego_tca)

    r_sec_t0, v_sec_t0 = scenario.r_sec_t0, scenario.v_sec_t0
    current_request_2d = target_2d.copy()
    best_error_m = np.inf
    best_r_sec_t0, best_v_sec_t0 = r_sec_t0, v_sec_t0
    n_calls = 0

    for _ in range(max_iters):
        times_s, r1, r2 = _fly_passive_pair(ego_r0, ego_v0, r_sec_t0, v_sec_t0, nominal_tca_s, sim_rate_s)
        n_calls += 1
        idx = int(np.argmin(np.abs(times_s - nominal_tca_s)))
        achieved_2d = basis.T @ (r2[idx] - r1[idx])
        error_m = float(np.linalg.norm(achieved_2d - target_2d))
        if error_m < best_error_m:
            best_error_m = error_m
            best_r_sec_t0, best_v_sec_t0 = r_sec_t0, v_sec_t0
        if error_m < tol_m:
            break
        bias = achieved_2d - current_request_2d
        current_request_2d = target_2d - bias
        r_rel_corrected = basis @ current_request_2d
        r_sec_tca_corrected = scenario.r_ego_tca + r_rel_corrected
        v_sec_tca_corrected = scenario.v_ego_tca + v_rel_target
        r_sec_t0, v_sec_t0 = propagate_state(r_sec_tca_corrected, v_sec_tca_corrected, -nominal_tca_s)

    corrected = replace(scenario, r_sec_t0=best_r_sec_t0, v_sec_t0=best_v_sec_t0)
    return corrected, {"n_basilisk_calls": n_calls, "final_error_m": best_error_m}
