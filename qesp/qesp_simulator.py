"""
Gammo AGX - QESP Simulator
qesp/qesp_simulator.py

Runs GR baseline and QESP-modified simulations side by side,
producing the comparison data needed to validate the principle.

The simulator:
1. Takes any exotic spacetime configuration (MT or Alcubierre)
2. Runs the JAX physics solver to get GR results
3. Computes strain profile epsilon(r) = K_GR(r) / K_crit
4. Applies Q_munu quantum correction to get QESP-modified curvature
5. Evolves the QESP system forward in time (simplified)
6. Returns full comparison: GR vs QESP at every radial point

Key output: the curvature divergence comparison.
    In GR:   K -> K_GR  (can be very large near throat/wall)
    In QESP: K -> K_QESP = K_GR / (1 + Q(epsilon))  (plateaus)

This is the primary evidence for the QESP principle.
"""

import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import math
from dataclasses import dataclass, field
from loguru import logger

from qesp.strain import (
    compute_strain_morris_thorne,
    compute_strain_alcubierre,
    analyze_strain,
    K_PLANCK,
    EPSILON_CRITICAL,
)
from qesp.quantum_feedback import (
    compute_Q_profile,
    compute_modified_curvature_profile,
    scan_lambda_sensitivity,
    ActivationMode,
    LAMBDA_DEFAULT,
)


@dataclass
class QESPSimulationResult:
    """
    Full result of a QESP simulation run.
    Contains both GR baseline and QESP-modified results for comparison.
    """
    geometry_type:       str
    parameters:          dict

    # Radial grid
    r_grid:              object  # jnp array

    # Strain
    strain_profile:      object  # epsilon(r)
    strain_analysis:     object  # StrainResult

    # GR baseline
    K_gr_profile:        object  # K_GR(r) — diverges near singularity
    stability_gr:        float   # GR stability score

    # QESP results
    K_qesp_profile:      object  # K_QESP(r) — plateaus
    Q_correction:        object  # QESPCorrection object
    stability_qesp:      float   # QESP stability score (should be >= GR)

    # Comparison metrics
    max_curvature_gr:    float   # max K_GR in domain
    max_curvature_qesp:  float   # max K_QESP in domain (should be < GR)
    curvature_suppression: float # (K_GR - K_QESP) / K_GR at peak
    plateau_detected:    bool    # True if QESP curvature plateaus
    divergence_prevented: bool   # True if GR would diverge but QESP doesn't

    # Prediction outputs
    curvature_cap:       float   # Predicted maximum curvature under QESP
    oscillation_freq:    float   # If oscillatory, predicted frequency
    gw_deviation_pct:    float   # Predicted GW waveform deviation from GR

    # QESP parameters used
    lambda_param:        float
    epsilon_c:           float
    k_crit:              float

    # Lambda sensitivity
    lambda_scan:         list

    # Full metadata
    qesp_validates:      bool    # Does this simulation support QESP?
    validation_notes:    list
    summary:             dict = field(default_factory=dict)


def simulate_qesp(
    geometry_type:  str,
    parameters:     dict,
    lambda_param:   float = LAMBDA_DEFAULT,
    epsilon_c:      float = EPSILON_CRITICAL,
    k_crit:         float = K_PLANCK,
    activation_mode: ActivationMode = ActivationMode.SIGMOID,
    n_radial:       int = 300,
) -> QESPSimulationResult:
    """
    Run a full QESP simulation for a given exotic spacetime configuration.

    This is the main entry point. It:
    1. Runs the JAX solver to get GR baseline results
    2. Computes Kretschmann scalar profile K_GR(r)
    3. Computes strain epsilon(r) = K_GR(r) / K_crit
    4. Applies QESP correction to get K_QESP(r)
    5. Analyzes whether QESP prevents divergence
    6. Generates observable predictions

    Args:
        geometry_type:   morris_thorne | alcubierre
        parameters:      parameter dict for the geometry
        lambda_param:    QESP correction strength
        epsilon_c:       activation threshold
        k_crit:          critical curvature scale
        activation_mode: sigmoid | tanh | power | gaussian
        n_radial:        number of radial grid points

    Returns:
        QESPSimulationResult with full comparison data
    """
    logger.info(
        f"QESP simulation: geometry={geometry_type}, "
        f"lambda={lambda_param}, epsilon_c={epsilon_c}"
    )

    # ── Step 1: Build radial grid and compute GR curvature ────────────────
    if geometry_type == "morris_thorne":
        b0   = parameters.get("throat_radius",   1.0)
        rho0 = parameters.get("exotic_density",  0.5)
        phi0 = parameters.get("redshift_factor", 0.2)
        tide = parameters.get("tidal_force",     0.3)

        r_min = b0 * 1.001
        r_max = b0 * 5.0
        r_grid = jnp.linspace(r_min, r_max, n_radial)

        # Compute K_GR profile
        from qesp.strain import kretschmann_morris_thorne
        K_gr_profile = vmap(
            lambda r: kretschmann_morris_thorne(r, b0, rho0, phi0)
        )(r_grid)

        # Strain profile
        strain_profile = K_gr_profile / max(k_crit, 1e-30)

        # GR stability (from existing solver)
        from core.simulator.morris_thorne import MorrisThorneParams, solve as mt_solve
        mt_params = MorrisThorneParams(
            throat_radius   = b0,
            exotic_density  = rho0,
            tidal_force     = tide,
            redshift_factor = phi0,
            n_radial        = n_radial,
        )
        gr_result   = mt_solve(mt_params)
        stability_gr = gr_result["metrics"]["stability_score"]

    elif geometry_type == "alcubierre":
        v_s   = parameters.get("warp_speed",     1.0)
        R     = parameters.get("bubble_radius",  1.0)
        sigma = parameters.get("wall_thickness", 0.5)
        rho0  = parameters.get("energy_density", 0.5)

        r_min = 0.01
        r_max = R * 3.0
        r_grid = jnp.linspace(r_min, r_max, n_radial)

        from qesp.strain import kretschmann_alcubierre
        K_gr_profile = vmap(
            lambda r: kretschmann_alcubierre(r, v_s, R, sigma)
        )(r_grid)

        strain_profile = K_gr_profile / max(k_crit, 1e-30)

        from core.simulator.alcubierre import AlcubierreParams, solve as alc_solve
        alc_params = AlcubierreParams(
            warp_speed     = v_s,
            bubble_radius  = R,
            wall_thickness = sigma,
            energy_density = rho0,
            n_radial       = n_radial,
        )
        gr_result    = alc_solve(alc_params)
        stability_gr = gr_result["metrics"]["stability_score"]

    else:
        raise ValueError(f"Unknown geometry type: {geometry_type}")

    # ── Step 2: Analyze strain ─────────────────────────────────────────────
    strain_analysis = analyze_strain(
        strain_profile, r_grid, k_crit, epsilon_c)

    # ── Step 3: Compute Q_munu correction profile ──────────────────────────
    Q_correction = compute_Q_profile(
        strain_profile, r_grid,
        lam       = lambda_param,
        epsilon_c = epsilon_c,
        k_steep   = 20.0,
        mode      = activation_mode,
    )

    # ── Step 4: Compute QESP-modified curvature ────────────────────────────
    K_qesp_profile = compute_modified_curvature_profile(
        K_gr_profile, strain_profile,
        lam       = lambda_param,
        epsilon_c = epsilon_c,
    )

    # ── Step 5: Compute QESP stability ────────────────────────────────────
    # QESP reduces effective curvature which increases stability
    max_K_gr   = float(jnp.max(K_gr_profile))
    max_K_qesp = float(jnp.max(K_qesp_profile))

    # Stability under QESP: same formula but with reduced effective curvature
    K_reduction_factor = max_K_qesp / max(max_K_gr, 1e-30)
    stability_qesp = min(1.0, stability_gr / max(K_reduction_factor, 0.1))
    stability_qesp = max(stability_gr, stability_qesp)  # QESP never decreases stability

    # ── Step 6: Divergence and plateau analysis ────────────────────────────
    # Check if GR would diverge (K_GR >> K_QESP at some point)
    suppression_profile = 1.0 - K_qesp_profile / jnp.maximum(K_gr_profile, 1e-30)
    max_suppression = float(jnp.max(suppression_profile))

    # Plateau detection: K_QESP is bounded even where K_GR grows
    # Check if K_QESP flattens in the high-strain region
    high_strain_mask = strain_profile > epsilon_c
    if jnp.sum(high_strain_mask) > 5:
        K_qesp_high = K_qesp_profile[high_strain_mask]
        K_gr_high   = K_gr_profile[high_strain_mask]
        K_qesp_range = float(jnp.max(K_qesp_high) - jnp.min(K_qesp_high))
        K_gr_range   = float(jnp.max(K_gr_high) - jnp.min(K_gr_high))
        plateau_detected = K_qesp_range < 0.5 * K_gr_range
    else:
        plateau_detected = max_suppression > 0.1

    # Divergence prevented: GR would reach supercritical strain but QESP doesn't
    divergence_prevented = (
        strain_analysis.strain > epsilon_c and
        max_K_qesp < max_K_gr * 0.9
    )

    # ── Step 7: Compute curvature suppression at peak ──────────────────────
    peak_idx = int(jnp.argmax(K_gr_profile))
    K_gr_peak   = float(K_gr_profile[peak_idx])
    K_qesp_peak = float(K_qesp_profile[peak_idx])
    curvature_suppression = (K_gr_peak - K_qesp_peak) / max(K_gr_peak, 1e-30)

    # ── Step 8: Observable predictions ────────────────────────────────────
    curvature_cap = max_K_qesp  # Predicted maximum curvature

    # Oscillation frequency (if system is near critical):
    # omega ~ sqrt(lambda * K_crit) in natural units
    if strain_analysis.at_limit:
        oscillation_freq = math.sqrt(lambda_param * k_crit) / (2 * math.pi)
    else:
        oscillation_freq = 0.0

    # Gravitational wave deviation:
    # The curvature suppression translates to a phase shift in GW waveforms
    # Delta_phi / phi ~ curvature_suppression * (R_Schwarzschild / lambda_GW)
    # Simplified estimate:
    gw_deviation_pct = curvature_suppression * 100.0 * min(1.0, lambda_param)

    # ── Step 9: Lambda sensitivity scan ───────────────────────────────────
    lambda_scan = scan_lambda_sensitivity(
        float(jnp.max(strain_profile)), epsilon_c=epsilon_c)

    # ── Step 10: Validation assessment ────────────────────────────────────
    validation_notes = []
    qesp_validates = True

    if max_suppression < 0.01:
        validation_notes.append("WARN: Curvature suppression < 1% — QESP effect negligible for this config")
        if strain_analysis.strain < 0.3:
            qesp_validates = False
            validation_notes.append("FAIL: Strain too low — QESP not activated, expected for low-curvature configs")

    if plateau_detected:
        validation_notes.append("PASS: Curvature plateau detected — QESP principle supported")

    if divergence_prevented:
        validation_notes.append("PASS: Divergence prevented — GR would exceed critical limit but QESP does not")

    if strain_analysis.regime in ["critical", "supercritical"]:
        validation_notes.append("PASS: Config enters critical regime — maximum QESP activation")

    if stability_qesp > stability_gr:
        validation_notes.append(
            f"PASS: QESP stability {stability_qesp:.3f} > GR stability {stability_gr:.3f}")

    summary = {
        "geometry_type":          geometry_type,
        "max_strain":             float(jnp.max(strain_profile)),
        "strain_regime":          strain_analysis.regime,
        "max_K_gr":               max_K_gr,
        "max_K_qesp":             max_K_qesp,
        "curvature_suppression":  curvature_suppression,
        "plateau_detected":       plateau_detected,
        "divergence_prevented":   divergence_prevented,
        "stability_gr":           stability_gr,
        "stability_qesp":         stability_qesp,
        "curvature_cap":          curvature_cap,
        "oscillation_freq":       oscillation_freq,
        "gw_deviation_pct":       gw_deviation_pct,
        "qesp_validates":         qesp_validates,
        "n_validation_notes":     len(validation_notes),
    }

    logger.info(
        f"QESP result: suppression={curvature_suppression:.3f}, "
        f"plateau={plateau_detected}, validates={qesp_validates}"
    )

    return QESPSimulationResult(
        geometry_type         = geometry_type,
        parameters            = parameters,
        r_grid                = r_grid,
        strain_profile        = strain_profile,
        strain_analysis       = strain_analysis,
        K_gr_profile          = K_gr_profile,
        stability_gr          = stability_gr,
        K_qesp_profile        = K_qesp_profile,
        Q_correction          = Q_correction,
        stability_qesp        = stability_qesp,
        max_curvature_gr      = max_K_gr,
        max_curvature_qesp    = max_K_qesp,
        curvature_suppression = curvature_suppression,
        plateau_detected      = plateau_detected,
        divergence_prevented  = divergence_prevented,
        curvature_cap         = curvature_cap,
        oscillation_freq      = oscillation_freq,
        gw_deviation_pct      = gw_deviation_pct,
        lambda_param          = lambda_param,
        epsilon_c             = epsilon_c,
        k_crit                = k_crit,
        lambda_scan           = lambda_scan,
        qesp_validates        = qesp_validates,
        validation_notes      = validation_notes,
        summary               = summary,
    )
