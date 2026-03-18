"""
Gammo AGX - Spacetime Descriptor Extractor

Converts raw simulation outputs into compact, structured feature vectors
that AI models can reason over and learn from.

Raw simulation grids cannot be fed directly to AI models -- they are
too large, too sparse, and lack semantic structure. The descriptor
system solves this by extracting the physically meaningful properties
of each geometry into a fixed-length vector.

Each descriptor vector has 64 dimensions covering:
    - Curvature properties (dims 0-15)
    - Energy distribution (dims 16-31)
    - Stability metrics (dims 32-47)
    - Quantum margins (dims 48-63)

This enables:
    - AI pattern recognition across thousands of geometries
    - Semantic similarity search in Supabase pgvector
    - Transfer learning across geometry types
    - Clustering of configuration families
"""

import math
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from typing import Optional


# Descriptor vector dimensionality
DESCRIPTOR_DIM = 64

# Dimension ranges
CURVATURE_DIMS   = slice(0, 16)
ENERGY_DIMS      = slice(16, 32)
STABILITY_DIMS   = slice(32, 48)
QUANTUM_DIMS     = slice(48, 64)


@dataclass
class DescriptorVector:
    """A compact feature vector representing a spacetime geometry."""
    vector:          list          # 64-dimensional float list
    geometry_type:   str
    norm:            float         # L2 norm of the vector
    curvature_summary:   dict = field(default_factory=dict)
    energy_summary:      dict = field(default_factory=dict)
    stability_summary:   dict = field(default_factory=dict)
    quantum_summary:     dict = field(default_factory=dict)


def safe_log(x: float, default: float = 0.0) -> float:
    """Safe logarithm that handles zero and negative values."""
    if x <= 0:
        return default
    return math.log10(abs(x))


def safe_normalize(x: float, scale: float = 1.0) -> float:
    """Normalize a value to approximately [-1, 1] range."""
    return max(-1.0, min(1.0, x / max(scale, 1e-10)))


def extract_morris_thorne(
    simulation_result: dict,
    params: dict,
) -> DescriptorVector:
    """
    Extract descriptor vector from a Morris-Thorne simulation result.

    Args:
        simulation_result: output from core.simulator.morris_thorne.solve()
        params: input parameters dict

    Returns:
        DescriptorVector with 64-dimensional feature vector
    """
    metrics = simulation_result.get("metrics", {})
    r_grid  = simulation_result.get("r_grid", [])
    nec     = simulation_result.get("nec", [])
    b_r     = simulation_result.get("b_r", [])
    z_upper = simulation_result.get("z_upper", [])

    # Convert JAX arrays to numpy if needed
    if hasattr(r_grid, "tolist"):
        r_grid  = list(r_grid)
        nec     = list(nec)
        b_r     = list(b_r)
        z_upper = list(z_upper)

    b0   = params.get("throat_radius",   1.0)
    rho0 = params.get("exotic_density",  0.5)
    tide = params.get("tidal_force",     0.3)
    phi0 = params.get("redshift_factor", 0.2)

    vector = [0.0] * DESCRIPTOR_DIM

    # ── CURVATURE DIMENSIONS (0-15) ──────────────────────────────────────
    # These encode the geometric properties of the spacetime

    # 0: Throat radius (normalized to [0,1] over [0.3, 3.0])
    vector[0] = (b0 - 0.3) / 2.7

    # 1: Shape function slope at throat (always -1 for MT b=b0^2/r)
    vector[1] = -1.0

    # 2: Embedding depth (max z / b0) — how deep the funnel goes
    max_z = max(z_upper) if z_upper else b0 * 2
    vector[2] = safe_normalize(max_z / b0, scale=5.0)

    # 3: Curvature concentration — how quickly geometry flattens
    if len(z_upper) > 10:
        z_gradient = [z_upper[i+1] - z_upper[i] for i in range(len(z_upper)-1)]
        curvature_conc = abs(z_gradient[0]) / max(abs(z_gradient[-1]), 1e-10)
        vector[3] = safe_normalize(math.log10(max(curvature_conc, 1.0)), scale=3.0)
    else:
        vector[3] = 0.5

    # 4: Shape function decay rate (b(r)/r at r=2*b0)
    shape_at_2b0 = b0**2 / (2 * b0) / (2 * b0)  # = 1/4 always for MT
    vector[4] = shape_at_2b0

    # 5: Asymptotic flatness measure (b(r_max)/r_max)
    r_max = b0 * 5.0
    flatness = b0**2 / r_max**2
    vector[5] = safe_normalize(flatness, scale=0.1)

    # 6-9: Redshift profile (phi at multiple radii)
    for i, r_factor in enumerate([1.1, 1.5, 2.0, 3.0]):
        r_val = b0 * r_factor
        phi_r = phi0 * b0 / r_val
        vector[6 + i] = safe_normalize(phi_r, scale=1.0)

    # 10-12: Shape function values at key radii
    for i, r_factor in enumerate([1.5, 2.0, 3.0]):
        r_val = b0 * r_factor
        b_val = b0**2 / r_val
        vector[10 + i] = safe_normalize(b_val / r_val, scale=1.0)

    # 13: Traversal time (normalized)
    trav_time = metrics.get("traversal_time", b0 * math.pi)
    vector[13] = safe_normalize(trav_time, scale=20.0)

    # 14: Geometry class encoding
    geo_class = metrics.get("geometry_class", "STANDARD")
    class_map = {"MICRO": 0.0, "STANDARD": 0.33, "MACRO": 0.67, "EXOTIC+": 1.0}
    vector[14] = class_map.get(geo_class, 0.33)

    # 15: Throat compactness (b0 / r_max)
    vector[15] = b0 / (b0 * 5.0)  # = 0.2 always, but useful for cross-geometry

    curvature_summary = {
        "throat_radius":     b0,
        "embedding_depth":   max_z,
        "flatness_measure":  flatness,
        "traversal_time":    trav_time,
        "geometry_class":    geo_class,
    }

    # ── ENERGY DIMENSIONS (16-31) ─────────────────────────────────────────
    # These encode the exotic energy properties

    # 16: Exotic energy density (log-normalized)
    vector[16] = safe_normalize(safe_log(rho0), scale=2.0)

    # 17: Energy requirement magnitude (log-normalized)
    energy_req = metrics.get("energy_requirement", 0.0)
    vector[17] = safe_normalize(safe_log(abs(energy_req)), scale=5.0)

    # 18: NEC violation extent
    nec_violations = [v for v in nec if v < 0] if nec else []
    nec_fraction = len(nec_violations) / max(len(nec), 1)
    vector[18] = nec_fraction

    # 19: Minimum NEC value (most negative point)
    min_nec = min(nec) if nec else -rho0
    vector[19] = safe_normalize(safe_log(abs(min_nec)), scale=3.0)

    # 20: Energy density at throat vs outer region ratio
    vector[20] = safe_normalize(rho0 / max(abs(energy_req), 1e-10), scale=10.0)

    # 21: Casimir gap (orders of magnitude)
    casimir_gap = metrics.get("casimir_gap_oom", 47.0)
    vector[21] = safe_normalize(casimir_gap, scale=50.0)

    # 22-27: Energy profile at different radii
    for i, r_factor in enumerate([1.1, 1.3, 1.6, 2.0, 2.5, 3.5]):
        r_val = b0 * r_factor
        # Stress-energy decays as 1/r^4 for MT
        energy_at_r = rho0 * (b0 / r_val)**4
        vector[22 + i] = safe_normalize(safe_log(energy_at_r), scale=3.0)

    # 28: Total integrated negative energy
    integrated_neg = rho0 * b0**2 / (8 * math.pi)
    vector[28] = safe_normalize(safe_log(integrated_neg), scale=3.0)

    # 29: Energy-to-mass ratio (dimensionless)
    vector[29] = safe_normalize(rho0 * b0**3, scale=10.0)

    # 30-31: Reserved for future energy metrics
    vector[30] = safe_normalize(phi0 * rho0, scale=1.0)
    vector[31] = safe_normalize(tide * rho0, scale=1.0)

    energy_summary = {
        "exotic_density":    rho0,
        "energy_requirement": energy_req,
        "nec_fraction":      nec_fraction,
        "casimir_gap_oom":   casimir_gap,
        "integrated_neg_energy": integrated_neg,
    }

    # ── STABILITY DIMENSIONS (32-47) ─────────────────────────────────────
    # These encode the dynamical stability properties

    # 32: Overall stability score
    stability = metrics.get("stability_score", 0.0)
    vector[32] = stability

    # 33: BSSN stability flag
    bssn_stable = metrics.get("bssn_stable", True)
    vector[33] = 1.0 if bssn_stable else 0.0

    # 34: Constraint violation error
    constraint_err = metrics.get("constraint_error", 0.0)
    vector[34] = safe_normalize(constraint_err, scale=1.0)

    # 35: Tidal force magnitude
    vector[35] = tide

    # 36: Tidal force * exotic density product
    vector[36] = safe_normalize(tide * rho0, scale=1.0)

    # 37: Redshift factor
    vector[37] = phi0

    # 38: Stability gradient (how stability changes with throat size)
    # For MT: stability ~ (1/(tide+0.1)) * rho0 * 0.5 + (1-phi0) * 0.3
    # Gradient w.r.t. b0 is approximately 0 (stability doesn't depend on b0 directly)
    vector[38] = safe_normalize(1.0 / (tide + 0.1), scale=10.0)

    # 39: Traversability score (inverse of tidal force)
    traversability = 1.0 - min(tide, 0.99)
    vector[39] = traversability

    # 40: Metric regularity (no singularities in domain)
    vector[40] = 1.0  # MT is always regular outside throat

    # 41-44: Stability at different parameter extremes
    # Stability sensitivity to parameter variations
    delta = 0.1
    stab_high_tide = max(0.0, min(1.0,
        (1.0 / (tide + delta + 0.1)) * rho0 * 0.5 + (1.0 - phi0) * 0.3
    ))
    stab_low_tide = max(0.0, min(1.0,
        (1.0 / (max(tide - delta, 0.01) + 0.1)) * rho0 * 0.5 + (1.0 - phi0) * 0.3
    ))
    vector[41] = stab_high_tide
    vector[42] = stab_low_tide
    vector[43] = abs(stab_high_tide - stab_low_tide)  # sensitivity
    vector[44] = safe_normalize(stability * traversability, scale=1.0)

    # 45-47: Reserved for BSSN evolution metrics
    vector[45] = safe_normalize(constraint_err * tide, scale=1.0)
    vector[46] = safe_normalize(stability * (1.0 - constraint_err), scale=1.0)
    vector[47] = 1.0 if stability > 0.7 and bssn_stable else 0.0

    stability_summary = {
        "stability_score":   stability,
        "bssn_stable":       bssn_stable,
        "constraint_error":  constraint_err,
        "tidal_force":       tide,
        "traversability":    traversability,
    }

    # ── QUANTUM DIMENSIONS (48-63) ────────────────────────────────────────
    # These encode quantum constraint satisfaction

    # 48: Ford-Roman status (1=satisfied, 0=violated)
    ford_roman = metrics.get("ford_roman_status", "unknown")
    vector[48] = 1.0 if ford_roman == "satisfied" else 0.0

    # 49: Ford-Roman violation factor
    integrated_neg_energy = rho0 * b0**2 / (8 * math.pi)
    fr_bound = 3.0 / (32.0 * math.pi**2)
    fr_factor = abs(integrated_neg_energy) / max(fr_bound, 1e-300)
    vector[49] = safe_normalize(math.log10(max(fr_factor, 1e-10)), scale=3.0)

    # 50: NEC violation flag (expected for traversable wormhole)
    nec_violated = metrics.get("null_energy_violated", True)
    vector[50] = 1.0 if nec_violated else 0.0

    # 51: WEC (Weak Energy Condition) — violated if rho < 0
    vector[51] = 1.0  # Always violated for exotic matter

    # 52: Quantum inequality margin (signed)
    qi_margin = fr_bound - abs(integrated_neg_energy)
    vector[52] = safe_normalize(qi_margin, scale=0.1)

    # 53: Casimir achievability score
    # How achievable is the required energy density via Casimir?
    casimir_score = max(0.0, 1.0 - casimir_gap / 50.0)
    vector[53] = casimir_score

    # 54: Energy condition violation severity
    violation_severity = fr_factor / max(stability, 0.01)
    vector[54] = safe_normalize(math.log10(max(violation_severity, 1e-10)), scale=3.0)

    # 55: Quantum viability index (composite)
    qi_viability = (
        (1.0 if ford_roman == "satisfied" else 0.0) * 0.4 +
        (1.0 - min(fr_factor / 10.0, 1.0)) * 0.3 +
        casimir_score * 0.3
    )
    vector[55] = qi_viability

    # 56-59: Ford-Roman check at multiple timescales
    for i, t0_exp in enumerate([-1, 0, 1, 2]):
        t0 = 10.0**t0_exp
        fr_bound_t = 3.0 / (32.0 * math.pi**2 * t0**4)
        passes = abs(integrated_neg_energy) <= fr_bound_t
        vector[56 + i] = 1.0 if passes else 0.0

    # 60: Planck-scale proximity
    planck_proximity = safe_log(rho0) / 3.0  # normalized
    vector[60] = safe_normalize(planck_proximity, scale=1.0)

    # 61: Quantum coherence length (b0 * t_planck/t0)
    vector[61] = safe_normalize(b0 / 1.0, scale=3.0)

    # 62-63: Reserved for future quantum metrics
    vector[62] = safe_normalize(rho0 * b0, scale=3.0)
    vector[63] = safe_normalize(tide * b0 / max(rho0, 0.01), scale=10.0)

    quantum_summary = {
        "ford_roman_status":  ford_roman,
        "fr_violation_factor": fr_factor,
        "qi_margin":          qi_margin,
        "casimir_score":      casimir_score,
        "qi_viability":       qi_viability,
    }

    # ── FINAL ASSEMBLY ────────────────────────────────────────────────────
    # Ensure all values are valid floats in [-1, 1] range where applicable
    vector = [float(max(-2.0, min(2.0, v))) for v in vector]

    # Compute L2 norm
    norm = math.sqrt(sum(v**2 for v in vector))

    logger.debug(
        f"Descriptor extracted: dim={len(vector)}, norm={norm:.3f}, "
        f"stability={stability:.3f}, ford_roman={ford_roman}"
    )

    return DescriptorVector(
        vector=vector,
        geometry_type="morris_thorne",
        norm=norm,
        curvature_summary=curvature_summary,
        energy_summary=energy_summary,
        stability_summary=stability_summary,
        quantum_summary=quantum_summary,
    )


def extract(
    geometry_type:     str,
    simulation_result: dict,
    params:            dict,
) -> Optional[DescriptorVector]:
    """
    Main entry point for descriptor extraction.
    Routes to the appropriate geometry-specific extractor.

    Args:
        geometry_type:     e.g. "morris_thorne"
        simulation_result: raw solver output
        params:            input parameters

    Returns:
        DescriptorVector or None if extraction fails
    """
    try:
        if geometry_type == "morris_thorne":
            return extract_morris_thorne(simulation_result, params)
        else:
            logger.warning(f"No descriptor extractor for geometry: {geometry_type}")
            return None
    except Exception as e:
        logger.error(f"Descriptor extraction failed for {geometry_type}: {e}")
        return None


def descriptor_to_list(descriptor: DescriptorVector) -> list:
    """Convert descriptor to a plain Python list for Supabase storage."""
    return [float(v) for v in descriptor.vector]


def cosine_similarity(v1: list, v2: list) -> float:
    """Compute cosine similarity between two descriptor vectors."""
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a**2 for a in v1))
    norm2 = math.sqrt(sum(b**2 for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
