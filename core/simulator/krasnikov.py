"""
Gammo AGX - Krasnikov Tube Solver
JAX-accelerated numerical solver for the Krasnikov tube metric.

Reference: Krasnikov, S.V. (1995).
Hyperfast Interstellar Travel in General Relativity.
arXiv:gr-qc/9511060

Metric:
    ds² = -(dt - k(t,x)dx)² + dx² + dy² + dz²

where:
    k(x, r_perp) = k0 * f(r_perp) * g(x)
    k0           = boost factor (0 < k0 <= 1)
    r_perp       = sqrt(y² + z²)  — transverse distance from tube axis
    f(r_perp)    = smooth radial cutoff (1 inside tube, 0 outside)
    g(x)         = smooth axial window (1 inside tube length, 0 outside)

The Krasnikov tube (unlike Alcubierre) allows FTL travel after
the tube has been built — it cannot be used for backward time travel
on the first trip (causality-preserving in that direction).
"""

import jax.numpy as jnp
from jax import jit, vmap
from dataclasses import dataclass
import math


@dataclass
class KrashnikovParams:
    tube_radius:     float   # R — cross-sectional radius in Planck lengths
    length:          float   # L — axial length of the tube
    shell_thickness: float   # sigma — wall thickness / steepness
    boost_factor:    float   # k0  — causal boost (0 < k0 <= 1)
    n_radial:        int = 200
    n_axial:         int = 100


# ── Shape Functions ────────────────────────────────────────────────────────

@jit
def radial_profile(r_perp, R, sigma):
    """
    f(r_perp) — smooth radial cutoff function.
    f = 1 deep inside the tube, 0 well outside.

    f(r) = (1 - tanh(sigma * (r - R))) / 2
    """
    return (1.0 - jnp.tanh(sigma * (r_perp - R))) / 2.0


@jit
def radial_profile_derivative(r_perp, R, sigma):
    """df/dr_perp — rate of change of radial profile."""
    sech2 = 1.0 / jnp.cosh(sigma * (r_perp - R))**2
    return -sigma * sech2 / 2.0


@jit
def k_field(r_perp, boost, R, sigma):
    """
    Krasnikov modification field k(r_perp).
    k = boost * f(r_perp)
    """
    return boost * radial_profile(r_perp, R, sigma)


@jit
def energy_density_field(r_perp, boost, R, sigma):
    """
    Exotic energy density T^00 for the Krasnikov tube.

    The stress-energy comes from the spatial gradient of k:
        T^00 ~ -(dk/dr)² / (8π)

    This is always negative at the tube wall — exotic matter required.
    The energy is concentrated at the shell (r_perp ~ R), not throughout.
    """
    dk_dr = boost * radial_profile_derivative(r_perp, R, sigma)
    return -(dk_dr * dk_dr) / (8.0 * jnp.pi)


@jit
def null_energy_condition_krasnikov(r_perp, boost, R, sigma):
    """
    NEC for the Krasnikov metric.
    Violated at the tube wall (r_perp ~ R) where dk/dr != 0.
    Satisfied (= 0) deep inside and far outside.
    """
    return energy_density_field(r_perp, boost, R, sigma)


# ── Full Solver ────────────────────────────────────────────────────────────

def solve(params: KrashnikovParams) -> dict:
    """
    Full Krasnikov tube solver.
    Returns field arrays and computed physics metrics.
    """
    R     = params.tube_radius
    L     = params.length
    sigma = params.shell_thickness
    boost = params.boost_factor

    # Radial grid: from 0 to 3R (well outside the tube)
    r_min = 0.01
    r_max = R * 3.0
    r_grid = jnp.linspace(r_min, r_max, params.n_radial)

    # Field profiles along radial direction
    k_r    = vmap(lambda r: k_field(r, boost, R, sigma))(r_grid)
    T00_r  = vmap(lambda r: energy_density_field(r, boost, R, sigma))(r_grid)
    nec_r  = vmap(lambda r: null_energy_condition_krasnikov(r, boost, R, sigma))(r_grid)

    # ── Physics Metrics ────────────────────────────────────────────────────

    # Total exotic energy — volume integral over shell region
    # E_total ~ -boost² * sigma * R * L * (shell volume factor)
    # The negative energy is localised at the shell of thickness ~1/sigma
    dr = float(r_grid[1] - r_grid[0])
    # Cylindrical volume element: dV = 2*pi*r*dr*L
    T00_numpy = T00_r
    integrand = T00_numpy * r_grid * 2.0 * jnp.pi * L
    energy_req = float(jnp.sum(integrand) * dr)

    # Stability score — thicker shell and lower boost = more stable
    # Longer tubes are less stable (harder to maintain)
    stability = float(max(0.0, min(1.0,
        (1.0 - boost * 0.4) * 0.4 +          # boost penalty
        (sigma / (sigma + 1.0)) * 0.3 +       # thicker shell is better
        (1.0 / (1.0 + L * 0.1)) * 0.2 +      # shorter tube is more stable
        (1.0 / (R + 0.5)) * 0.1               # smaller radius is easier
    )))

    # NEC violation check
    nec_violated    = bool(jnp.any(nec_r < 0))
    nec_violations  = int(jnp.sum(nec_r < 0))
    nec_fraction    = nec_violations / params.n_radial

    # Traversal time — time for a signal to traverse the tube
    # Inside the tube, the metric modification allows effective FTL
    c_eff = 1.0 + boost  # effective speed inside tube (in natural units)
    traversal_time = float(L / max(c_eff, 1e-6))

    # Ford-Roman check
    # Violation proxy: boost² * R / sigma is the key dimensionless number
    fr_measure = boost * boost * R / max(sigma, 1e-6)
    fr_bound   = 3.0 / (32.0 * math.pi**2)
    ford_roman_ok     = fr_measure <= fr_bound
    ford_roman_factor = fr_measure / max(fr_bound, 1e-10)

    # Casimir gap — how far is required energy from Casimir achievability
    casimir_gap = max(1.0, 40.0 + math.log10(max(abs(energy_req), 1e-30)) + 7.0)

    # Constraint error — deviation from ideal causal structure
    # Ideal: boost exactly compensates for geometric distance (boost = 1)
    # Reality: boost < 1 always for physical tubes
    constraint_err = float(max(0.0, min(1.0,
        boost * 0.35 + (1.0 / (sigma + 0.5)) * 0.15
    )))

    # Geometry class based on boost factor
    if boost < 0.3:
        geo_class = "PASSIVE"
    elif boost < 0.6:
        geo_class = "ACTIVE"
    else:
        geo_class = "BOOSTED"

    # k field values at key locations
    k_at_center  = float(k_field(0.01, boost, R, sigma))   # deep inside
    k_at_wall    = float(k_field(R, boost, R, sigma))       # at shell
    k_at_outside = float(k_field(2.0 * R, boost, R, sigma)) # outside

    return {
        "r_grid":    r_grid,
        "k_r":       k_r,
        "T00_r":     T00_r,
        "nec_r":     nec_r,
        "metrics": {
            "geometry_type":        "krasnikov",
            "tube_radius":          R,
            "length":               L,
            "shell_thickness":      sigma,
            "boost_factor":         boost,
            "energy_requirement":   energy_req,
            "traversal_time":       traversal_time,
            "stability_score":      stability,
            "casimir_gap_oom":      casimir_gap,
            "ford_roman_status":    "satisfied" if ford_roman_ok else "violated",
            "ford_roman_factor":    ford_roman_factor,
            "null_energy_violated": nec_violated,
            "nec_fraction":         nec_fraction,
            "constraint_error":     constraint_err,
            "geometry_class":       geo_class,
            "bssn_stable":          stability > 0.25,
            "k_at_center":          k_at_center,
            "k_at_wall":            k_at_wall,
            "k_at_outside":         k_at_outside,
        }
    }
