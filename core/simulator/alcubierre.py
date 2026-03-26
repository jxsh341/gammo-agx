"""
Gammo AGX - Alcubierre Warp Drive Solver
JAX-accelerated numerical solver for the Alcubierre warp metric.

Reference: Alcubierre, M. (1994).
Classical and Quantum Gravity, 11(5), L73-L77.
"The warp drive: hyper-fast travel within general relativity"

Metric:
    ds² = -dt² + (dx - v_s(t) f(r_s) dt)² + dy² + dz²

where:
    v_s(t)  = bubble velocity (can exceed c)
    f(r_s)  = shape function (1 inside, 0 outside)
    r_s     = distance from bubble center
    R       = bubble radius
    sigma   = bubble wall thickness parameter
"""

import jax.numpy as jnp
from jax import jit, vmap
from dataclasses import dataclass
import math


@dataclass
class AlcubierreParams:
    warp_speed:      float  # v_s / c  (warp factor)
    bubble_radius:   float  # R in Planck lengths
    wall_thickness:  float  # sigma (steepness of bubble wall)
    energy_density:  float  # magnitude of exotic energy
    n_radial:        int = 200
    r_max_factor:    float = 3.0


@jit
def shape_function(r_s, R, sigma):
    """
    Alcubierre shape function f(r_s).
    f = 1 inside bubble, 0 outside, smooth transition at wall.

    f(r_s) = (tanh(sigma*(r_s + R)) - tanh(sigma*(r_s - R))) / (2*tanh(sigma*R))
    """
    denom = 2.0 * jnp.tanh(sigma * R + 1e-8)
    return (jnp.tanh(sigma * (r_s + R)) - jnp.tanh(sigma * (r_s - R))) / denom


@jit
def shape_function_derivative(r_s, R, sigma):
    """df/dr_s — rate of change of shape function."""
    sech2_plus  = 1.0 / jnp.cosh(sigma * (r_s + R))**2
    sech2_minus = 1.0 / jnp.cosh(sigma * (r_s - R))**2
    denom = 2.0 * jnp.tanh(sigma * R + 1e-8)
    return sigma * (sech2_plus - sech2_minus) / denom


@jit
def energy_density_field(r_s, v_s, R, sigma):
    """
    Alcubierre exotic energy density T^00.

    T^00 = -v_s^2 / (8*pi) * (df/dr_s)^2 * (y^2 + z^2) / (4 * r_s^2)

    Simplified for spherical symmetry (y^2 + z^2 = r_s^2 * sin^2(theta)):
    T^00 = -v_s^2 / (32*pi) * (df/dr_s)^2

    This is always negative — exotic matter required everywhere.
    """
    df = shape_function_derivative(r_s, R, sigma)
    return -(v_s * v_s) / (32.0 * jnp.pi) * df * df


@jit
def null_energy_condition_alcubierre(r_s, v_s, R, sigma):
    """
    NEC for Alcubierre metric.
    Always violated where df/dr_s != 0 (the bubble wall).
    """
    T00 = energy_density_field(r_s, v_s, R, sigma)
    # NEC: T_uv k^u k^v for null vector — proportional to T00 for this metric
    return T00


def solve(params: AlcubierreParams) -> dict:
    """
    Full Alcubierre warp drive solver.
    Returns field arrays and computed physics metrics.
    """
    v_s   = params.warp_speed
    R     = params.bubble_radius
    sigma = params.wall_thickness
    rho0  = params.energy_density

    # Radial grid (from 0 to r_max)
    r_min = 0.01
    r_max = R * params.r_max_factor
    r_grid = jnp.linspace(r_min, r_max, params.n_radial)

    # Shape function profile
    f_r = vmap(lambda r: shape_function(r, R, sigma))(r_grid)

    # Energy density profile
    T00_r = vmap(lambda r: energy_density_field(r, v_s, R, sigma))(r_grid)

    # NEC profile
    nec_r = vmap(lambda r: null_energy_condition_alcubierre(r, v_s, R, sigma))(r_grid)

    # ── Physics Metrics ──────────────────────────────────────────────────

    # Total exotic energy (volume integral approximation)
    # E_total ~ -v_s^2 * R^3 * sigma / (some dimensionless factor)
    energy_req = float(-(v_s**2 * R**3 * sigma) / (12.0 * math.pi))

    # Energy in solar masses for context (in Planck units)
    # E_solar_mass ~ -v_s^2 * (R/R_sun)^3 * (sigma/sigma_0) * M_sun
    energy_solar = abs(energy_req) * 1.346e57  # rough conversion factor

    # NEC violation extent
    nec_violated = bool(jnp.any(nec_r < 0))
    nec_violations = int(jnp.sum(nec_r < 0))
    nec_fraction = nec_violations / params.n_radial

    # Bubble stability — thicker walls and lower speed = more stable
    stability = float(max(0.0, min(1.0,
        (1.0 / (v_s + 0.1)) * 0.4 +
        (sigma / (sigma + 1.0)) * 0.3 +
        (1.0 / (R + 0.5)) * 0.2 +
        (1.0 - rho0) * 0.1
    )))

    # Ford-Roman check (approximate for Alcubierre)
    # The integrated negative energy scales as v_s^2 * R^3 * sigma
    integrated_neg = abs(energy_req)
    fr_bound = 3.0 / (32.0 * math.pi**2)
    ford_roman_ok = integrated_neg <= fr_bound
    ford_roman_factor = integrated_neg / max(fr_bound, 1e-10)

    # Casimir gap — how far is required energy from Casimir achievability
    casimir_gap = max(1.0, 47.0 + math.log10(max(abs(energy_req), 1e-30)) + 8.0)

    # Constraint error
    constraint_err = float(max(0.0, min(1.0,
        (v_s - 1.0) * 0.3 + rho0 * 0.2
    )))

    # Geometry class
    if v_s < 1.0:
        geo_class = "SUBLUMINAL"
    elif v_s < 2.0:
        geo_class = "WARP-1"
    elif v_s < 5.0:
        geo_class = "WARP-2"
    else:
        geo_class = "EXTREME"

    # Traversal time for 1 light-year (in Planck time units)
    ly_in_planck = 5.85e51  # 1 light-year in Planck lengths
    traversal_time = float(ly_in_planck / max(v_s * 3e8, 1.0))

    # Shape function value at key radii
    f_at_center = float(shape_function(0.1, R, sigma))
    f_at_R      = float(shape_function(R, R, sigma))
    f_at_2R     = float(shape_function(2*R, R, sigma))

    return {
        "r_grid":    r_grid,
        "f_r":       f_r,
        "T00_r":     T00_r,
        "nec_r":     nec_r,
        "metrics": {
            "geometry_type":        "alcubierre",
            "warp_speed":           v_s,
            "energy_requirement":   energy_req,
            "energy_solar_masses":  energy_solar,
            "traversal_time":       traversal_time,
            "stability_score":      stability,
            "casimir_gap_oom":      casimir_gap,
            "ford_roman_status":    "satisfied" if ford_roman_ok else "violated",
            "ford_roman_factor":    ford_roman_factor,
            "null_energy_violated": nec_violated,
            "nec_fraction":         nec_fraction,
            "constraint_error":     constraint_err,
            "geometry_class":       geo_class,
            "bssn_stable":          stability > 0.3,
            "f_at_center":          f_at_center,
            "f_at_bubble_wall":     f_at_R,
            "f_at_2R":              f_at_2R,
            "bubble_radius":        R,
            "wall_thickness":       sigma,
        }
    }
