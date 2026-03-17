"""
Gammo AGX — Morris-Thorne Wormhole Solver
Numerical solver for the Morris-Thorne traversable wormhole metric.

Reference: Morris, M.S. & Thorne, K.S. (1988).
Wormholes in spacetime and their use for interstellar travel.
American Journal of Physics, 56(5), 395-412.

Metric:
    ds² = -e^(2Φ(r)) dt² + dl² + r²(l)(dθ² + sin²θ dφ²)

Shape function:
    b(r) = b₀²/r  (Morris-Thorne standard choice)

Embedding:
    dz/dr = ±1 / sqrt(r/b(r) - 1)
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from dataclasses import dataclass
from loguru import logger


@dataclass
class MorrisThorneParams:
    """Parameters for the Morris-Thorne wormhole metric."""
    throat_radius: float      # b₀ — minimum throat radius (Planck lengths)
    exotic_density: float     # ρ  — negative energy density at throat
    tidal_force: float        # tide — tidal acceleration at throat
    redshift_factor: float    # Φ  — gravitational redshift function value
    n_radial: int = 200       # Number of radial grid points
    r_max_factor: float = 5.0 # r_max = throat_radius * r_max_factor


@jit
def shape_function(r: jnp.ndarray, b0: float) -> jnp.ndarray:
    """
    Morris-Thorne shape function b(r) = b₀²/r.
    Must satisfy b(r) ≤ r for all r ≥ b₀.
    """
    return (b0 * b0) / jnp.maximum(r, b0 * 1e-6)


@jit
def embedding_function(r: jnp.ndarray, b0: float) -> jnp.ndarray:
    """
    Flamm's paraboloid embedding: z(r) = ∫ dr / sqrt(r/b(r) - 1)
    Analytic result for b(r) = b₀²/r:
        z(r) = sqrt(r² - b₀²)
    """
    return jnp.sqrt(jnp.maximum(r * r - b0 * b0, 0.0))


@jit
def redshift_function(r: jnp.ndarray, phi0: float, b0: float) -> jnp.ndarray:
    """
    Redshift function Φ(r). For traversability, |dΦ/dr| must be finite.
    Simple choice: Φ(r) = phi0 * b0 / r
    """
    return phi0 * b0 / jnp.maximum(r, b0 * 1e-6)


@jit
def stress_energy_tensor_tt(r: jnp.ndarray, b0: float, rho0: float) -> jnp.ndarray:
    """
    T^t_t component — energy density.
    For Morris-Thorne: T^t_t = -b'(r) / (8π r²)
    With b(r) = b₀²/r: b'(r) = -b₀²/r²
    So: T^t_t = b₀² / (8π r⁴)  — but exotic matter requires negative energy
    We scale by -rho0 to parameterize exotic matter density.
    """
    b_prime = -(b0 * b0) / jnp.maximum(r * r, 1e-10)
    return -b_prime / (8.0 * jnp.pi * jnp.maximum(r * r, 1e-10)) * rho0


@jit
def null_energy_condition(
    r: jnp.ndarray,
    b0: float,
    rho0: float,
    phi0: float
) -> jnp.ndarray:
    """
    Check null energy condition: T_μν k^μ k^ν ≥ 0
    For Morris-Thorne, NEC is violated at the throat (required for traversability).
    Returns the NEC value — negative means violated.
    """
    T_tt = stress_energy_tensor_tt(r, b0, rho0)
    T_rr = -T_tt * 0.5  # Simplified approximation
    nec = T_tt + T_rr
    return nec


def solve(params: MorrisThorneParams) -> dict:
    """
    Full Morris-Thorne wormhole solver.

    Returns a dict containing:
    - r_grid: radial coordinate array
    - z_upper: upper sheet embedding
    - z_lower: lower sheet embedding
    - b_r: shape function values
    - phi_r: redshift function values
    - T_tt: stress-energy T^t_t component
    - nec: null energy condition values
    - metrics: computed physics metrics
    """
    logger.debug(f"Solving Morris-Thorne: b₀={params.throat_radius:.3f}, ρ={params.exotic_density:.3f}")

    b0   = params.throat_radius
    rho0 = params.exotic_density
    phi0 = params.redshift_factor

    # Radial grid from throat to r_max
    r_min = b0 * 1.001  # Just outside throat
    r_max = b0 * params.r_max_factor
    r_grid = jnp.linspace(r_min, r_max, params.n_radial)

    # Compute geometric quantities
    b_r     = vmap(lambda r: shape_function(r, b0))(r_grid)
    z_upper = vmap(lambda r: embedding_function(r, b0))(r_grid)
    z_lower = -z_upper
    phi_r   = vmap(lambda r: redshift_function(r, phi0, b0))(r_grid)

    # Compute stress-energy
    T_tt = vmap(lambda r: stress_energy_tensor_tt(r, b0, rho0))(r_grid)
    nec  = vmap(lambda r: null_energy_condition(r, b0, rho0, phi0))(r_grid)

    # Physics metrics
    nec_violation   = bool(jnp.any(nec < 0))
    min_nec         = float(jnp.min(nec))
    energy_req      = float(-rho0 * b0 * b0 / (8.0 * jnp.pi))
    traversal_time  = float(b0 * jnp.pi / (1.0 - params.tidal_force * 0.5))
    max_tidal       = float(params.tidal_force * rho0)

    metrics = {
        "energy_requirement":   energy_req,
        "traversal_time":       traversal_time,
        "nec_violated":         nec_violation,
        "min_nec_value":        min_nec,
        "max_tidal_force":      max_tidal,
        "throat_radius":        b0,
        "geometry_type":        "morris_thorne",
    }

    logger.debug(f"Morris-Thorne solved — NEC violated: {nec_violation}, E_req: {energy_req:.4e}")

    return {
        "r_grid":   r_grid,
        "z_upper":  z_upper,
        "z_lower":  z_lower,
        "b_r":      b_r,
        "phi_r":    phi_r,
        "T_tt":     T_tt,
        "nec":      nec,
        "params":   params,
        "metrics":  metrics,
    }
