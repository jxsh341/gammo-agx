"""
Gammo AGX - Ford-Roman Quantum Inequality Checker

The Ford-Roman quantum inequalities set fundamental limits on how
negative the energy density can be and for how long. They are the
primary physical constraint on exotic matter configurations.

For a massless scalar field in 4D Minkowski space:
    integral[ rho(tau) * f(tau) dtau ] >= -3 / (32 * pi^2 * t0^4)

where f(tau) is a Lorentzian sampling function of width t0.

This is not just a theoretical curiosity -- it is the constraint
that determines whether a wormhole configuration is physically
realizable or merely mathematically consistent.

References:
    Ford, L.H. & Roman, T.A. (1995). Phys. Rev. D, 51, 4277.
    Ford, L.H. & Roman, T.A. (1996). Phys. Rev. D, 53, 5496.
    Fewster, C.J. & Teo, E. (1999). Phys. Rev. D, 59, 104016.
"""

import math
from dataclasses import dataclass, field
from loguru import logger
from typing import Optional


# Physical constants
HBAR = 1.0545718e-34
C    = 2.99792458e8
PI   = math.pi

# Ford-Roman constant for massless scalar field in 4D
# C_FR = 3 / (32 * pi^2)
C_FORD_ROMAN = 3.0 / (32.0 * PI**2)


@dataclass
class FordRomanResult:
    """Result of a Ford-Roman quantum inequality check."""
    satisfied:          bool
    bound_value:        float    # Right-hand side of inequality
    integrated_energy:  float    # Left-hand side (negative energy integral)
    margin:             float    # bound_value - |integrated_energy| (positive = satisfied)
    violation_factor:   float    # |integrated_energy| / bound_value (< 1 = satisfied)
    sampling_time:      float    # t0 used in calculation
    status:             str      # "satisfied" | "marginal" | "violated"
    details:            dict = field(default_factory=dict)


def lorentzian_sampling(tau: float, t0: float) -> float:
    """
    Lorentzian sampling function f(tau) = (t0/pi) / (tau^2 + t0^2)

    This is the standard Ford-Roman sampling function.
    Normalized so that integral f(tau) dtau = 1.

    Args:
        tau: proper time
        t0:  sampling timescale

    Returns:
        Sampling function value
    """
    return (t0 / PI) / (tau**2 + t0**2)


def ford_roman_bound(sampling_time: float) -> float:
    """
    Compute the Ford-Roman lower bound for a given sampling time.

    Bound = -C_FR / t0^4  (in natural units where hbar=c=1)
    In SI: Bound = -3 * hbar * c / (32 * pi^2 * t0^4 * c^4) * c^4

    For Planck units (hbar=c=1), the bound is:
        Bound = -C_FR / t0^4

    Args:
        sampling_time: t0 in Planck time units

    Returns:
        Ford-Roman bound value (negative)
    """
    return -C_FORD_ROMAN / sampling_time**4


def check_morris_thorne(
    throat_radius:   float,
    exotic_density:  float,
    sampling_time:   float = 1.0,
) -> FordRomanResult:
    """
    Check Ford-Roman quantum inequality for a Morris-Thorne wormhole.

    For MT with b(r) = b0^2/r, the integrated negative energy is:
        integral(rho * f) ~ rho0 * b0^2 / (8*pi) * f_integral

    The bound is satisfied when:
        rho0 * b0^2 / (8*pi) <= C_FR / t0^4

    Args:
        throat_radius:  b0 in Planck lengths
        exotic_density: rho0 (magnitude of negative energy density)
        sampling_time:  t0 in Planck time units

    Returns:
        FordRomanResult with satisfaction status and details
    """
    # Integrated negative energy for MT geometry
    # This is the left-hand side of the Ford-Roman inequality
    integrated_energy = exotic_density * throat_radius**2 / (8.0 * PI)

    # Ford-Roman bound (right-hand side, expressed as magnitude)
    bound_magnitude = C_FORD_ROMAN / sampling_time**4

    # Check satisfaction
    satisfied = abs(integrated_energy) <= bound_magnitude
    margin = bound_magnitude - abs(integrated_energy)
    violation_factor = abs(integrated_energy) / max(bound_magnitude, 1e-300)

    if violation_factor < 0.8:
        status = "satisfied"
    elif violation_factor < 1.0:
        status = "marginal"
    else:
        status = "violated"

    details = {
        "throat_radius":       throat_radius,
        "exotic_density":      exotic_density,
        "sampling_time":       sampling_time,
        "C_ford_roman":        C_FORD_ROMAN,
        "bound_magnitude":     bound_magnitude,
        "integrated_energy":   integrated_energy,
        "violation_factor":    violation_factor,
    }

    logger.debug(
        f"Ford-Roman check: status={status}, "
        f"factor={violation_factor:.3f}, margin={margin:.4e}"
    )

    return FordRomanResult(
        satisfied=satisfied,
        bound_value=bound_magnitude,
        integrated_energy=integrated_energy,
        margin=margin,
        violation_factor=violation_factor,
        sampling_time=sampling_time,
        status=status,
        details=details,
    )


def check_with_multiple_timescales(
    throat_radius:  float,
    exotic_density: float,
    timescales:     Optional[list] = None,
) -> dict:
    """
    Check Ford-Roman inequality across multiple sampling timescales.

    The inequality must hold for ALL sampling timescales -- a single
    violation is sufficient to rule out the configuration.

    Args:
        throat_radius:  b0 in Planck lengths
        exotic_density: rho0
        timescales:     list of t0 values to check (default: logarithmic range)

    Returns:
        dict with results for each timescale and overall status
    """
    if timescales is None:
        # Check across 5 orders of magnitude
        timescales = [10**i for i in range(-2, 3)]

    results = {}
    all_satisfied = True

    for t0 in timescales:
        result = check_morris_thorne(throat_radius, exotic_density, t0)
        results[f"t0_{t0}"] = {
            "status":           result.status,
            "violation_factor": result.violation_factor,
            "margin":           result.margin,
        }
        if not result.satisfied:
            all_satisfied = False

    return {
        "overall_satisfied": all_satisfied,
        "timescale_results": results,
        "worst_violation":   max(
            r["violation_factor"] for r in results.values()
        ),
        "best_margin":       max(
            r["margin"] for r in results.values()
        ),
    }


def compute_maximum_exotic_density(
    throat_radius:  float,
    sampling_time:  float = 1.0,
) -> float:
    """
    Compute the maximum exotic matter density allowed by
    the Ford-Roman inequality for a given throat radius.

    rho_max = 8*pi * C_FR / (b0^2 * t0^4)

    Args:
        throat_radius: b0 in Planck lengths
        sampling_time: t0 in Planck time units

    Returns:
        Maximum allowed exotic density (positive, represents magnitude)
    """
    if throat_radius <= 0:
        return 0.0

    return (8.0 * PI * C_FORD_ROMAN) / (throat_radius**2 * sampling_time**4)


def quantum_inequality_margin(
    throat_radius:  float,
    exotic_density: float,
    sampling_time:  float = 1.0,
) -> float:
    """
    Compute the signed quantum inequality margin.

    Positive margin = satisfies the bound (how much room is left)
    Negative margin = violates the bound (how far over the limit)

    Useful for the knowledge store and hypothesis scoring.

    Args:
        throat_radius:  b0
        exotic_density: rho0
        sampling_time:  t0

    Returns:
        Signed margin value
    """
    result = check_morris_thorne(throat_radius, exotic_density, sampling_time)
    return result.margin


def filter_by_ford_roman(
    throat_radius:  float,
    exotic_density: float,
    tidal_force:    float,
    strict:         bool = False,
) -> tuple:
    """
    Quick Ford-Roman filter for the discovery loop.

    Args:
        throat_radius:  b0
        exotic_density: rho0
        tidal_force:    tide
        strict:         if True, reject marginal cases too

    Returns:
        (passes_filter: bool, reason: str)
    """
    result = check_morris_thorne(throat_radius, exotic_density)

    if result.status == "violated":
        return False, (
            f"Ford-Roman violated: factor={result.violation_factor:.3f} "
            f"(integrated_energy={result.integrated_energy:.4e} > "
            f"bound={result.bound_value:.4e})"
        )

    if strict and result.status == "marginal":
        return False, (
            f"Ford-Roman marginal (strict mode): "
            f"factor={result.violation_factor:.3f}"
        )

    return True, f"Ford-Roman {result.status}: factor={result.violation_factor:.3f}"
