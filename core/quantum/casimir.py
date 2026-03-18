"""
Gammo AGX - Casimir Effect Vacuum Energy Engine

Models negative energy density from quantum vacuum fluctuations.
The Casimir effect is the only experimentally confirmed source of
negative energy density -- critical for exotic metric viability.

The central question Gammo AGX is built around:
    Under what physical conditions does the energy requirement for a
    stable Morris-Thorne wormhole throat approach what is achievable
    via the Casimir effect?

References:
    Casimir, H.B.G. (1948). Proc. Kon. Ned. Akad. Wetensch., 51, 793.
    Ford, L.H. & Roman, T.A. (1995). Phys. Rev. D, 51, 4277.
    Visser, M. (1995). Lorentzian Wormholes. AIP Press.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from typing import Optional


# ── Physical Constants (SI units) ──────────────────────────────────────────
HBAR  = 1.0545718e-34   # Reduced Planck constant (J·s)
C     = 2.99792458e8    # Speed of light (m/s)
PI    = math.pi
KB    = 1.380649e-23    # Boltzmann constant (J/K)

# Planck units for comparison
PLANCK_LENGTH  = 1.616255e-35   # m
PLANCK_ENERGY  = 1.956e9        # J
PLANCK_DENSITY = 5.155e96       # kg/m³  (enormous)


class CasimirGeometry(Enum):
    PARALLEL_PLATE  = "parallel_plate"
    SPHERICAL_SHELL = "spherical_shell"
    CYLINDRICAL     = "cylindrical"
    TOROIDAL        = "toroidal"


@dataclass
class CasimirConfig:
    """Configuration for a Casimir energy calculation."""
    geometry:    CasimirGeometry = CasimirGeometry.PARALLEL_PLATE
    separation:  float = 1e-9    # Plate separation or shell radius (m)
    area:        float = 1e-6    # Plate area m² (parallel plate only)
    radius:      float = 1e-9    # Shell/cylinder radius (m)
    length:      float = 1e-6    # Cylinder/tube length (m)
    temperature: float = 0.0     # Temperature K (0 = zero-temperature limit)
    material:    str   = "ideal" # Conductor type: ideal, gold, silicon


@dataclass
class CasimirResult:
    """Result of a Casimir energy calculation."""
    geometry:        str
    energy_density:  float   # J/m³  (negative)
    total_energy:    float   # J     (negative)
    force:           float   # N     (negative = attractive)
    pressure:        float   # Pa
    is_negative:     bool
    separation:      float
    planck_ratio:    float   # |energy_density| / planck_density
    gap_orders:      float   # log10(|required| / |achieved|) vs wormhole
    details:         dict = field(default_factory=dict)


# ── Core Casimir Calculations ───────────────────────────────────────────────

def parallel_plate_energy_density(
    separation: float,
    temperature: float = 0.0,
) -> float:
    """
    Casimir energy density between two infinite parallel conducting plates.

    Zero temperature:
        rho = -pi² hbar c / (720 d⁴)

    Finite temperature correction (leading order):
        rho_T = rho_0 * (1 + (2*pi*d*kB*T / hbar*c)^2 * ...)

    Args:
        separation: plate separation d in meters
        temperature: temperature in Kelvin

    Returns:
        Energy density in J/m³ (negative)
    """
    if separation <= 0:
        raise ValueError(f"Separation must be positive, got {separation}")

    rho_0 = -(PI**2 * HBAR * C) / (720.0 * separation**4)

    if temperature > 0:
        # Thermal correction factor (approximate)
        thermal_param = (2 * PI * separation * KB * temperature) / (HBAR * C)
        correction = 1.0 + thermal_param**2 / 3.0
        return rho_0 * correction

    return rho_0


def parallel_plate_force(separation: float, area: float) -> float:
    """
    Casimir force between two parallel conducting plates.

    F = -pi² hbar c A / (240 d⁴)

    Args:
        separation: plate separation d in meters
        area: plate area A in m²

    Returns:
        Force in Newtons (negative = attractive)
    """
    return -(PI**2 * HBAR * C * area) / (240.0 * separation**4)


def spherical_shell_energy(radius: float) -> float:
    """
    Casimir energy for a perfectly conducting spherical shell.

    E = 0.0462 hbar c / (2a)    (Boyer 1968 result)

    Note: Unlike parallel plates, the spherical Casimir energy is
    POSITIVE (repulsive) for a conducting shell. Negative energy
    requires specific boundary conditions.

    Args:
        radius: shell radius a in meters

    Returns:
        Total Casimir energy in Joules
    """
    return 0.0462 * HBAR * C / (2.0 * radius)


def cylindrical_energy_density(radius: float, length: float) -> float:
    """
    Casimir energy density inside a conducting cylindrical shell.

    rho ~ -hbar c / (720 * pi * r⁴)   (approximate)

    Args:
        radius: cylinder radius in meters
        length: cylinder length in meters

    Returns:
        Energy density in J/m³ (negative)
    """
    volume = PI * radius**2 * length
    energy = -(HBAR * C) / (720.0 * PI * radius**4) * length
    return energy / volume if volume > 0 else 0.0


def toroidal_energy_density(
    major_radius: float,
    minor_radius: float,
) -> float:
    """
    Approximate Casimir energy density for a toroidal cavity.
    Based on proximity force approximation for d << R.

    Args:
        major_radius: R - distance from torus center to tube center (m)
        minor_radius: r - radius of the tube (m)

    Returns:
        Energy density in J/m³ (approximate, negative)
    """
    if minor_radius <= 0 or major_radius <= minor_radius:
        return 0.0

    # Effective separation = minor_radius for proximity force approximation
    d_eff = minor_radius
    rho_plate = parallel_plate_energy_density(d_eff)

    # Geometric correction factor for toroidal curvature
    correction = 1.0 - minor_radius / (2.0 * major_radius)
    return rho_plate * correction


# ── Main Compute Function ───────────────────────────────────────────────────

def compute_casimir(config: CasimirConfig) -> CasimirResult:
    """
    Compute Casimir energy for the given configuration.

    Args:
        config: CasimirConfig specifying geometry and parameters

    Returns:
        CasimirResult with energy density, force, and analysis
    """
    logger.debug(
        f"Computing Casimir: geometry={config.geometry.value}, "
        f"separation={config.separation:.2e}m, T={config.temperature}K"
    )

    if config.geometry == CasimirGeometry.PARALLEL_PLATE:
        rho = parallel_plate_energy_density(config.separation, config.temperature)
        force = parallel_plate_force(config.separation, config.area)
        total = rho * config.area * config.separation
        pressure = force / config.area if config.area > 0 else 0.0
        details = {
            "separation_nm": config.separation * 1e9,
            "area_um2": config.area * 1e12,
        }

    elif config.geometry == CasimirGeometry.SPHERICAL_SHELL:
        total = spherical_shell_energy(config.separation)
        volume = (4.0 / 3.0) * PI * config.separation**3
        rho = total / volume
        force = -total / config.separation
        pressure = force / (4 * PI * config.separation**2)
        details = {"radius_nm": config.separation * 1e9}

    elif config.geometry == CasimirGeometry.CYLINDRICAL:
        rho = cylindrical_energy_density(config.radius, config.length)
        volume = PI * config.radius**2 * config.length
        total = rho * volume
        force = total / config.length if config.length > 0 else 0.0
        pressure = rho / 3.0
        details = {
            "radius_nm": config.radius * 1e9,
            "length_um": config.length * 1e6,
        }

    elif config.geometry == CasimirGeometry.TOROIDAL:
        rho = toroidal_energy_density(config.radius, config.separation)
        volume = 2 * PI**2 * config.radius * config.separation**2
        total = rho * volume
        force = total / (2 * PI * config.radius)
        pressure = rho / 3.0
        details = {
            "major_radius_nm": config.radius * 1e9,
            "minor_radius_nm": config.separation * 1e9,
        }

    else:
        raise ValueError(f"Unknown geometry: {config.geometry}")

    # Planck ratio: how exotic is this?
    planck_ratio = abs(rho) / PLANCK_DENSITY

    # Gap in orders of magnitude vs typical wormhole requirement
    # Typical MT wormhole needs ~10^-3 to 10^-1 Planck density of exotic matter
    typical_requirement = 1e-3 * PLANCK_DENSITY
    gap_orders = math.log10(typical_requirement / max(abs(rho), 1e-300))

    result = CasimirResult(
        geometry=config.geometry.value,
        energy_density=rho,
        total_energy=total,
        force=force,
        pressure=pressure,
        is_negative=rho < 0,
        separation=config.separation,
        planck_ratio=planck_ratio,
        gap_orders=gap_orders,
        details=details,
    )

    logger.debug(
        f"Casimir result: rho={rho:.3e} J/m³, "
        f"gap={gap_orders:.1f} OOM from wormhole requirement"
    )

    return result


# ── Gap Analysis ────────────────────────────────────────────────────────────

def compute_energy_gap(
    wormhole_energy_requirement: float,
    casimir_config: Optional[CasimirConfig] = None,
) -> dict:
    """
    Compute the gap between current Casimir capability and
    what an exotic metric actually requires.

    This is the central scientific measurement of Gammo AGX --
    how close are we to making a wormhole energetically viable?

    Args:
        wormhole_energy_requirement: required exotic energy density (J/m³)
        casimir_config: Casimir configuration to compare against
                       (uses best-case parallel plate if None)

    Returns:
        dict with gap analysis
    """
    # Best achievable Casimir at ~1nm separation
    if casimir_config is None:
        casimir_config = CasimirConfig(
            geometry=CasimirGeometry.PARALLEL_PLATE,
            separation=1e-9,
            area=1e-4,
        )

    casimir_result = compute_casimir(casimir_config)
    casimir_density = casimir_result.energy_density

    if casimir_density == 0 or wormhole_energy_requirement == 0:
        return {
            "gap_orders_of_magnitude": 47.0,
            "casimir_density": casimir_density,
            "required_density": wormhole_energy_requirement,
            "achievability": "not_computed",
        }

    gap = math.log10(
        abs(wormhole_energy_requirement) / max(abs(casimir_density), 1e-300)
    )

    if gap <= 0:
        achievability = "achievable"
    elif gap <= 10:
        achievability = "near_term"
    elif gap <= 30:
        achievability = "long_term"
    else:
        achievability = "theoretical_only"

    return {
        "gap_orders_of_magnitude": round(gap, 2),
        "casimir_density":         casimir_density,
        "required_density":        wormhole_energy_requirement,
        "achievability":           achievability,
        "casimir_geometry":        casimir_config.geometry.value,
        "casimir_separation_m":    casimir_config.separation,
    }


# ── Scanning Utilities ──────────────────────────────────────────────────────

def scan_separation_range(
    sep_min: float = 1e-10,
    sep_max: float = 1e-6,
    n_points: int = 50,
    geometry: CasimirGeometry = CasimirGeometry.PARALLEL_PLATE,
) -> list[dict]:
    """
    Scan Casimir energy density across a range of separations.
    Useful for finding optimal configurations.

    Args:
        sep_min: minimum separation in meters
        sep_max: maximum separation in meters
        n_points: number of scan points
        geometry: Casimir geometry to use

    Returns:
        List of dicts with separation and energy density
    """
    results = []
    log_min = math.log10(sep_min)
    log_max = math.log10(sep_max)
    step = (log_max - log_min) / (n_points - 1)

    for i in range(n_points):
        sep = 10 ** (log_min + i * step)
        config = CasimirConfig(geometry=geometry, separation=sep)
        try:
            result = compute_casimir(config)
            results.append({
                "separation_m":  sep,
                "separation_nm": sep * 1e9,
                "energy_density": result.energy_density,
                "gap_orders":    result.gap_orders,
            })
        except Exception as e:
            logger.warning(f"Casimir scan failed at sep={sep:.2e}: {e}")

    return results


def find_optimal_separation(
    target_energy_density: float,
    geometry: CasimirGeometry = CasimirGeometry.PARALLEL_PLATE,
    tol: float = 0.1,
) -> Optional[float]:
    """
    Find the plate separation needed to achieve a target energy density.

    For parallel plates: d = (pi² hbar c / (720 |rho|))^(1/4)

    Args:
        target_energy_density: desired energy density in J/m³ (negative)
        geometry: Casimir geometry
        tol: tolerance for convergence

    Returns:
        Required separation in meters, or None if not achievable
    """
    if target_energy_density >= 0:
        return None

    if geometry == CasimirGeometry.PARALLEL_PLATE:
        # Analytic inversion of rho = -pi² hbar c / (720 d⁴)
        d = ((PI**2 * HBAR * C) / (720.0 * abs(target_energy_density))) ** 0.25
        return d

    # For other geometries, use binary search on scan results
    scan = scan_separation_range(n_points=200, geometry=geometry)
    for point in scan:
        if abs(point["energy_density"] - target_energy_density) / abs(target_energy_density) < tol:
            return point["separation_m"]

    return None
