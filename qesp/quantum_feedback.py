"""
Gammo AGX - QESP Quantum Feedback Engine
qesp/quantum_feedback.py

Implements the quantum correction tensor Q_munu that activates
near the critical curvature limit — the mathematical heart of QESP.

Modified Einstein equations under QESP:
    G_munu + Q_munu(epsilon) = 8*pi*T_munu

where:
    Q_munu = lambda * A(epsilon) * g_munu

A(epsilon) is the activation function — smooth, zero below threshold,
maximum near epsilon = 1.

Physical interpretation:
    - Q_munu acts as a quantum pressure term
    - It opposes further curvature increase near the limit
    - It is proportional to g_munu (isotropic quantum pressure)
    - lambda sets the strength (tied to Planck scale)
    - The activation function A(epsilon) ensures it only matters
      near the critical curvature — not in ordinary spacetime

Activation function options:
    1. SIGMOID  — smooth logistic: A = 1 / (1 + exp(-k*(epsilon - epsilon_c)))
    2. TANH     — symmetric variant: A = 0.5 * (1 + tanh(k*(epsilon - epsilon_c)))
    3. POWER    — A = max(0, epsilon - epsilon_c)^n / (1 + max(0,...))
    4. GAUSSIAN — A = exp(-(epsilon - 1)^2 / (2*sigma^2)) near epsilon=1

All options satisfy:
    A(epsilon << epsilon_c) ~ 0    (no correction in normal spacetime)
    A(epsilon >= 1) ~ 1            (maximum correction at critical limit)

Reference: Ford & Roman (1995) quantum inequalities for comparison
           Ashtekar & Bojowald (2006) LQC quantum bounce for context
"""

import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


# Default QESP parameters
LAMBDA_DEFAULT    = 1.0    # Quantum correction strength (Planck units)
EPSILON_C_DEFAULT = 0.85   # Activation threshold
K_STEEP_DEFAULT   = 20.0   # Steepness of sigmoid transition


class ActivationMode(Enum):
    SIGMOID  = "sigmoid"
    TANH     = "tanh"
    POWER    = "power"
    GAUSSIAN = "gaussian"


@dataclass
class QmunuResult:
    """Result of Q_munu computation at a point or over a grid."""
    Q_scalar:        float         # Scalar magnitude of correction
    activation:      float         # A(epsilon) value [0, 1]
    epsilon:         float         # Strain at this point
    lambda_eff:      float         # Effective lambda used
    correction_active: bool        # True if correction meaningfully active
    mode:            str           # Activation function used
    details:         dict = field(default_factory=dict)


@dataclass
class QESPCorrection:
    """Full QESP correction for a simulation result."""
    Q_profile:       object        # Array Q(r) over radial grid
    activation_profile: object     # A(epsilon(r)) over radial grid
    strain_profile:  object        # epsilon(r) over radial grid
    max_correction:  float         # Maximum |Q| in domain
    mean_correction: float         # Mean |Q| in domain
    correction_region: object      # Boolean mask where A > 0.1
    energy_shift:    float         # Integrated energy correction
    peak_location:   float         # r where Q is maximum
    summary:         dict = field(default_factory=dict)


# ── Activation Functions ───────────────────────────────────────────────────

@jit
def sigmoid_activation(epsilon, epsilon_c=EPSILON_C_DEFAULT, k=K_STEEP_DEFAULT):
    """
    Sigmoid (logistic) activation function.

    A(epsilon) = 1 / (1 + exp(-k * (epsilon - epsilon_c)))

    Properties:
        A(epsilon << epsilon_c) -> 0
        A(epsilon_c) = 0.5
        A(epsilon >> epsilon_c) -> 1

    This is the primary recommended activation function for QESP.
    The sigmoid is differentiable everywhere and has a clear
    physical interpretation: nothing happens until epsilon
    approaches epsilon_c, then quantum correction smoothly
    activates over a transition region of width ~4/k.

    Args:
        epsilon:   strain value
        epsilon_c: activation threshold
        k:         steepness (higher = sharper transition)

    Returns:
        Activation value in [0, 1]
    """
    return 1.0 / (1.0 + jnp.exp(-k * (epsilon - epsilon_c)))


@jit
def tanh_activation(epsilon, epsilon_c=EPSILON_C_DEFAULT, k=K_STEEP_DEFAULT):
    """
    Hyperbolic tangent activation.

    A(epsilon) = 0.5 * (1 + tanh(k * (epsilon - epsilon_c)))

    Mathematically equivalent to sigmoid but sometimes numerically
    better behaved in JAX.
    """
    return 0.5 * (1.0 + jnp.tanh(k * (epsilon - epsilon_c)))


@jit
def power_activation(epsilon, epsilon_c=EPSILON_C_DEFAULT, n=3.0):
    """
    Power-law activation.

    A(epsilon) = max(0, epsilon - epsilon_c)^n / (1 + max(0, epsilon - epsilon_c)^n)

    Properties:
        A = 0 below threshold (exact, not approximate)
        Smooth power-law onset above threshold
        Bounded by 1 at large epsilon

    Physically motivated by analogy with Landau-Ginzburg
    order parameter transitions.
    """
    delta = jnp.maximum(epsilon - epsilon_c, 0.0)
    delta_n = delta ** n
    return delta_n / (1.0 + delta_n)


@jit
def gaussian_activation(epsilon, epsilon_c=1.0, sigma_g=0.15):
    """
    Gaussian activation centered at epsilon = 1 (critical limit).

    A(epsilon) = exp(-(epsilon - 1)^2 / (2 * sigma_g^2))

    Properties:
        Maximum exactly at epsilon = 1
        Falls off symmetrically on both sides
        Models a resonant quantum response at the critical limit

    This is appropriate if the quantum effect is strongest
    AT the critical limit rather than above it.
    """
    return jnp.exp(-(epsilon - epsilon_c) ** 2 / (2.0 * sigma_g ** 2))


def get_activation_fn(mode: ActivationMode):
    """Return the JAX-jitted activation function for a given mode."""
    mapping = {
        ActivationMode.SIGMOID:  sigmoid_activation,
        ActivationMode.TANH:     tanh_activation,
        ActivationMode.POWER:    power_activation,
        ActivationMode.GAUSSIAN: gaussian_activation,
    }
    return mapping[mode]


# ── Q_munu Computation ─────────────────────────────────────────────────────

@jit
def Q_scalar_from_strain(
    epsilon:   float,
    lam:       float = LAMBDA_DEFAULT,
    epsilon_c: float = EPSILON_C_DEFAULT,
    k_steep:   float = K_STEEP_DEFAULT,
):
    """
    Compute the scalar magnitude of Q_munu at a given strain.

    Q_scalar = lambda * A(epsilon)

    This is the trace of the correction tensor divided by 4
    (in 4D spacetime with g_munu of signature -+++ and trace = -2).

    The full tensor Q_munu = Q_scalar * g_munu where g_munu
    is the metric tensor at that point.

    Physical meaning:
        Q_scalar = 0    -> no quantum correction (normal spacetime)
        Q_scalar = 0.5*lambda  -> half-activated (epsilon = epsilon_c)
        Q_scalar = lambda      -> fully activated (epsilon >> epsilon_c)

    Args:
        epsilon:   strain at this point
        lam:       quantum correction strength (lambda parameter)
        epsilon_c: activation threshold
        k_steep:   sigmoid steepness

    Returns:
        Q_scalar value
    """
    A = sigmoid_activation(epsilon, epsilon_c, k_steep)
    return lam * A


def compute_Q_profile(
    strain_profile: jnp.ndarray,
    r_grid:         jnp.ndarray,
    lam:            float = LAMBDA_DEFAULT,
    epsilon_c:      float = EPSILON_C_DEFAULT,
    k_steep:        float = K_STEEP_DEFAULT,
    mode:           ActivationMode = ActivationMode.SIGMOID,
) -> QESPCorrection:
    """
    Compute full Q_munu correction profile over a radial grid.

    This is the main function called by the QESP simulator.
    It takes a strain profile epsilon(r) and returns the
    quantum correction Q(r) at every point.

    Args:
        strain_profile: epsilon(r) array from strain.py
        r_grid:         radial coordinate array
        lam:            quantum correction strength
        epsilon_c:      activation threshold
        k_steep:        sigmoid steepness
        mode:           activation function to use

    Returns:
        QESPCorrection with full profiles and summary statistics
    """
    act_fn = get_activation_fn(mode)

    # Compute activation A(epsilon) at every point
    activation_profile = vmap(
        lambda eps: act_fn(eps, epsilon_c, k_steep)
    )(strain_profile)

    # Q(r) = lambda * A(epsilon(r))
    Q_profile = lam * activation_profile

    # Summary statistics
    max_Q    = float(jnp.max(Q_profile))
    mean_Q   = float(jnp.mean(Q_profile))
    peak_idx = int(jnp.argmax(Q_profile))
    r_peak   = float(r_grid[peak_idx])

    # Region where correction is meaningfully active (A > 0.1)
    correction_region = activation_profile > 0.1
    n_active = int(jnp.sum(correction_region))

    # Integrated energy correction (trapezoidal approximation)
    if len(r_grid) > 1:
        dr = float(r_grid[1] - r_grid[0])
        energy_shift = float(jnp.sum(Q_profile) * dr)
    else:
        energy_shift = 0.0

    summary = {
        "max_correction":    max_Q,
        "mean_correction":   mean_Q,
        "peak_at_r":         r_peak,
        "n_active_points":   n_active,
        "frac_active":       n_active / max(len(r_grid), 1),
        "energy_shift":      energy_shift,
        "lambda":            lam,
        "epsilon_c":         epsilon_c,
        "activation_mode":   mode.value,
    }

    return QESPCorrection(
        Q_profile          = Q_profile,
        activation_profile = activation_profile,
        strain_profile     = strain_profile,
        max_correction     = max_Q,
        mean_correction    = mean_Q,
        correction_region  = correction_region,
        energy_shift       = energy_shift,
        peak_location      = r_peak,
        summary            = summary,
    )


# ── Modified Curvature ─────────────────────────────────────────────────────

@jit
def modified_kretschmann(
    K_gr:      float,
    epsilon:   float,
    lam:       float = LAMBDA_DEFAULT,
    epsilon_c: float = EPSILON_C_DEFAULT,
    k_steep:   float = K_STEEP_DEFAULT,
):
    """
    Compute QESP-modified Kretschmann scalar.

    In QESP, the quantum correction Q_munu modifies the effective
    curvature as seen by the field equations. The correction acts
    as a negative feedback on K:

    K_QESP(r) = K_GR(r) / (1 + Q_scalar(epsilon))

    Physical interpretation:
        - At low strain: K_QESP = K_GR (no change)
        - At critical strain: K_QESP = K_GR / (1 + 0.5*lambda)
        - At high strain: K_QESP -> K_GR / (1 + lambda) [plateaus]

    This is the key QESP prediction: curvature plateaus instead
    of diverging at the critical limit.

    Args:
        K_gr:      GR Kretschmann scalar
        epsilon:   strain = K_gr / K_crit
        lam:       correction strength
        epsilon_c: activation threshold
        k_steep:   sigmoid steepness

    Returns:
        QESP-modified Kretschmann scalar
    """
    Q = Q_scalar_from_strain(epsilon, lam, epsilon_c, k_steep)
    return K_gr / (1.0 + Q)


def compute_modified_curvature_profile(
    K_gr_profile:   jnp.ndarray,
    strain_profile: jnp.ndarray,
    lam:            float = LAMBDA_DEFAULT,
    epsilon_c:      float = EPSILON_C_DEFAULT,
    k_steep:        float = K_STEEP_DEFAULT,
) -> jnp.ndarray:
    """
    Compute QESP-modified curvature profile over a radial grid.

    Args:
        K_gr_profile:   GR Kretschmann K(r) array
        strain_profile: epsilon(r) array
        lam:            correction strength
        epsilon_c:      activation threshold
        k_steep:        sigmoid steepness

    Returns:
        K_QESP(r) array
    """
    return vmap(
        lambda K, eps: modified_kretschmann(K, eps, lam, epsilon_c, k_steep)
    )(K_gr_profile, strain_profile)


# ── Parameter Space Scan ───────────────────────────────────────────────────

def scan_lambda_sensitivity(
    strain_max: float,
    lambda_values: list = None,
    epsilon_c: float = EPSILON_C_DEFAULT,
) -> list:
    """
    Scan how the maximum curvature suppression varies with lambda.

    This is used for calibrating the QESP parameter and generating
    the lambda sensitivity analysis in the prediction output.

    Args:
        strain_max:    maximum strain in the configuration
        lambda_values: list of lambda values to test
        epsilon_c:     activation threshold

    Returns:
        List of dicts with lambda, Q_max, K_suppression_factor
    """
    if lambda_values is None:
        lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = []
    for lam in lambda_values:
        A_max = float(sigmoid_activation(
            jnp.array(strain_max), epsilon_c, K_STEEP_DEFAULT))
        Q_max = lam * A_max
        suppression = 1.0 / (1.0 + Q_max)
        results.append({
            "lambda":              lam,
            "activation_at_max":   A_max,
            "Q_max":               Q_max,
            "suppression_factor":  suppression,
            "K_QESP_over_K_GR":   suppression,
            "curvature_reduction_pct": (1.0 - suppression) * 100.0,
        })

    return results
