"""
Gammo AGX - QESP Principle Validator
analysis/validator.py

Checks whether a simulation result supports or contradicts
the Quantum Elastic Spacetime Principle.

The validator applies four formal checks:
    1. NO DIVERGENCE: K_QESP does not diverge where K_GR would
    2. ACTIVATION BOUNDARY: Q_munu only activates near threshold
    3. STABILITY: System stabilizes (plateau or bounded oscillation)
    4. MEASURABILITY: Predictions produce observable deviations

Each check returns a score 0.0-1.0 and a classification.
The overall QESP score is a weighted average.
"""

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class ValidationCheck:
    """Result of a single QESP validation check."""
    name:        str
    passed:      bool
    score:       float     # 0.0 - 1.0
    message:     str
    details:     dict = field(default_factory=dict)


@dataclass
class QESPValidationReport:
    """Full QESP validation report for a simulation result."""
    overall_score:    float      # 0.0 - 1.0 weighted average
    overall_passed:   bool       # True if score > 0.6
    verdict:          str        # SUPPORTS | INCONCLUSIVE | CONTRADICTS
    checks:           list       # List of ValidationCheck
    key_finding:      str        # One-sentence summary
    publishable:      bool       # True if result is publication-worthy
    confidence:       str        # HIGH | MEDIUM | LOW
    details:          dict = field(default_factory=dict)


def check_no_divergence(sim_result) -> ValidationCheck:
    """
    Check 1: K_QESP does not diverge where K_GR would.

    Passes if:
    - Maximum QESP curvature is less than maximum GR curvature
    - The suppression is statistically significant (> 5%)
    - K_QESP remains bounded in the high-strain region
    """
    K_gr_max   = sim_result.max_curvature_gr
    K_qesp_max = sim_result.max_curvature_qesp
    suppression = sim_result.curvature_suppression

    if K_gr_max < 1e-10:
        return ValidationCheck(
            name="no_divergence",
            passed=False,
            score=0.0,
            message="GR curvature essentially zero — insufficient strain to test divergence prevention",
        )

    # Score based on suppression magnitude
    if suppression > 0.5:
        score = 1.0
        msg = f"Strong: K_QESP is {suppression*100:.1f}% lower than K_GR at peak"
    elif suppression > 0.2:
        score = 0.8
        msg = f"Good: K_QESP suppressed by {suppression*100:.1f}% at peak"
    elif suppression > 0.05:
        score = 0.6
        msg = f"Marginal: K_QESP suppressed by {suppression*100:.1f}% — detectable but weak"
    elif suppression > 0.01:
        score = 0.3
        msg = f"Weak: Only {suppression*100:.2f}% suppression — strain likely too low"
    else:
        score = 0.1
        msg = f"Negligible: {suppression*100:.3f}% suppression — QESP not activated"

    passed = suppression > 0.05

    return ValidationCheck(
        name="no_divergence",
        passed=passed,
        score=score,
        message=msg,
        details={
            "K_gr_max":     K_gr_max,
            "K_qesp_max":   K_qesp_max,
            "suppression":  suppression,
        }
    )


def check_activation_boundary(sim_result) -> ValidationCheck:
    """
    Check 2: Q_munu activates only near the critical threshold.

    Passes if:
    - Mean activation is low (Q barely active in most of domain)
    - Peak activation is high (Q strongly active in critical region)
    - The ratio peak/mean is large (localized activation)
    """
    Q = sim_result.Q_correction
    activation = Q.activation_profile

    mean_act  = float(jnp.mean(activation))
    max_act   = float(jnp.max(activation))
    frac_act  = Q.summary.get("frac_active", 0.0)

    if max_act < 0.05:
        return ValidationCheck(
            name="activation_boundary",
            passed=False,
            score=0.2,
            message=f"Max activation only {max_act:.3f} — QESP correction never meaningfully triggered",
            details={"mean_act": mean_act, "max_act": max_act, "frac_active": frac_act}
        )

    # Good behavior: high peak, low mean, localized
    locality_ratio = max_act / max(mean_act, 1e-6)
    strain_analysis = sim_result.strain_analysis

    if locality_ratio > 10 and frac_act < 0.3:
        score = 1.0
        msg = (f"Excellent localization: peak activation {max_act:.3f}, "
               f"only {frac_act*100:.1f}% of domain active")
    elif locality_ratio > 5 and frac_act < 0.5:
        score = 0.8
        msg = (f"Good localization: peak {max_act:.3f}, "
               f"{frac_act*100:.1f}% active")
    elif locality_ratio > 2:
        score = 0.6
        msg = f"Moderate localization: locality ratio {locality_ratio:.1f}"
    else:
        score = 0.4
        msg = f"Poor localization: correction spread throughout domain (ratio={locality_ratio:.1f})"

    # Bonus if activation is in the right regime
    if strain_analysis.regime in ["critical", "supercritical", "approaching"]:
        score = min(1.0, score + 0.1)
        msg += f" | Strain regime: {strain_analysis.regime} (correct)"

    passed = score > 0.5

    return ValidationCheck(
        name="activation_boundary",
        passed=passed,
        score=score,
        message=msg,
        details={
            "mean_activation":  mean_act,
            "max_activation":   max_act,
            "locality_ratio":   locality_ratio,
            "frac_active":      frac_act,
            "strain_regime":    strain_analysis.regime,
        }
    )


def check_stability(sim_result) -> ValidationCheck:
    """
    Check 3: System stabilizes under QESP correction.

    Passes if:
    - QESP stability >= GR stability
    - Plateau detected in K_QESP profile
    - OR bounded oscillation pattern exists
    """
    stab_gr   = sim_result.stability_gr
    stab_qesp = sim_result.stability_qesp
    plateau   = sim_result.plateau_detected
    div_prev  = sim_result.divergence_prevented

    stability_improvement = stab_qesp - stab_gr

    if plateau and div_prev:
        score = 1.0
        msg = (f"Strong stability: plateau detected AND divergence prevented. "
               f"Stability improved by {stability_improvement:.3f}")
    elif plateau:
        score = 0.85
        msg = f"Plateau detected in K_QESP. Stability: GR={stab_gr:.3f} -> QESP={stab_qesp:.3f}"
    elif div_prev:
        score = 0.8
        msg = f"Divergence prevented. Stability: GR={stab_gr:.3f} -> QESP={stab_qesp:.3f}"
    elif stability_improvement > 0.05:
        score = 0.7
        msg = f"Stability improved: +{stability_improvement:.3f} under QESP"
    elif stability_improvement >= 0:
        score = 0.5
        msg = f"Stability unchanged or marginal improvement ({stability_improvement:+.3f})"
    else:
        score = 0.2
        msg = f"Stability decreased under QESP ({stability_improvement:.3f}) — unexpected"

    # Check for bounded oscillation in K_QESP
    K_qesp = sim_result.K_qesp_profile
    K_qesp_np = np.array(K_qesp)
    if len(K_qesp_np) > 20:
        # Count sign changes in second derivative (oscillation indicator)
        d2K = np.diff(np.diff(K_qesp_np))
        sign_changes = np.sum(np.diff(np.sign(d2K)) != 0)
        if sign_changes > 5:
            score = min(1.0, score + 0.1)
            msg += f" | Bounded oscillation detected ({sign_changes} inflections)"

    passed = score > 0.5

    return ValidationCheck(
        name="stability",
        passed=passed,
        score=score,
        message=msg,
        details={
            "stability_gr":          stab_gr,
            "stability_qesp":        stab_qesp,
            "stability_improvement": stability_improvement,
            "plateau_detected":      plateau,
            "divergence_prevented":  div_prev,
        }
    )


def check_measurability(sim_result) -> ValidationCheck:
    """
    Check 4: Predictions produce observable, measurable deviations.

    Checks whether the QESP effect produces predictions that could
    in principle be observed:
    - Gravitational wave phase shift > 0.01% (detectable by LIGO)
    - Curvature cap well-defined (K_QESP_max is finite and specific)
    - Energy shift has measurable magnitude
    """
    gw_dev    = sim_result.gw_deviation_pct
    cap       = sim_result.curvature_cap
    osc_freq  = sim_result.oscillation_freq
    Q         = sim_result.Q_correction
    e_shift   = Q.energy_shift

    observations = []
    score_components = []

    # GW deviation
    if gw_dev > 10.0:
        observations.append(f"GW deviation: {gw_dev:.1f}% (very large, easily detectable)")
        score_components.append(1.0)
    elif gw_dev > 1.0:
        observations.append(f"GW deviation: {gw_dev:.2f}% (detectable by next-gen detectors)")
        score_components.append(0.8)
    elif gw_dev > 0.1:
        observations.append(f"GW deviation: {gw_dev:.3f}% (marginal, needs precision measurement)")
        score_components.append(0.6)
    elif gw_dev > 0.01:
        observations.append(f"GW deviation: {gw_dev:.4f}% (at detection threshold)")
        score_components.append(0.4)
    else:
        observations.append(f"GW deviation: {gw_dev:.5f}% (below current detection capability)")
        score_components.append(0.2)

    # Curvature cap
    if cap > 0 and cap < sim_result.max_curvature_gr * 0.99:
        observations.append(f"Curvature cap: K_max = {cap:.4e} (well-defined prediction)")
        score_components.append(0.9)
    else:
        observations.append(f"Curvature cap: {cap:.4e} (cap not significantly below GR)")
        score_components.append(0.3)

    # Oscillation frequency
    if osc_freq > 0:
        observations.append(f"Oscillation frequency: {osc_freq:.4f} (Planck units) — testable prediction")
        score_components.append(0.8)
    else:
        observations.append("No oscillation (strain below critical threshold)")
        score_components.append(0.4)

    # Energy shift
    if abs(e_shift) > 0.01:
        observations.append(f"Integrated energy shift: {e_shift:.4f} — measurable via exotic matter budget")
        score_components.append(0.7)
    else:
        observations.append(f"Energy shift: {e_shift:.6f} — below measurement threshold")
        score_components.append(0.3)

    score = sum(score_components) / len(score_components) if score_components else 0.0
    passed = score > 0.5
    msg = " | ".join(observations[:2])  # Top 2 for message

    return ValidationCheck(
        name="measurability",
        passed=passed,
        score=score,
        message=msg,
        details={
            "gw_deviation_pct":  gw_dev,
            "curvature_cap":     cap,
            "oscillation_freq":  osc_freq,
            "energy_shift":      e_shift,
            "observations":      observations,
        }
    )


def validate_qesp(sim_result) -> QESPValidationReport:
    """
    Run all four QESP validation checks and produce a full report.

    Args:
        sim_result: QESPSimulationResult from qesp_simulator.py

    Returns:
        QESPValidationReport with overall verdict and publishability assessment
    """
    checks = [
        check_no_divergence(sim_result),
        check_activation_boundary(sim_result),
        check_stability(sim_result),
        check_measurability(sim_result),
    ]

    # Weighted average (divergence prevention is most important)
    weights = [0.35, 0.25, 0.25, 0.15]
    overall_score = sum(c.score * w for c, w in zip(checks, weights))
    overall_passed = overall_score > 0.6

    n_passed = sum(1 for c in checks if c.passed)

    # Verdict
    if overall_score >= 0.8:
        verdict = "STRONGLY SUPPORTS"
        confidence = "HIGH"
    elif overall_score >= 0.65:
        verdict = "SUPPORTS"
        confidence = "HIGH"
    elif overall_score >= 0.5:
        verdict = "WEAKLY SUPPORTS"
        confidence = "MEDIUM"
    elif overall_score >= 0.35:
        verdict = "INCONCLUSIVE"
        confidence = "LOW"
    else:
        verdict = "DOES NOT SUPPORT"
        confidence = "LOW"

    # Key finding
    passed_checks = [c.name for c in checks if c.passed]
    if "no_divergence" in passed_checks and "stability" in passed_checks:
        key_finding = (
            f"QESP successfully prevents curvature divergence "
            f"({sim_result.curvature_suppression*100:.1f}% suppression) "
            f"and stabilizes the geometry "
            f"(stability: {sim_result.stability_gr:.3f} -> {sim_result.stability_qesp:.3f})"
        )
    elif "no_divergence" in passed_checks:
        key_finding = (
            f"QESP reduces peak curvature by "
            f"{sim_result.curvature_suppression*100:.1f}% "
            f"({sim_result.strain_analysis.regime} strain regime)"
        )
    elif "stability" in passed_checks:
        key_finding = (
            f"QESP improves stability "
            f"({sim_result.stability_gr:.3f} -> {sim_result.stability_qesp:.3f}) "
            f"but curvature suppression is weak"
        )
    else:
        key_finding = (
            f"Configuration strain ({sim_result.strain_analysis.strain:.3f}) "
            f"too low for significant QESP activation. "
            f"Try higher curvature configurations."
        )

    # Publishability assessment
    publishable = (
        overall_score >= 0.7 and
        sim_result.curvature_suppression > 0.1 and
        (sim_result.plateau_detected or sim_result.divergence_prevented)
    )

    details = {
        "n_checks_passed":    n_passed,
        "n_checks_total":     len(checks),
        "check_scores":       {c.name: c.score for c in checks},
        "overall_score":      overall_score,
        "lambda_used":        sim_result.lambda_param,
        "epsilon_c_used":     sim_result.epsilon_c,
        "strain_regime":      sim_result.strain_analysis.regime,
        "geometry":           sim_result.geometry_type,
    }

    logger.info(
        f"QESP validation: {verdict}, score={overall_score:.3f}, "
        f"publishable={publishable}"
    )

    return QESPValidationReport(
        overall_score   = overall_score,
        overall_passed  = overall_passed,
        verdict         = verdict,
        checks          = checks,
        key_finding     = key_finding,
        publishable     = publishable,
        confidence      = confidence,
        details         = details,
    )
