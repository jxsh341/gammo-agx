"""
Gammo AGX - QESP Test Suite
tests/test_qesp.py

Run with: python -m pytest tests/test_qesp.py -v
OR:        python tests/test_qesp.py

Tests all four QESP components:
    1. Strain engine
    2. Quantum feedback
    3. QESP simulator
    4. Validator + predictions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp
import math


# ── Color helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed_count = 0
failed_count = 0
test_results = []

def test(name, condition, detail=""):
    global passed_count, failed_count
    if condition:
        passed_count += 1
        status = f"{GREEN}PASS{RESET}"
        test_results.append(("PASS", name))
    else:
        failed_count += 1
        status = f"{RED}FAIL{RESET}"
        test_results.append(("FAIL", name))
    detail_str = f" ({detail})" if detail else ""
    print(f"  {status}  {name}{detail_str}")


def section(title):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: STRAIN ENGINE
# ══════════════════════════════════════════════════════════════════════════
section("1. STRAIN ENGINE (qesp/strain.py)")

from qesp.strain import (
    kretschmann_morris_thorne as kretschmann_scalar_morris_thorne,
    kretschmann_alcubierre as kretschmann_scalar_alcubierre,
    compute_strain_morris_thorne,
    compute_strain_alcubierre,
    analyze_strain,
    strain_at_point,
    K_PLANCK,
    EPSILON_CRITICAL,
)

# Test 1.1: MT Kretschmann scalar is positive
r_test = jnp.array(1.5)
K_mt = float(kretschmann_scalar_morris_thorne(r_test, 1.0, 0.5, 0.2))
test("MT Kretschmann scalar is positive", K_mt > 0, f"K={K_mt:.4e}")

# Test 1.2: MT Kretschmann increases toward throat
K_far   = float(kretschmann_scalar_morris_thorne(jnp.array(5.0), 1.0, 0.5, 0.2))
K_near  = float(kretschmann_scalar_morris_thorne(jnp.array(1.1), 1.0, 0.5, 0.2))
test("MT Kretschmann higher near throat", K_near > K_far,
     f"K_near={K_near:.4e}, K_far={K_far:.4e}")

# Test 1.3: Alcubierre Kretschmann peaks at bubble wall
K_center = float(kretschmann_scalar_alcubierre(jnp.array(0.1), 1.0, 1.0, 0.5))
K_wall   = float(kretschmann_scalar_alcubierre(jnp.array(1.0), 1.0, 1.0, 0.5))
K_far2   = float(kretschmann_scalar_alcubierre(jnp.array(3.0), 1.0, 1.0, 0.5))
test("Alcubierre Kretschmann peaks at bubble wall",
     K_wall >= K_center and K_wall >= K_far2,
     f"center={K_center:.4e}, wall={K_wall:.4e}, far={K_far2:.4e}")

# Test 1.4: Strain profile has correct shape
r_grid = jnp.linspace(1.001, 5.0, 100)
strain_mt = compute_strain_morris_thorne(r_grid, 1.0, 0.5, 0.2)
test("MT strain profile has 100 points", len(strain_mt) == 100, f"len={len(strain_mt)}")
test("MT strain all non-negative", float(jnp.min(strain_mt)) >= 0,
     f"min={float(jnp.min(strain_mt)):.4e}")

# Test 1.5: Strain analysis returns correct regime
strain_result = analyze_strain(strain_mt, r_grid)
test("Strain analysis regime is string", isinstance(strain_result.regime, str),
     f"regime={strain_result.regime}")
test("Strain result has all fields",
     hasattr(strain_result, 'kretschmann') and
     hasattr(strain_result, 'strain') and
     hasattr(strain_result, 'regime'),
     "fields: kretschmann, strain, regime")

# Test 1.6: strain_at_point convenience function
eps = strain_at_point(1.5, "morris_thorne",
                      {"throat_radius": 1.0, "exotic_density": 0.5, "redshift_factor": 0.2})
test("strain_at_point returns float", isinstance(eps, float), f"eps={eps:.4e}")
test("strain_at_point non-negative", eps >= 0, f"eps={eps:.4e}")

# Test 1.7: Higher warp speed -> higher Alcubierre curvature
K_slow = float(kretschmann_scalar_alcubierre(jnp.array(1.0), 0.5, 1.0, 0.5))
K_fast = float(kretschmann_scalar_alcubierre(jnp.array(1.0), 2.0, 1.0, 0.5))
test("Higher warp speed produces higher curvature", K_fast > K_slow,
     f"v=0.5: {K_slow:.4e}, v=2.0: {K_fast:.4e}")

# Test 1.8: Planck constant defined
test("K_PLANCK is positive", K_PLANCK > 0, f"K_PLANCK={K_PLANCK}")
test("EPSILON_CRITICAL in (0,1)", 0 < EPSILON_CRITICAL < 1,
     f"eps_c={EPSILON_CRITICAL}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: QUANTUM FEEDBACK ENGINE
# ══════════════════════════════════════════════════════════════════════════
section("2. QUANTUM FEEDBACK ENGINE (qesp/quantum_feedback.py)")

from qesp.quantum_feedback import (
    sigmoid_activation,
    tanh_activation,
    power_activation,
    gaussian_activation,
    Q_scalar_from_strain,
    compute_Q_profile,
    compute_modified_curvature_profile,
    scan_lambda_sensitivity,
    ActivationMode,
    LAMBDA_DEFAULT,
    EPSILON_C_DEFAULT,
)

# Test 2.1: Sigmoid activation function properties
A_low  = float(sigmoid_activation(jnp.array(0.0)))
A_mid  = float(sigmoid_activation(jnp.array(EPSILON_C_DEFAULT)))
A_high = float(sigmoid_activation(jnp.array(2.0)))
test("Sigmoid: low strain -> near 0", A_low < 0.05, f"A(0)={A_low:.4f}")
test("Sigmoid: at threshold -> ~0.5", 0.4 < A_mid < 0.6, f"A(eps_c)={A_mid:.4f}")
test("Sigmoid: high strain -> near 1", A_high > 0.95, f"A(2.0)={A_high:.4f}")

# Test 2.2: Monotonically increasing
epsilons = jnp.linspace(0.0, 2.0, 50)
activations = [float(sigmoid_activation(e)) for e in epsilons]
is_monotone = all(activations[i] <= activations[i+1] for i in range(len(activations)-1))
test("Sigmoid is monotonically increasing", is_monotone)

# Test 2.3: Tanh activation equivalent to sigmoid
A_tanh_low  = float(tanh_activation(jnp.array(0.0)))
A_tanh_high = float(tanh_activation(jnp.array(2.0)))
test("Tanh: low -> near 0", A_tanh_low < 0.05, f"A={A_tanh_low:.4f}")
test("Tanh: high -> near 1", A_tanh_high > 0.95, f"A={A_tanh_high:.4f}")

# Test 2.4: Power activation is exactly 0 below threshold
A_power_below = float(power_activation(jnp.array(EPSILON_C_DEFAULT - 0.1)))
A_power_above = float(power_activation(jnp.array(EPSILON_C_DEFAULT + 0.2)))
test("Power: exactly 0 below threshold", A_power_below == 0.0,
     f"A(eps_c-0.1)={A_power_below}")
test("Power: positive above threshold", A_power_above > 0,
     f"A(eps_c+0.2)={A_power_above:.4f}")

# Test 2.5: Q_scalar properties
Q_zero   = float(Q_scalar_from_strain(jnp.array(0.0)))
Q_crit   = float(Q_scalar_from_strain(jnp.array(EPSILON_C_DEFAULT)))
Q_max    = float(Q_scalar_from_strain(jnp.array(5.0)))
test("Q_scalar near 0 at low strain", Q_zero < 0.1 * LAMBDA_DEFAULT,
     f"Q(0)={Q_zero:.4f}")
test("Q_scalar near lambda/2 at threshold",
     0.3 * LAMBDA_DEFAULT < Q_crit < 0.7 * LAMBDA_DEFAULT,
     f"Q(eps_c)={Q_crit:.4f}")
test("Q_scalar near lambda at high strain",
     Q_max > 0.9 * LAMBDA_DEFAULT,
     f"Q(5.0)={Q_max:.4f}, lambda={LAMBDA_DEFAULT}")

# Test 2.6: Compute Q profile
r_grid = jnp.linspace(1.001, 5.0, 100)
strain_mt = compute_strain_morris_thorne(r_grid, 1.0, 0.5, 0.2)
Q_corr = compute_Q_profile(strain_mt, r_grid)
test("Q profile has correct length", len(Q_corr.Q_profile) == 100,
     f"len={len(Q_corr.Q_profile)}")
test("Q profile all non-negative", float(jnp.min(Q_corr.Q_profile)) >= 0,
     f"min={float(jnp.min(Q_corr.Q_profile)):.4e}")
test("Q correction has summary dict", isinstance(Q_corr.summary, dict))
test("Q correction has energy_shift", hasattr(Q_corr, 'energy_shift'))

# Test 2.7: Modified curvature <= GR curvature
K_gr_profile = strain_mt * K_PLANCK
K_qesp_profile = compute_modified_curvature_profile(K_gr_profile, strain_mt)
test("QESP curvature <= GR curvature everywhere",
     float(jnp.all(K_qesp_profile <= K_gr_profile + 1e-10)),
     f"max K_QESP={float(jnp.max(K_qesp_profile)):.4e}, max K_GR={float(jnp.max(K_gr_profile)):.4e}")

# Test 2.8: Lambda sensitivity scan
scan = scan_lambda_sensitivity(0.9)
test("Lambda scan returns list", isinstance(scan, list))
test("Lambda scan has 6 entries", len(scan) == 6, f"len={len(scan)}")
test("Lambda scan suppression increases with lambda",
     scan[-1]["curvature_reduction_pct"] > scan[0]["curvature_reduction_pct"],
     f"min={scan[0]['curvature_reduction_pct']:.1f}%, max={scan[-1]['curvature_reduction_pct']:.1f}%")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: QESP SIMULATOR
# ══════════════════════════════════════════════════════════════════════════
section("3. QESP SIMULATOR (qesp/qesp_simulator.py)")

from qesp.qesp_simulator import simulate_qesp

# Test 3.1: Morris-Thorne simulation completes
print(f"  {YELLOW}Running MT QESP simulation...{RESET}")
mt_params = {
    "throat_radius":   1.0,
    "exotic_density":  0.5,
    "tidal_force":     0.3,
    "redshift_factor": 0.2,
}
mt_result = simulate_qesp("morris_thorne", mt_params, n_radial=100)
test("MT simulation completes", mt_result is not None)
test("MT has r_grid", len(mt_result.r_grid) == 100, f"len={len(mt_result.r_grid)}")
test("MT K_GR profile non-negative",
     float(jnp.min(mt_result.K_gr_profile)) >= 0)
test("MT K_QESP <= K_GR everywhere",
     float(jnp.all(mt_result.K_qesp_profile <= mt_result.K_gr_profile + 1e-8)),
     f"max ratio={float(jnp.max(mt_result.K_qesp_profile / jnp.maximum(mt_result.K_gr_profile, 1e-30))):.4f}")
test("MT stability_qesp >= stability_gr",
     mt_result.stability_qesp >= mt_result.stability_gr - 1e-6,
     f"GR={mt_result.stability_gr:.3f}, QESP={mt_result.stability_qesp:.3f}")
test("MT curvature_suppression in [0,1]",
     0.0 <= mt_result.curvature_suppression <= 1.0,
     f"suppression={mt_result.curvature_suppression:.4f}")
test("MT has validation_notes list",
     isinstance(mt_result.validation_notes, list))
test("MT has lambda_scan",
     isinstance(mt_result.lambda_scan, list) and len(mt_result.lambda_scan) > 0)

# Test 3.2: Alcubierre simulation completes
print(f"  {YELLOW}Running Alcubierre QESP simulation...{RESET}")
alc_params = {
    "warp_speed":     1.0,
    "bubble_radius":  1.0,
    "wall_thickness": 0.5,
    "energy_density": 0.5,
}
alc_result = simulate_qesp("alcubierre", alc_params, n_radial=100)
test("Alcubierre simulation completes", alc_result is not None)
test("Alcubierre K_QESP <= K_GR",
     float(jnp.all(alc_result.K_qesp_profile <= alc_result.K_gr_profile + 1e-8)),
     f"max K_GR={alc_result.max_curvature_gr:.4e}, max K_QESP={alc_result.max_curvature_qesp:.4e}")
test("Alcubierre has curvature_cap > 0",
     alc_result.curvature_cap > 0,
     f"cap={alc_result.curvature_cap:.4e}")

# Test 3.3: Higher lambda -> more suppression
mt_lo = simulate_qesp("morris_thorne", mt_params, lambda_param=0.1, n_radial=50)
mt_hi = simulate_qesp("morris_thorne", mt_params, lambda_param=5.0, n_radial=50)
test("Higher lambda -> more curvature suppression",
     mt_hi.curvature_suppression >= mt_lo.curvature_suppression,
     f"lambda=0.1: {mt_lo.curvature_suppression:.4f}, lambda=5.0: {mt_hi.curvature_suppression:.4f}")

# Test 3.4: Higher warp speed -> more suppression (more curvature)
alc_slow = simulate_qesp("alcubierre",
    {"warp_speed": 0.5, "bubble_radius": 1.0, "wall_thickness": 0.5, "energy_density": 0.5},
    n_radial=50)
alc_fast = simulate_qesp("alcubierre",
    {"warp_speed": 3.0, "bubble_radius": 1.0, "wall_thickness": 0.5, "energy_density": 0.5},
    n_radial=50)
test("Higher warp speed -> more curvature in Alcubierre",
     alc_fast.max_curvature_gr >= alc_slow.max_curvature_gr,
     f"v=0.5: {alc_slow.max_curvature_gr:.4e}, v=3.0: {alc_fast.max_curvature_gr:.4e}")

# Test 3.5: Summary dict has required keys
required_keys = ["max_strain", "strain_regime", "max_K_gr", "max_K_qesp",
                 "curvature_suppression", "stability_gr", "stability_qesp",
                 "qesp_validates"]
for key in required_keys:
    test(f"Summary has key: {key}", key in mt_result.summary, f"key={key}")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: VALIDATOR + PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════
section("4. VALIDATOR + PREDICTIONS")

from analysis.validator import validate_qesp
from analysis.predictions import generate_predictions

# Test 4.1: Validation report structure
print(f"  {YELLOW}Running validation...{RESET}")
report = validate_qesp(mt_result)
test("Report has overall_score", hasattr(report, 'overall_score'),
     f"score={report.overall_score:.3f}")
test("Report score in [0,1]",
     0.0 <= report.overall_score <= 1.0,
     f"score={report.overall_score:.3f}")
test("Report has verdict string",
     isinstance(report.verdict, str) and len(report.verdict) > 0,
     f"verdict={report.verdict}")
test("Report has 4 checks", len(report.checks) == 4,
     f"n_checks={len(report.checks)}")
test("Report has key_finding",
     isinstance(report.key_finding, str) and len(report.key_finding) > 10)
test("Report has publishable flag",
     isinstance(report.publishable, bool))
test("Report has confidence",
     report.confidence in ["HIGH", "MEDIUM", "LOW"],
     f"confidence={report.confidence}")

# Test 4.2: Each check has required fields
for check in report.checks:
    test(f"Check '{check.name}' has score in [0,1]",
         0.0 <= check.score <= 1.0,
         f"score={check.score:.3f}")
    test(f"Check '{check.name}' has message",
         isinstance(check.message, str) and len(check.message) > 5)

# Test 4.3: Predictions structure
print(f"  {YELLOW}Generating predictions...{RESET}")
pred_set = generate_predictions(mt_result, report)
test("PredictionSet has predictions list",
     isinstance(pred_set.predictions, list) and len(pred_set.predictions) > 0,
     f"n={len(pred_set.predictions)}")
test("PredictionSet has paper_abstract",
     isinstance(pred_set.paper_abstract, str) and len(pred_set.paper_abstract) > 100)
test("PredictionSet has strongest prediction",
     pred_set.strongest is not None)
test("All predictions have deviation_pct",
     all(hasattr(p, 'deviation_pct') for p in pred_set.predictions))
test("All predictions have observatory",
     all(hasattr(p, 'observatory') for p in pred_set.predictions))
test("All predictions have type A, B, or C",
     all(p.type in ['A', 'B', 'C'] for p in pred_set.predictions))

# Test 4.4: Alcubierre validation
alc_report = validate_qesp(alc_result)
test("Alcubierre validation completes",
     alc_report is not None and hasattr(alc_report, 'verdict'))
test("Alcubierre has verdict",
     isinstance(alc_report.verdict, str),
     f"verdict={alc_report.verdict}")

# Test 4.5: Physical consistency checks
# Curvature suppression must be non-negative
test("Curvature suppression non-negative",
     mt_result.curvature_suppression >= 0,
     f"suppression={mt_result.curvature_suppression:.4f}")
# Stability must not decrease
test("QESP stability never decreases",
     mt_result.stability_qesp >= mt_result.stability_gr - 1e-6,
     f"GR={mt_result.stability_gr:.3f}, QESP={mt_result.stability_qesp:.3f}")
# Curvature cap must be finite
test("Curvature cap is finite",
     math.isfinite(mt_result.curvature_cap),
     f"cap={mt_result.curvature_cap:.4e}")
# GW deviation must be non-negative
test("GW deviation non-negative",
     mt_result.gw_deviation_pct >= 0,
     f"dev={mt_result.gw_deviation_pct:.4f}%")


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: INTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════════
section("5. INTEGRATION TEST (full pipeline)")

print(f"  {YELLOW}Running full QESP pipeline...{RESET}")

# High-curvature configuration to maximize QESP effect
high_curv_params = {
    "throat_radius":   0.5,   # Small throat -> high curvature
    "exotic_density":  0.9,
    "tidal_force":     0.8,
    "redshift_factor": 0.9,
}

full_result = simulate_qesp(
    "morris_thorne", high_curv_params,
    lambda_param=2.0,   # Stronger correction
    epsilon_c=0.7,      # Lower threshold
    n_radial=200,
)
full_report   = validate_qesp(full_result)
full_preds    = generate_predictions(full_result, full_report)

test("Full pipeline completes", full_result is not None)
test("High-curvature suppression > 1%",
     full_result.curvature_suppression > 0.01,
     f"suppression={full_result.curvature_suppression*100:.2f}%")
test("Full report has verdict", isinstance(full_report.verdict, str),
     f"verdict={full_report.verdict}")
test("Full predictions generated", len(full_preds.predictions) > 0,
     f"n={len(full_preds.predictions)}")
test("Paper abstract generated", len(full_preds.paper_abstract) > 200)

# Print the full results
print(f"\n  {BOLD}Full Pipeline Results:{RESET}")
print(f"    Strain regime:        {full_result.strain_analysis.regime}")
print(f"    Max strain:           {float(jnp.max(full_result.strain_profile)):.4f}")
print(f"    GR stability:         {full_result.stability_gr:.4f}")
print(f"    QESP stability:       {full_result.stability_qesp:.4f}")
print(f"    Curvature suppression:{full_result.curvature_suppression*100:.2f}%")
print(f"    Plateau detected:     {full_result.plateau_detected}")
print(f"    Divergence prevented: {full_result.divergence_prevented}")
print(f"    QESP verdict:         {full_report.verdict}")
print(f"    Score:                {full_report.overall_score:.3f}")
print(f"    Publishable:          {full_report.publishable}")
print(f"    Key finding:          {full_report.key_finding[:100]}...")
print(f"\n  {BOLD}Predictions:{RESET}")
for pred in full_preds.predictions[:4]:
    print(f"    [{pred.type}] {pred.name}: {pred.deviation_pct:.3f}% deviation [{pred.confidence}]")
print(f"\n  {BOLD}Draft Abstract (first 300 chars):{RESET}")
print(f"    {full_preds.paper_abstract[:300]}...")


# ══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}TEST RESULTS{RESET}")
print(f"{'='*60}")
print(f"  {GREEN}Passed: {passed_count}{RESET}")
print(f"  {RED}Failed: {failed_count}{RESET}")
total = passed_count + failed_count
pct = (passed_count / total * 100) if total > 0 else 0
print(f"  Total:  {total}  ({pct:.1f}%)")

if failed_count > 0:
    print(f"\n{RED}Failed tests:{RESET}")
    for status, name in test_results:
        if status == "FAIL":
            print(f"  - {name}")
else:
    print(f"\n{GREEN}{BOLD}All tests passed.{RESET}")

print(f"\n{'='*60}")

if __name__ == "__main__":
    sys.exit(0 if failed_count == 0 else 1)