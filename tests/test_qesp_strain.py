"""
Tests for qesp/strain.py

Run with: python -m pytest tests/test_qesp_strain.py -v
"""

import pytest
import math
from qesp.strain import (
    kretschmann_schwarzschild,
    kretschmann_morris_thorne,
    kretschmann_alcubierre,
    compute_strain,
    strain_profile_morris_thorne,
    strain_profile_alcubierre,
    detect_critical_radius,
    DEFAULT_K_CRIT,
    EPSILON_FLOOR,
)


class TestKretschmannSchwarzschild:

    def test_basic_computation(self):
        """K = 48 M^2 / r^6 in Planck units."""
        K = kretschmann_schwarzschild(r=2.0, M=1.0)
        expected = 48.0 * 1.0 / (2.0 ** 6)
        assert abs(K - expected) < 1e-10

    def test_diverges_at_origin(self):
        """K must diverge as r -> 0 (singularity)."""
        K_small = kretschmann_schwarzschild(r=0.001, M=1.0)
        K_large = kretschmann_schwarzschild(r=1.0, M=1.0)
        assert K_small > K_large * 1000  # Much larger near origin

    def test_zero_radius_returns_inf(self):
        """r=0 must return infinity."""
        K = kretschmann_schwarzschild(r=0.0, M=1.0)
        assert math.isinf(K)

    def test_scales_with_mass_squared(self):
        """K scales as M^2."""
        K1 = kretschmann_schwarzschild(r=1.0, M=1.0)
        K2 = kretschmann_schwarzschild(r=1.0, M=2.0)
        assert abs(K2 / K1 - 4.0) < 1e-10  # 2^2 = 4

    def test_decreases_with_radius(self):
        """K must decrease as r increases."""
        K1 = kretschmann_schwarzschild(r=1.0, M=1.0)
        K2 = kretschmann_schwarzschild(r=2.0, M=1.0)
        K3 = kretschmann_schwarzschild(r=5.0, M=1.0)
        assert K1 > K2 > K3

    def test_positive(self):
        """K is always non-negative."""
        for r in [0.5, 1.0, 2.0, 10.0]:
            K = kretschmann_schwarzschild(r=r, M=1.0)
            assert K >= 0


class TestKretschmannMorrisThorne:

    def test_basic_computation(self):
        """K must be positive and finite for valid parameters."""
        K = kretschmann_morris_thorne(r=1.1, b0=1.0, rho0=0.5)
        assert K > 0
        assert math.isfinite(K)

    def test_peaks_near_throat(self):
        """Curvature must be highest near the throat."""
        b0 = 1.0
        K_throat = kretschmann_morris_thorne(r=b0*1.001, b0=b0, rho0=0.5)
        K_far    = kretschmann_morris_thorne(r=b0*5.0,   b0=b0, rho0=0.5)
        assert K_throat > K_far

    def test_throat_clamping(self):
        """r < b0 must be clamped to throat."""
        K_inside = kretschmann_morris_thorne(r=0.5, b0=1.0, rho0=0.5)
        K_throat = kretschmann_morris_thorne(r=1.001, b0=1.0, rho0=0.5)
        # Should be similar (both evaluate at throat)
        assert abs(K_inside - K_throat) / K_throat < 0.01

    def test_increases_with_density(self):
        """Higher exotic density -> higher curvature."""
        K_low  = kretschmann_morris_thorne(r=1.1, b0=1.0, rho0=0.1)
        K_high = kretschmann_morris_thorne(r=1.1, b0=1.0, rho0=0.9)
        assert K_high > K_low

    def test_floor_enforced(self):
        """K must never return exactly zero."""
        K = kretschmann_morris_thorne(r=100.0, b0=1.0, rho0=0.01)
        assert K >= EPSILON_FLOOR


class TestKretschmannAlcubierre:

    def test_basic_computation(self):
        """K must be positive and finite for valid warp parameters."""
        K = kretschmann_alcubierre(r_s=1.0, v_s=1.0, R=1.0, sigma=0.5)
        assert K > 0
        assert math.isfinite(K)

    def test_peaks_at_bubble_wall(self):
        """Curvature peaks at r_s = R (bubble wall)."""
        v_s, R, sigma = 2.0, 1.0, 8.0
        K_wall   = kretschmann_alcubierre(r_s=R, v_s=v_s, R=R, sigma=sigma)
        K_center = kretschmann_alcubierre(r_s=0.1, v_s=v_s, R=R, sigma=sigma)
        K_far    = kretschmann_alcubierre(r_s=3*R, v_s=v_s, R=R, sigma=sigma)
        assert K_wall > K_center
        assert K_wall > K_far

    def test_scales_with_warp_speed(self):
        """Higher warp speed -> higher curvature at wall."""
        K_slow = kretschmann_alcubierre(r_s=1.0, v_s=0.5, R=1.0, sigma=0.5)
        K_fast = kretschmann_alcubierre(r_s=1.0, v_s=2.0, R=1.0, sigma=0.5)
        assert K_fast > K_slow

    def test_sharper_wall_higher_curvature(self):
        """Thicker wall (higher sigma) -> sharper transition -> higher K at wall."""
        K_thin  = kretschmann_alcubierre(r_s=1.0, v_s=1.0, R=1.0, sigma=0.1)
        K_thick = kretschmann_alcubierre(r_s=1.0, v_s=1.0, R=1.0, sigma=2.0)
        assert K_thick > K_thin


class TestComputeStrain:

    def test_normal_regime(self):
        """Low curvature -> normal regime, strain < 0.5."""
        result = compute_strain(kretschmann=0.1, k_crit=1.0)
        assert result.strain == pytest.approx(0.1)
        assert result.regime == "normal"
        assert not result.threshold_exceeded

    def test_critical_regime(self):
        """Near-critical -> critical regime."""
        result = compute_strain(kretschmann=0.92, k_crit=1.0)
        assert 0.8 <= result.strain < 1.0
        assert result.regime == "critical"
        assert not result.threshold_exceeded

    def test_supercritical(self):
        """Above threshold -> supercritical, threshold_exceeded=True."""
        result = compute_strain(kretschmann=1.5, k_crit=1.0)
        assert result.strain == pytest.approx(1.5)
        assert result.regime == "supercritical"
        assert result.threshold_exceeded

    def test_strain_formula(self):
        """epsilon = K / K_crit."""
        K = 0.456
        k_crit = 0.9
        result = compute_strain(K, k_crit)
        assert result.strain == pytest.approx(K / k_crit, rel=1e-6)

    def test_distance_to_threshold(self):
        """distance = 1 - strain."""
        result = compute_strain(kretschmann=0.6, k_crit=1.0)
        assert result.distance_to_threshold == pytest.approx(0.4, rel=1e-6)

    def test_invalid_k_crit(self):
        """k_crit <= 0 must raise ValueError."""
        with pytest.raises(ValueError):
            compute_strain(0.5, k_crit=0.0)
        with pytest.raises(ValueError):
            compute_strain(0.5, k_crit=-1.0)


class TestStrainProfiles:

    def test_mt_profile_length(self):
        """Profile must have n_points entries."""
        profile = strain_profile_morris_thorne(
            b0=1.0, rho0=0.5, n_points=50
        )
        assert len(profile) == 50

    def test_mt_profile_structure(self):
        """Each profile point must have required keys."""
        profile = strain_profile_morris_thorne(b0=1.0, rho0=0.5, n_points=10)
        for point in profile:
            assert "r" in point
            assert "strain" in point
            assert "regime" in point
            assert "kretschmann" in point

    def test_mt_profile_max_near_throat(self):
        """Strain must be highest at or near the throat."""
        profile = strain_profile_morris_thorne(b0=1.0, rho0=0.5, n_points=100)
        # First 10% of radial points (near throat) should have higher strain
        near_throat_max = max(p["strain"] for p in profile[:10])
        far_max = max(p["strain"] for p in profile[80:])
        assert near_throat_max >= far_max

    def test_alc_profile_structure(self):
        """Alcubierre profile must have required keys."""
        profile = strain_profile_alcubierre(v_s=1.0, R=1.0, sigma=0.5, n_points=20)
        for point in profile:
            assert "r_s" in point
            assert "strain" in point

    def test_detect_critical_radius(self):
        """Must find radius where strain exceeds threshold."""
        profile = strain_profile_morris_thorne(
            b0=1.0, rho0=5.0, k_crit=0.001, n_points=100
        )
        r_crit = detect_critical_radius(profile, threshold=0.5)
        assert r_crit is not None
        assert r_crit > 0

    def test_detect_critical_radius_none(self):
        """Returns None when threshold never exceeded."""
        profile = strain_profile_morris_thorne(
            b0=1.0, rho0=0.01, k_crit=1e10, n_points=50
        )
        r_crit = detect_critical_radius(profile, threshold=0.9)
        assert r_crit is None
