"""
Tests for qesp/quantum_feedback.py

Run with: python -m pytest tests/test_qesp_feedback.py -v
"""

import pytest
import math
from qesp.quantum_feedback import (
    sigmoid_activation,
    tanh_activation,
    power_activation,
    gaussian_activation,
    Q_scalar_from_strain,
    modified_kretschmann,
    ActivationMode,
    LAMBDA_DEFAULT,
    EPSILON_C_DEFAULT,
    K_STEEP_DEFAULT,
)


class TestSigmoidActivation:

    def test_zero_at_low_strain(self):
        """Sigmoid must be near 0 far below threshold."""
        A = sigmoid_activation(epsilon=0.0, epsilon_c=0.85, k=10.0)
        assert A < 0.01

    def test_half_at_threshold(self):
        """Sigmoid must be exactly 0.5 at epsilon = epsilon_c."""
        A = sigmoid_activation(epsilon=0.85, epsilon_c=0.85, k=10.0)
        assert abs(A - 0.5) < 1e-6

    def test_one_at_high_strain(self):
        """Sigmoid must be near 1 far above threshold."""
        A = sigmoid_activation(epsilon=2.0, epsilon_c=0.85, k=10.0)
        assert A > 0.99

    def test_monotonically_increasing(self):
        """Sigmoid must be monotonically increasing."""
        epsilons = [0.0, 0.3, 0.6, 0.85, 1.0, 1.5, 2.0]
        activations = [sigmoid_activation(e, 0.85, 10.0) for e in epsilons]
        for i in range(len(activations) - 1):
            assert activations[i] <= activations[i+1]


class TestActivationFunctions:

    def test_tanh_unit_interval(self):
        for e in [0.0, 0.5, 0.85, 1.0, 2.0]:
            A = tanh_activation(e, epsilon_c=0.85, k=10.0)
            assert 0.0 <= A <= 1.0

    def test_power_zero_below_threshold(self):
        A = power_activation(0.5, epsilon_c=0.85)
        assert A == 0.0

    def test_gaussian_peaks_at_threshold(self):
        A1 = gaussian_activation(1.0, epsilon_c=1.0, sigma_g=0.15)
        A2 = gaussian_activation(0.5, epsilon_c=1.0, sigma_g=0.15)
        A3 = gaussian_activation(1.5, epsilon_c=1.0, sigma_g=0.15)
        assert A1 == 1.0
        assert A1 > A2
        assert A1 > A3


class TestQScalarFromStrain:

    def test_inactive_far_below_threshold(self):
        result = Q_scalar_from_strain(epsilon=0.1)
        assert result < 0.05

    def test_activates_near_threshold(self):
        result = Q_scalar_from_strain(epsilon=EPSILON_C_DEFAULT)
        assert result == pytest.approx(0.5 * LAMBDA_DEFAULT, abs=0.05)

    def test_saturates_above_threshold(self):
        result = Q_scalar_from_strain(epsilon=2.0)
        assert result > 0.9 * LAMBDA_DEFAULT

    def test_lambda_zero_means_no_correction(self):
        result = Q_scalar_from_strain(epsilon=1.5, lam=0.0)
        assert result == 0.0


class TestModifiedCurvature:

    def test_qesp_less_than_gr_at_high_strain(self):
        K_gr = 10.0
        epsilon = 10.0
        K_qesp = modified_kretschmann(K_gr, epsilon)
        assert K_qesp < K_gr

    def test_gr_unchanged_at_low_strain(self):
        K_gr = 0.001
        epsilon = 0.001
        K_qesp = modified_kretschmann(K_gr, epsilon)
        assert K_qesp == pytest.approx(K_gr, rel=0.01)
