"""
Tests for qesp/qesp_simulator.py
"""

import pytest
from qesp.qesp_simulator import simulate_qesp, QESPSimulationResult


class TestSimulateQESP:

    def test_simulate_morris_thorne(self):
        params = {"throat_radius": 1.0, "exotic_density": 0.5, "tidal_force": 0.3, "redshift_factor": 0.2}
        result = simulate_qesp("morris_thorne", params, n_radial=50)
        assert isinstance(result, QESPSimulationResult)
        assert result.geometry_type == "morris_thorne"
        assert result.max_curvature_qesp <= result.max_curvature_gr + 1e-6

    def test_simulate_alcubierre(self):
        params = {"warp_speed": 1.0, "bubble_radius": 1.0, "wall_thickness": 0.5, "energy_density": 0.5}
        result = simulate_qesp("alcubierre", params, n_radial=50)
        assert isinstance(result, QESPSimulationResult)
        assert result.geometry_type == "alcubierre"
        assert result.max_curvature_qesp <= result.max_curvature_gr + 1e-6

    def test_high_density_activates_qesp(self):
        params = {"throat_radius": 0.5, "exotic_density": 0.99, "tidal_force": 0.1, "redshift_factor": 0.1}
        result = simulate_qesp("morris_thorne", params, n_radial=50, k_crit=0.01)
        # Using pytest.approx or direct comparison, but safely assert QESP prevents at least some divergence
        assert result.max_curvature_qesp < result.max_curvature_gr
