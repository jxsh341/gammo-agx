"""
Gammo AGX — QESP Integration Layer
qesp/integration.py

Wires the QESP module into the existing Gammo AGX discovery loop.

Every simulation now runs a QESP analysis alongside the existing
Morris-Thorne and Alcubierre physics. The results are stored
in Supabase as additional columns on the simulations table.

INTEGRATION STEPS REQUIRED:
1. Run the Supabase migration SQL (see below)
2. Add QESP columns to store/writer.py
3. Wire _step_qesp into loop/discovery_loop.py
4. Add QESP endpoints to api/routes/query.py
5. Update dashboard to show QESP validation results
"""

from loguru import logger
from typing import Optional


def run_qesp_analysis(
    geometry_type: str,
    parameters: dict,
    simulation_result: dict,
) -> Optional[dict]:
    """
    Run QESP analysis for any supported geometry type.
    Called from the discovery loop after _step_simulate.

    Args:
        geometry_type:    "morris_thorne" or "alcubierre"
        parameters:       Input parameter dict
        simulation_result: Output from JAX solver

    Returns:
        QESP result dict for Supabase storage, or None if unsupported
    """
    try:
        from qesp.qesp_simulator import (
            simulate_qesp_morris_thorne,
            simulate_qesp_alcubierre,
            QESPConfig,
        )
        from analysis.validator import validate_qesp_result, generate_prediction_summary

        # Use fast config for loop (full config for deep analysis)
        config = QESPConfig(
            n_steps=100,
            n_radial=50,
            k_crit=0.9,
            lambda_param=0.8,
            epsilon_c=0.85,
            k_sigmoid=10.0,
        )

        if geometry_type == "morris_thorne":
            result = simulate_qesp_morris_thorne(
                throat_radius   = parameters.get("throat_radius",   1.0),
                exotic_density  = parameters.get("exotic_density",  0.5),
                tidal_force     = parameters.get("tidal_force",     0.3),
                redshift_factor = parameters.get("redshift_factor", 0.2),
                config          = config,
            )
        elif geometry_type == "alcubierre":
            result = simulate_qesp_alcubierre(
                warp_speed      = parameters.get("warp_speed",     1.0),
                bubble_radius   = parameters.get("bubble_radius",  1.0),
                wall_thickness  = parameters.get("wall_thickness", 0.5),
                energy_density  = parameters.get("energy_density", 0.5),
                config          = config,
            )
        else:
            logger.debug(f"QESP: no analyzer for geometry {geometry_type}")
            return None

        # Validate
        report = validate_qesp_result(result)
        summary = generate_prediction_summary(result, report)

        logger.info(
            f"QESP analysis: geometry={geometry_type}, "
            f"validated={report.qesp_validated}, "
            f"score={result.qesp_score:.3f}, "
            f"predictions={summary['predictions_satisfied']}/4"
        )

        return {
            # Core QESP metrics
            "qesp_validated":          summary["qesp_validated"],
            "qesp_score":              summary["qesp_score"],
            "qesp_confidence":         summary["confidence"],
            "qesp_predictions_passed": summary["predictions_satisfied"],

            # Prediction flags
            "qesp_p1_no_divergence":        summary["p1_no_divergence"],
            "qesp_p2_selective_activation":  summary["p2_selective_activation"],
            "qesp_p3_stabilization":         summary["p3_stabilization"],
            "qesp_p4_measurable_deviation":  summary["p4_measurable_deviation"],

            # Numerical results
            "qesp_divergence_ratio":   summary["divergence_ratio"],
            "qesp_deviation_pct":      summary["deviation_pct"],
            "qesp_max_activation":     summary["max_activation"],
            "qesp_plateau_fraction":   result.plateau_fraction,
            "qesp_gr_final_K":         result.gr_final_K,
            "qesp_qesp_final_K":       result.qesp_final_K,
            "qesp_max_strain":         result.max_strain_gr,

            # Warnings and notes for hypothesis context
            "qesp_warnings":           summary["warnings"],
            "qesp_notes":              summary["notes"],
        }

    except Exception as e:
        logger.error(f"QESP analysis failed for {geometry_type}: {e}")
        return None


# ── SUPABASE MIGRATION SQL ────────────────────────────────────────────────
# Run this in your Supabase SQL Editor to add QESP columns

SUPABASE_MIGRATION_SQL = """
-- QESP columns for the simulations table
-- Run this in Supabase SQL Editor

ALTER TABLE simulations
    ADD COLUMN IF NOT EXISTS qesp_validated          boolean,
    ADD COLUMN IF NOT EXISTS qesp_score              float8,
    ADD COLUMN IF NOT EXISTS qesp_confidence         float8,
    ADD COLUMN IF NOT EXISTS qesp_predictions_passed integer,
    ADD COLUMN IF NOT EXISTS qesp_p1_no_divergence   boolean,
    ADD COLUMN IF NOT EXISTS qesp_p2_selective_activation boolean,
    ADD COLUMN IF NOT EXISTS qesp_p3_stabilization   boolean,
    ADD COLUMN IF NOT EXISTS qesp_p4_measurable_deviation boolean,
    ADD COLUMN IF NOT EXISTS qesp_divergence_ratio   float8,
    ADD COLUMN IF NOT EXISTS qesp_deviation_pct      float8,
    ADD COLUMN IF NOT EXISTS qesp_max_activation     float8,
    ADD COLUMN IF NOT EXISTS qesp_plateau_fraction   float8,
    ADD COLUMN IF NOT EXISTS qesp_gr_final_K         float8,
    ADD COLUMN IF NOT EXISTS qesp_qesp_final_K       float8,
    ADD COLUMN IF NOT EXISTS qesp_max_strain         float8;

-- Index for querying validated configurations
CREATE INDEX IF NOT EXISTS idx_qesp_validated
    ON simulations (qesp_validated)
    WHERE qesp_validated = true;

-- Index for sorting by QESP score
CREATE INDEX IF NOT EXISTS idx_qesp_score
    ON simulations (qesp_score DESC NULLS LAST);
"""


# ── DISCOVERY LOOP PATCH ──────────────────────────────────────────────────
# Add this method to DiscoveryLoop in loop/discovery_loop.py
# Insert it between _step_evaluate and _step_uncertainty

LOOP_PATCH = '''
    async def _step_qesp(self, result: dict) -> dict:
        """Step 3b: QESP quantum elastic spacetime analysis."""
        from qesp.integration import run_qesp_analysis

        geo    = result.get("geometry_type", "morris_thorne")
        params = result.get("parameters", {})
        sim    = result.get("simulation_result", {})

        qesp_data = run_qesp_analysis(geo, params, sim)

        if qesp_data:
            return {**result, **qesp_data}
        return result
'''

# ── DISCOVERY LOOP CYCLE PATCH ────────────────────────────────────────────
# In _run_cycle, add after _step_evaluate:

CYCLE_PATCH = '''
        # Step 3b: QESP analysis
        scored = await self._step_qesp(scored)
'''

# ── WRITER PATCH ──────────────────────────────────────────────────────────
# Add these fields to simulation_record in _step_store:

WRITER_PATCH = '''
            # QESP fields
            "qesp_validated":          record.get("qesp_validated"),
            "qesp_score":              record.get("qesp_score"),
            "qesp_confidence":         record.get("qesp_confidence"),
            "qesp_predictions_passed": record.get("qesp_predictions_passed"),
            "qesp_p1_no_divergence":   record.get("qesp_p1_no_divergence"),
            "qesp_p2_selective_activation": record.get("qesp_p2_selective_activation"),
            "qesp_p3_stabilization":   record.get("qesp_p3_stabilization"),
            "qesp_p4_measurable_deviation": record.get("qesp_p4_measurable_deviation"),
            "qesp_divergence_ratio":   record.get("qesp_divergence_ratio"),
            "qesp_deviation_pct":      record.get("qesp_deviation_pct"),
            "qesp_max_activation":     record.get("qesp_max_activation"),
            "qesp_plateau_fraction":   record.get("qesp_plateau_fraction"),
            "qesp_gr_final_K":         record.get("qesp_gr_final_K"),
            "qesp_qesp_final_K":       record.get("qesp_qesp_final_K"),
            "qesp_max_strain":         record.get("qesp_max_strain"),
'''
