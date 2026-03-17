"""
Supabase schema constants and table names
"""

# Table names
SIMULATIONS_TABLE = "simulations"
LITERATURE_TABLE  = "literature_embeddings"
HYPOTHESES_TABLE  = "hypotheses"
METRICS_TABLE     = "discovered_metrics"
LOOP_STATE_TABLE  = "loop_state"

# Schema version
SCHEMA_VERSION = "1.0.0"

# Simulation record schema
SIMULATION_SCHEMA = {
    "id":                  "uuid",
    "created_at":          "timestamptz",
    "geometry_type":       "text",       # morris_thorne | alcubierre | krasnikov | schwarzschild
    "parameters":          "jsonb",      # full parameter dict
    "descriptor_vector":   "vector(64)", # pgvector embedding
    "stability_score":     "float8",
    "energy_requirement":  "float8",
    "casimir_gap_oom":     "float8",
    "ford_roman_status":   "text",       # satisfied | violated | marginal
    "null_energy_violated":"bool",
    "constraint_error":    "float8",
    "traversal_time":      "float8",
    "bssn_stable":         "bool",
    "hypothesis":          "text",
    "hypothesis_confidence":"float8",
    "uncertainty_type":    "text",       # aleatoric | epistemic
    "novelty_flag":        "bool",
    "novelty_score":       "float8",
    "geometry_class":      "text",
    "simulation_duration_ms": "int4",
    "model_used":          "text",       # gemma3 | deepseek_r1
    "loop_iteration":      "int8",
}
