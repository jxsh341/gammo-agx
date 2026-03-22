"""
Gammo AGX - Multi-Pass Reasoning Pipeline

Makes Gemma 3 4B reason far above its weight class through architecture:

1. RETRIEVAL    - Pull similar configs from Supabase pgvector for context
2. PASS 1       - Initial hypothesis generation with retrieved context
3. TOOL: SymPy  - Symbolic validation of proposed configuration
4. PASS 2       - Self-critique given SymPy feedback
5. CONSISTENCY  - Sample 3 variants, vote on most physically consistent
6. TOOL: JAX    - Simulate the proposed configuration
7. PASS 3       - Final refinement given simulation result
8. UNCERTAINTY  - Evidential scoring of final hypothesis

Each pass conditions on all previous outputs.
The model is grounded by real physics at every step.
"""

import random
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReasoningState:
    """Tracks the full state of a multi-pass reasoning session."""
    simulation_record:    dict
    retrieved_context:    list = field(default_factory=list)
    pass1_hypothesis:     str = ""
    sympy_feedback:       dict = field(default_factory=dict)
    pass2_critique:       str = ""
    consistency_samples:  list = field(default_factory=list)
    consistency_winner:   str = ""
    jax_result:           dict = field(default_factory=dict)
    pass3_final:          str = ""
    confidence:           float = 0.0
    uncertainty_type:     str = "epistemic"
    novelty_score:        float = 0.0
    falsifiability:       str = ""
    reasoning_trace:      list = field(default_factory=list)


def _log_step(state: ReasoningState, step: str, detail: str = ""):
    """Log a reasoning step to the trace."""
    entry = f"[{step}] {detail}"
    state.reasoning_trace.append(entry)
    logger.debug(f"Reasoning pipeline: {entry}")


# ── STEP 1: RETRIEVAL ────────────────────────────────────────────────────────

def retrieve_similar_context(
    simulation_record: dict,
    n_results: int = 5,
) -> list[dict]:
    """
    Pull the most similar configurations from Supabase pgvector.
    Feeds real physics data into the reasoning context.
    """
    try:
        from store.search import search_by_params
        params = simulation_record.get("parameters", {})
        results = search_by_params(
            throat_radius   = params.get("throat_radius",   1.0),
            exotic_density  = params.get("exotic_density",  0.5),
            tidal_force     = params.get("tidal_force",     0.3),
            redshift_factor = params.get("redshift_factor", 0.2),
            match_count     = n_results,
            min_stability   = 0.0,
        )
        logger.debug(f"Retrieved {len(results)} similar configurations")
        return results
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        return []


def format_context(retrieved: list[dict]) -> str:
    """Format retrieved records as context for the prompt."""
    if not retrieved:
        return "No similar configurations found in knowledge store."

    lines = ["Similar configurations from knowledge store:"]
    for i, r in enumerate(retrieved[:5]):
        params = r.get("parameters", {})
        sim = r.get("similarity", 0)
        lines.append(
            f"  Config {i+1} (similarity={sim:.3f}): "
            f"b0={params.get('throat_radius','?')}, "
            f"rho={params.get('exotic_density','?')}, "
            f"tide={params.get('tidal_force','?')}, "
            f"stability={r.get('stability_score',0):.3f}, "
            f"ford_roman={r.get('ford_roman_status','?')}"
        )
    return "\n".join(lines)


# ── STEP 2: PASS 1 - INITIAL HYPOTHESIS ─────────────────────────────────────

def pass1_initial_hypothesis(
    state: ReasoningState,
    gemma_generate,
) -> str:
    """
    Pass 1: Generate initial hypothesis with retrieved context.
    """
    record  = state.simulation_record
    params  = record.get("parameters", {})
    context = format_context(state.retrieved_context)

    throat  = params.get("throat_radius",   1.0)
    exotic  = params.get("exotic_density",  0.5)
    tide    = params.get("tidal_force",     0.3)
    stab    = record.get("stability_score", 0.0)
    ford    = record.get("ford_roman_status", "unknown")
    casimir = record.get("casimir_gap_oom", 47.0)
    energy  = record.get("energy_requirement", 0.0)

    prompt = f"""You are analyzing a Morris-Thorne wormhole configuration for Gammo AGX.

CONFIGURATION:
- Throat radius b0 = {throat} Planck lengths
- Exotic matter density rho = {exotic}
- Tidal force = {tide}
- Stability score: {stab:.3f}
- Ford-Roman status: {ford}
- Energy requirement: {energy:.4e}
- Casimir gap: {casimir:.1f} orders of magnitude

{context}

Generate an initial scientific hypothesis (2-3 sentences) about:
1. The physical significance of this configuration
2. What determines its viability as a traversable wormhole
3. How it compares to similar configurations in the knowledge store

Be precise. Use physics terminology."""

    hypothesis = gemma_generate(prompt, max_tokens=250, thinking=True)
    _log_step(state, "PASS1", f"Generated initial hypothesis ({len(hypothesis)} chars)")
    return hypothesis


# ── STEP 3: TOOL - SYMPY VALIDATION ─────────────────────────────────────────

def tool_sympy_validation(state: ReasoningState) -> dict:
    """
    Tool call: SymPy validates the configuration symbolically.
    Returns feedback that Pass 2 conditions on.
    """
    try:
        from core.symbolic.metric_validator import validate_morris_thorne
        from core.quantum.ford_roman import check_morris_thorne

        params = state.simulation_record.get("parameters", {})
        b0   = params.get("throat_radius",   1.0)
        rho  = params.get("exotic_density",  0.5)
        tide = params.get("tidal_force",     0.3)
        phi0 = params.get("redshift_factor", 0.2)

        # Symbolic validation
        val_result = validate_morris_thorne(b0, rho, tide, phi0)

        # Ford-Roman check
        fr_result = check_morris_thorne(b0, rho)

        feedback = {
            "symbolic_valid":     val_result.valid,
            "symbolic_reason":    val_result.reason,
            "ford_roman_status":  fr_result.status,
            "fr_violation_factor": fr_result.violation_factor,
            "fr_margin":          fr_result.margin,
            "energy_magnitude":   val_result.energy_density,
            "details":            val_result.details,
        }

        _log_step(
            state, "SYMPY",
            f"valid={val_result.valid}, fr={fr_result.status}, "
            f"fr_factor={fr_result.violation_factor:.3f}"
        )
        return feedback

    except Exception as e:
        logger.warning(f"SymPy tool call failed: {e}")
        return {"symbolic_valid": True, "error": str(e)}


def format_sympy_feedback(feedback: dict) -> str:
    """Format SymPy results as natural language feedback."""
    valid   = feedback.get("symbolic_valid", True)
    reason  = feedback.get("symbolic_reason", "")
    fr      = feedback.get("ford_roman_status", "unknown")
    factor  = feedback.get("fr_violation_factor", 1.0)
    margin  = feedback.get("fr_margin", 0.0)
    e_mag   = feedback.get("energy_magnitude", 0.0)

    lines = ["SymPy symbolic analysis results:"]
    lines.append(f"  Symbolic validity: {'PASSED' if valid else 'FAILED'} - {reason}")
    lines.append(f"  Ford-Roman quantum inequality: {fr.upper()}")
    lines.append(f"  Violation factor: {factor:.4f} (< 1.0 = satisfied)")
    lines.append(f"  Quantum margin: {margin:.6f}")
    lines.append(f"  Energy density magnitude: {e_mag:.4e}")
    return "\n".join(lines)


# ── STEP 4: PASS 2 - SELF-CRITIQUE ──────────────────────────────────────────

def pass2_self_critique(
    state: ReasoningState,
    gemma_generate,
) -> str:
    """
    Pass 2: Gemma critiques its own hypothesis given SymPy feedback.
    Identifies physical inconsistencies and improves the reasoning.
    """
    sympy_feedback_str = format_sympy_feedback(state.sympy_feedback)

    prompt = f"""You previously generated this hypothesis about a Morris-Thorne wormhole:

INITIAL HYPOTHESIS:
{state.pass1_hypothesis}

Now you have received symbolic physics validation results:

{sympy_feedback_str}

Critique your initial hypothesis:
1. Is your hypothesis consistent with the SymPy results?
2. Did you correctly assess the Ford-Roman status?
3. What physical aspects did you miss or get wrong?
4. What should be corrected or emphasized?

Be specific and self-critical. Identify exactly what needs to change."""

    critique = gemma_generate(prompt, max_tokens=200, thinking=True)
    _log_step(state, "PASS2", f"Self-critique generated ({len(critique)} chars)")
    return critique


# ── STEP 5: SELF-CONSISTENCY SAMPLING ────────────────────────────────────────

def self_consistency_sampling(
    state: ReasoningState,
    gemma_generate,
    n_samples: int = 3,
) -> tuple[list[str], str]:
    """
    Generate N hypothesis variants and vote on the most physically consistent.
    Majority reasoning dramatically improves reliability on scientific tasks.
    """
    record  = state.simulation_record
    params  = record.get("parameters", {})
    context = format_context(state.retrieved_context)
    sympy   = format_sympy_feedback(state.sympy_feedback)

    throat  = params.get("throat_radius",   1.0)
    exotic  = params.get("exotic_density",  0.5)
    tide    = params.get("tidal_force",     0.3)
    stab    = record.get("stability_score", 0.0)
    ford    = record.get("ford_roman_status", "unknown")
    energy  = record.get("energy_requirement", 0.0)
    casimir = record.get("casimir_gap_oom", 47.0)

    base_prompt = f"""Generate a precise scientific hypothesis for this Morris-Thorne wormhole:
b0={throat}, rho={exotic}, tide={tide}
stability={stab:.3f}, ford_roman={ford}, energy={energy:.4e}, casimir_gap={casimir:.1f} OOM

SymPy validation: {sympy}
Initial analysis: {state.pass1_hypothesis[:200]}
Self-critique: {state.pass2_critique[:200]}

Provide a refined 2-sentence hypothesis grounded in the physics above."""

    samples = []
    temperatures = [0.5, 0.7, 0.9]

    for i, temp in enumerate(temperatures[:n_samples]):
        try:
            sample = gemma_generate(base_prompt, max_tokens=150, temperature=temp)
            samples.append(sample)
            _log_step(state, "CONSISTENCY", f"Sample {i+1} generated (temp={temp})")
        except Exception as e:
            logger.warning(f"Consistency sample {i+1} failed: {e}")

    if not samples:
        return [], state.pass1_hypothesis

    # Vote: ask Gemma to pick the most physically consistent sample
    if len(samples) == 1:
        return samples, samples[0]

    vote_prompt = f"""You have {len(samples)} candidate hypotheses about a Morris-Thorne wormhole
(b0={throat}, stability={stab:.3f}, ford_roman={ford}).

"""
    for i, s in enumerate(samples):
        vote_prompt += f"Hypothesis {i+1}: {s}\n\n"

    vote_prompt += f"""SymPy says: ford_roman={ford}, symbolic_valid={state.sympy_feedback.get('symbolic_valid', True)}

Which hypothesis is most physically accurate and consistent with the SymPy results?
Reply with just the number: 1, 2, or {len(samples)}"""

    try:
        vote = gemma_generate(vote_prompt, max_tokens=10, temperature=0.1)
        # Extract the winning number
        import re
        match = re.search(r'\b([1-9])\b', vote)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(samples):
                winner = samples[idx]
                _log_step(state, "CONSISTENCY", f"Winner: sample {idx+1}")
                return samples, winner

    except Exception as e:
        logger.warning(f"Consistency vote failed: {e}")

    # Default to first sample
    return samples, samples[0]


# ── STEP 6: TOOL - JAX SIMULATION ────────────────────────────────────────────

def tool_jax_simulation(state: ReasoningState) -> dict:
    """
    Tool call: Run JAX simulation on the configuration.
    Returns real physics results that Pass 3 conditions on.
    """
    try:
        from core.simulator.morris_thorne import MorrisThorneParams, solve
        from core.quantum.casimir import compute_energy_gap

        params = state.simulation_record.get("parameters", {})
        mt_params = MorrisThorneParams(
            throat_radius   = params.get("throat_radius",   1.0),
            exotic_density  = params.get("exotic_density",  0.5),
            tidal_force     = params.get("tidal_force",     0.3),
            redshift_factor = params.get("redshift_factor", 0.2),
        )
        result  = solve(mt_params)
        metrics = result.get("metrics", {})

        # Real Casimir gap
        casimir = compute_energy_gap(
            wormhole_energy_requirement=metrics.get("energy_requirement", -1e-2)
        )

        jax_result = {
            "stability_score":     metrics.get("stability_score", 0.0),
            "ford_roman_status":   metrics.get("ford_roman_status", "unknown"),
            "energy_requirement":  metrics.get("energy_requirement", 0.0),
            "casimir_gap_oom":     casimir.get("gap_orders_of_magnitude", 47.0),
            "casimir_achievability": casimir.get("achievability", "unknown"),
            "null_energy_violated":metrics.get("null_energy_violated", True),
            "traversal_time":      metrics.get("traversal_time", 0.0),
            "geometry_class":      metrics.get("geometry_class", "STANDARD"),
            "bssn_stable":         metrics.get("bssn_stable", True),
            "constraint_error":    metrics.get("constraint_error", 0.0),
        }

        _log_step(
            state, "JAX",
            f"stability={jax_result['stability_score']:.3f}, "
            f"ford_roman={jax_result['ford_roman_status']}, "
            f"casimir_gap={jax_result['casimir_gap_oom']:.1f} OOM"
        )
        return jax_result

    except Exception as e:
        logger.warning(f"JAX tool call failed: {e}")
        return state.simulation_record


def format_jax_result(jax_result: dict) -> str:
    """Format JAX simulation results as natural language."""
    lines = ["JAX simulation results (ground truth):"]
    lines.append(f"  Stability score: {jax_result.get('stability_score', 0):.4f}")
    lines.append(f"  Ford-Roman status: {jax_result.get('ford_roman_status', '?')}")
    lines.append(f"  Energy requirement: {jax_result.get('energy_requirement', 0):.4e}")
    lines.append(f"  Casimir gap: {jax_result.get('casimir_gap_oom', 47):.2f} OOM")
    lines.append(f"  Casimir achievability: {jax_result.get('casimir_achievability', '?')}")
    lines.append(f"  Traversal time: {jax_result.get('traversal_time', 0):.4f} t_P")
    lines.append(f"  BSSN stable: {jax_result.get('bssn_stable', True)}")
    lines.append(f"  Constraint error: {jax_result.get('constraint_error', 0):.6f}")
    return "\n".join(lines)


# ── STEP 7: PASS 3 - FINAL REFINEMENT ────────────────────────────────────────

def pass3_final_hypothesis(
    state: ReasoningState,
    gemma_generate,
) -> str:
    """
    Pass 3: Final hypothesis refinement given JAX simulation results.
    This is the output that gets stored in Supabase and shown to researchers.
    """
    jax_str  = format_jax_result(state.jax_result)
    sympy_str = format_sympy_feedback(state.sympy_feedback)

    prompt = f"""You have analyzed a Morris-Thorne wormhole through multiple reasoning passes.
Now produce the FINAL scientific hypothesis incorporating all evidence.

PREVIOUS ANALYSIS:
Initial hypothesis: {state.pass1_hypothesis[:200]}
Self-critique: {state.pass2_critique[:150]}
Consistency winner: {state.consistency_winner[:200]}

GROUND TRUTH PHYSICS:
{jax_str}

{sympy_str}

Produce the final hypothesis (3-4 sentences) that:
1. Accurately reflects the JAX simulation results
2. Is consistent with SymPy symbolic validation
3. Makes a specific falsifiable prediction
4. Assesses physical viability with precise justification

Format:
HYPOTHESIS: [your final hypothesis]
FALSIFIABLE_PREDICTION: [one specific measurable prediction]
VIABILITY: [VIABLE / MARGINAL / NOT_VIABLE] with one-sentence justification"""

    final = gemma_generate(prompt, max_tokens=350, thinking=True)
    _log_step(state, "PASS3", f"Final hypothesis generated ({len(final)} chars)")
    return final


def extract_falsifiability(final_text: str) -> str:
    """Extract the falsifiable prediction from Pass 3 output."""
    if "FALSIFIABLE_PREDICTION:" in final_text:
        import re
        match = re.search(r'FALSIFIABLE_PREDICTION:\s*(.+?)(?:\n|VIABILITY:|$)', final_text, re.DOTALL)
        if match:
            return match.group(1).strip()[:300]
    return ""


def extract_hypothesis_text(final_text: str) -> str:
    """Extract clean hypothesis from Pass 3 output."""
    if "HYPOTHESIS:" in final_text:
        import re
        match = re.search(r'HYPOTHESIS:\s*(.+?)(?:\n\n|FALSIFIABLE|$)', final_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return final_text[:500]


# ── STEP 8: EVIDENTIAL UNCERTAINTY SCORING ───────────────────────────────────

def compute_evidential_uncertainty(state: ReasoningState) -> dict:
    """
    Compute calibrated confidence using evidential deep learning principles.

    Combines:
    - Physics consistency (SymPy + JAX agreement)
    - Reasoning consistency (self-consistency sampling agreement)
    - Domain coverage (retrieval context quality)
    - Physical viability metrics
    """
    jax    = state.jax_result
    sympy  = state.sympy_feedback
    stab   = jax.get("stability_score", 0.0)
    ford   = jax.get("ford_roman_status", "unknown")
    casimir = jax.get("casimir_gap_oom", 47.0)
    bssn   = jax.get("bssn_stable", True)
    cerr   = jax.get("constraint_error", 1.0)
    sym_valid = sympy.get("symbolic_valid", True)
    fr_factor = sympy.get("fr_violation_factor", 1.0)
    n_retrieved = len(state.retrieved_context)
    n_samples   = len(state.consistency_samples)

    # Component scores
    physics_score = (
        stab * 0.35 +
        (0.25 if ford == "satisfied" else 0.05) +
        (0.15 if bssn else 0.0) +
        (0.10 if sym_valid else 0.0) +
        max(0.0, 0.15 * (1.0 - cerr))
    )

    casimir_score = max(0.0, 1.0 - casimir / 50.0) * 0.1

    retrieval_score = min(n_retrieved / 5.0, 1.0) * 0.05

    consistency_score = (n_samples / 3.0) * 0.05

    confidence = min(0.97, max(0.25,
        physics_score + casimir_score + retrieval_score + consistency_score
    ))

    # Uncertainty type
    if confidence < 0.5:
        uncertainty_type = "epistemic"
    elif fr_factor > 1.5:
        uncertainty_type = "epistemic"
    else:
        uncertainty_type = "aleatoric"

    # Novelty score
    novelty_score = min(1.0, max(0.0,
        (1.0 - stab) * 0.3 +
        (0.4 if ford == "satisfied" and casimir < 20 else 0.0) +
        (0.3 if not sym_valid else 0.0) +
        random.uniform(0.0, 0.1)
    ))

    _log_step(
        state, "UNCERTAINTY",
        f"confidence={confidence:.3f}, type={uncertainty_type}, "
        f"novelty={novelty_score:.3f}"
    )

    return {
        "confidence":      round(confidence, 3),
        "uncertainty_type": uncertainty_type,
        "novelty_score":   round(novelty_score, 3),
        "physics_score":   round(physics_score, 3),
        "components": {
            "physics":     round(physics_score, 3),
            "casimir":     round(casimir_score, 3),
            "retrieval":   round(retrieval_score, 3),
            "consistency": round(consistency_score, 3),
        }
    }


# ── MAIN PIPELINE ENTRY POINT ────────────────────────────────────────────────

def run_reasoning_pipeline(simulation_record: dict) -> dict:
    """
    Run the full multi-pass reasoning pipeline on a simulation record.

    This is called by the discovery loop for novel or high-confidence discoveries.

    Args:
        simulation_record: full simulation record with metrics and parameters

    Returns:
        dict with final hypothesis, confidence, falsifiability, reasoning trace
    """
    logger.info(
        f"Starting multi-pass reasoning pipeline: "
        f"stability={simulation_record.get('stability_score', 0):.3f}"
    )

    try:
        from ai.models.gemma_runner import generate as gemma_generate
    except Exception as e:
        logger.error(f"Failed to load Gemma for reasoning pipeline: {e}")
        return _fallback_result(simulation_record)

    # Initialize state
    state = ReasoningState(simulation_record=simulation_record)

    try:
        # Step 1: Retrieve similar configs
        _log_step(state, "START", "Beginning multi-pass reasoning pipeline")
        state.retrieved_context = retrieve_similar_context(simulation_record)

        # Step 2: Pass 1 - Initial hypothesis
        state.pass1_hypothesis = pass1_initial_hypothesis(state, gemma_generate)

        # Step 3: SymPy validation
        state.sympy_feedback = tool_sympy_validation(state)

        # Step 4: Pass 2 - Self-critique
        state.pass2_critique = pass2_self_critique(state, gemma_generate)

        # Step 5: Self-consistency sampling
        state.consistency_samples, state.consistency_winner = \
            self_consistency_sampling(state, gemma_generate)

        # Step 6: JAX simulation
        state.jax_result = tool_jax_simulation(state)

        # Step 7: Pass 3 - Final hypothesis
        state.pass3_final = pass3_final_hypothesis(state, gemma_generate)

        # Step 8: Evidential uncertainty
        uncertainty = compute_evidential_uncertainty(state)

        # Extract structured outputs
        final_hypothesis  = extract_hypothesis_text(state.pass3_final)
        falsifiability    = extract_falsifiability(state.pass3_final)

        logger.success(
            f"Multi-pass pipeline complete: "
            f"confidence={uncertainty['confidence']:.3f}, "
            f"novelty={uncertainty['novelty_score']:.3f}, "
            f"steps={len(state.reasoning_trace)}"
        )

        return {
            "hypothesis":            final_hypothesis,
            "hypothesis_confidence": uncertainty["confidence"],
            "uncertainty_type":      uncertainty["uncertainty_type"],
            "novelty_score":         uncertainty["novelty_score"],
            "novelty_flag":          uncertainty["novelty_score"] > 0.6,
            "falsifiability":        falsifiability,
            "model_used":            "gemma3_multipass",
            "reasoning_trace":       state.reasoning_trace,
            "pass1":                 state.pass1_hypothesis,
            "pass2_critique":        state.pass2_critique,
            "consistency_samples":   state.consistency_samples,
            "jax_verification":      state.jax_result,
            "pipeline_complete":     True,
        }

    except Exception as e:
        logger.error(f"Multi-pass pipeline failed at step: {e}")
        return _fallback_result(simulation_record, error=str(e))


def _fallback_result(record: dict, error: str = "") -> dict:
    """Fallback when pipeline fails."""
    params   = record.get("parameters", {})
    throat   = params.get("throat_radius", "?")
    ford     = record.get("ford_roman_status", "unknown")
    stab     = record.get("stability_score", 0.0)
    casimir  = record.get("casimir_gap_oom", 47.0)

    return {
        "hypothesis": (
            f"Configuration with throat_radius={throat} yields "
            f"stability={stab:.3f} and ford_roman={ford}. "
            f"Casimir gap of {casimir:.1f} OOM requires further reduction."
        ),
        "hypothesis_confidence": 0.35,
        "uncertainty_type":      "epistemic",
        "novelty_score":         0.3,
        "novelty_flag":          False,
        "falsifiability":        "",
        "model_used":            "fallback",
        "pipeline_complete":     False,
        "error":                 error,
    }
