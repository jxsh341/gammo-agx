"""
Gammo AGX - Semantic Search
Searches the knowledge store using vector similarity via Supabase pgvector.

Enables queries like:
    "find configurations similar to this one"
    "find the most stable wormholes"
    "find novel discoveries near this parameter region"
"""

import math
from loguru import logger
from store.supabase_client import get_client
from core.descriptors.extractor import extract, descriptor_to_list
from core.simulator.morris_thorne import MorrisThorneParams, solve


def search_by_vector(
    query_vector: list,
    match_count: int = 10,
    min_stability: float = 0.0,
) -> list[dict]:
    """
    Search simulations by descriptor vector similarity.

    Args:
        query_vector: 64-dimensional descriptor vector
        match_count:  number of results to return
        min_stability: minimum stability score filter

    Returns:
        List of similar simulation records with similarity scores
    """
    try:
        client = get_client()
        result = client.rpc(
            "match_simulations",
            {
                "query_vector":  query_vector,
                "match_count":   match_count,
                "min_stability": min_stability,
            }
        ).execute()

        if result.data:
            logger.debug(f"Semantic search returned {len(result.data)} results")
            return result.data
        return []

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def search_by_params(
    throat_radius:   float = 1.0,
    exotic_density:  float = 0.5,
    tidal_force:     float = 0.3,
    redshift_factor: float = 0.2,
    match_count:     int = 10,
    min_stability:   float = 0.0,
) -> list[dict]:
    """
    Search for simulations similar to a given parameter configuration.
    Runs the solver, extracts the descriptor, then searches by similarity.

    Args:
        throat_radius:   b0
        exotic_density:  rho0
        tidal_force:     tide
        redshift_factor: phi0
        match_count:     number of results
        min_stability:   minimum stability filter

    Returns:
        List of similar simulation records
    """
    try:
        # Run solver to get simulation result
        params = MorrisThorneParams(
            throat_radius   = throat_radius,
            exotic_density  = exotic_density,
            tidal_force     = tidal_force,
            redshift_factor = redshift_factor,
        )
        result = solve(params)

        # Extract descriptor vector
        params_dict = {
            "throat_radius":   throat_radius,
            "exotic_density":  exotic_density,
            "tidal_force":     tidal_force,
            "redshift_factor": redshift_factor,
        }
        descriptor = extract("morris_thorne", result, params_dict)
        if descriptor is None:
            logger.error("Failed to extract descriptor for similarity search")
            return []

        query_vector = descriptor_to_list(descriptor)
        return search_by_vector(query_vector, match_count, min_stability)

    except Exception as e:
        logger.error(f"Parameter-based search failed: {e}")
        return []


def search_by_natural_language(
    query: str,
    match_count: int = 10,
) -> list[dict]:
    """
    Parse a natural language query and search the knowledge store.

    Handles queries like:
        "find stable wormholes where Ford-Roman is satisfied"
        "show me novel discoveries with high stability"
        "find configurations similar to throat_radius=1.0"

    Args:
        query: natural language query string
        match_count: number of results

    Returns:
        List of matching simulation records
    """
    q = query.lower()

    # Extract parameter hints from query
    throat = 1.0
    exotic = 0.5
    tidal  = 0.3
    redshift = 0.2
    min_stability = 0.0

    # Parse throat radius
    if "throat" in q or "b0" in q:
        import re
        match = re.search(r'throat[_\s]*radius[=:\s]+([0-9.]+)', q)
        if match:
            throat = float(match.group(1))

    # Parse stability requirement
    if "stable" in q or "stability" in q:
        if "very stable" in q or "high stability" in q:
            min_stability = 0.8
        elif "stable" in q:
            min_stability = 0.6

    # Parse ford-roman
    if "ford" in q or "roman" in q or "satisfied" in q:
        exotic = 0.2  # lower exotic density more likely to satisfy FR

    # Parse novelty
    novelty_only = "novel" in q or "discovery" in q or "discoveries" in q

    if novelty_only:
        # Query structured for novel discoveries
        from store.query import get_novel_discoveries
        return get_novel_discoveries(limit=match_count)

    # Default: search by constructed parameter vector
    return search_by_params(
        throat_radius   = throat,
        exotic_density  = exotic,
        tidal_force     = tidal,
        redshift_factor = redshift,
        match_count     = match_count,
        min_stability   = min_stability,
    )


def find_most_stable(limit: int = 10) -> list[dict]:
    """Find the most stable configurations in the knowledge store."""
    from store.query import query_simulations
    return query_simulations(
        min_stability=0.0,
        limit=limit,
        order_by="stability_score",
        ascending=False,
    )


def find_ford_roman_satisfied(limit: int = 10) -> list[dict]:
    """Find configurations where Ford-Roman bounds are satisfied."""
    from store.query import query_simulations
    return query_simulations(
        ford_roman_status="satisfied",
        limit=limit,
        order_by="stability_score",
        ascending=False,
    )


def find_similar_to_record(
    record_id: str,
    match_count: int = 10,
) -> list[dict]:
    """
    Find simulations similar to an existing record in the knowledge store.

    Args:
        record_id: UUID of the reference simulation
        match_count: number of similar records to return

    Returns:
        List of similar records
    """
    try:
        client = get_client()

        # Fetch the reference record
        result = client.table("simulations")\
            .select("descriptor_vector, parameters")\
            .eq("id", record_id)\
            .execute()

        if not result.data:
            logger.error(f"Record {record_id} not found")
            return []

        record = result.data[0]
        descriptor = record.get("descriptor_vector")

        if descriptor is None:
            logger.error(f"Record {record_id} has no descriptor vector")
            return []

        return search_by_vector(descriptor, match_count)

    except Exception as e:
        logger.error(f"Similar record search failed: {e}")
        return []
