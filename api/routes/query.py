"""
Gammo AGX — Natural Language Query Routes
Human query interface for the knowledge store.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from store.query import query_simulations, get_record_count, get_novel_discoveries

router = APIRouter()


class NaturalLanguageQuery(BaseModel):
    query: str
    limit: int = 20


class StructuredQuery(BaseModel):
    geometry_type: str | None = None
    min_stability: float | None = None
    ford_roman_status: str | None = None
    novelty_only: bool = False
    limit: int = 50


@router.post("/natural")
async def natural_language_query(request: NaturalLanguageQuery):
    """
    Parse and execute a natural language query against the knowledge store.
    Example: 'find stable wormholes where Ford-Roman is satisfied'
    """
    # TODO: Wire to AI query parser
    # For now, extract key terms and run structured query
    q = request.query.lower()
    params = {}
    if "wormhole" in q or "morris" in q:
        params["geometry_type"] = "morris_thorne"
    if "stable" in q or "stability" in q:
        params["min_stability"] = 0.6
    if "ford" in q or "roman" in q or "satisfied" in q:
        params["ford_roman_status"] = "satisfied"
    if "novel" in q or "discovery" in q:
        params["novelty_only"] = True

    results = query_simulations(**params, limit=request.limit)
    return {"query": request.query, "results": results, "count": len(results)}


@router.post("/structured")
async def structured_query(request: StructuredQuery):
    """Execute a structured query with explicit filters."""
    results = query_simulations(
        geometry_type=request.geometry_type,
        min_stability=request.min_stability,
        ford_roman_status=request.ford_roman_status,
        novelty_only=request.novelty_only,
        limit=request.limit,
    )
    return {"results": results, "count": len(results)}


@router.get("/stats")
async def get_stats():
    """Get knowledge store statistics."""
    return {
        "total_records": get_record_count(),
        "morris_thorne": get_record_count("morris_thorne"),
        "alcubierre": get_record_count("alcubierre"),
        "krasnikov": get_record_count("krasnikov"),
        "novel_discoveries": len(get_novel_discoveries(limit=1000)),
    }
