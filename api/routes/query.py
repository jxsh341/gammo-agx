"""
Gammo AGX - Query Routes
Natural language and structured queries over the knowledge store.
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
    """Parse and execute a natural language query against the knowledge store."""
    from store.search import search_by_natural_language
    results = search_by_natural_language(request.query, request.limit)
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
        "total_records":    get_record_count(),
        "morris_thorne":    get_record_count("morris_thorne"),
        "alcubierre":       get_record_count("alcubierre"),
        "krasnikov":        get_record_count("krasnikov"),
        "novel_discoveries": len(get_novel_discoveries(limit=1000)),
    }


@router.post("/similar")
async def find_similar(
    throat_radius:  float = 1.0,
    exotic_density: float = 0.5,
    tidal_force:    float = 0.3,
    min_stability:  float = 0.0,
    limit:          int   = 10,
):
    """Find simulations similar to a given parameter configuration."""
    from store.search import search_by_params
    results = search_by_params(
        throat_radius  = throat_radius,
        exotic_density = exotic_density,
        tidal_force    = tidal_force,
        match_count    = limit,
        min_stability  = min_stability,
    )
    return {"results": results, "count": len(results)}


@router.get("/stable")
async def get_most_stable(limit: int = 10):
    """Get the most stable configurations in the knowledge store."""
    from store.search import find_most_stable
    results = find_most_stable(limit=limit)
    return {"results": results, "count": len(results)}


@router.get("/ford-roman")
async def get_ford_roman_satisfied(limit: int = 10):
    """Get configurations where Ford-Roman bounds are satisfied."""
    from store.search import find_ford_roman_satisfied
    results = find_ford_roman_satisfied(limit=limit)
    return {"results": results, "count": len(results)}