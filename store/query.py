"""
Gammo AGX — Knowledge Store Query Interface
Natural language and structured queries over simulation records
"""

from typing import Any
from loguru import logger
from store.supabase_client import get_client
from config.supabase_config import SIMULATIONS_TABLE


def query_simulations(
    geometry_type: str | None = None,
    min_stability: float | None = None,
    ford_roman_status: str | None = None,
    novelty_only: bool = False,
    limit: int = 50,
    order_by: str = "created_at",
    ascending: bool = False,
) -> list[dict]:
    """
    Structured query over simulation records.

    Example:
        results = query_simulations(
            geometry_type="morris_thorne",
            min_stability=0.7,
            ford_roman_status="satisfied",
            limit=20
        )
    """
    try:
        client = get_client()
        q = client.table(SIMULATIONS_TABLE).select("*")

        if geometry_type:
            q = q.eq("geometry_type", geometry_type)
        if min_stability is not None:
            q = q.gte("stability_score", min_stability)
        if ford_roman_status:
            q = q.eq("ford_roman_status", ford_roman_status)
        if novelty_only:
            q = q.eq("novelty_flag", True)

        q = q.order(order_by, desc=not ascending).limit(limit)
        result = q.execute()
        return result.data or []
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []


def get_record_count(geometry_type: str | None = None) -> int:
    """Get total simulation record count."""
    try:
        client = get_client()
        q = client.table(SIMULATIONS_TABLE).select("id", count="exact")
        if geometry_type:
            q = q.eq("geometry_type", geometry_type)
        result = q.execute()
        return result.count or 0
    except Exception as e:
        logger.error(f"Count query failed: {e}")
        return 0


def get_best_configurations(
    metric: str = "stability_score",
    limit: int = 10,
) -> list[dict]:
    """Return top N configurations by a given metric."""
    try:
        client = get_client()
        result = (
            client.table(SIMULATIONS_TABLE)
            .select("*")
            .order(metric, desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        logger.error(f"Best config query failed: {e}")
        return []


def get_novel_discoveries(limit: int = 20) -> list[dict]:
    """Return all flagged novel discoveries, most recent first."""
    return query_simulations(novelty_only=True, limit=limit)
