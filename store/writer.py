"""
Gammo AGX — Knowledge Store Writer
Writes simulation records to Supabase
"""

from datetime import datetime, timezone
from typing import Any
from loguru import logger
from store.supabase_client import get_client
from config.supabase_config import SIMULATIONS_TABLE


def write_simulation(record: dict[str, Any]) -> dict | None:
    """
    Write a simulation result to the Supabase knowledge store.

    Args:
        record: Simulation result dict matching SIMULATION_SCHEMA

    Returns:
        The inserted record with generated id, or None on failure
    """
    try:
        client = get_client()
        record["created_at"] = datetime.now(timezone.utc).isoformat()
        result = client.table(SIMULATIONS_TABLE).insert(record).execute()
        if result.data:
            logger.info(
                f"Simulation written — geometry: {record.get('geometry_type')}, "
                f"stability: {record.get('stability_score', 0):.3f}, "
                f"novelty: {record.get('novelty_flag', False)}"
            )
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to write simulation record: {e}")
        return None


def write_hypothesis(hypothesis: dict[str, Any]) -> dict | None:
    """Write a standalone hypothesis record."""
    try:
        client = get_client()
        hypothesis["created_at"] = datetime.now(timezone.utc).isoformat()
        result = client.table("hypotheses").insert(hypothesis).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Failed to write hypothesis: {e}")
        return None


def write_discovered_metric(metric: dict[str, Any]) -> dict | None:
    """Write a novel AI-discovered metric to the knowledge store."""
    try:
        client = get_client()
        metric["created_at"] = datetime.now(timezone.utc).isoformat()
        result = client.table("discovered_metrics").insert(metric).execute()
        logger.success(f"Novel metric written: {metric.get('metric_name', 'unnamed')}")
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Failed to write discovered metric: {e}")
        return None
