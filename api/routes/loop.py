"""
Gammo AGX — Loop Control Routes
Start, stop, and inspect the autonomous discovery loop.
"""

from fastapi import APIRouter
from api.main import discovery_loop

router = APIRouter()


@router.get("/status")
async def get_loop_status():
    """Get current discovery loop status."""
    return discovery_loop.get_status()


@router.post("/start")
async def start_loop():
    """Start the discovery loop if not running."""
    import asyncio
    if not discovery_loop.state.running:
        asyncio.create_task(discovery_loop.run())
        return {"status": "started"}
    return {"status": "already_running"}


@router.post("/stop")
async def stop_loop():
    """Stop the discovery loop after current cycle."""
    discovery_loop.stop()
    return {"status": "stopping"}


@router.get("/feed")
async def get_loop_feed():
    """Get recent loop activity for live dashboard feed."""
    from store.query import query_simulations
    recent = query_simulations(limit=20)
    return {
        "loop_status": discovery_loop.get_status(),
        "recent_simulations": recent,
    }
