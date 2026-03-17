"""
Gammo AGX — FastAPI Backend
Main application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from api.routes import simulation, query, hypothesis, loop, discovery, dataset
from loop.discovery_loop import DiscoveryLoop
from config.settings import settings

# Global discovery loop instance
discovery_loop = DiscoveryLoop()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background services on app startup."""
    logger.info("Gammo AGX API starting up")

    # Auto-start discovery loop if configured
    if settings.loop_auto_start:
        import asyncio
        asyncio.create_task(discovery_loop.run())
        logger.success("Discovery loop auto-started")

    yield

    # Shutdown
    discovery_loop.stop()
    logger.info("Gammo AGX API shutting down")


app = FastAPI(
    title="Gammo AGX",
    description="Autonomous Hybrid Research Engine for Exotic Spacetime Physics",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Tauri app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(simulation.router,  prefix="/simulation",  tags=["Simulation"])
app.include_router(query.router,       prefix="/query",       tags=["Query"])
app.include_router(hypothesis.router,  prefix="/hypothesis",  tags=["Hypothesis"])
app.include_router(loop.router,        prefix="/loop",        tags=["Loop"])
app.include_router(discovery.router,   prefix="/discovery",   tags=["Discovery"])
app.include_router(dataset.router,     prefix="/dataset",     tags=["Dataset"])


@app.get("/")
async def root():
    return {
        "name": "Gammo AGX",
        "version": "0.1.0",
        "status": "operational",
        "loop_running": discovery_loop.state.running,
        "total_simulations": discovery_loop.state.total_simulations,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
