from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from api.routes import simulation, query, hypothesis, loop, discovery, dataset
from api.state import discovery_loop
from config.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Gammo AGX API starting up")
    if settings.loop_auto_start:
        import asyncio
        asyncio.create_task(discovery_loop.run())
        logger.success("Discovery loop auto-started")
    yield
    discovery_loop.stop()
    logger.info("Gammo AGX API shutting down")


app = FastAPI(
    title="Gammo AGX",
    description="Autonomous Hybrid Research Engine for Exotic Spacetime Physics",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(loop.router,  prefix="/loop",  tags=["Loop"])
app.include_router(query.router, prefix="/query", tags=["Query"])


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