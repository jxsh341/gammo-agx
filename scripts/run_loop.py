"""
Gammo AGX — Standalone Discovery Loop Runner
Run with: python scripts/run_loop.py
"""

import asyncio
from loguru import logger
from loop.discovery_loop import DiscoveryLoop


async def main():
    logger.info("Starting Gammo AGX Discovery Loop (standalone mode)")
    loop = DiscoveryLoop()
    try:
        await loop.run()
    except KeyboardInterrupt:
        logger.info("Loop stopped by user")
        loop.stop()


if __name__ == "__main__":
    asyncio.run(main())
