"""
Gammo AGX — Supabase Client
Single client instance shared across all components
"""

from supabase import create_client, Client
from config.settings import settings
from loguru import logger

_client: Client | None = None


def get_client() -> Client:
    """Returns the shared Supabase client, initializing if needed."""
    global _client
    if _client is None:
        if not settings.supabase_url or not settings.supabase_anon_key:
            raise ValueError(
                "Supabase credentials not configured. "
                "Copy .env.example to .env and add your credentials."
            )
        _client = create_client(settings.supabase_url, settings.supabase_anon_key)
        logger.info("Supabase client initialized")
    return _client
