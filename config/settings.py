"""
Gammo AGX — Central Configuration
All settings loaded from environment variables via .env
"""

from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_key: str = ""

    # AI Models
    gemma_model_path: str = "models/gemma-3-4b-instruct-q4_k_m.gguf"
    deepseek_model_path: str = "models/deepseek-r1-distill-qwen-7b-q4_k_m.gguf"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change_this"

    # Physics
    jax_enable_x64: bool = True
    jax_platform_name: str = "gpu"

    # Discovery Loop
    loop_interval_seconds: int = 30
    loop_max_concurrent: int = 2
    loop_auto_start: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/gammo_agx.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
