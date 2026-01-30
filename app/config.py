"""
Sahayog Setu - Configuration Module
Loads environment variables and provides app-wide settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App Info
    app_name: str = "Sahayog Setu API"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True
    
    # Database (Supabase PostgreSQL)
    database_url: str
    
    # Supabase Direct API (optional, for real-time features)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    
    # External APIs
    vedas_api_key: Optional[str] = "wUHMJtqXrBdEz_wFQkIdgQ"
    
    # API Settings
    api_prefix: str = "/api/v1"
    
    # CORS
    cors_origins: str = "*"  # Comma-separated list in production
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid reading .env file on every request.
    """
    return Settings()
