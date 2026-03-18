"""Application configuration via pydantic-settings.

All values are read from environment variables (and optionally a .env file).
Validation and type coercion happen at startup — a missing required variable
raises a clear ValidationError rather than failing silently at runtime.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env first so that every library that reads
# os.environ directly (openai, httpx, etc.) sees the values too.
load_dotenv()


class Config(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # ignore unknown env vars
    )

    # --- API keys ---
    OPENAI_API_KEY: str | None = None

    # --- Database (persistent LangGraph checkpointer) ---
    # Accepts both POSTGRES_URL and the Heroku-style DATABASE_URL alias.
    # Example: postgresql+psycopg://user:password@localhost:5432/chat_agent
    POSTGRES_URL: str | None = None
    DATABASE_URL: str | None = None  # alias — resolved in validator below

    # --- Server ---
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 7860

    # --- UI branding ---
    APP_NAME: str = "AI Chat Agent"

    # --- Agent ---
    AGENT_MODEL: str = "gpt-4o-mini"
    AGENT_TEMPERATURE: float = 0.2

    @model_validator(mode="after")
    def _resolve_postgres_url(self) -> "Config":
        """Fall back to DATABASE_URL when POSTGRES_URL is not set."""
        if not self.POSTGRES_URL and self.DATABASE_URL:
            self.POSTGRES_URL = self.DATABASE_URL
        return self


# Global config instance — validated eagerly at import time.
config = Config()
