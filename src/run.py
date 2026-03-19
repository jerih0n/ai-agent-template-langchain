"""Entrypoint for the LangChain Chat Agent template."""

from __future__ import annotations

import asyncio
import logging

import gradio as gr

from app.ai.agent import ChatAgent
from app.core.config import config
from app.database.sql_database.migrations_manager import run_migrations
from app.ui import create_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_required_config() -> None:
    """Fail fast with a clear message when required environment variables are missing."""
    missing: list[str] = []
    if not config.OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not config.POSTGRES_URL:
        missing.append("POSTGRES_URL")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy env.example to .env and fill in the required values."
        )


async def _run_migrations() -> None:
    """Run all pending database migrations and surface failures to startup."""
    logger.info("Running database migrations...")
    executed = await run_migrations()
    if executed:
        logger.info(
            "Executed %d migration(s): %s", len(executed), ", ".join(executed)
        )
    else:
        logger.info("No pending migrations to run.")


def main() -> None:
    """Run preflight checks, migrate the database, build the agent, and launch the UI."""
    try:
        _validate_required_config()
        asyncio.run(_run_migrations())
        agent = ChatAgent()
    except Exception as exc:
        logger.error("Application startup failed: %s", exc)
        raise SystemExit(1) from exc

    app = create_ui(agent)
    app.launch(
        server_name=config.APP_HOST,
        server_port=config.APP_PORT,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
