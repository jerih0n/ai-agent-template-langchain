"""Entrypoint for the LangChain Chat Agent template."""

from __future__ import annotations

import asyncio
import logging
import sys

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


async def _run_migrations() -> None:
    """Run all pending database migrations; log but do not crash on failure."""
    try:
        logger.info("Running database migrations...")
        executed = await run_migrations()
        if executed:
            logger.info(
                "Executed %d migration(s): %s", len(executed), ", ".join(executed)
            )
        else:
            logger.info("No pending migrations to run.")
    except Exception:
        logger.exception("Migration error — continuing anyway")
        # Do not exit: allow the app to start even if the DB is unavailable
        # (useful during local development without a running Postgres instance).


def main() -> None:
    """Run migrations, build the agent, and launch the Gradio UI."""
    asyncio.run(_run_migrations())

    agent = ChatAgent()
    app = create_ui(agent)
    app.launch(
        server_name=config.APP_HOST,
        server_port=config.APP_PORT,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
