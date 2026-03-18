"""
Short-lived (thread-level) memory management helpers.

Encapsulates LangGraph checkpointer creation for persisting an agent's
per-thread state and exposes thin async helpers for thread CRUD used by the UI.

Reference: https://docs.langchain.com/oss/python/langchain/short-term-memory
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-not-found]

from app.database.sql_database.commands.threads_commands import (
    insert_thread,
    get_all_threads as _get_all_threads,
    delete_thread as _delete_thread,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpointer factory
# ---------------------------------------------------------------------------

def create_checkpointer(
    *,
    postgres_url: str | None,
    use_postgres_checkpointer: bool = True,
) -> tuple[Any, Any | None]:
    """
    Create a LangGraph checkpointer for `create_agent(..., checkpointer=...)`.

    Uses the synchronous PostgresSaver (psycopg3 sync) — safe on all platforms
    including Windows ProactorEventLoop.  The agent calls sync `invoke` via
    run_in_executor so the event loop is never blocked.

    Returns:
        (checkpointer, context_manager)  — context_manager is None for InMemorySaver.
    """
    if not use_postgres_checkpointer:
        return InMemorySaver(), None

    if not postgres_url:
        raise ValueError("postgres_url is required when use_postgres_checkpointer=True.")

    cm = PostgresSaver.from_conn_string(postgres_url)
    checkpointer = cm.__enter__()
    checkpointer.setup()
    return checkpointer, cm


# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------

async def create_new_thread(postgres_url: str | None = None) -> str:
    """
    Generate a new UUID thread_id and persist it in the threads table.

    Returns:
        The generated thread_id string.
    """
    thread_id = str(uuid4())
    try:
        await insert_thread(thread_id, postgres_url)
        return thread_id
    except ValueError as exc:
        raise ValueError(f"Failed to create thread: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error creating new thread in database: {exc}") from exc


async def get_all_threads(postgres_url: str | None = None) -> list[dict]:
    """
    Return all threads from the database ordered newest-first.

    Returns:
        List of dicts with keys: id, thread_id, created_at, summary.
    """
    try:
        return await _get_all_threads(postgres_url)
    except Exception as exc:
        raise RuntimeError(f"Error fetching all threads: {exc}") from exc


async def delete_thread(thread_id: str, postgres_url: str | None = None) -> bool:
    """
    Delete a thread record by thread_id.

    Returns:
        True if deleted, False if thread_id was not found.
    """
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id cannot be empty")
    try:
        return await _delete_thread(thread_id, postgres_url)
    except Exception as exc:
        raise RuntimeError(f"Error deleting thread: {exc}") from exc


# ---------------------------------------------------------------------------
# Message history loader
# ---------------------------------------------------------------------------

async def get_thread_messages(checkpointer: Any, *, thread_id: str) -> list[dict[str, str]]:
    """
    Load conversation history for a thread from a LangGraph checkpointer.

    Reads the stored checkpoint and converts messages into Gradio's
    {"role": ..., "content": ...} format.

    Args:
        checkpointer: A LangGraph checkpointer (PostgresSaver or InMemorySaver).
        thread_id:    The conversation thread identifier.

    Returns:
        List of {"role": ..., "content": ...} dicts, or [] if no history.
    """
    if checkpointer is None:
        return []

    config_dict = {"configurable": {"thread_id": thread_id}}

    try:
        loop = asyncio.get_running_loop()
        checkpoint_tuple = await loop.run_in_executor(
            None,
            lambda: checkpointer.get_tuple(config_dict),
        )
    except Exception:
        logger.exception("Failed to load checkpoint for thread %s", thread_id)
        return []

    if checkpoint_tuple is None:
        return []

    raw_messages = (
        checkpoint_tuple.checkpoint
        .get("channel_values", {})
        .get("messages", [])
    )

    history: list[dict[str, str]] = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
        # SystemMessage and other internal messages are intentionally skipped

    return history
