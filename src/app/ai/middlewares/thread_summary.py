"""
After-agent middleware: auto-summarise a new thread after the first reply.

Registered via `create_agent(middleware=[summarise_if_new])` in agent.py.
Fires once after the agent completes its response, but only on the very first
exchange (exactly 1 human + 1 AI message) so that each thread gets a label
without repeated LLM calls.

Fully synchronous — uses psycopg3 (sync) for the DB write and llm.invoke
(sync) for the LLM call.  Safe to run inside a thread-pool executor.

Reference: https://docs.langchain.com/oss/python/langchain/middleware/custom
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain.agents.middleware import after_agent, AgentState  # type: ignore[import-not-found]
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_config  # type: ignore[import-not-found]
from langgraph.runtime import Runtime  # type: ignore[import-not-found]

from app.core.config import config
from app.database.sql_database.postgres_db_helper import get_connection_sync

logger = logging.getLogger(__name__)

# --- Module-level caches (initialised once at import time) ---

_SUMMARY_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "thread_summary_prompt.md"

try:
    _SUMMARY_PROMPT_TEMPLATE: str = _SUMMARY_PROMPT_PATH.read_text(encoding="utf-8").strip()
except OSError:
    _SUMMARY_PROMPT_TEMPLATE = (
        "In 5 words or fewer, what is this conversation about? "
        "Reply with only the short topic label, no punctuation.\n\n"
        "First user message: {message}"
    )

# Single LLM instance shared across all middleware invocations.
_summary_llm: ChatOpenAI | None = None


def _get_summary_llm() -> ChatOpenAI:
    """Return the cached summary LLM, creating it on first call."""
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = ChatOpenAI(
            model=config.AGENT_MODEL,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY,
        )
    return _summary_llm


def _update_thread_sync(thread_id: str, summary: str) -> None:
    """Write the generated summary to the threads table via a sync psycopg3 connection."""
    conn = get_connection_sync()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE threads SET summary = %s WHERE thread_id = %s",
                (summary, thread_id),
            )
        conn.commit()
    finally:
        conn.close()


@after_agent
def summarise_if_new(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    After-agent hook — generates and stores a short thread label.

    Only fires on the very first exchange (2 messages: 1 human + 1 AI).
    thread_id is read from the LangGraph config context variable set during
    graph execution.
    """
    try:
        cfg = get_config()
        thread_id: str | None = cfg.get("configurable", {}).get("thread_id") if cfg else None
        if not thread_id:
            return None

        messages = state.get("messages", [])
        if len(messages) != 2:
            return None

        first_user_msg = next(
            (m for m in messages if isinstance(m, HumanMessage)), None
        )
        if not first_user_msg:
            return None

        prompt = _SUMMARY_PROMPT_TEMPLATE.format(message=first_user_msg.content[:500])
        response = _get_summary_llm().invoke(prompt)
        summary = response.content.strip()[:100]

        if summary:
            _update_thread_sync(thread_id, summary)

    except Exception:
        logger.exception("summarise_if_new middleware failed")

    return None
