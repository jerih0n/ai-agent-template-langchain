"""Main LangChain-based AI agent logic."""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import config
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from app.ai.memory.short_lived_memory.short_lived_memory_manager import (
    create_checkpointer,
    get_thread_messages,
)
from app.ai.middlewares import summarise_if_new

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "default_system_prompt.md"


def load_default_system_prompt() -> str:
    """Load the default system prompt from `app/ai/prompts/default_system_prompt.md`."""
    try:
        return DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "You are a helpful AI assistant."


@dataclass
class AgentResponse:
    """
    Normalised response returned by ChatAgent.send_message.

    Extend this dataclass to surface additional fields
    (e.g. tool_calls, usage metadata) as your agent grows.
    """

    content: str


class ChatAgent:
    """
    Wraps a LangChain agent with a LangGraph PostgresSaver checkpointer.

    Uses the synchronous PostgresSaver (psycopg3 sync) — compatible with all
    platforms including Windows ProactorEventLoop.  `send_message` runs the
    sync `invoke` call inside `run_in_executor` so the async event loop is
    never blocked.

    Environment variables:
        OPENAI_API_KEY  — required
        POSTGRES_URL    — required for the Postgres checkpointer
    """

    def __init__(
        self,
        *,
        postgres_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
        use_postgres_checkpointer: bool = True,
    ) -> None:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required to use ChatAgent.")

        self.model_name = model or config.AGENT_MODEL
        self.temperature = (
            float(temperature) if temperature is not None else config.AGENT_TEMPERATURE
        )
        self.system_prompt = (
            system_prompt if system_prompt is not None else load_default_system_prompt()
        )

        resolved_url = postgres_url or config.POSTGRES_URL
        self._checkpointer, self._checkpointer_cm = create_checkpointer(
            postgres_url=resolved_url,
            use_postgres_checkpointer=use_postgres_checkpointer,
        )
        self._agent = self._build_agent()

    @classmethod
    async def create(cls, **kwargs: Any) -> "ChatAgent":
        """
        Async-compatible factory — delegates to __init__ (which is synchronous).

        Usage:
            agent = await ChatAgent.create(model="gpt-4o", temperature=0.5)
        """
        return cls(**kwargs)

    def _build_agent(self):
        """Build the LangChain agent graph."""
        self._llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=config.OPENAI_API_KEY,
        )
        return create_agent(
            model=self._llm,
            tools=[],  # ← Add your tools here
            system_prompt=self.system_prompt,
            checkpointer=self._checkpointer,
            middleware=[summarise_if_new],
        )

    async def send_message(self, *, thread_id: str, message: str) -> AgentResponse:
        """
        Send a message to the agent and return the assistant's reply.

        The synchronous `invoke` call runs inside `asyncio.run_in_executor` so
        the event loop is never blocked (psycopg3 sync; safe on Windows
        ProactorEventLoop).
        """
        config_dict = {"configurable": {"thread_id": thread_id}}
        input_dict = {"messages": [HumanMessage(content=message)]}

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._agent.invoke(input_dict, config_dict),
            )
        except Exception:
            traceback.print_exc()
            raise

        messages = result.get("messages", [])
        assistant_msg = messages[-1] if messages else AIMessage(content="")

        return AgentResponse(
            content=getattr(assistant_msg, "content", str(assistant_msg)),
        )

    async def get_messages(self, *, thread_id: str) -> list[dict[str, str]]:
        """
        Load conversation history for a thread from the checkpointer.

        Returns a list of {"role": ..., "content": ...} dicts compatible
        with Gradio's Chatbot message format.
        """
        return await get_thread_messages(self._checkpointer, thread_id=thread_id)

    def close(self) -> None:
        """Release DB resources held by the Postgres checkpointer."""
        if self._checkpointer_cm is not None:
            self._checkpointer_cm.__exit__(None, None, None)
            self._checkpointer_cm = None
