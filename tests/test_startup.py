from __future__ import annotations

import pytest

import run
from app.ai.agent import ChatAgent


def test_validate_required_config_requires_openai_and_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run.config, "OPENAI_API_KEY", None)
    monkeypatch.setattr(run.config, "POSTGRES_URL", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY, POSTGRES_URL"):
        run._validate_required_config()


def test_chat_agent_requires_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.ai.agent.config.OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("app.ai.agent.config.POSTGRES_URL", None)

    with pytest.raises(ValueError, match="persistent thread memory"):
        ChatAgent()
