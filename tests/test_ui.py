from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ui import ui as ui_module


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def send_message(self, *, thread_id: str, message: str) -> SimpleNamespace:
        self.calls.append((thread_id, message))
        return SimpleNamespace(content="Hi there")

    async def get_messages(self, *, thread_id: str) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": f"Loaded {thread_id}"}]


@pytest.mark.asyncio
async def test_load_threads_starts_in_new_thread_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_all_threads() -> list[dict[str, str | None]]:
        return [{"thread_id": "thread-1", "summary": "Existing Thread", "created_at": None}]

    monkeypatch.setattr(ui_module, "get_all_threads", fake_get_all_threads)

    agent = FakeAgent()
    _, load_threads, _, _, _, _ = ui_module._make_handlers(agent)

    thread_update, history, delete_update = await load_threads()

    assert thread_update["value"] is None
    assert thread_update["choices"] == [("Existing Thread", "thread-1")]
    assert history == []
    assert delete_update["interactive"] is False


@pytest.mark.asyncio
async def test_send_message_creates_thread_on_first_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_new_thread() -> str:
        return "thread-1"

    async def fake_fetch_thread_choices() -> list[tuple[str, str]]:
        return [("Existing Thread", "thread-1")]

    monkeypatch.setattr(ui_module, "create_new_thread", fake_create_new_thread)
    monkeypatch.setattr(ui_module, "_fetch_thread_choices", fake_fetch_thread_choices)

    agent = FakeAgent()
    send_message, _, _, _, _, _ = ui_module._make_handlers(agent)

    history, cleared_message, thread_update, delete_update = await send_message(
        "Hello",
        [],
        None,
    )

    assert agent.calls == [("thread-1", "Hello")]
    assert history == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    assert cleared_message == ""
    assert thread_update["value"] == "thread-1"
    assert delete_update["interactive"] is True
