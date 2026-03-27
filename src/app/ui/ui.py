"""Gradio UI for the LangChain Chat Agent template."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import gradio as gr

from app.ai.agent import ChatAgent
from app.ai.memory.short_lived_memory.short_lived_memory_manager import (
    create_new_thread,
    delete_thread,
    get_all_threads,
)
from app.core.config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_thread_id(raw_value) -> str | None:
    """
    Safely extract a thread_id string from a Gradio Dropdown value.

    Gradio 6.9 may return a (label, value) tuple instead of a plain string.
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
        return str(raw_value[1])
    return str(raw_value)


def _delete_button_update(thread_id: str | None) -> gr.update:
    """Enable delete only when a persisted thread is selected."""
    return gr.update(interactive=bool(_extract_thread_id(thread_id)))


async def _fetch_thread_choices() -> list[tuple[str, str]]:
    """
    Fetch all threads and return as (display_label, thread_id) tuples.
    Returns [] on any error so the UI degrades gracefully.
    """
    try:
        threads = await get_all_threads()
        choices: list[tuple[str, str]] = []
        for thread in threads:
            thread_id = thread["thread_id"]
            summary = thread.get("summary") or "New Thread"
            created_at = thread.get("created_at")
            if created_at:
                date_str = (
                    created_at.strftime("%Y-%m-%d %H:%M")
                    if hasattr(created_at, "strftime")
                    else str(created_at)
                )
                label = f"{summary} ({date_str})"
            else:
                label = summary
            choices.append((label, thread_id))
        return choices
    except Exception:
        logger.exception("Error loading threads")
        return []


# ---------------------------------------------------------------------------
# Gradio event handlers (depend on the injected agent)
# ---------------------------------------------------------------------------

def _make_handlers(agent: ChatAgent):
    """
    Return all Gradio event-handler coroutines closed over the given agent.

    Using a factory keeps the agent dependency explicit and the module
    testable without a live database connection.
    """

    async def send_message(
        message: str,
        history: list[dict[str, str]],
        thread_id: Optional[str],
    ):
        """Send a message to the agent; create a thread only on first send.

        Yields twice so the user message appears in the chatbot immediately,
        before the (potentially slow) agent response arrives.
        """
        thread_id = _extract_thread_id(thread_id)

        if not message.strip():
            yield history, "", gr.update(), _delete_button_update(thread_id)
            return

        if not thread_id:
            try:
                thread_id = await create_new_thread()
            except Exception as exc:
                logger.exception("Error auto-creating thread")
                yield (
                    history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": f"Failed to create thread: {exc}"},
                    ],
                    "",
                    gr.update(),
                    _delete_button_update(None),
                )
                return

        # Show user message immediately with a live "thinking" placeholder.
        start_ts = time.monotonic()
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Thinking... (0s)"},
        ]
        yield history, "", gr.update(), _delete_button_update(thread_id)

        # Run the model call in the background and update elapsed time every second.
        task = asyncio.create_task(agent.send_message(thread_id=thread_id, message=message))
        while not task.done():
            elapsed_s = int(time.monotonic() - start_ts)
            history[-1] = {"role": "assistant", "content": f"Thinking... ({elapsed_s}s)"}
            yield history, "", gr.update(), _delete_button_update(thread_id)
            await asyncio.sleep(1)

        try:
            response = await task
            history[-1] = {"role": "assistant", "content": response.content}
        except Exception as exc:
            logger.exception("Error sending message to agent")
            history[-1] = {"role": "assistant", "content": f"Error: {exc}"}

        choices = await _fetch_thread_choices()
        yield (
            history,
            "",
            gr.update(choices=choices, value=thread_id),
            _delete_button_update(thread_id),
        )

    async def load_threads() -> tuple[gr.update, list[dict[str, str]], gr.update]:
        """Populate the thread dropdown and start in a fresh unsaved conversation."""
        choices = await _fetch_thread_choices()
        return (
            gr.update(choices=choices, value=None),
            [],
            _delete_button_update(None),
        )

    async def select_thread(thread_id) -> tuple[list[dict[str, str]], gr.update]:
        """Load conversation history when the user picks a persisted thread."""
        thread_id = _extract_thread_id(thread_id)
        if not thread_id:
            return [], _delete_button_update(None)
        try:
            history = await agent.get_messages(thread_id=thread_id)
            return history, _delete_button_update(thread_id)
        except Exception:
            logger.exception("Error loading messages for thread %s", thread_id)
            return [], _delete_button_update(None)

    async def create_new_conversation() -> tuple[gr.update, list, gr.update]:
        """Create a fresh thread and switch the UI to it."""
        try:
            new_thread_id = await create_new_thread()
            choices = await _fetch_thread_choices()
            return (
                gr.update(choices=choices, value=new_thread_id),
                [],
                _delete_button_update(new_thread_id),
            )
        except Exception:
            logger.exception("Error creating new conversation")
            choices = await _fetch_thread_choices()
            return (
                gr.update(choices=choices, value=None),
                [],
                _delete_button_update(None),
            )

    async def delete_conversation(thread_id) -> tuple[gr.update, list, gr.update]:
        """Delete the selected thread and clear the chat view."""
        thread_id = _extract_thread_id(thread_id)
        if not thread_id:
            choices = await _fetch_thread_choices()
            return (
                gr.update(choices=choices, value=None),
                [],
                _delete_button_update(None),
            )

        try:
            await delete_thread(thread_id)
        except Exception:
            logger.exception("Error deleting thread %s", thread_id)

        choices = await _fetch_thread_choices()
        return (
            gr.update(choices=choices, value=None),
            [],
            _delete_button_update(None),
        )

    async def refresh_after_send(thread_id) -> gr.update:
        """
        Refresh the thread list 2 s after a message is sent.

        The delay gives the @after_agent middleware time to write the
        auto-generated summary before the dropdown label is reloaded.
        """
        await asyncio.sleep(2)
        thread_id = _extract_thread_id(thread_id)
        choices = await _fetch_thread_choices()
        return gr.update(choices=choices, value=thread_id)

    return (
        send_message,
        load_threads,
        select_thread,
        create_new_conversation,
        delete_conversation,
        refresh_after_send,
    )


# ---------------------------------------------------------------------------
# UI factory
# ---------------------------------------------------------------------------

def create_ui(agent: ChatAgent) -> gr.Blocks:
    """
    Build and return the Gradio Blocks chat interface.

    Args:
        agent: An initialised ChatAgent instance.

    Returns:
        Gradio Blocks app ready for ``app.launch()``.
    """
    (
        send_message,
        load_threads,
        select_thread,
        create_new_conversation,
        delete_conversation,
        refresh_after_send,
    ) = _make_handlers(agent)

    app_name = config.APP_NAME

    with gr.Blocks(title=app_name) as app:
        gr.Markdown(f"# {app_name}")

        with gr.Row():
            # ── Left sidebar ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### Threads")
                thread_list = gr.Dropdown(
                    choices=[],
                    label="Select Thread",
                    value=None,
                    interactive=True,
                    allow_custom_value=False,
                )
                with gr.Row():
                    new_thread_btn = gr.Button(
                        "New Conversation",
                        variant="primary",
                        scale=3,
                    )
                    delete_thread_btn = gr.Button(
                        "Delete Thread",
                        variant="stop",
                        scale=1,
                        interactive=False,
                    )

            # ── Chat area ─────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        lines=1,
                        max_lines=6,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        # ── Event wiring ──────────────────────────────────────────────────
        for trigger in (send_btn.click, msg_input.submit):
            trigger(
                fn=send_message,
                inputs=[msg_input, chatbot, thread_list],
                outputs=[chatbot, msg_input, thread_list, delete_thread_btn],
            ).then(
                fn=refresh_after_send,
                inputs=[thread_list],
                outputs=[thread_list],
            )

        app.load(fn=load_threads, outputs=[thread_list, chatbot, delete_thread_btn])
        thread_list.change(
            fn=select_thread,
            inputs=[thread_list],
            outputs=[chatbot, delete_thread_btn],
        )
        new_thread_btn.click(
            fn=create_new_conversation,
            outputs=[thread_list, chatbot, delete_thread_btn],
        )
        delete_thread_btn.click(
            fn=delete_conversation,
            inputs=[thread_list],
            outputs=[thread_list, chatbot, delete_thread_btn],
        )

    return app
