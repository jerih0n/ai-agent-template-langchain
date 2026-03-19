from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import tool


@tool(
    "utc_now",
    description="Return the current date and time in UTC, formatted as an ISO-8601 string. Takes no input.",
)
def utc_now() -> str:
    """Get the current UTC datetime as an ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()

