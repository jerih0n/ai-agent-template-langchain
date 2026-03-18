"""PostgreSQL connection helpers using asyncpg (async) and psycopg3 (sync).

Two drivers are intentionally used:
  - asyncpg  : used by thread CRUD commands (async, high-performance)
  - psycopg3 : used by the LangGraph PostgresSaver checkpointer and the
               thread_summary middleware (sync, runs inside thread-pool executor)

Both drivers accept standard postgresql:// URLs once the SQLAlchemy-style
"+psycopg" scheme suffix is stripped.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import asyncpg  # type: ignore[import-not-found]
import psycopg  # type: ignore[import-not-found]

from app.core.config import config


def parse_postgres_url(url: str) -> dict[str, Any]:
    """
    Parse a PostgreSQL connection URL into asyncpg keyword arguments.

    Supports:
        postgresql://user:pass@host:port/dbname
        postgresql+psycopg://user:pass@host:port/dbname   (SQLAlchemy style)

    Uses stdlib ``urllib.parse.urlparse`` to correctly handle URL-encoded
    credentials, IPv6 hostnames, and query-string parameters.

    Returns:
        Dict of keyword arguments suitable for ``asyncpg.connect(**params)``.
    """
    # Strip SQLAlchemy scheme driver suffix  (postgresql+psycopg -> postgresql)
    clean_url = re.sub(r"\+\w+://", "://", url)
    parsed = urlparse(clean_url)

    params: dict[str, Any] = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") or None,
    }
    if parsed.username:
        params["user"] = parsed.username
    if parsed.password:
        params["password"] = parsed.password

    return params


def get_connection_sync(postgres_url: str | None = None) -> psycopg.Connection:
    """
    Return a synchronous psycopg3 connection.

    Used where no event loop is available — e.g. inside thread-pool-executor
    callbacks such as the thread_summary middleware.

    Raises:
        ValueError:       If no POSTGRES_URL is configured.
        ConnectionError:  If the database is unreachable.
    """
    url = postgres_url or config.POSTGRES_URL
    if not url:
        raise ValueError("POSTGRES_URL is required for database operations.")

    # psycopg3 accepts standard postgresql:// URLs directly.
    clean_url = re.sub(r"\+\w+://", "://", url)
    try:
        return psycopg.connect(clean_url)
    except Exception as exc:
        raise ConnectionError(f"Failed to connect to database (sync): {exc}") from exc


async def get_connection(postgres_url: str | None = None) -> asyncpg.Connection:
    """
    Return an asyncpg connection.

    Raises:
        ValueError:       If no POSTGRES_URL is configured.
        ConnectionError:  If the database is unreachable.

    Note:
        Each call opens a new connection.  For high-throughput workloads,
        consider replacing this with an ``asyncpg.Pool``.
    """
    url = postgres_url or config.POSTGRES_URL
    if not url:
        raise ValueError("POSTGRES_URL is required for database operations.")

    conn_params = parse_postgres_url(url)
    try:
        return await asyncpg.connect(**conn_params)
    except Exception as exc:
        raise ConnectionError(f"Failed to connect to database: {exc}") from exc
