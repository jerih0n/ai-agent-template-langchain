"""Async CRUD commands for the threads table (asyncpg)."""

from __future__ import annotations

from typing import Any

import asyncpg  # type: ignore[import-not-found]

from app.database.sql_database.postgres_db_helper import get_connection


async def insert_thread(
    thread_id: str,
    postgres_url: str | None = None,
) -> dict[str, Any]:
    """
    Insert a new thread record into the threads table.

    Args:
        thread_id:    Unique thread identifier.
        postgres_url: Optional PostgreSQL URL; defaults to config.POSTGRES_URL.

    Returns:
        Dict with keys: id, thread_id, created_at, summary.

    Raises:
        ValueError:   If thread_id is empty or already exists.
        RuntimeError: If the insert fails for any other reason.
    """
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id cannot be empty")

    conn = await get_connection(postgres_url)
    try:
        row = await conn.fetchrow(
            """
            INSERT INTO threads (thread_id, summary)
            VALUES ($1, $2)
            RETURNING id, thread_id, created_at, summary
            """,
            thread_id,
            "New Thread",
        )

        if not row:
            raise RuntimeError("Failed to insert thread — no row returned")

        return {
            "id": row["id"],
            "thread_id": row["thread_id"],
            "created_at": row["created_at"],
            "summary": row["summary"],
        }
    except asyncpg.UniqueViolationError as exc:
        raise ValueError(f"Thread '{thread_id}' already exists") from exc
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Error inserting thread: {exc}") from exc
    finally:
        await conn.close()


async def get_all_threads(postgres_url: str | None = None) -> list[dict[str, Any]]:
    """
    Return all threads ordered newest-first.

    Returns:
        List of dicts with keys: id, thread_id, created_at, summary.
    """
    conn = await get_connection(postgres_url)
    try:
        rows = await conn.fetch(
            """
            SELECT id, thread_id, created_at, summary
            FROM threads
            ORDER BY created_at DESC
            """
        )
        return [
            {
                "id": row["id"],
                "thread_id": row["thread_id"],
                "created_at": row["created_at"],
                "summary": row["summary"],
            }
            for row in rows
        ]
    except Exception as exc:
        raise RuntimeError(f"Error fetching all threads: {exc}") from exc
    finally:
        await conn.close()


async def delete_thread(
    thread_id: str,
    postgres_url: str | None = None,
) -> bool:
    """
    Delete a thread record by thread_id.

    Returns:
        True if a row was deleted, False if thread_id was not found.

    Raises:
        ValueError:   If thread_id is empty.
        RuntimeError: If the delete fails.
    """
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id cannot be empty")

    conn = await get_connection(postgres_url)
    try:
        result = await conn.execute(
            "DELETE FROM threads WHERE thread_id = $1",
            thread_id,
        )
        # asyncpg returns a status string like "DELETE 1" or "DELETE 0"
        deleted_count = int(result.split()[-1])
        return deleted_count > 0
    except Exception as exc:
        raise RuntimeError(f"Error deleting thread: {exc}") from exc
    finally:
        await conn.close()
