"""Database migration manager (asyncpg).

Discovers *.sql files in the migrations/ folder, tracks executed migrations
in a ``_migrations`` table, and runs any that are pending.

Can be run directly from the command line:
    python -m app.database.sql_database.migrations_manager
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-not-found]

from app.core.config import config
from app.database.sql_database.postgres_db_helper import parse_postgres_url

logger = logging.getLogger(__name__)


class MigrationsManager:
    """
    Runs pending SQL migrations and tracks them in a ``_migrations`` table.

    Migration files must live in the ``migrations/`` directory next to this
    module and be named with a sortable prefix (e.g. ``001_create_foo.sql``).
    """

    def __init__(self, postgres_url: str | None = None) -> None:
        self.postgres_url = postgres_url or config.POSTGRES_URL
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL is required for migrations.")

        self._conn_params: dict[str, Any] = parse_postgres_url(self.postgres_url)
        self.migrations_dir = Path(__file__).parent / "migrations"
        self._target_database: str | None = self._conn_params.get("database")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_pending_migrations(self) -> list[str]:
        """
        Run all pending migrations that have not yet been executed.

        Returns:
            List of migration names that were executed in this run.
        """
        await self._ensure_database_exists()

        conn = await asyncpg.connect(**self._conn_params)
        try:
            await self._ensure_migrations_table(conn)

            all_migrations = self._discover_migrations()
            executed = await self._get_executed_migrations(conn)

            pending = [
                (name, path)
                for name, path in all_migrations
                if name not in executed
            ]

            executed_names: list[str] = []
            for migration_name, file_path in pending:
                await self._run_migration(conn, migration_name, file_path)
                executed_names.append(migration_name)
                logger.info("Executed migration: %s", migration_name)

            if not executed_names:
                logger.info("No pending migrations to run.")

            return executed_names
        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _ensure_database_exists(self) -> None:
        """Create the target database if it does not already exist."""
        if not self._target_database:
            return

        default_params = {**self._conn_params, "database": "postgres"}
        try:
            conn = await asyncpg.connect(**default_params)
        except Exception:
            conn = await asyncpg.connect(**{**self._conn_params, "database": "template1"})

        try:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                self._target_database,
            )
            if exists is None:
                await conn.execute(f'CREATE DATABASE "{self._target_database}"')
                logger.info("Created database: %s", self._target_database)
            else:
                logger.debug("Database already exists: %s", self._target_database)
        finally:
            await conn.close()

    async def _ensure_migrations_table(self, conn: asyncpg.Connection) -> None:
        """Create the _migrations tracking table if it does not exist."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(255) UNIQUE NOT NULL,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def _discover_migrations(self) -> list[tuple[str, Path]]:
        """Return (name, path) tuples for all *.sql files, sorted by filename."""
        if not self.migrations_dir.exists():
            return []
        return [
            (path.stem, path)
            for path in sorted(self.migrations_dir.glob("*.sql"))
        ]

    async def _get_executed_migrations(self, conn: asyncpg.Connection) -> set[str]:
        """Return the set of migration names already recorded in _migrations."""
        rows = await conn.fetch("SELECT migration_name FROM _migrations")
        return {row["migration_name"] for row in rows}

    async def _run_migration(
        self,
        conn: asyncpg.Connection,
        migration_name: str,
        file_path: Path,
    ) -> None:
        """Execute one migration file inside a transaction and record it."""
        sql_content = file_path.read_text(encoding="utf-8")
        async with conn.transaction():
            await conn.execute(sql_content)
            await conn.execute(
                "INSERT INTO _migrations (migration_name) VALUES ($1)",
                migration_name,
            )


async def run_migrations(postgres_url: str | None = None) -> list[str]:
    """
    Convenience coroutine — run all pending migrations and return their names.
    """
    manager = MigrationsManager(postgres_url)
    return await manager.run_pending_migrations()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_migrations())
