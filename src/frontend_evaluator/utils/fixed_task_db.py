"""SQLite-backed storage for canonical query fixed tasks and query-agent bindings."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class FixedTaskDB:
    """Persist canonical fixed tasks for queries and their agent bindings."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self._ensure_schema()

    @staticmethod
    def _query_hash(query_text: str) -> str:
        return hashlib.sha256(query_text.strip().encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL UNIQUE,
                    query_text TEXT NOT NULL UNIQUE,
                    fixed_task_count INTEGER NOT NULL,
                    fixed_tasks_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL UNIQUE,
                    agent_name TEXT,
                    stage TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS query_agent_fixed_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER NOT NULL,
                    agent_id INTEGER NOT NULL,
                    fixed_task_count INTEGER NOT NULL,
                    fixed_tasks_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(query_id, agent_id),
                    FOREIGN KEY(query_id) REFERENCES queries(id) ON DELETE CASCADE,
                    FOREIGN KEY(agent_id) REFERENCES agents(id) ON DELETE CASCADE
                );
                """
            )

    def get_query_tasks(self, query_text: str) -> Optional[List[Dict[str, Any]]]:
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return None

        with self._connect() as conn:
            row = conn.execute(
                "SELECT fixed_tasks_json FROM queries WHERE query_hash = ?",
                (self._query_hash(normalized_query),),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["fixed_tasks_json"]))

    def ensure_query_tasks(self, query_text: str, fixed_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return []

        existing = self.get_query_tasks(normalized_query)
        if existing is not None:
            return existing

        payload_json = json.dumps(fixed_tasks, ensure_ascii=False)
        query_hash = self._query_hash(normalized_query)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queries (query_hash, query_text, fixed_task_count, fixed_tasks_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_hash) DO NOTHING
                """,
                (query_hash, normalized_query, len(fixed_tasks), payload_json),
            )
        return self.get_query_tasks(normalized_query) or list(fixed_tasks)

    def upsert_query_tasks(self, query_text: str, fixed_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return []

        payload_json = json.dumps(fixed_tasks, ensure_ascii=False)
        query_hash = self._query_hash(normalized_query)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queries (query_hash, query_text, fixed_task_count, fixed_tasks_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_hash) DO UPDATE SET
                    query_text = excluded.query_text,
                    fixed_task_count = excluded.fixed_task_count,
                    fixed_tasks_json = excluded.fixed_tasks_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (query_hash, normalized_query, len(fixed_tasks), payload_json),
            )
        return self.get_query_tasks(normalized_query) or list(fixed_tasks)

    def _ensure_agent(self, agent_id: str, agent_name: str, stage: str) -> int:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agents (agent_id, agent_name, stage)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    agent_name = excluded.agent_name,
                    stage = excluded.stage,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (agent_id, agent_name, stage),
            )
            row = conn.execute(
                "SELECT id FROM agents WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to ensure agent row for {agent_id}")
        return int(row["id"])

    def _query_id(self, query_text: str) -> Optional[int]:
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM queries WHERE query_hash = ?",
                (self._query_hash(normalized_query),),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    def get_query_agent_tasks(self, query_text: str, agent_id: str) -> Optional[List[Dict[str, Any]]]:
        normalized_query = str(query_text or "").strip()
        normalized_agent_id = str(agent_id or "").strip()
        if not normalized_query or not normalized_agent_id:
            return None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT qaf.fixed_tasks_json
                FROM query_agent_fixed_tasks AS qaf
                JOIN queries AS q ON q.id = qaf.query_id
                JOIN agents AS a ON a.id = qaf.agent_id
                WHERE q.query_hash = ? AND a.agent_id = ?
                """,
                (self._query_hash(normalized_query), normalized_agent_id),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["fixed_tasks_json"]))

    def upsert_agent_specific_tasks(
        self,
        *,
        query_text: str,
        agent_id: str,
        agent_name: str,
        stage: str,
        fixed_tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Store agent-specific tasks in query_agent_fixed_tasks independently.

        Unlike bind_query_tasks_to_agent, this method does NOT reference or
        overwrite the shared ``queries`` table — it stores the provided tasks
        directly so each agent can have its own generated tasks.
        """
        normalized_query = str(query_text or "").strip()
        if not normalized_query:
            return []

        payload_json = json.dumps(fixed_tasks, ensure_ascii=False)
        query_hash = self._query_hash(normalized_query)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queries (query_hash, query_text, fixed_task_count, fixed_tasks_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_hash) DO NOTHING
                """,
                (query_hash, normalized_query, len(fixed_tasks), payload_json),
            )
            conn.execute(
                """
                INSERT INTO agents (agent_id, agent_name, stage)
                VALUES (?, ?, ?)
                ON CONFLICT(agent_id) DO UPDATE SET
                    agent_name = excluded.agent_name,
                    stage = excluded.stage,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (agent_id, agent_name, stage),
            )
            query_row = conn.execute(
                "SELECT id FROM queries WHERE query_hash = ?", (query_hash,)
            ).fetchone()
            agent_row = conn.execute(
                "SELECT id FROM agents WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            if query_row is not None and agent_row is not None:
                conn.execute(
                    """
                    INSERT INTO query_agent_fixed_tasks (query_id, agent_id, fixed_task_count, fixed_tasks_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(query_id, agent_id) DO UPDATE SET
                        fixed_task_count = excluded.fixed_task_count,
                        fixed_tasks_json = excluded.fixed_tasks_json,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (int(query_row["id"]), int(agent_row["id"]), len(fixed_tasks), payload_json),
                )
        return list(fixed_tasks)

    def bind_query_tasks_to_agent(
        self,
        *,
        query_text: str,
        agent_id: str,
        agent_name: str,
        stage: str,
        fixed_tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        canonical_tasks = self.ensure_query_tasks(query_text, fixed_tasks)
        query_id = self._query_id(query_text)
        if query_id is None:
            raise RuntimeError("Failed to resolve query row after ensuring query tasks")
        agent_row_id = self._ensure_agent(agent_id, agent_name, stage)
        payload_json = json.dumps(canonical_tasks, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO query_agent_fixed_tasks (query_id, agent_id, fixed_task_count, fixed_tasks_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(query_id, agent_id) DO UPDATE SET
                    fixed_task_count = excluded.fixed_task_count,
                    fixed_tasks_json = excluded.fixed_tasks_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (query_id, agent_row_id, len(canonical_tasks), payload_json),
            )
        return canonical_tasks