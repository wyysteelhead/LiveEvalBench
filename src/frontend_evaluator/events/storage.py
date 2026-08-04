"""SQLite storage for evaluation events and tasks."""

import sqlite3
import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import EvaluationEvent, EvaluationTask
from ..utils.logger import logger


class EventStorage:
    """Stores evaluation events and tasks in SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage.

        Args:
            db_path: Path to SQLite database file.
                     If None, uses evaluations.db in the project root.
        """
        if db_path is None:
            # Project root is 4 levels up from this file:
            # src/frontend_evaluator/events/storage.py -> project root
            project_root = Path(__file__).parents[3]
            db_path = str(project_root / "evaluations.db")
        
        self.db_path = db_path
        self._init_db()
        logger.debug(f"EventStorage initialized with database: {self.db_path}")

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                markdown_file TEXT NOT NULL,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                verdict INTEGER,
                reason TEXT,
                error TEXT,
                log_path TEXT,
                planner_report TEXT
            )
        """)

        conn.commit()

        # Migration: add planner_report column if it doesn't exist
        # Check after commit to ensure table exists
        cursor.execute("PRAGMA table_info(tasks)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'planner_report' not in columns:
            logger.info("Migrating database: adding planner_report column")
            cursor.execute("ALTER TABLE tasks ADD COLUMN planner_report TEXT")
            conn.commit()
        if 'log_path' not in columns:
            logger.info("Migrating database: adding log_path column")
            cursor.execute("ALTER TABLE tasks ADD COLUMN log_path TEXT")
            conn.commit()

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                iteration INTEGER,
                event_type TEXT NOT NULL,
                data TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        """)

        # Screenshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS screenshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                event_id INTEGER,
                timestamp REAL NOT NULL,
                screenshot_b64 TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES tasks(id),
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_task_id ON events(task_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_screenshots_task_id ON screenshots(task_id)"
        )

        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")

        conn.commit()
        conn.close()

        logger.debug(f"Database initialized at {self.db_path}")

    def save_task(self, task: EvaluationTask):
        """Save or update a task.

        Args:
            task: Task to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO tasks
            (id, markdown_file, query, status, created_at, started_at, completed_at, verdict, reason, error, log_path, planner_report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task.id,
                task.markdown_file,
                task.query,
                task.status,
                task.created_at,
                task.started_at,
                task.completed_at,
                task.verdict,
                task.reason,
                task.error,
                task.log_path,
                json.dumps(task.planner_report) if task.planner_report else None,
            ),
        )

        conn.commit()
        conn.close()

    def save_event(self, event: EvaluationEvent):
        """Save an event.

        Args:
            event: Event to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save event
        cursor.execute(
            """
            INSERT INTO events (task_id, timestamp, iteration, event_type, data)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                event.task_id,
                event.timestamp,
                event.iteration,
                event.event_type,
                json.dumps(event.data),
            ),
        )

        event_id = cursor.lastrowid

        # Save screenshot if present
        if event.screenshot:
            cursor.execute(
                """
                INSERT INTO screenshots (task_id, event_id, timestamp, screenshot_b64)
                VALUES (?, ?, ?, ?)
            """,
                (event.task_id, event_id, event.timestamp, event.screenshot),
            )

        conn.commit()
        conn.close()

    def get_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return EvaluationTask(
            id=row['id'],
            markdown_file=row['markdown_file'],
            query=row['query'],
            status=row['status'],
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            verdict=bool(row['verdict']) if row['verdict'] is not None else None,
            reason=row['reason'],
            error=row['error'],
            log_path=row['log_path'] if 'log_path' in row.keys() else None,
            planner_report=json.loads(row['planner_report']) if row['planner_report'] else None,
        )

    def get_all_tasks(self) -> List[EvaluationTask]:
        """Get all tasks.

        Returns:
            List of tasks
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            tasks.append(
                EvaluationTask(
                    id=row['id'],
                    markdown_file=row['markdown_file'],
                    query=row['query'],
                    status=row['status'],
                    created_at=row['created_at'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    verdict=bool(row['verdict']) if row['verdict'] is not None else None,
                    reason=row['reason'],
                    error=row['error'],
                    log_path=row['log_path'] if 'log_path' in row.keys() else None,
                    planner_report=json.loads(row['planner_report']) if row['planner_report'] else None,
                )
            )

        return tasks

    def get_tasks_summary(self) -> List[Dict[str, Any]]:
        """Get lightweight task summaries without loading full planner_report blobs.

        Uses json_extract to pull only overall_score from the planner_report JSON,
        avoiding the cost of deserializing large report objects for every task.

        Returns:
            List of task summary dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id, markdown_file, query, status,
                created_at, started_at, completed_at,
                verdict, reason, error,
                json_extract(planner_report, '$.overall_score') AS overall_score,
                CASE WHEN planner_report IS NULL THEN NULL
                     ELSE (
                         SELECT COUNT(*)
                         FROM json_each(planner_report, '$.behavior_checks')
                         WHERE json_extract(value, '$.verdict.verdict') IS NULL
                            OR json_extract(value, '$.verdict.verdict') != 'not_applicable'
                     )
                END AS interaction_count
            FROM tasks
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_stats(self) -> Dict[str, int]:
        """Get aggregate task statistics using a single SQL query.

        Returns:
            Dict with total, pending, running, completed, failed, passed, failed_verdict counts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'pending'   THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status = 'running'   THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = 'failed'    THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN verdict = 1          THEN 1 ELSE 0 END) AS passed,
                SUM(CASE WHEN verdict = 0          THEN 1 ELSE 0 END) AS failed_verdict
            FROM tasks
        """)
        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"total": 0, "pending": 0, "running": 0, "completed": 0,
                    "failed": 0, "passed": 0, "failed_verdict": 0}

        return {
            "total":          row[0] or 0,
            "pending":        row[1] or 0,
            "running":        row[2] or 0,
            "completed":      row[3] or 0,
            "failed":         row[4] or 0,
            "passed":         row[5] or 0,
            "failed_verdict": row[6] or 0,
        }

    def get_task_events(self, task_id: str) -> List[EvaluationEvent]:
        """Get all events for a task.

        Args:
            task_id: Task ID

        Returns:
            List of events
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT e.*, s.screenshot_b64
            FROM events e
            LEFT JOIN screenshots s ON e.id = s.event_id
            WHERE e.task_id = ?
            ORDER BY e.timestamp
        """,
            (task_id,),
        )

        events = []
        for row in cursor.fetchall():
            events.append(
                EvaluationEvent(
                    task_id=row['task_id'],
                    timestamp=row['timestamp'],
                    iteration=row['iteration'],
                    event_type=row['event_type'],
                    data=json.loads(row['data']),
                    screenshot=row['screenshot_b64'],
                )
            )

        conn.close()
        return events

    def update_planner_report(self, task_id: str, planner_report: dict):
        """Update only the planner_report field for an existing task.

        Args:
            task_id: Task ID
            planner_report: New planner report dict
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE tasks SET planner_report = ? WHERE id = ?",
            (json.dumps(planner_report), task_id),
        )
        conn.commit()
        conn.close()

    def cleanup_old_data(self, days: int = 7):
        """Delete data older than specified days.

        Args:
            days: Number of days to keep
        """
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM tasks WHERE completed_at < ?", (cutoff,))
        cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        cursor.execute("DELETE FROM screenshots WHERE timestamp < ?", (cutoff,))

        deleted_tasks = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted_tasks} old tasks")



    def get_tasks_paginated(self, page: int = 1, page_size: int = 10, status_filter: Optional[str] = None) -> tuple[List[EvaluationTask], int]:
        """Get tasks with pagination and optional filtering.

        Args:
            page: Page number (1-based)
            page_size: Number of tasks per page
            status_filter: Optional status filter ('passed', 'failed', 'error', 'running')

        Returns:
            Tuple of (List of tasks, total count)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build WHERE clause based on filter
        where_clause = ""
        params = []

        if status_filter:
            if status_filter == 'passed':
                where_clause = "WHERE status = ? AND verdict = ?"
                params = ['completed', 1]
            elif status_filter == 'failed':
                where_clause = "WHERE status = ? AND verdict = ?"
                params = ['completed', 0]
            elif status_filter == 'error':
                where_clause = "WHERE status = ?"
                params = ['failed']
            elif status_filter == 'running':
                where_clause = "WHERE status = ?"
                params = ['running']

        # Get total count with filter
        count_query = f"SELECT COUNT(*) FROM tasks {where_clause}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]

        # Get paginated tasks with filter
        offset = (page - 1) * page_size
        query = f"SELECT * FROM tasks {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        cursor.execute(query, params + [page_size, offset])
        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            tasks.append(
                EvaluationTask(
                    id=row['id'],
                    markdown_file=row['markdown_file'],
                    query=row['query'],
                    status=row['status'],
                    created_at=row['created_at'],
                    started_at=row['started_at'],
                    completed_at=row['completed_at'],
                    verdict=bool(row['verdict']) if row['verdict'] is not None else None,
                    reason=row['reason'],
                    error=row['error'],
                    log_path=row['log_path'] if 'log_path' in row.keys() else None,
                    planner_report=json.loads(row['planner_report']) if row['planner_report'] else None,
                )
            )

        return tasks, total_count