"""Data models for evaluation events and tasks."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import time


@dataclass
class EvaluationEvent:
    """Represents a single event during evaluation."""

    task_id: str
    timestamp: float
    iteration: int
    event_type: str  # "tool_executing" | "tool_completed" | "tool_failed" | "verdict_submitted" | "planner_phase"
    data: Dict[str, Any]
    screenshot: Optional[str] = None  # base64 encoded PNG


@dataclass
class EvaluationTask:
    """Represents an evaluation task."""

    id: str
    markdown_file: str
    query: str
    status: str  # "pending" | "running" | "completed" | "failed"
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    verdict: Optional[bool] = None
    reason: Optional[str] = None
    error: Optional[str] = None
    log_path: Optional[str] = None
    # Planner-specific: structured report dict (None for single-query tasks)
    planner_report: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        task_id: str,
        markdown_file: str,
        query: str,
        log_path: Optional[str] = None,
    ) -> "EvaluationTask":
        """Create a new pending task."""
        return cls(
            id=task_id,
            markdown_file=markdown_file,
            query=query,
            status="pending",
            created_at=time.time(),
            log_path=log_path,
        )

    def start(self):
        """Mark task as running."""
        self.status = "running"
        self.started_at = time.time()

    def complete(self, verdict: bool, reason: str, planner_report: Optional[Dict[str, Any]] = None):
        """Mark task as completed."""
        self.status = "completed"
        self.completed_at = time.time()
        self.verdict = verdict
        self.reason = reason
        self.planner_report = planner_report

    def fail(self, error: str):
        """Mark task as failed."""
        self.status = "failed"
        self.completed_at = time.time()
        self.error = error
