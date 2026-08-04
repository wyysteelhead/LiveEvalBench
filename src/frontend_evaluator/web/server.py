"""FastAPI server for web visualization."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..events.storage import EventStorage
from ..events.models import EvaluationTask, EvaluationEvent
from ..utils.logger import logger


def _normalize_planner_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize planner report shape for backward/forward compatibility."""
    if not isinstance(report, dict):
        return {}

    normalized = dict(report)
    behavior_checks = normalized.get("behavior_checks") or []
    normalized_checks: List[Dict[str, Any]] = []
    for i, check in enumerate(behavior_checks):
        if not isinstance(check, dict):
            continue
        c = dict(check)
        query_id = c.get("query_id", f"behavior_{i}")
        c.setdefault("query_id", query_id)
        c.setdefault("standard_id", query_id.split("__", 1)[0] if "__" in query_id else query_id)
        c.setdefault("query_text", "")
        c.setdefault("category", "behavior")
        c.setdefault("guideline_ref", "")
        c.setdefault("execution_group_id", query_id)
        c.setdefault("execution_standard_ids", [c["standard_id"]])
        c.setdefault("scenario_id", None)
        c.setdefault("scenario_index", None)
        c.setdefault("scenario_weight", 1.0)
        c.setdefault("check_nodes", [])
        c.setdefault("steps", [])
        c.setdefault("dom_elements", [])
        c.setdefault("iterations", 0)
        verdict = c.get("verdict")
        if not isinstance(verdict, dict):
            # Legacy shape: may have `passed` and `reason` at top level
            passed = bool(c.get("passed", False))
            reason = str(c.get("reason", ""))
            c["verdict"] = {"verdict": "passed" if passed else "failed", "passed": passed, "reason": reason}
        else:
            v = dict(verdict)
            if "verdict" not in v:
                v["verdict"] = "passed" if v.get("passed") else "failed"
            v["passed"] = bool(v.get("verdict") == "passed" or v.get("passed", False))
            v.setdefault("reason", "")
            c["verdict"] = v
        normalized_checks.append(c)
    normalized["behavior_checks"] = normalized_checks
    normalized.setdefault("visual_checks", [])

    # Future query-intent section defaults (safe for old records).
    normalized.setdefault("query_decomposition", {
        "source_requirements": [],
        "dom_requirements": [],
        "behavior_requirements": [],
        "visual_requirements": [],
        "interaction_visual_requirements": [],
    })
    phase_scores = normalized.get("phase_requirement_scores")
    if not isinstance(phase_scores, dict):
        phase_scores = {}
    normalized["phase_requirement_scores"] = phase_scores

    # Backward/forward compatibility for merged interaction+visual phase.
    if "interaction_visual" not in phase_scores:
        behavior_phase = phase_scores.get("behavior", {})
        visual_phase = phase_scores.get("visual", {})
        b_reqs = behavior_phase.get("requirements", 0) if isinstance(behavior_phase, dict) else 0
        v_reqs = visual_phase.get("requirements", 0) if isinstance(visual_phase, dict) else 0
        b_score = behavior_phase.get("score") if isinstance(behavior_phase, dict) else None
        v_score = visual_phase.get("score") if isinstance(visual_phase, dict) else None
        merged_score = b_score if b_score is not None else v_score
        phase_scores["interaction_visual"] = {
            "requirements": int(b_reqs or 0) + int(v_reqs or 0),
            "score": merged_score,
        }

    merged = phase_scores.get("interaction_visual", {})
    if "behavior" not in phase_scores:
        phase_scores["behavior"] = {
            "requirements": len(normalized["query_decomposition"].get("behavior_requirements") or []),
            "score": merged.get("score") if isinstance(merged, dict) else None,
        }
    if "visual" not in phase_scores:
        phase_scores["visual"] = {
            "requirements": len(normalized["query_decomposition"].get("visual_requirements") or []),
            "score": merged.get("score") if isinstance(merged, dict) else None,
        }

    normalized.setdefault("query_fulfillment_score", None)
    normalized.setdefault("interaction_visual_standard_scores", {})
    return normalized


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)


def create_app(event_storage: EventStorage) -> FastAPI:
    """Create FastAPI application."""

    def _queries_path(markdown_file: str) -> Path:
        p = Path(markdown_file)
        if p.is_dir():
            return p.parent / (p.name + ".queries.json")
        return p.with_suffix(".queries.json")
    app = FastAPI(
        title="Frontend Evaluator Dashboard",
        description="Real-time visualization of frontend evaluation tasks",
        version="0.1.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # WebSocket connection manager
    manager = ConnectionManager()

    # Store event_storage in app state
    app.state.event_storage = event_storage
    app.state.ws_manager = manager

    # REST API endpoints
    @app.get("/api/tasks")
    async def get_tasks():
        """Get all tasks (lightweight summary, no planner_report blobs)."""
        try:
            summaries = event_storage.get_tasks_summary()
            tasks = []
            for row in summaries:
                started = row.get("started_at")
                completed = row.get("completed_at")
                verdict_raw = row.get("verdict")
                tasks.append({
                    "id": row["id"],
                    "markdown_file": row["markdown_file"],
                    "query": row["query"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "started_at": started,
                    "completed_at": completed,
                    "duration_seconds": (
                        round(completed - started, 1)
                        if started and completed else None
                    ),
                    "verdict": bool(verdict_raw) if verdict_raw is not None else None,
                    "reason": row.get("reason"),
                    "error": row.get("error"),
                    "overall_score": row.get("overall_score"),
                    "interaction_count": row.get("interaction_count"),
                })
            return {"tasks": tasks}
        except Exception as e:
            logger.error(f"Failed to get tasks: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get a specific task."""
        try:
            task = event_storage.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            return {
                "id": task.id,
                "markdown_file": task.markdown_file,
                "query": task.query,
                "status": task.status,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "verdict": task.verdict,
                "reason": task.reason,
                "error": task.error,
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tasks/{task_id}/events")
    async def get_task_events(task_id: str):
        """Get all events for a task."""
        try:
            events = event_storage.get_task_events(task_id)
            return {
                "events": [
                    {
                        "event_index": idx,
                        "task_id": event.task_id,
                        "timestamp": event.timestamp,
                        "iteration": event.iteration,
                        "event_type": event.event_type,
                        "data": event.data,
                        "has_screenshot": event.screenshot is not None,
                        "screenshot_url": (
                            f"/api/tasks/{task_id}/events/{idx}/screenshot"
                            if event.screenshot is not None
                            else None
                        ),
                    }
                    for idx, event in enumerate(events)
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get events for task {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tasks/{task_id}/planner_report")
    async def get_planner_report(task_id: str):
        """Get the Planner report for a task (Planner mode only)."""
        try:
            task = event_storage.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            if not task.planner_report:
                raise HTTPException(status_code=404, detail="No planner report for this task")
            return _normalize_planner_report(task.planner_report)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get planner report for {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    class UpdateQueriesBody(BaseModel):
        queries: List[Dict[str, Any]]

    @app.put("/api/tasks/{task_id}/queries")
    async def update_task_queries(task_id: str, body: UpdateQueriesBody):
        """Save edited behavior queries to the .queries.json sidecar file."""
        try:
            task = event_storage.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")
            path = _queries_path(task.markdown_file)
            path.write_text(
                json.dumps(body.queries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return {"ok": True, "path": str(path), "count": len(body.queries)}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to save queries for {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tasks/{task_id}/events/{event_index}/screenshot")
    async def get_event_screenshot(task_id: str, event_index: int):
        """Get screenshot for a specific event."""
        try:
            events = event_storage.get_task_events(task_id)
            if event_index < 0 or event_index >= len(events):
                raise HTTPException(status_code=404, detail="Event not found")

            event = events[event_index]
            if not event.screenshot:
                raise HTTPException(status_code=404, detail="Screenshot not found")

            return {"screenshot": event.screenshot}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get screenshot: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stats")
    async def get_stats():
        """Get overall statistics via a single aggregation query."""
        try:
            return event_storage.get_stats()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await manager.connect(websocket)
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # Root endpoint - serve simple HTML dashboard
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the dashboard HTML."""
        html_path = Path(__file__).parent / "static" / "index.html"
        if html_path.exists():
            return html_path.read_text()
        else:
            return "<h1>Dashboard not found</h1>"

    return app


async def broadcast_event(app: FastAPI, event: EvaluationEvent):
    """Broadcast an event to all WebSocket clients.

    Args:
        app: FastAPI application
        event: Event to broadcast
    """
    manager = app.state.ws_manager
    await manager.broadcast(
        {
            "type": "event",
            "task_id": event.task_id,
            "timestamp": event.timestamp,
            "iteration": event.iteration,
            "event_type": event.event_type,
            "data": event.data,
        }
    )


async def broadcast_task_update(app: FastAPI, task: EvaluationTask):
    """Broadcast a task update to all WebSocket clients.

    Args:
        app: FastAPI application
        task: Task to broadcast
    """
    manager = app.state.ws_manager
    await manager.broadcast(
        {
            "type": "task_update",
            "task_id": task.id,
            "status": task.status,
            "verdict": task.verdict,
            "reason": task.reason,
            "error": task.error,
        }
    )
