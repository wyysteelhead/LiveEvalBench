"""Trajectory persistence for agentic runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class TrajectoryStore:
    """File-backed trajectory store."""

    def __init__(self, base_dir: str = "artifacts/agentic_trajectories"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        task_id: str,
        agent_id: str,
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Persist one trajectory payload and return relative reference path."""
        safe_task = task_id.replace("/", "_")
        safe_agent = agent_id.replace("/", "_")
        out_dir = self.base_dir / safe_task
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_agent}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_path)

