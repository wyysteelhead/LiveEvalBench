"""Unified planner rubric loader."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


def _default_rubric_path() -> Path:
    # .../src/frontend_evaluator/planner/rubric_config.py -> repo root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "planner_rubric.json"


@lru_cache(maxsize=1)
def load_planner_rubric() -> Dict[str, Any]:
    """Load unified planner rubric JSON from disk."""
    configured_path = os.getenv("PLANNER_RUBRIC_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
    else:
        path = _default_rubric_path()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid planner rubric format in {path}")
    return raw


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def get_source_scanner_checks() -> List[Dict[str, Any]]:
    rubric = load_planner_rubric()
    return _as_list(rubric.get("source", {}).get("scanner_checks"))


def get_source_agent_standards() -> List[Dict[str, Any]]:
    rubric = load_planner_rubric()
    return _as_list(rubric.get("source", {}).get("agent_checks"))


def get_dom_auditor_checks() -> List[Dict[str, Any]]:
    rubric = load_planner_rubric()
    return _as_list(rubric.get("dom", {}).get("auditor_checks"))


def get_dom_agent_standards() -> List[Dict[str, Any]]:
    rubric = load_planner_rubric()
    return _as_list(rubric.get("dom", {}).get("agent_checks"))


def get_interaction_visual_standards() -> List[Dict[str, Any]]:
    rubric = load_planner_rubric()
    return _as_list(rubric.get("interaction_visual", {}).get("standards"))


def get_interaction_visual_execution_groups() -> List[Dict[str, Any]]:
    rubric = load_planner_rubric()
    return _as_list(rubric.get("interaction_visual", {}).get("execution_groups"))
