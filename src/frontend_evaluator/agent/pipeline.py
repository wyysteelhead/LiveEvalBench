"""Dependency-aware pipeline helpers for persona execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .agent_registry import AgentRegistry
from .config import AgentConfig


@dataclass
class AgentHandoff:
    """Structured handoff emitted by one persona for downstream personas."""

    producer_agent_id: str
    stage: str
    status: str
    summary: str
    facts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "producer_agent_id": self.producer_agent_id,
            "stage": self.stage,
            "status": self.status,
            "summary": self.summary,
            "facts": self.facts,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class PipelineAgentResult:
    """Execution result tracked at pipeline level."""

    agent_id: str
    stage: str
    status: str
    report: Dict[str, Any] | None = None
    handoff: AgentHandoff | None = None
    blocked_by: List[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "stage": self.stage,
            "status": self.status,
        }
        if self.report is not None:
            payload["report"] = self.report
        if self.handoff is not None:
            payload["handoff"] = self.handoff.to_dict()
        if self.blocked_by:
            payload["blocked_by"] = self.blocked_by
        if self.error:
            payload["error"] = self.error
        return payload


def build_dependency_layers(agents: Sequence[AgentConfig]) -> List[List[AgentConfig]]:
    """Expose dependency layers for callers without touching registry internals."""
    return AgentRegistry.build_dependency_layers(list(agents))


def format_dependency_handoffs(handoffs: Sequence[AgentHandoff]) -> str:
    """Render upstream handoffs into a compact prompt block."""
    if not handoffs:
        return ""

    chunks: List[str] = ["=== Upstream persona handoffs ==="]
    for handoff in handoffs:
        chunk_lines = [
            f"agent_id: {handoff.producer_agent_id}",
            f"stage: {handoff.stage}",
            f"status: {handoff.status}",
            f"summary: {handoff.summary}",
        ]
        if handoff.facts:
            for key, value in handoff.facts.items():
                if value in (None, "", [], {}):
                    continue
                if isinstance(value, (dict, list)):
                    rendered = json.dumps(value, ensure_ascii=False)
                else:
                    rendered = str(value)
                chunk_lines.append(f"{key}: {rendered}")
        if handoff.warnings:
            chunk_lines.append(f"warnings: {json.dumps(handoff.warnings, ensure_ascii=False)}")
        if handoff.blockers:
            chunk_lines.append(f"blockers: {json.dumps(handoff.blockers, ensure_ascii=False)}")
        if handoff.evidence_refs:
            chunk_lines.append(f"evidence_refs: {json.dumps(handoff.evidence_refs, ensure_ascii=False)}")
        chunks.append("\n".join(chunk_lines))
    return "\n\n".join(chunks)