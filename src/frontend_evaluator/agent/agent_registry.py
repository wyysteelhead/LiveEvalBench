"""Agent registry: loads and validates agent JSON configurations."""

import json
from collections import deque
from pathlib import Path
from typing import Dict, List, Union

from pydantic import ValidationError

from .config import AgentConfig
from .. import tools as _tools  # noqa: F401
from ..tools.registry import TOOL_REGISTRY
from ..utils.logger import logger


class AgentRegistryError(Exception):
    """Raised when agent configuration loading or validation fails."""


class AgentRegistry:
    """Loads agent configs from a directory of JSON files."""

    VERDICT_TOOLS = {"submit_verdict", "submit_group_verdict"}

    def __init__(self, agents_dir: Union[str, Path] = "agents"):
        self.agents_dir = Path(agents_dir)

    def load(self) -> List[AgentConfig]:
        """Load, validate, deduplicate, and return enabled agent configs.

        Returns:
            List of enabled AgentConfig instances.

        Raises:
            AgentRegistryError: On duplicate IDs, invalid tools, or parse errors.
        """
        if not self.agents_dir.exists():
            logger.warning(f"Agents directory not found: {self.agents_dir}")
            return []

        json_files = sorted(self.agents_dir.glob("*.json"))
        if not json_files:
            logger.info(f"No agent JSON files in {self.agents_dir}")
            return []

        configs: List[AgentConfig] = []
        seen_ids: dict[str, str] = {}  # agent_id -> filename

        for path in json_files:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise AgentRegistryError(
                    f"Failed to read {path.name}: {exc}"
                ) from exc

            try:
                config = AgentConfig.model_validate(raw)
            except ValidationError as exc:
                raise AgentRegistryError(
                    f"Validation failed for {path.name}: {exc}"
                ) from exc

            # Check for duplicate IDs across files
            if config.id in seen_ids:
                raise AgentRegistryError(
                    f"Duplicate agent id '{config.id}' in {path.name} "
                    f"(first seen in {seen_ids[config.id]})"
                )
            seen_ids[config.id] = path.name

            if config.enabled:
                self._validate_tools(config, path.name)

            configs.append(config)

        # Return only enabled agents
        enabled = [c for c in configs if c.enabled]
        self._validate_dependencies(enabled)
        logger.info(
            f"Loaded {len(enabled)} enabled agent(s) "
            f"({len(configs) - len(enabled)} disabled) from {self.agents_dir}"
        )
        return enabled

    def load_in_dependency_order(self) -> List[AgentConfig]:
        """Load enabled agents and return them in dependency-safe order."""
        return self.topological_sort(self.load())

    @staticmethod
    def _validate_tools(config: AgentConfig, filename: str = "") -> None:
        """Check allowed_tools against TOOL_REGISTRY; require a verdict tool.

        Raises:
            AgentRegistryError: If unknown tools found or verdict tool missing.
        """
        registered = set(TOOL_REGISTRY.keys())
        unknown = [t for t in config.allowed_tools if t not in registered]
        if unknown:
            raise AgentRegistryError(
                f"Agent '{config.id}' ({filename}) references unknown tools: {unknown}. "
                f"Available: {sorted(registered)}"
            )

        has_verdict = bool(
            set(config.allowed_tools) & AgentRegistry.VERDICT_TOOLS
        )
        if not has_verdict:
            raise AgentRegistryError(
                f"Agent '{config.id}' ({filename}) must include at least one "
                f"verdict tool ({AgentRegistry.VERDICT_TOOLS})"
            )

    @staticmethod
    def _validate_dependencies(configs: List[AgentConfig]) -> None:
        """Validate that all dependencies exist and do not form cycles."""
        by_id = {config.id: config for config in configs}
        missing: Dict[str, List[str]] = {}
        for config in configs:
            for dep_id in config.depends_on:
                if dep_id not in by_id:
                    missing.setdefault(config.id, []).append(dep_id)

        if missing:
            details = "; ".join(
                f"{agent_id} -> {sorted(dep_ids)}" for agent_id, dep_ids in sorted(missing.items())
            )
            raise AgentRegistryError(f"Missing dependencies in agent configs: {details}")

        AgentRegistry.topological_sort(configs)

    @staticmethod
    def topological_sort(configs: List[AgentConfig]) -> List[AgentConfig]:
        """Return configs in topological order or raise on cycles."""
        by_id = {config.id: config for config in configs}
        indegree = {config.id: 0 for config in configs}
        outgoing: Dict[str, List[str]] = {config.id: [] for config in configs}

        for config in configs:
            for dep_id in config.depends_on:
                outgoing.setdefault(dep_id, []).append(config.id)
                indegree[config.id] += 1

        queue = deque(sorted(agent_id for agent_id, degree in indegree.items() if degree == 0))
        ordered: List[AgentConfig] = []

        while queue:
            agent_id = queue.popleft()
            ordered.append(by_id[agent_id])
            for downstream_id in sorted(outgoing.get(agent_id, [])):
                indegree[downstream_id] -= 1
                if indegree[downstream_id] == 0:
                    queue.append(downstream_id)

        if len(ordered) != len(configs):
            remaining = sorted(agent_id for agent_id, degree in indegree.items() if degree > 0)
            raise AgentRegistryError(
                f"Cyclic dependencies detected between agents: {remaining}"
            )

        return ordered

    @staticmethod
    def build_dependency_layers(configs: List[AgentConfig]) -> List[List[AgentConfig]]:
        """Return configs grouped into parallelizable dependency layers."""
        ordered = AgentRegistry.topological_sort(configs)
        by_id = {config.id: config for config in ordered}
        layer_by_id: Dict[str, int] = {}
        layers: List[List[AgentConfig]] = []

        for config in ordered:
            layer_index = 0
            if config.depends_on:
                layer_index = max(layer_by_id[dep_id] for dep_id in config.depends_on) + 1
            layer_by_id[config.id] = layer_index
            while len(layers) <= layer_index:
                layers.append([])
            layers[layer_index].append(by_id[config.id])

        return layers
