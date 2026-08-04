from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..batch import BatchPlannerExecutor
from .checklist import ChecklistItem, coerce_checklist_item, run_checklist
from ..cli_support.agentic_core import run_agentic_single
from ..cli_support.common import files_from_fixture_dir, files_from_markdown
from ..cli_support.open_core import run_open_single
from ..parser.artifact_parser import ArtifactParser
from ..planner.task_planner import PlannedTask
from ..utils.config import Config
from ..utils.logger import setup_logger


def checklist_to_planned_tasks(
    checklist: Sequence[ChecklistItem | Mapping[str, Any] | str],
    *,
    prefix: str = "checklist",
) -> list[PlannedTask]:
    """Convert a fixed checklist into PlannedTask objects and bypass LLM task planning."""
    tasks: list[PlannedTask] = []
    for index, item in enumerate(checklist, start=1):
        checklist_item = coerce_checklist_item(item, index=index, prefix=prefix)
        item_id = checklist_item.item_id or f"{prefix}_{index}"
        title = checklist_item.title or f"Checklist item {index}"
        tasks.append(
            PlannedTask(
                task_id=item_id,
                title=title,
                task_text=checklist_item.instruction,
                phase="interaction_visual",
                task_type="functional_core",
                generated_from="external_checklist",
                source_requirement_refs=[item_id],
                source_requirement_texts=[checklist_item.instruction],
                covers_standard_ids=list(checklist_item.covers_standard_ids),
                rubric_gap_only=False,
                scenario_id=f"{item_id}_s1",
                scenario_weight=float(checklist_item.scenario_weight or 1.0),
                multi_step=bool(checklist_item.multi_step),
                expected_signals=list(checklist_item.expected_signals),
                preconditions=list(checklist_item.preconditions),
                based_on={"external_checklist": True},
            )
        )
    return tasks


class FrontendEvaluationAPI:
    """Programmatic wrapper around the existing evaluation entrypoints."""

    def __init__(self, *, config: Config | None = None, env_file: str | None = None):
        self.config = config or Config(env_file=env_file)
        self.logger = setup_logger("frontend_evaluation_api")
        self.logger.setLevel(self.config.log_level.upper())
        for handler in self.logger.handlers:
            handler.setLevel(self.config.log_level.upper())

    async def run_agentic(
        self,
        *,
        files: Mapping[str, str] | None = None,
        fixture_dir: str | Path | None = None,
        markdown: str | None = None,
        markdown_path: str | Path | None = None,
        query: str = "",
        sandbox_provider: str | None = None,
        agents_dir: str = "agents",
        max_parallel: int | None = None,
        checklist: Sequence[ChecklistItem | Mapping[str, Any] | str] | None = None,
        agent_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        resolved_files = _resolve_files(
            files=files,
            fixture_dir=fixture_dir,
            markdown=markdown,
            markdown_path=markdown_path,
        )
        planned_tasks = checklist_to_planned_tasks(checklist) if checklist else None
        return await run_agentic_single(
            config=self.config,
            logger=self.logger,
            files=resolved_files,
            query=query,
            sandbox_provider=sandbox_provider or self.config.sandbox_provider,
            agents_dir=agents_dir,
            max_parallel=max_parallel,
            planned_tasks=planned_tasks,
            agent_ids=list(agent_ids) if agent_ids else None,
        )

    async def run_checklist(
        self,
        *,
        files: Mapping[str, str] | None = None,
        fixture_dir: str | Path | None = None,
        markdown: str | None = None,
        markdown_path: str | Path | None = None,
        checklist: Sequence[ChecklistItem | Mapping[str, Any] | str],
        query: str = "",
        sandbox_provider: str | None = None,
    ) -> dict[str, Any]:
        resolved_files = _resolve_files(
            files=files,
            fixture_dir=fixture_dir,
            markdown=markdown,
            markdown_path=markdown_path,
        )
        resolved_checklist = [
            coerce_checklist_item(item, index=index, prefix="checklist")
            for index, item in enumerate(checklist, start=1)
        ]
        return await run_checklist(
            config=self.config,
            logger=self.logger,
            files=resolved_files,
            checklist=resolved_checklist,
            query=query,
            sandbox_provider=sandbox_provider or self.config.sandbox_provider,
        )

    async def run_open(
        self,
        *,
        raw_markdown: str | None = None,
        markdown_path: str | Path | None = None,
        query: str = "",
        agents_dir: str = "agents",
        max_parallel: int | None = None,
        checklist: Sequence[ChecklistItem | Mapping[str, Any] | str] | None = None,
        agent_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        resolved_markdown = _resolve_markdown(raw_markdown=raw_markdown, markdown_path=markdown_path)
        planned_tasks = checklist_to_planned_tasks(checklist) if checklist else None
        return await run_open_single(
            config=self.config,
            raw_markdown=resolved_markdown,
            query=query,
            agents_dir=agents_dir,
            max_parallel=max_parallel,
            planned_tasks=planned_tasks,
            agent_ids=list(agent_ids) if agent_ids else None,
        )

    async def run_benchmark_jsonl(
        self,
        jsonl_path: str | Path,
        *,
        user_query: str | None = None,
        max_parallel: int = 1,
        save_to_db: bool = True,
        db_path: str | None = None,
        regenerate_queries: bool = False,
    ) -> list[Any]:
        executor = BatchPlannerExecutor(
            max_parallel=max_parallel,
            db_path=db_path,
            save_to_db=save_to_db,
            regenerate_queries=regenerate_queries,
        )
        return await executor.execute_jsonl(str(jsonl_path), user_query=user_query)

    async def run_benchmark_paths(
        self,
        paths: Iterable[str | Path],
        *,
        user_query: str | None = None,
        max_parallel: int = 1,
        save_to_db: bool = True,
        db_path: str | None = None,
        regenerate_queries: bool = False,
    ) -> list[Any]:
        executor = BatchPlannerExecutor(
            max_parallel=max_parallel,
            db_path=db_path,
            save_to_db=save_to_db,
            regenerate_queries=regenerate_queries,
        )
        normalized_paths = [str(path) for path in paths]
        return await executor.execute(normalized_paths, user_query=user_query)


def _resolve_files(
    *,
    files: Mapping[str, str] | None,
    fixture_dir: str | Path | None,
    markdown: str | None,
    markdown_path: str | Path | None,
) -> dict[str, str]:
    provided_sources = [files is not None, fixture_dir is not None, markdown is not None, markdown_path is not None]
    if sum(bool(value) for value in provided_sources) != 1:
        raise ValueError("Provide exactly one source: files, fixture_dir, markdown, or markdown_path")

    if files is not None:
        return {str(path): str(content) for path, content in files.items()}

    if fixture_dir is not None:
        return files_from_fixture_dir(Path(fixture_dir))

    if markdown_path is not None:
        return files_from_markdown(Path(markdown_path))

    parsed = ArtifactParser.parse(markdown or "")
    if not parsed:
        raise ValueError("No files could be parsed from markdown input")
    return parsed


def _resolve_markdown(*, raw_markdown: str | None, markdown_path: str | Path | None) -> str:
    if (raw_markdown is None) == (markdown_path is None):
        raise ValueError("Provide exactly one source: raw_markdown or markdown_path")

    if markdown_path is not None:
        return Path(markdown_path).read_text(encoding="utf-8")

    resolved = (raw_markdown or "").strip()
    if not resolved:
        raise ValueError("raw_markdown is empty")
    return resolved