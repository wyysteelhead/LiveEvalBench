"""Batch executor for running Planner on JSONL samples."""

import asyncio
import json
import re
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task_queue import TaskQueue
from ..events.models import EvaluationTask
from ..events.storage import EventStorage
from ..events.emitter import EventEmitter
from ..parser import ArtifactParser
from ..planner import Planner
from ..utils.config import Config
from ..utils.logger import logger


_JSONL_LOCATOR_PREFIX = "jsonl|"
_FENCED_BLOCK_PATTERN = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


@dataclass
class JsonlSample:
    """A validated JSONL sample row."""

    sample_id: str
    query: str
    files: Dict[str, str]
    jsonl_path: Path
    line_no: int


class BatchPlannerExecutor:
    """Runs Planner on JSONL samples.

    Results are always saved to SQLite database for later viewing via web dashboard.
    """

    def __init__(
        self,
        max_parallel: int = 1,
        db_path: Optional[str] = None,
        save_to_db: bool = True,
        regenerate_queries: bool = False,
    ):
        self.max_parallel = max_parallel
        self.queue = TaskQueue(max_parallel)
        self.event_storage = EventStorage(db_path=db_path) if save_to_db else None
        self.config = Config()
        self.regenerate_queries = regenerate_queries
        self._cli_user_query: Optional[str] = None
        self._jsonl_cache: Dict[str, JsonlSample] = {}

        project_root = Path(__file__).parents[3]
        self.task_log_dir = project_root / "logs" / "tasks"
        self.task_log_dir.mkdir(parents=True, exist_ok=True)

    def _create_task_log_path(self, task: EvaluationTask) -> Path:
        stem = self._task_display_name(task.markdown_file)
        safe_stem = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem) or "task"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.task_log_dir / f"{ts}_{safe_stem}_{task.id[:8]}.log"

    def _append_task_log(self, log_path: Path, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} | {message}\n"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def _task_display_name(self, source_ref: str) -> str:
        meta = self._parse_jsonl_locator(source_ref)
        if meta:
            return f"{meta['sample_id']}@L{meta['line_no']}"
        return Path(source_ref).name

    def _build_jsonl_locator(self, jsonl_path: Path, line_no: int, sample_id: str) -> str:
        return f"{_JSONL_LOCATOR_PREFIX}{jsonl_path.resolve()}|{line_no}|{sample_id}"

    def _parse_jsonl_locator(self, source_ref: str) -> Optional[Dict[str, Any]]:
        if not source_ref.startswith(_JSONL_LOCATOR_PREFIX):
            return None
        body = source_ref[len(_JSONL_LOCATOR_PREFIX):]
        parts = body.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid JSONL locator: {source_ref}")
        path_str, line_no_str, sample_id = parts
        try:
            line_no = int(line_no_str)
        except ValueError as e:
            raise ValueError(f"Invalid JSONL locator line number: {source_ref}") from e
        return {
            "jsonl_path": Path(path_str),
            "line_no": line_no,
            "sample_id": sample_id,
        }

    def _parse_code_payload(self, code_payload: Any, line_no: int) -> Dict[str, str]:
        if isinstance(code_payload, dict):
            files: Dict[str, str] = {}
            for raw_path, raw_content in code_payload.items():
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError(f"line {line_no}: code map contains invalid file path")
                if not isinstance(raw_content, str):
                    raise ValueError(f"line {line_no}: code map value for '{raw_path}' must be string")
                files[raw_path] = raw_content
            self._validate_files_map(files, line_no)
            return files

        if not isinstance(code_payload, str) or not code_payload.strip():
            raise ValueError(
                f"line {line_no}: field 'code' must be either a non-empty artifact string "
                "or a file-content map"
            )

        payload = code_payload.strip()

        # Support JSON-encoded map strings: "{\"app/page.tsx\":\"...\"}"
        maybe_json_map = self._try_parse_json_string_map(payload, line_no)
        if maybe_json_map is not None:
            return maybe_json_map

        try:
            files = ArtifactParser().parse(payload)
            self._validate_files_map(files, line_no)
            return files
        except Exception as e:
            artifact_err = str(e)

        # Support a single fenced code block without filename header.
        fenced = self._try_parse_single_fenced_block(payload, line_no)
        if fenced is not None:
            return fenced

        # Support raw source code without markdown fences.
        raw = self._try_parse_raw_source(payload, line_no)
        if raw is not None:
            return raw

        raise ValueError(
            f"line {line_no}: failed to parse 'code'. "
            f"Expected one of: "
            f"(1) file map object, "
            f"(2) markdown artifacts with '# filename' + fenced block, "
            f"(3) single fenced block, or "
            f"(4) raw source string. "
            f"artifact_parse_error={artifact_err}"
        )

    def _try_parse_json_string_map(self, payload: str, line_no: int) -> Optional[Dict[str, str]]:
        if not payload.startswith("{"):
            return None
        try:
            parsed = json.loads(payload)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        files: Dict[str, str] = {}
        for raw_path, raw_content in parsed.items():
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"line {line_no}: code JSON map contains invalid file path")
            if not isinstance(raw_content, str):
                raise ValueError(f"line {line_no}: code JSON map value for '{raw_path}' must be string")
            files[raw_path] = raw_content
        self._validate_files_map(files, line_no)
        return files

    def _try_parse_single_fenced_block(self, payload: str, line_no: int) -> Optional[Dict[str, str]]:
        blocks = _FENCED_BLOCK_PATTERN.findall(payload)
        if len(blocks) != 1:
            return None
        lang, content = blocks[0]
        file_path = self._infer_filename_from_lang(lang.strip().lower())
        files = {file_path: content.rstrip()}
        self._validate_files_map(files, line_no)
        return files

    def _try_parse_raw_source(self, payload: str, line_no: int) -> Optional[Dict[str, str]]:
        if payload.startswith("```"):
            return None
        # Heuristic: treat as a single frontend entry file.
        file_path = self._infer_filename_from_content(payload)
        files = {file_path: payload}
        self._validate_files_map(files, line_no)
        return files

    def _infer_filename_from_lang(self, lang: str) -> str:
        if lang in {"tsx", "typescriptreact"}:
            return "app/page.tsx"
        if lang in {"ts", "typescript"}:
            return "src/main.ts"
        if lang in {"jsx", "javascriptreact"}:
            return "src/main.jsx"
        if lang in {"js", "javascript", "mjs", "cjs"}:
            return "src/main.js"
        if lang in {"html"}:
            return "index.html"
        if lang in {"css"}:
            return "src/index.css"
        if lang in {"json"}:
            return "package.json"
        return "src/main.txt"

    def _infer_filename_from_content(self, content: str) -> str:
        lowered = content.lower()
        if "export default" in content and ("jsx" in lowered or "tsx" in lowered or "<div" in lowered):
            return "app/page.tsx"
        if "<html" in lowered or "<!doctype html" in lowered:
            return "index.html"
        if "react" in lowered and ("function " in lowered or "const " in lowered):
            return "src/main.jsx"
        if lowered.strip().startswith("{") and "\"dependencies\"" in lowered:
            return "package.json"
        return "src/main.js"

    def _validate_files_map(self, files: Dict[str, str], line_no: int) -> None:
        if not files:
            raise ValueError(f"line {line_no}: parsed code has no files")

        for path, content in files.items():
            p = Path(path)
            if p.is_absolute():
                raise ValueError(f"line {line_no}: absolute paths are not allowed in code files: {path}")
            if ".." in p.parts:
                raise ValueError(f"line {line_no}: parent directory traversal is not allowed: {path}")
            if not isinstance(content, str):
                raise ValueError(f"line {line_no}: file content for '{path}' must be string")

    def _load_jsonl_sample_from_locator(self, source_ref: str) -> JsonlSample:
        cached = self._jsonl_cache.get(source_ref)
        if cached:
            return cached

        meta = self._parse_jsonl_locator(source_ref)
        if not meta:
            raise ValueError(f"Not a JSONL locator: {source_ref}")

        jsonl_path: Path = meta["jsonl_path"]
        line_no: int = meta["line_no"]
        expected_sample_id: str = meta["sample_id"]

        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        row_text = None
        with jsonl_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if idx == line_no:
                    row_text = line
                    break

        if row_text is None:
            raise ValueError(f"JSONL locator line not found: {source_ref}")

        sample = self._parse_jsonl_row(row_text, jsonl_path, line_no)
        if sample.sample_id != expected_sample_id:
            raise ValueError(
                f"JSONL locator sample id mismatch at {jsonl_path}:{line_no}. "
                f"expected='{expected_sample_id}', got='{sample.sample_id}'"
            )

        self._jsonl_cache[source_ref] = sample
        return sample

    def _parse_jsonl_row(self, row_text: str, jsonl_path: Path, line_no: int) -> JsonlSample:
        stripped = row_text.strip()
        if not stripped:
            raise ValueError(f"line {line_no}: empty line is not allowed in JSONL input")

        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {line_no}: invalid JSON: {e.msg}") from e

        if not isinstance(row, dict):
            raise ValueError(f"line {line_no}: each JSONL row must be an object")

        missing = {"id", "query", "code"} - set(row.keys())
        if missing:
            missing_fields = ", ".join(sorted(missing))
            raise ValueError(f"line {line_no}: missing required field(s): {missing_fields}")

        sample_id = str(row["id"]).strip()
        if not sample_id:
            raise ValueError(f"line {line_no}: field 'id' cannot be empty")

        query = row["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"line {line_no}: field 'query' must be a non-empty string")

        files = self._parse_code_payload(row["code"], line_no)

        return JsonlSample(
            sample_id=sample_id,
            query=query.strip(),
            files=files,
            jsonl_path=jsonl_path,
            line_no=line_no,
        )

    def _load_jsonl_samples(self, jsonl_path: str) -> List[JsonlSample]:
        path = Path(jsonl_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        samples: List[JsonlSample] = []
        seen_ids = set()
        with path.open("r", encoding="utf-8") as f:
            for line_no, row_text in enumerate(f, start=1):
                sample = self._parse_jsonl_row(row_text, path, line_no)
                if sample.sample_id in seen_ids:
                    raise ValueError(f"line {line_no}: duplicate id '{sample.sample_id}'")
                seen_ids.add(sample.sample_id)
                samples.append(sample)

        if not samples:
            raise ValueError(f"JSONL file has no samples: {jsonl_path}")
        return samples

    def _load_files(self, source_ref: str) -> Dict[str, str]:
        """Load source files from a JSONL locator or legacy path."""
        meta = self._parse_jsonl_locator(source_ref)
        if meta:
            return self._load_jsonl_sample_from_locator(source_ref).files

        # Legacy path mode (kept for older DB reruns)
        p = Path(source_ref)
        if p.is_dir():
            files = {}
            for f in p.rglob("*"):
                if f.is_file() and f.suffix in {
                    ".tsx", ".ts", ".jsx", ".js", ".json", ".css", ".html", ".mjs", ".cjs"
                }:
                    files[str(f.relative_to(p))] = f.read_text(encoding="utf-8")
            return files

        content = p.read_text(encoding="utf-8")
        return ArtifactParser().parse(content)

    async def execute_jsonl(self, jsonl_path: str, user_query: Optional[str] = None) -> List[EvaluationTask]:
        """Run Planner on all rows in a JSONL file."""
        samples = self._load_jsonl_samples(jsonl_path)
        self._cli_user_query = user_query
        self.queue.tasks = []

        logger.info(
            "Starting JSONL batch planner: samples=%s, max_parallel=%s, source=%s",
            len(samples),
            self.max_parallel,
            jsonl_path,
        )

        for sample in samples:
            task_id = str(uuid.uuid4())
            locator = self._build_jsonl_locator(sample.jsonl_path, sample.line_no, sample.sample_id)
            self._jsonl_cache[locator] = sample
            label = sample.query
            task = EvaluationTask.create(task_id, locator, label)
            self.queue.tasks.append(task)

        start_time = time.time()
        tasks = await self.queue.run_all(self._execute_single, self.event_storage)
        duration = time.time() - start_time

        self._print_summary(tasks, duration)
        return tasks

    async def execute(self, paths: List[str], user_query: Optional[str] = None) -> List[EvaluationTask]:
        """Deprecated legacy path-mode execution."""
        logger.warning(
            "Legacy path-mode execute() called; JSONL mode is now the primary input format."
        )
        logger.info(f"Starting batch planner: {len(paths)} tasks, max_parallel={self.max_parallel}")

        self.queue.tasks = []
        self._cli_user_query = user_query
        for path in paths:
            task_id = str(uuid.uuid4())
            label = user_query or "planner"
            task = EvaluationTask.create(task_id, path, label)
            self.queue.tasks.append(task)

        start_time = time.time()
        tasks = await self.queue.run_all(self._execute_single, self.event_storage)
        duration = time.time() - start_time

        self._print_summary(tasks, duration)
        return tasks

    def _queries_path(self, source_ref: str) -> Path:
        """Return the .queries.json sidecar path for JSONL or legacy source refs."""
        meta = self._parse_jsonl_locator(source_ref)
        if meta:
            jsonl_path: Path = meta["jsonl_path"]
            sample_id: str = meta["sample_id"]
            safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in sample_id)
            sidecar_dir = jsonl_path.parent / ".queries" / jsonl_path.stem
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            return sidecar_dir / f"{safe_id}.queries.json"

        p = Path(source_ref)
        if p.is_dir():
            return p.parent / (p.name + ".queries.json")
        return p.with_suffix(".queries.json")

    def _load_user_query(self, source_ref: str) -> Optional[str]:
        """Load user query from JSONL row or legacy sidecar .query file."""
        meta = self._parse_jsonl_locator(source_ref)
        if meta:
            sample = self._load_jsonl_sample_from_locator(source_ref)
            return sample.query

        p = Path(source_ref)
        if p.is_dir():
            sidecar = p.parent / (p.name + ".query")
        else:
            sidecar = p.with_suffix(".query")
        if sidecar.exists():
            value = sidecar.read_text(encoding="utf-8").strip()
            return value or None
        return None

    async def _execute_single(self, task: EvaluationTask, event_storage: Optional[EventStorage]):
        task.start()
        log_path = self._create_task_log_path(task)
        task.log_path = str(log_path)
        self._append_task_log(log_path, f"Task started: id={task.id}")
        self._append_task_log(log_path, f"Input: {task.markdown_file}")
        if event_storage:
            event_storage.save_task(task)

        event_emitter = None
        if event_storage:
            event_emitter = EventEmitter(task.id, enabled=True)

            async def _save_event(event):
                event_storage.save_event(event)

            event_emitter.on(_save_event)

        try:
            files = self._load_files(task.markdown_file)
            if not files:
                raise ValueError(f"No source files found in: {task.markdown_file}")
            self._append_task_log(
                log_path,
                (
                    f"Parsed files: count={len(files)}, "
                    f"has_package_json={'package.json' in files}, files={sorted(files.keys())}"
                ),
            )

            user_query = self._load_user_query(task.markdown_file) or self._cli_user_query
            if user_query:
                task.query = user_query
                self._append_task_log(log_path, f"Using query: {user_query[:200]}")
                if event_storage:
                    event_storage.save_task(task)
            else:
                raise ValueError("Missing query: JSONL rows must define a non-empty 'query'")

            queries_path = self._queries_path(task.markdown_file)
            self._append_task_log(
                log_path,
                f"queries_path={queries_path} exists={queries_path.exists()} regenerate={self.regenerate_queries}",
            )

            planner = Planner()
            report = await planner.run(
                files=files,
                config=self.config,
                event_emitter=event_emitter,
                user_query=user_query,
                queries_path=queries_path,
                regenerate_queries=self.regenerate_queries,
            )

            passed = report.overall_score >= 0.5
            reason = f"Overall score: {report.overall_score:.1%}"
            task.complete(passed, reason, planner_report=report.to_dict())
            if report.build_error:
                self._append_task_log(log_path, f"Build error: {report.build_error}")
            self._append_task_log(log_path, f"Task completed: verdict={passed}, reason={reason}")
            logger.info(f"Task {task.id} done: {reason}")

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.fail(str(e))
            self._append_task_log(log_path, f"Task failed: {e}")
            self._append_task_log(log_path, traceback.format_exc())

        finally:
            self._append_task_log(log_path, f"Task final status: {task.status}")
            if event_storage:
                event_storage.save_task(task)

    async def rerun_task(self, task_id: str) -> Optional[EvaluationTask]:
        """Re-run behavior checks for an existing task using its edited .queries.json sidecar."""
        if not self.event_storage:
            raise RuntimeError("rerun_task requires save_to_db=True (event_storage must be set)")

        task = self.event_storage.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return None

        if not task.planner_report:
            logger.error(f"Task {task_id} has no planner_report to re-run from")
            return None

        queries_path = self._queries_path(task.markdown_file)
        if not queries_path.exists():
            logger.error(f"No .queries.json sidecar found at {queries_path}")
            return None

        logger.info(f"Re-running task {task_id} from {task.markdown_file}")
        task.start()
        self.event_storage.save_task(task)

        try:
            files = self._load_files(task.markdown_file)
            if not files:
                raise ValueError(f"No source files found in: {task.markdown_file}")

            planner = Planner()
            report = await planner.rerun(
                files=files,
                original_report=task.planner_report,
                queries_path=queries_path,
                config=self.config,
            )

            passed = report.overall_score >= 0.5
            reason = f"Overall score: {report.overall_score:.1%}"
            task.complete(passed, reason, planner_report=report.to_dict())
            logger.info(f"Re-run task {task_id} done: {reason}")

        except Exception as e:
            logger.error(f"Re-run task {task_id} failed: {e}")
            task.fail(str(e))

        finally:
            self.event_storage.save_task(task)

        return task

    async def _start_web_server(self):
        import uvicorn
        from ..web.server import create_app

    def _print_summary(self, tasks: List[EvaluationTask], duration: float):
        total = len(tasks)
        passed = sum(1 for t in tasks if t.verdict is True)
        failed = sum(1 for t in tasks if t.verdict is False)
        errors = sum(1 for t in tasks if t.status == "failed")

        print("\n" + "=" * 60)
        print("BATCH PLANNER SUMMARY")
        print("=" * 60)
        print(f"Total tasks:  {total}")
        print(f"Passed:       {passed}")
        print(f"Failed:       {failed}")
        print(f"Errors:       {errors}")
        print(f"Duration:     {duration:.1f}s")
        print("=" * 60)

        for i, task in enumerate(tasks, 1):
            name = self._task_display_name(task.markdown_file)
            if task.status == "failed":
                status = f"⚠ ERROR: {task.error}"
            elif task.verdict is True:
                status = f"✓ PASSED — {task.reason}"
            else:
                status = f"✗ FAILED — {task.reason}"
            print(f"  [{i}/{total}] {name}: {status}")
            if task.log_path:
                print(f"      log: {task.log_path}")
        print()
