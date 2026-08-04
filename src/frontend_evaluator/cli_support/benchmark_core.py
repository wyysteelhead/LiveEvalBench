from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frontend_evaluator.batch import BatchPlannerExecutor
    from frontend_evaluator.utils.config import Config


def resolve_default_jsonl(args: Any) -> Path | None:
    if args.jsonl:
        return Path(args.jsonl)
    if any([args.fixture, args.markdown, args.batch_dir, args.rerun_task]):
        return None

    default = Path("fixtures/samples.jsonl")
    return default if default.exists() else None


def create_benchmark_executor(args: Any, config: "Config") -> "BatchPlannerExecutor":
    from frontend_evaluator.batch import BatchPlannerExecutor

    return BatchPlannerExecutor(
        max_parallel=args.max_parallel,
        db_path=args.db_path,
        save_to_db=args.save_to_db,
        regenerate_queries=args.regenerate_queries,
    )


async def run_benchmark_entry(args: Any, config: "Config") -> int:
    from frontend_evaluator.utils.logger import logger

    if args.executor:
        os.environ["EXECUTOR_BACKEND"] = args.executor

    if args.rerun_task and any([args.jsonl, args.fixture, args.markdown, args.batch_dir]):
        print("Error: --rerun-task cannot be combined with input source flags", file=sys.stderr)
        return 1

    source_flags = [bool(args.jsonl), bool(args.fixture), bool(args.markdown), bool(args.batch_dir)]
    if sum(source_flags) > 1:
        print(
            "Error: choose only one input source among --jsonl / --fixture / --markdown / --batch-dir",
            file=sys.stderr,
        )
        return 1

    executor = create_benchmark_executor(args, config)
    jsonl_path = resolve_default_jsonl(args)

    if args.rerun_task:
        try:
            task = await executor.rerun_task(args.rerun_task)
            if task is None:
                print(f"Error: task '{args.rerun_task}' not found or cannot be re-run", file=sys.stderr)
                return 1
            if task.status == "failed":
                print(f"Re-run failed: {task.error}", file=sys.stderr)
                return 1
            print(f"Re-run complete: {task.reason}")
            return 0
        except KeyboardInterrupt:
            logger.info("Re-run interrupted by user")
            return 1
        except Exception as exc:
            logger.error(f"Re-run failed: {exc}")
            return 1

    try:
        tasks = []
        if args.fixture or args.markdown or args.batch_dir:
            if args.batch_dir:
                base_path = Path(args.batch_dir)
                if not base_path.is_dir():
                    print(f"Error: Batch directory not found: {args.batch_dir}", file=sys.stderr)
                    return 1

                if "*" in args.batch_pattern or "?" in args.batch_pattern:
                    pattern = str(base_path / args.batch_pattern)
                    paths = glob.glob(pattern, recursive=True)
                else:
                    pattern = str(base_path / f"**/{args.batch_pattern}")
                    paths = glob.glob(pattern, recursive=True)

                if not paths:
                    print(
                        f"Error: No files found in {args.batch_dir} matching pattern {args.batch_pattern}",
                        file=sys.stderr,
                    )
                    return 1

                filtered_paths = []
                for path in paths:
                    path_obj = Path(path)
                    if path_obj.is_dir() or path_obj.suffix in {".md"}:
                        filtered_paths.append(path)

                if not filtered_paths:
                    print("Error: No valid fixtures or markdown files found", file=sys.stderr)
                    return 1

                logger.info("Legacy mode: found %d item(s) for batch processing", len(filtered_paths))
                tasks = await executor.execute(filtered_paths, user_query=args.query)
            elif args.fixture:
                logger.info("Legacy single fixture: %s", args.fixture)
                tasks = await executor.execute([args.fixture], user_query=args.query)
            else:
                logger.info("Legacy single markdown: %s", args.markdown)
                tasks = await executor.execute([args.markdown], user_query=args.query)
        elif jsonl_path:
            if not jsonl_path.exists() or not jsonl_path.is_file():
                print(
                    f"Error: JSONL file not found: {jsonl_path}\n"
                    "Provide --jsonl PATH or use explicit legacy flags --fixture/--markdown/--batch-dir.",
                    file=sys.stderr,
                )
                return 1
            if args.jsonl:
                logger.info("JSONL mode: %s", jsonl_path)
            else:
                logger.info("No source flag provided; defaulting to JSONL: fixtures/samples.jsonl")
            tasks = await executor.execute_jsonl(str(jsonl_path), user_query=args.query)
        else:
            return 1

        if args.output and len(tasks) == 1:
            task = tasks[0]
            if task.planner_report:
                output_path = Path(args.output)
                output_path.write_text(json.dumps(task.planner_report, indent=2), encoding="utf-8")
                logger.info("JSON report written to: %s", args.output)

        if any(task.verdict is False or task.status == "failed" for task in tasks):
            return 1
        return 0
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        return 1
