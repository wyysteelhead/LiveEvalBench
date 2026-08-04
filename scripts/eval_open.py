#!/usr/bin/env python3
"""eval_open.py — permissive-mode autonomous environment evaluation.

Guardrail: open (all commands logged, no domain restrictions).
Two-phase flow:
  Phase 1 (Build): Code Engineer agent writes files, runs npm install/dev,
                   starts Chromium, writes artifacts.json.
  Phase 2 (Evaluate): Evaluator agents connect via CDP and test the app.

Usage:
    python scripts/eval_open.py --jsonl fixtures/samples.jsonl
    python scripts/eval_open.py --jsonl fixtures/samples.jsonl --max-parallel 4

    # Resume is the default — existing output is auto-detected:
    python scripts/eval_open.py --jsonl data.jsonl --output report.json

    # Point to a specific report to resume from:
    python scripts/eval_open.py --jsonl data.jsonl --output report.json --resume-report report.ckpt-0003.json

    # Re-run non-passed rows on resume:
    python scripts/eval_open.py --jsonl data.jsonl --output report.json --rerun-nonpassed

    # Start completely fresh (ignore existing output):
    python scripts/eval_open.py --jsonl data.jsonl --output report.json --no-resume

    # Multi-process sharding (4 processes, each handling 1/4 of the rows):
    python scripts/eval_open.py --jsonl data.jsonl --shard 0 --shard-count 4 --output out.json &
    python scripts/eval_open.py --jsonl data.jsonl --shard 1 --shard-count 4 --output out.json &
    python scripts/eval_open.py --jsonl data.jsonl --shard 2 --shard-count 4 --output out.json &
    python scripts/eval_open.py --jsonl data.jsonl --shard 3 --shard-count 4 --output out.json &
    wait
    # Results: out.shard00.json, out.shard01.json, out.shard02.json, out.shard03.json

    # 分片写入（每 50 条结果写一个小文件，减少 NFS 大文件压力）：
    python scripts/eval_open.py --jsonl data.jsonl --output report.json --chunk-size 50
    # 分片存储在 report.shards/0000.jsonl, report.shards/0001.jsonl ...
    # 中途合并查看：python scripts/tools/merge_shards.py report.shards/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Force polling mode for all file watchers to avoid inotify exhaustion.
# In containerised / multi-process eval environments the kernel limits
# fs.inotify.max_user_instances (128) and max_user_watches (51200) are
# easily hit by concurrent webpack/Next.js/Vite dev servers, causing EMFILE
# on sockets and CDP connections.  Polling uses CPU instead of inotify.
os.environ.setdefault("CHOKIDAR_USEPOLLING", "true")      # chokidar / Vite
os.environ.setdefault("WATCHPACK_POLLING", "true")         # webpack 5 / Next.js
os.environ.setdefault("CHOKIDAR_INTERVAL", "2000")         # poll every 2s (default is 100ms)
os.environ.setdefault("WATCHPACK_POLLING_INTERVAL", "2000") # same for watchpack


def _check_node_version_or_exit() -> None:
    """Fail fast when the active Node is too old for vitest/rolldown.

    Node 22+ is required because rolldown imports `util.styleText`, which was
    added in Node 22. Running with older Node leads to opaque "SyntaxError:
    The requested module 'node:util' does not provide an export named
    'styleText'" failures deep inside Phase 2 — much easier to catch up front.
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        print("FATAL: 'node' not on PATH. Activate Node 22+ via nvm before running.", file=sys.stderr)
        sys.exit(2)
    try:
        out = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5,
        )
    except Exception as e:  # noqa: BLE001 — sanity check only
        print(f"FATAL: failed to invoke 'node --version': {e}", file=sys.stderr)
        sys.exit(2)
    ver = (out.stdout or "").strip().lstrip("v")
    try:
        major = int(ver.split(".", 1)[0])
    except ValueError:
        print(f"FATAL: cannot parse node version '{ver}'", file=sys.stderr)
        sys.exit(2)
    if major < 22:
        print(
            f"FATAL: node {ver} < 22. vitest/rolldown require Node 22+ "
            f"(util.styleText). Run: nvm install 22 --lts && nvm use 22",
            file=sys.stderr,
        )
        sys.exit(2)


# Disabled — re-enable once everyone has Node 22 installed.
# _check_node_version_or_exit()


from frontend_evaluator.cli_support.common import find_latest_checkpoint, load_jsonl_rows, load_resume_index, write_json_output
from frontend_evaluator.cli_support.open_core import run_open_batch
from frontend_evaluator.open_monitor import (
    OpenEvalMonitorWriter,
    build_monitor_attach_command,
    build_monitor_task_descriptors,
)
from frontend_evaluator.utils.config import Config
from frontend_evaluator.utils.logger import setup_logger
from tools.guardrail import audit_log_entry

logger = setup_logger("eval_open")
GUARDRAIL_MODE = "open"


def _add_log_file(logger: logging.Logger, path: Path) -> None:
    """Add a file handler to *logger* via QueueHandler so NFS writes don't block the event loop."""
    import queue
    from logging import FileHandler, Formatter
    from logging.handlers import QueueHandler, QueueListener

    path.parent.mkdir(parents=True, exist_ok=True)
    fh = FileHandler(str(path), mode="a", encoding="utf-8")
    fh.setLevel(logger.level)
    fh.setFormatter(Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    log_queue: queue.Queue = queue.Queue(-1)
    qh = QueueHandler(log_queue)
    qh.setLevel(logger.level)
    listener = QueueListener(log_queue, fh, respect_handler_level=True)
    listener.start()
    logger.addHandler(qh)
    import atexit
    atexit.register(listener.stop)


def _collect_rerun_nonpassed(
    existing_index: dict[str, dict],
    rows: list[dict],
    monitor_path: Path | None = None,
) -> set[str]:
    """Collect non-passed ``sample_id`` values from *existing_index* and
    reset their monitor state to ``"pending"`` so ``--rerun-nonpassed`` can
    re-run them.

    Unlike ``_cleanup_rerun_nonpassed``, this does **not** modify the
    report JSONL file — results are overwritten lazily when each row is
    actually re-run via ``_append_row_to_jsonl``.

    Returns the set of ``sample_id`` values that were collected.
    """
    import json as _json

    rerun_ids: set[str] = set()
    for row in rows:
        sid = str(row.get("id", "") or row.get("sample_id", "")).strip()
        if sid and sid in existing_index:
            status = str(existing_index[sid].get("status", "")).lower()
            if status in {"failed", "inconclusive", "error"}:
                rerun_ids.add(sid)

    # ── Reset monitor state: set collected tasks back to "pending" ──
    if monitor_path and monitor_path.exists() and rerun_ids:
        try:
            monitor = _json.loads(monitor_path.read_text(encoding="utf-8"))
            changed = 0
            for task in monitor.get("tasks") if isinstance(monitor, dict) else []:
                if not isinstance(task, dict):
                    continue
                if str(task.get("sample_id", "")).strip() in rerun_ids:
                    # Preserve last failure so the monitor shows previous attempt history
                    task["last_attempt"] = {
                        "status": task.get("status"),
                        "phase": task.get("phase"),
                        "verdict_status": task.get("verdict_status"),
                        "reason": task.get("reason"),
                        "error": task.get("error"),
                        "started_at": task.get("started_at"),
                        "completed_at": task.get("completed_at"),
                    }
                    task["status"] = "pending"
                    task["phase"] = "pending"
                    task["verdict_status"] = None
                    task["reason"] = ""
                    task["error"] = None
                    task["started_at"] = None
                    task["completed_at"] = None
                    task["agents"] = []
                    task["agent_summary"] = {"total": 0, "pending": 0, "running": 0, "finished": 0}
                    changed += 1
            if changed:
                _refresh_monitor_summary(monitor)
                monitor_path.write_text(
                    _json.dumps(monitor, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info("Monitor cleanup: reset %d tasks to pending", changed)
        except Exception as exc:
            logger.warning("Monitor cleanup failed (non-fatal): %s", exc)

    return rerun_ids


def _refresh_monitor_summary(monitor: dict) -> None:
    """Recalculate ``summary`` counters from the tasks list."""
    tasks = monitor.get("tasks") or []
    total = len(tasks)
    pending = sum(1 for t in tasks if isinstance(t, dict) and t.get("status") == "pending")
    running = sum(1 for t in tasks if isinstance(t, dict) and t.get("status") == "running")
    finished = sum(1 for t in tasks if isinstance(t, dict) and t.get("status") in ("completed", "failed"))
    monitor["summary"] = {
        "tasks_total": total,
        "tasks_pending": pending,
        "tasks_running": running,
        "tasks_finished": finished,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Permissive-mode autonomous frontend evaluator (open guardrail — all ops logged)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--jsonl", metavar="PATH",
                        help="Path to JSONL file (id/query/code per line).")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Maximum parallel evaluations (default from env MAX_PARALLEL, or 1).")
    parser.add_argument("--output", metavar="PATH",
                        help="Write JSON results to this file.")
    parser.add_argument("--audit-log", metavar="PATH",
                        help="Append audit log entries (JSONL) to this file.")
    parser.add_argument("--env", metavar="PATH",
                        help="Path to .env file.")
    parser.add_argument("--fixed-task-db-path", metavar="PATH",
                        help="Override the SQLite path used to persist query fixed tasks.")
    parser.add_argument("--agents-dir", metavar="PATH", default="agents",
                        help="Directory containing agent JSON configs (default: agents).")
    parser.add_argument("--resume-report", metavar="PATH",
                        help="Resume from an existing results JSON; skip already-passed rows.")
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=True,
                        help="Start fresh — ignore existing output and checkpoints.")
    parser.add_argument("--rerun-nonpassed", action="store_true",
                        help="When resuming, rerun rows whose status is not 'passed'.")
    parser.add_argument("--force-rerun-agents", metavar="CSV", default="",
                        help="Comma-separated agent ids (e.g. 'build_engineer,code_tester'). For every "
                             "row already present in the resume report, rerun only these agents and reuse "
                             "the saved records of all other agents (overrides --rerun-nonpassed skip).")
    parser.add_argument("--monitor-state", metavar="PATH",
                        help="Write live monitor state JSON for scripts/eval_open_monitor.py.")
    parser.add_argument("--shard", type=int, default=None,
                        help="0-indexed shard index for multi-process execution (requires --shard-count).")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Write incremental results every N completed rows (default: 5).")
    parser.add_argument("--shard-count", type=int, default=None,
                        help="Total number of shards (e.g. 4 means split into 4 processes).")
    parser.add_argument("--log-file", metavar="PATH",
                        help="Write all log output to this file (default: {output}.log).")
    parser.add_argument("--row-timeout", type=float, default=None,
                        help="Per-row wall-clock timeout in seconds. Rows exceeding this limit are "
                             "cancelled and marked as failed. Default from env ROW_TIMEOUT (7200 = 2h). "
                             "Set to 0 to disable.")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Write results into rotating chunk files (report.shards/NNNN.jsonl) "
                             "instead of a single JSONL. Every N results open a new chunk. "
                             "Default from env CHUNK_SIZE (0 = disabled).")
    args = parser.parse_args()

    if not args.jsonl:
        parser.print_help()
        return 1

    if args.fixed_task_db_path:
        os.environ["FIXED_TASK_DB_PATH"] = args.fixed_task_db_path

    config = Config(env_file=args.env)
    max_parallel = args.max_parallel if args.max_parallel is not None else config.max_parallel
    logger.setLevel(config.log_level.upper())
    for handler in logger.handlers:
        handler.setLevel(config.log_level.upper())

    # ── File logging ────────────────────────────────────────────────
    log_file_path: Path | None = None
    if args.log_file:
        log_file_path = Path(args.log_file)
    elif args.output:
        log_file_path = Path(args.output).with_suffix(".log")
    if log_file_path:
        _add_log_file(logger, log_file_path)
        logger.info("Log file: %s", log_file_path)

    logger.info("eval_open: guardrail_mode=%s (all operations logged)", GUARDRAIL_MODE)

    audit_path = Path(args.audit_log) if args.audit_log else None
    output_path = Path(args.output) if args.output else None
    chunk_size: int = max(0, args.chunk_size if args.chunk_size is not None else config.chunk_size)

    def _log_audit(cmd: str, risk: str, cwd: str = ".") -> None:
        entry = audit_log_entry(cmd, risk, cwd)
        logger.info("audit: %s", json.dumps(entry))
        if audit_path:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

    # ── Resume ─────────────────────────────────────────────────────
    # By default, resume from the existing output file (and checkpoints
    # if available).  Pass --no-resume to start completely fresh.
    if args.resume and not args.resume_report and output_path:
        latest_ckpt = find_latest_checkpoint(output_path)
        if latest_ckpt:
            logger.info("Auto-resume: detected checkpoint %s", latest_ckpt)
            args.resume_report = str(latest_ckpt)
        elif output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Resume: using existing output file %s", output_path)
            args.resume_report = str(output_path)

    existing_index = {}
    if args.resume_report:
        resume_path = Path(args.resume_report)
        try:
            existing_index = load_resume_index(resume_path, id_keys=("id", "sample_id"))
            if existing_index:
                logger.info("Loaded %d rows from resume report", len(existing_index))
        except Exception as exc:
            logger.warning("Could not load resume report: %s", exc)

    rows = load_jsonl_rows(Path(args.jsonl))

    # Extract model info from ext_info so it flows into the report
    _model_names_seen: set[str] = set()
    for row in rows:
        ext_info = row.get("ext_info")
        if isinstance(ext_info, str):
            try:
                parsed = json.loads(ext_info)
                if isinstance(parsed, dict) and parsed.get("model_name"):
                    row["model"] = parsed["model_name"]
                    _model_names_seen.add(parsed["model_name"])
            except (json.JSONDecodeError, TypeError):
                pass
        elif isinstance(ext_info, dict) and ext_info.get("model_name"):
            row["model"] = ext_info["model_name"]
            _model_names_seen.add(ext_info["model_name"])
    if _model_names_seen:
        logger.info("Extracted model info: %s", ", ".join(sorted(_model_names_seen)))

    shard_suffix = ""
    if args.shard is not None and args.shard_count is not None:
        if args.shard < 0 or args.shard >= args.shard_count:
            logger.error("--shard %d out of range [0, %d)", args.shard, args.shard_count)
            return 1
        rows = rows[args.shard::args.shard_count]
        shard_suffix = f".shard{args.shard:02d}"
        logger.info("Shard %d/%d: processing %d rows (indices %d, %d, ...)",
                     args.shard, args.shard_count, len(rows), args.shard, args.shard + args.shard_count)
    elif args.shard is not None or args.shard_count is not None:
        logger.error("--shard and --shard-count must be used together")
        return 1

    if shard_suffix and output_path:
        output_path = output_path.with_name(f"{output_path.stem}{shard_suffix}{output_path.suffix}")

    # ── 分片写入（chunk）: 每 chunk_size 条写一个小 JSONL，减少 NFS 大文件压力 ──
    chunks_dir: Path | None = (
        output_path.parent / f"{output_path.stem}.shards"
        if output_path and chunk_size > 0
        else None
    )
    if chunks_dir is not None:
        logger.info("Chunk mode enabled: chunk_size=%d, dir=%s", chunk_size, chunks_dir)

    # 如果 chunks 目录已存在（上次运行留下的），从中重建 existing_index
    if chunks_dir is not None and chunks_dir.is_dir() and args.resume:
        chunk_files = sorted(chunks_dir.glob("*.jsonl"))
        if chunk_files:
            chunk_index: dict[str, dict] = {}
            for cf in chunk_files:
                chunk_index.update(load_resume_index(cf, id_keys=("id", "sample_id")))
            if chunk_index:
                # 合并：chunk 里已完成的优先覆盖之前的 resume_report（如果有）
                existing_index = {**existing_index, **chunk_index}
                logger.info(
                    "Chunk resume: loaded %d completed rows from %d chunk file(s) in %s",
                    len(chunk_index), len(chunk_files), chunks_dir,
                )
            # 初始化 chunk 状态：接续最后一个 chunk 文件
            last_chunk = chunk_files[-1]
            last_idx = int(last_chunk.stem)
            last_count = sum(1 for ln in last_chunk.open(encoding="utf-8") if ln.strip())
            _chunk_state = {
                "index": last_idx if last_count < chunk_size else last_idx + 1,
                "count": last_count if last_count < chunk_size else 0,
            }
        else:
            _chunk_state = {"index": 0, "count": 0}
    else:
        _chunk_state = {"index": 0, "count": 0}

    monitor_path = Path(args.monitor_state) if args.monitor_state else (
        output_path.with_name(f"{output_path.stem}.monitor.json")
        if output_path
        else Path(f"artifacts/reports/eval_open_monitor{shard_suffix}.json")
    )

    # ── Rerun-nonpassed: collect failed IDs, reorder rows so they run last ──
    _rerun_cleanup_sample_ids: set[str] = set()
    _force_rerun_agents: list[str] = [
        a.strip() for a in (args.force_rerun_agents or "").split(",") if a.strip()
    ]
    _force_rerun_sample_ids: set[str] = set()
    if _force_rerun_agents and existing_index:
        # Baseline file: records when this force-rerun campaign started.
        # On restart, rows whose generated_at > baseline have already been
        # force-rerun'd and should be skipped to avoid infinite re-processing.
        _fr_baseline_path = (
            chunks_dir / ".force_rerun_baseline" if chunks_dir else
            (output_path.parent / ".force_rerun_baseline" if output_path else None)
        )
        _fr_baseline: str = ""
        if _fr_baseline_path and _fr_baseline_path.exists():
            _fr_baseline = _fr_baseline_path.read_text(encoding="utf-8").strip()
            logger.info("force-rerun baseline (from file): %s", _fr_baseline)
        else:
            from datetime import datetime, timezone
            _fr_baseline = datetime.now(timezone.utc).isoformat()
            if _fr_baseline_path:
                _fr_baseline_path.parent.mkdir(parents=True, exist_ok=True)
                _fr_baseline_path.write_text(_fr_baseline + "\n", encoding="utf-8")
                logger.info("force-rerun baseline (new): %s → %s", _fr_baseline, _fr_baseline_path)

        # For chunk-purge: any row already in the report that has at least one
        # of the named agents must have its old shard line removed and a fresh
        # line appended after rerun. We do NOT drop it from existing_index —
        # run_open_batch needs the saved agent records to reuse non-rerun
        # agents (the "good_agents" merge).
        # Skip rows whose generated_at > baseline (already force-rerun'd this campaign).
        _force_rerun_sample_ids = set()
        _force_rerun_already_done = 0
        for sid, entry in existing_index.items():
            has_agent = any(
                isinstance(ag, dict) and str(ag.get("agent_id") or "").strip() in _force_rerun_agents
                for ag in (entry.get("result", {}) or {}).get("agents", []) or []
            )
            if not has_agent:
                continue
            row_generated = str((entry.get("result", {}) or {}).get("generated_at", ""))
            if _fr_baseline and row_generated > _fr_baseline:
                _force_rerun_already_done += 1
                continue
            _force_rerun_sample_ids.add(sid)

        logger.info(
            "force-rerun-agents=%s: %d existing rows queued for partial rerun "
            "(chunk-purge included), %d already done (skipped)",
            _force_rerun_agents, len(_force_rerun_sample_ids), _force_rerun_already_done,
        )

    # Monitor cleanup: stale completed/running entries whose chunk row was
    # purged in an earlier kill_restart (their sample_id is no longer in
    # existing_index). These show up as misleading "finished" counts on the
    # dashboard until the eval re-runs them. Mark them for monitor-reset too.
    _stale_monitor_sample_ids: set[str] = set()
    if _force_rerun_agents and monitor_path.exists():
        try:
            _mon_prev = json.loads(monitor_path.read_text(encoding="utf-8"))
            for _t in (_mon_prev.get("tasks") or []) if isinstance(_mon_prev, dict) else []:
                if not isinstance(_t, dict):
                    continue
                _sid = str(_t.get("sample_id") or "").strip()
                _st = str(_t.get("status") or "").lower()
                if _sid and _st in {"completed", "running"} and _sid not in existing_index:
                    _stale_monitor_sample_ids.add(_sid)
        except Exception as _exc:
            logger.warning("could not scan monitor for stale entries: %s", _exc)
        if _stale_monitor_sample_ids:
            logger.info(
                "monitor: %d stale completed/running entries (chunk already purged); will reset to pending",
                len(_stale_monitor_sample_ids),
            )

    if args.rerun_nonpassed and args.resume_report and Path(args.resume_report).exists():
        _rerun_cleanup_sample_ids = _collect_rerun_nonpassed(
            existing_index,
            rows,
            monitor_path=monitor_path if monitor_path.exists() else None,
        )
        if _rerun_cleanup_sample_ids:
            logger.info(
                "Collected %d non-passed rows for re-run",
                len(_rerun_cleanup_sample_ids),
            )
            existing_index = {
                sid: row
                for sid, row in existing_index.items()
                if sid not in _rerun_cleanup_sample_ids
            }
            # Reorder rows: non-failed first, failed last — so failed rows
            # get processed after all other (new) rows.
            failed_sample_ids = _rerun_cleanup_sample_ids
            non_failed_rows = [r for r in rows if str(r.get("id", "") or r.get("sample_id", "")).strip() not in failed_sample_ids]
            rerun_rows = [r for r in rows if str(r.get("id", "") or r.get("sample_id", "")).strip() in failed_sample_ids]
            rows = non_failed_rows + rerun_rows
            logger.info(
                "Reordered %d non-passed rows to the end for re-run (total rows: %d)",
                len(rerun_rows), len(rows),
            )

    # Merge force-rerun ids into the chunk-purge set (separate code path from
    # _collect_rerun_nonpassed — those rows stay in existing_index).
    if _force_rerun_sample_ids:
        _rerun_cleanup_sample_ids = _rerun_cleanup_sample_ids | _force_rerun_sample_ids

    # ── Chunk mode: lazy dedup (write-time) instead of startup purge ──
    # Previously we purged every rerun sample_id from shard files at startup,
    # but that was destructive: if eval was SIGTERM'd before processing some
    # of those sample_ids, their old records would be permanently lost (the
    # baseline-loss bug). We now keep the old chunk row as a safety fallback
    # and remove it lazily inside _append_row_to_jsonl when the new row is
    # actually written. Recompute _chunk_state for backwards-compat (no purge
    # mutation needed since chunks are untouched here).
    if _rerun_cleanup_sample_ids:
        logger.info(
            "%d sample_ids queued for rerun; old chunk rows kept as fallback "
            "and removed lazily when fresh results arrive",
            len(_rerun_cleanup_sample_ids),
        )

    # Build task descriptors AFTER reordering so task_ids match the
    # execution order (non-failed → failed).
    task_descriptors = build_monitor_task_descriptors(rows)

    monitor_writer = OpenEvalMonitorWriter(
        monitor_path,
        max_parallel=max_parallel,
        source_jsonl=args.jsonl,
        agents_dir=args.agents_dir,
    )
    await monitor_writer.initialize(task_descriptors,
                                      resume_from=monitor_path if monitor_path.exists() else None,
                                      reset_sample_ids=(_force_rerun_sample_ids | _stale_monitor_sample_ids) or None)
    logger.info("Monitor state written to %s", monitor_path)
    logger.info("Attach monitor in another terminal: %s", build_monitor_attach_command(monitor_path))

    # ── JSONL incremental checkpoint ──────────────────────────────────
    # Every completed row is appended as one JSON line so we never
    # rewrite the full file or keep all results in memory.  The JSONL
    # IS the canonical output — no final JSON array conversion needed.
    _append_lock = asyncio.Lock()

    async def _append_row_to_jsonl(row_result: dict[str, object], out_path: Path) -> None:
        """Append *row_result* as a single JSON line.

        When chunk mode is active (chunks_dir is set), writes to rotating chunk
        files (chunks_dir/NNNN.jsonl) instead of a single monolithic file.
        For rerun rows, the old chunk row (if any) is removed at write time so
        each sample_id appears exactly once. The kept-old-row-as-fallback
        approach (vs the old startup-purge approach) means we never lose data
        if eval is killed before reaching every queued rerun.

        In single-file mode, rerun rows are deduplicated in-place.
        """
        if chunks_dir is not None:
            # ── chunk 模式：写入分片文件，按需 lazy-dedup ──
            sample_id = str(row_result.get("id", "") or row_result.get("sample_id", "")).strip()
            is_rerun = bool(sample_id) and sample_id in _rerun_cleanup_sample_ids
            line = json.dumps(row_result, ensure_ascii=False) + "\n"
            try:
                async with _append_lock:
                    def _do_chunk() -> None:
                        chunks_dir.mkdir(parents=True, exist_ok=True)
                        # Lazy dedup: remove any existing row with this sample_id
                        # from earlier shards before appending the new one.
                        if is_rerun:
                            for _cf in sorted(chunks_dir.glob("*.jsonl")):
                                _kept: list[str] = []
                                _removed_in_file = False
                                try:
                                    with open(_cf, encoding="utf-8") as _fh:
                                        for _raw in _fh:
                                            _stripped = _raw.strip()
                                            if not _stripped:
                                                continue
                                            try:
                                                _entry = json.loads(_stripped)
                                                _sid = str(_entry.get("id", "") or _entry.get("sample_id", "")).strip()
                                                if _sid == sample_id:
                                                    _removed_in_file = True
                                                    continue
                                            except json.JSONDecodeError:
                                                pass
                                            _kept.append(_raw if _raw.endswith("\n") else _raw + "\n")
                                except FileNotFoundError:
                                    continue
                                if _removed_in_file:
                                    with open(_cf, "w", encoding="utf-8") as _fh:
                                        _fh.writelines(_kept)
                                    if _cf.exists() and _cf.stat().st_size == 0:
                                        _cf.unlink()
                            # After purging, mark sid as cleaned so subsequent
                            # writes for the same sid (shouldn't happen) skip dedup.
                            _rerun_cleanup_sample_ids.discard(sample_id)
                        # Append the new row to current shard
                        shard_path = chunks_dir / f"{_chunk_state['index']:04d}.jsonl"
                        with open(shard_path, "a", encoding="utf-8") as fh:
                            fh.write(line)
                            fh.flush()
                            os.fsync(fh.fileno())
                        _chunk_state["count"] += 1
                        if _chunk_state["count"] >= chunk_size:
                            _chunk_state["index"] += 1
                            _chunk_state["count"] = 0
                    await asyncio.to_thread(_do_chunk)
            except Exception as exc:
                logger.error("[chunk] append failed: %s", exc, exc_info=True)
            return

        # ── 单文件模式（原逻辑）──
        if not out_path:
            return
        sample_id = str(row_result.get("id", "") or row_result.get("sample_id", "")).strip()
        is_rerun = sample_id in _rerun_cleanup_sample_ids
        line = json.dumps(row_result, ensure_ascii=False) + "\n"
        try:
            async with _append_lock:
                def _do_append() -> None:
                    if is_rerun:
                        # Overwrite: remove old entry for this sample_id, then write new line
                        lines: list[str] = []
                        if out_path.exists():
                            with open(out_path, "r", encoding="utf-8") as fh:
                                for raw in fh:
                                    stripped = raw.strip()
                                    if not stripped:
                                        continue
                                    try:
                                        existing = json.loads(stripped)
                                        eid = str(existing.get("id", "") or existing.get("sample_id", "")).strip()
                                        if eid == sample_id:
                                            continue  # drop old entry
                                    except json.JSONDecodeError:
                                        pass
                                    lines.append(raw)
                        lines.append(line)
                        with open(out_path, "w", encoding="utf-8") as fh:
                            fh.writelines(lines)
                            fh.flush()
                            os.fsync(fh.fileno())
                    else:
                        with open(out_path, "a", encoding="utf-8") as fh:
                            fh.write(line)
                            fh.flush()
                            os.fsync(fh.fileno())
                await asyncio.to_thread(_do_append)
        except Exception as exc:
            logger.error("[jsonl] append failed: %s", exc, exc_info=True)

    async def _handle_monitor_event(task_id: str, payload: dict[str, object]) -> None:
        kind = str(payload.get("kind", "") or "")
        if kind == "task":
            await monitor_writer.mark_task_status(
                task_id,
                status=str(payload.get("status", "pending") or "pending"),
                phase=str(payload.get("phase", "pending") or "pending"),
                reason=str(payload.get("reason", "") or "") or None,
                error=str(payload.get("error", "") or "") or None,
                verdict_status=str(payload.get("verdict_status", "") or "") or None,
                row_wait_ms=(
                    int(payload.get("row_wait_ms", 0) or 0)
                    if isinstance(payload.get("row_wait_ms"), int)
                    else None
                ),
            )
            return
        if kind == "task_final":
            report = payload.get("report") if isinstance(payload.get("report"), dict) else None
            await monitor_writer.apply_final_result(
                task_id,
                lifecycle_status=str(payload.get("status", "completed") or "completed"),
                verdict_status=str(payload.get("verdict_status", "") or "") or None,
                reason=str(payload.get("reason", "") or "") or None,
                error=str(payload.get("error", "") or "") or None,
                report=report,
            )
            return
        if kind == "agent":
            await monitor_writer.upsert_agent(
                task_id,
                str(payload.get("agent_id", "") or "unknown"),
                status=str(payload.get("status", "pending") or "pending"),
                steps_completed=int(payload.get("steps_completed", 0) or 0),
                steps_total=int(payload.get("steps_total", 0) or 0),
                current_task_title=str(payload.get("current_task_title", "") or "") or None,
                current_task_id=str(payload.get("current_task_id", "") or "") or None,
                current_task_status=str(payload.get("current_task_status", "") or "") or None,
                current_task_completion_score=(
                    float(payload.get("current_task_completion_score"))
                    if isinstance(payload.get("current_task_completion_score"), (int, float))
                    else None
                ),
                current_task_steps_completed=(
                    int(payload.get("current_task_steps_completed", 0) or 0)
                    if isinstance(payload.get("current_task_steps_completed"), int)
                    else None
                ),
                current_task_steps_total=(
                    int(payload.get("current_task_steps_total", 0) or 0)
                    if isinstance(payload.get("current_task_steps_total"), int)
                    else None
                ),
                current_task_index=(
                    int(payload.get("current_task_index", 0) or 0)
                    if isinstance(payload.get("current_task_index"), int)
                    else None
                ),
                current_task_total=(
                    int(payload.get("current_task_total", 0) or 0)
                    if isinstance(payload.get("current_task_total"), int)
                    else None
                ),
                current_main_task_id=str(payload.get("current_main_task_id", "") or "") or None,
                current_main_task_title=str(payload.get("current_main_task_title", "") or "") or None,
                current_subtask_title=str(payload.get("current_subtask_title", "") or "") or None,
                running_subtasks=(
                    payload.get("running_subtasks")
                    if isinstance(payload.get("running_subtasks"), list)
                    else None
                ),
                task_synthesis_mode=str(payload.get("task_synthesis_mode", "") or "") or None,
                planned_task_tree=(
                    payload.get("planned_task_tree")
                    if isinstance(payload.get("planned_task_tree"), dict)
                    else None
                ),
                build_wait_ms=(
                    int(payload.get("build_wait_ms", 0) or 0)
                    if isinstance(payload.get("build_wait_ms"), int)
                    else None
                ),
                npm_wait_ms=(
                    int(payload.get("npm_wait_ms", 0) or 0)
                    if isinstance(payload.get("npm_wait_ms"), int)
                    else None
                ),
                npm_duration_ms=(
                    int(payload.get("npm_duration_ms", 0) or 0)
                    if isinstance(payload.get("npm_duration_ms"), int)
                    else None
                ),
                evaluate_agent_wait_ms=(
                    int(payload.get("evaluate_agent_wait_ms", 0) or 0)
                    if isinstance(payload.get("evaluate_agent_wait_ms"), int)
                    else None
                ),
                cdp_connect_ms=(
                    int(payload.get("cdp_connect_ms", 0) or 0)
                    if isinstance(payload.get("cdp_connect_ms"), int)
                    else None
                ),
                subtask_wait_ms=(
                    int(payload.get("subtask_wait_ms", 0) or 0)
                    if isinstance(payload.get("subtask_wait_ms"), int)
                    else None
                ),
                end_reason=str(payload.get("end_reason", "") or "") or None,
                queue_reason=str(payload.get("queue_reason", "") or "") or None,
            )

    # When writing to the same file used for resume, suppress on_result for
    # skipped rows (they are already in the file) to avoid duplicate rows.
    # In chunk mode we always suppress skipped rows (they are already in chunks).
    _same_file = chunks_dir is not None or (
        output_path is not None
        and args.resume_report is not None
        and os.path.normpath(str(output_path)) == os.path.normpath(args.resume_report)
    )

    final_results = await run_open_batch(
        config=config,
        rows=rows,
        max_parallel=max_parallel,
        agents_dir=args.agents_dir,
        existing_index=existing_index,
        rerun_nonpassed=args.rerun_nonpassed,
        save_interval=args.save_interval,
        audit_log_fn=_log_audit,
        task_descriptors=task_descriptors,
        task_progress_callback=_handle_monitor_event,
        suppress_skipped_on_result=_same_file,
        force_rerun_agents=_force_rerun_agents or None,
        on_result=lambda _index, row_result: _append_row_to_jsonl(row_result, output_path) if (output_path or chunks_dir) else None,
    )

    if output_path and output_path.exists():
        # Count lines to report progress
        line_count = 0
        with open(output_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    line_count += 1
        logger.info("Results written to %s (%d rows)", output_path, line_count)
    elif not output_path:
        logger.warning("No output path specified; results are not persisted.")

    failed = 0
    if output_path and output_path.exists():
        for raw in output_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
                if str(row.get("status", "")).lower() == "failed":
                    failed += 1
            except json.JSONDecodeError:
                pass
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
