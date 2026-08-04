#!/usr/bin/env python3
"""merge_shards.py — 合并 eval_open 分片 JSONL 文件。

合并时自动去重：每个 id 保留最后写入的一条（编号更大的分片优先），
rerun 重跑结果天然排在旧结果之后，因此去重后保留的就是最新一次的结果。

用法：
    # 直接指定分片目录
    python scripts/tools/merge_shards.py report0515.shards/

    # 自动在当前目录查找唯一一个 *.shards/ 目录
    python scripts/tools/merge_shards.py

    # 指定输出文件（默认输出到 stdout）
    python scripts/tools/merge_shards.py report0515.shards/ --output merged.jsonl

    # 只输出统计摘要，不输出内容
    python scripts/tools/merge_shards.py report0515.shards/ --stats

    # 增量 append 模式（autopilot 的 merge daemon 用），只追加自上次以来新增的
    # 行，不去重；依赖 eval chunk 模式保证 sample_id 已经唯一。state 文件记录
    # 每个 shard 上次处理到的 (mtime_ns, size, lines)。
    python scripts/tools/merge_shards.py report0515.shards/ \\
        --append-mode --output merged.jsonl --state merged.state.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _find_shards_dir() -> Path | None:
    candidates = sorted(Path(".").glob("*.shards"))
    dirs = [p for p in candidates if p.is_dir()]
    return dirs[0] if len(dirs) == 1 else None


def _load_shards(shards_dir: Path) -> list[dict]:
    shard_files = sorted(shards_dir.glob("*.jsonl"))
    if not shard_files:
        print(f"[merge_shards] 未找到分片文件：{shards_dir}", file=sys.stderr)
        return []

    rows: list[dict] = []
    for sf in shard_files:
        count = 0
        with open(sf, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"[merge_shards] 跳过损坏行 {sf.name}: {e}", file=sys.stderr)
        print(f"[merge_shards] {sf.name}: {count} 条", file=sys.stderr)
    return rows


def _dedup(rows: list[dict]) -> list[dict]:
    """保留每个 id/sample_id 的最后一条（适用于含 rerun 结果的分片）。"""
    seen: dict[str, int] = {}
    for i, row in enumerate(rows):
        rid = str(row.get("id", "") or row.get("sample_id", "")).strip()
        if rid:
            seen[rid] = i
    # 按原始顺序输出，去掉被覆盖的旧条目
    keep = set(seen.values())
    return [row for i, row in enumerate(rows) if i in keep]


def _print_stats(rows: list[dict]) -> None:
    total = len(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"总计: {total} 条")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        print(f"  {status}: {count} ({pct:.1f}%)")


def _load_append_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"version": 1, "shards": {}}
    try:
        with open(state_file, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "shards" not in data:
            return {"version": 1, "shards": {}}
        return data
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "shards": {}}


def _save_append_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(state_file.suffix + ".partial")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, state_file)


def _append_mode_merge(shards_dir: Path, output: Path, state_file: Path) -> int:
    """Incrementally append new lines from changed shards to *output*.

    Reads state (per-shard mtime_ns + size + lines), skips unchanged shards,
    re-opens changed shards and appends lines beyond `lines_appended`. Does
    NOT dedup — relies on eval's chunk-mode purge to keep sample_ids unique
    across shards.

    Returns total number of new lines appended this invocation.
    """
    state = _load_append_state(state_file)
    shards_state: dict = state.get("shards", {})

    output.parent.mkdir(parents=True, exist_ok=True)
    new_state: dict = {}
    appended = 0
    skipped = 0
    changed = 0

    with open(output, "a", encoding="utf-8") as out_fh:
        for shard_path in sorted(shards_dir.glob("*.jsonl")):
            shard_name = shard_path.name
            try:
                st = shard_path.stat()
            except FileNotFoundError:
                continue
            cur_mtime_ns = st.st_mtime_ns
            cur_size = st.st_size

            prev = shards_state.get(shard_name) or {}
            prev_mtime_ns = prev.get("mtime_ns")
            prev_size = prev.get("size")
            prev_lines = int(prev.get("lines", 0) or 0)

            if prev_mtime_ns == cur_mtime_ns and prev_size == cur_size:
                # Unchanged — preserve state verbatim.
                new_state[shard_name] = prev
                skipped += 1
                continue

            changed += 1
            cur_lines = 0
            try:
                with open(shard_path, encoding="utf-8") as shard_fh:
                    for ln in shard_fh:
                        cur_lines += 1
                        if cur_lines <= prev_lines:
                            continue
                        if not ln.endswith("\n"):
                            ln = ln + "\n"
                        if ln.strip():
                            out_fh.write(ln)
                            appended += 1
            except OSError as e:
                print(f"[merge_shards] WARN: 读取 {shard_name} 失败: {e}", file=sys.stderr)
                # Don't update state for this shard — retry next cycle.
                new_state[shard_name] = prev
                continue

            new_state[shard_name] = {
                "mtime_ns": cur_mtime_ns,
                "size": cur_size,
                "lines": cur_lines,
            }

    state["shards"] = new_state
    _save_append_state(state_file, state)

    print(
        f"[merge_shards] append 模式: 新增 {appended} 行 "
        f"(changed={changed} skipped={skipped} shards)",
        file=sys.stderr,
    )
    return appended


def main() -> int:
    parser = argparse.ArgumentParser(
        description="合并 eval_open 分片 JSONL 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("shards_dir", nargs="?", metavar="SHARDS_DIR",
                        help="分片目录路径（*.shards/）。省略时自动查找当前目录下唯一的 *.shards/ 目录。")
    parser.add_argument("--output", "-o", metavar="PATH",
                        help="输出文件路径。省略则输出到 stdout。")
    parser.add_argument("--stats", action="store_true",
                        help="只输出统计摘要，不输出 JSONL 内容。")
    parser.add_argument("--append-mode", action="store_true",
                        help="增量 append 模式：只追加自上次以来新增的行，不去重。"
                             "依赖 eval chunk 模式保证 sample_id 唯一。需 --output 和 --state。")
    parser.add_argument("--state", metavar="PATH",
                        help="append 模式的状态文件路径（记录每个 shard 的 mtime/size/lines）。")
    args = parser.parse_args()

    if args.shards_dir:
        shards_dir = Path(args.shards_dir)
    else:
        shards_dir = _find_shards_dir()
        if shards_dir is None:
            print("错误：未指定分片目录，且当前目录下没有唯一的 *.shards/ 目录。", file=sys.stderr)
            return 1
        print(f"[merge_shards] 自动选择：{shards_dir}", file=sys.stderr)

    if not shards_dir.is_dir():
        print(f"错误：{shards_dir} 不是目录或不存在。", file=sys.stderr)
        return 1

    if args.append_mode:
        if not args.output or not args.state:
            print("错误：--append-mode 需要同时指定 --output 和 --state。", file=sys.stderr)
            return 1
        _append_mode_merge(shards_dir, Path(args.output), Path(args.state))
        return 0

    rows = _load_shards(shards_dir)
    if not rows:
        return 1

    before = len(rows)
    rows = _dedup(rows)
    removed = before - len(rows)
    if removed:
        print(f"[merge_shards] 去重：移除 {removed} 条旧条目，保留 {len(rows)} 条", file=sys.stderr)

    if args.stats:
        _print_stats(rows)
        return 0

    output_lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in rows]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.writelines(output_lines)
        print(f"[merge_shards] 已写入 {len(rows)} 条 → {out_path}", file=sys.stderr)
    else:
        sys.stdout.writelines(output_lines)

    return 0


if __name__ == "__main__":
    sys.exit(main())
