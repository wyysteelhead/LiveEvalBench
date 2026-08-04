---
name: livebench-run
description: Auto-launch and manage a LiveEvalBench evaluation run AFTER setup is done (.env configured, benchmark.jsonl placed). Pre-flight checks then starts eval_open.py in the background, monitors shard progress, resumes/retries on stalls or failures, merges shards into a final report, and prints result counts + paths. Generic — uses a user-chosen runs/ dir, no /tmp hardcode, no NFS, no cron. Use when the user says "run the benchmark", "start eval", "launch experiment", or asks to auto-run after setup.
---

# livebench-run: Auto-Run the Benchmark

## Purpose
Given a configured `.env` (see `livebench-setup`) and a `benchmark.jsonl`, run the **full** evaluation automatically:
launch → monitor → resume/retry on trouble → merge → report. This is the "auto experiment" skill; it does NOT do first-time install (use `livebench-setup` first).

## Inputs (decide with the user; defaults shown)
- `BENCH` — path to benchmark.jsonl (e.g. `benchmark.jsonl` or `data/benchmark.jsonl`)
- `OUT` — output dir, e.g. `runs/exp1` (created if missing). All logs/shards/PIDs go here.
- `CHUNK` — chunk size, e.g. `10` (rows per shard file)
- `MODE` — `fresh` (default, `--no-resume`) or `resume` (continue an existing `OUT` run; `--resume` is default in eval_open so omit `--no-resume`)
- Parallelism is read from `.env` (`ROW_PARALLELISM`, `MAX_TOTAL_CHROMIUM_WORKERS`, `LLM_CALLS_PER_SECOND`) — do NOT override unless the user asks.

## Workflow

### Step 1 — Pre-flight (do NOT launch if any fails)
1. `.env` exists and has `MODEL_PROVIDER` + `MODEL_NAME` + a non-placeholder API key for that provider.
2. `BENCH` exists; `wc -l $BENCH` > 0.
3. `OUT` not already being written by a live process:
   ```bash
   ps aux | grep "eval_open.py.*--output $OUT" | grep -v grep
   ```
   If a healthy process is alive on the same `OUT` → **adopt** it (skip to monitoring). If `OUT/eval.pid` exists but process dead → offer `resume`.
4. `python scripts/eval_open.py --help` runs (deps installed). If not → tell user to run `livebench-setup` first.

### Step 2 — Launch (fresh)
```bash
mkdir -p $OUT
nohup python scripts/eval_open.py \
  --jsonl "$BENCH" \
  --output "$OUT/report.json" \
  --chunk-size $CHUNK \
  --monitor-state "$OUT/monitor.json" \
  --log-file "$OUT/run.log" \
  --no-resume \
  > "$OUT/nohup.log" 2>&1 &
echo $! > "$OUT/eval.pid"
```
For `resume` mode: drop `--no-resume` (eval_open resumes by default, skipping rows already in `OUT/report.shards/*.jsonl`). Never mix runs by resuming one run's output from another run's dir.

Record: PID, start time, expected total rows = `wc -l < $BENCH`.

### Step 3 — Monitor
Poll every few minutes:
```bash
done=$(cat $OUT/report.shards/*.jsonl 2>/dev/null | wc -l)
tail -1 $OUT/run.log
```
Live dashboard (optional, foreground): `python scripts/visualization/eval_open_monitor.py --state $OUT/monitor.json`
Each row is done when it lands in a shard file. Report `done / total` + current agent in flight (from log).

### Step 4 — Stall / failure handling (on-demand, NOT cron)
Kill + resume ONLY when BOTH hold:
- `done` count unchanged across 2 checks (~20 min), AND
- `run.log` latest timestamp > 3 min behind now, OR LLM `APIConnectionError`/`timed out` count surges in the log.
Resume procedure:
```bash
kill -9 $(cat $OUT/eval.pid); pkill -9 -f "eval_open.py.*--output $OUT"; pkill -9 chrome; sleep 6
# then re-launch in resume mode (drop --no-resume):
nohup python scripts/eval_open.py --jsonl "$BENCH" --output "$OUT/report.json" \
  --chunk-size $CHUNK --monitor-state "$OUT/monitor.json" --log-file "$OUT/run.log" \
  > "$OUT/nohup.log" 2>&1 &
echo $! > "$OUT/eval.pid"
```
eval_open prints `Chunk resume: loaded N completed rows` and skips them — **zero data loss** because every completed row is in a shard file.
Do NOT kill for memory pressure alone (the run can be tight on RAM and still produce clean verdicts). Only kill for true stalls as defined above.

### Step 5 — Retry inconclusive / failed rows (after the run completes)
If the final shard set has `inconclusive` / `error` / missing verdicts, do a retry pass (in-place resume, re-runs only the non-passed rows):
```bash
python scripts/eval_open.py --jsonl "$BENCH" --output "$OUT/report.json" \
  --chunk-size $CHUNK --rerun-nonpassed --monitor-state "$OUT/monitor.json" --log-file "$OUT/rerun.log"
```
Repeat until success rate > 95% or 5 passes add no new successes.

### Step 6 — Finalize & report
1. Merge shards into one file:
   ```bash
   python scripts/tools/merge_shards.py "$OUT/report.shards" --output "$OUT/report.merged.jsonl" --stats
   ```
2. Count verdicts:
   ```bash
   python - <<'PY'
   import json,collections
   c=collections.Counter()
   for l in open("OUT/report.merged.jsonl"):
       r=json.loads(l); c[r.get("result",{}).get("overall_verdict","?")]+=1
   print(c)
   PY
   ```
   (replace `OUT` with the real path)
3. View results in the web dashboard:
   ```bash
   python scripts/visualization/eval_open_web.py   # point it at $OUT/report.merged.jsonl per its --help
   ```
4. Print to the user: total rows, pass/partial/fail/inconclusive counts, merged file path, web dashboard command.

## Critical rules
- **One process per `--output`** — never start a second eval_open on the same `OUT` while one is alive.
- **No cron, no daemon, no NFS backup** during the run — those were internal-only ops. This skill acts on-demand when the user invokes it (or a scheduled wakeup re-invokes it).
- **Independence between runs**: use a fresh `OUT` dir per independent run; only `resume` within the same `OUT`.
- All logs/shards live under the user-chosen `OUT` (e.g. `runs/exp1`), not `/tmp` — survives shell restarts, easy to inspect.

## When NOT to use
- First-time install / `.env` not set → `livebench-setup` first.
- Computing paper statistics (inter-judge agreement, multi-run variance, significance) → those are separate analysis tasks, not this skill.