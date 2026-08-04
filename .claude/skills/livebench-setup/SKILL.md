---
name: livebench-setup
description: First-time onboarding assistant for LiveEvalBench. Checks prerequisites, installs Python/Node/Playwright deps, guides the user to fill in .env (LLM provider + API key + model + parallelism), helps place the benchmark.jsonl dataset, and runs a sanity check. Ends by handing off to the livebench-run skill to auto-launch the experiment. Use when the user says "setup", "install", "configure", "getting started", "first time", or asks how to run the benchmark.
---

# livebench-setup: First-time Setup Assistant

## Purpose
Get a brand-new user from a fresh clone to "ready to run the benchmark" in one guided pass:
check prerequisites → install deps → configure `.env` → place dataset → sanity check → hand off to `livebench-run`.

This skill **does not run the full experiment** — that's `livebench-run`'s job. It only sets up and verifies.

## Workflow

### Step 1 — Prerequisites check
Verify (tell the user what's missing, don't auto-install system packages):
- **Python 3.10+** — `python3 --version`
- **Node.js 22+** — `node --version` (needed by vitest/rolldown for `util.styleText`; if `node` is at a non-standard path, a symlink `ln -sf $(which node) /usr/local/bin/node` may be needed)
- **uv** (recommended) or **pip**
- An **LLM API key** for one provider (Anthropic / OpenAI / Google / any OpenAI-compatible).

### Step 2 — Install dependencies
From the repo root (`LiveEvalBench/`):
```bash
# recommended
uv sync
# or
pip install -e .
# browser driver for local execution
playwright install chromium
```
Sanity: `python scripts/eval_open.py --help` should print options without error.

### Step 3 — Configure `.env` (THE file the user must edit)
```bash
cp .env.example .env
```
Then open `.env` and fill in **at minimum** (this is the single file the user provides required info in):

| Variable | What to set |
|---|---|
| `MODEL_PROVIDER` | `anthropic` / `openai` / `google` / `custom` (OpenAI-compatible) |
| `MODEL_NAME` | the judge/cognition model, e.g. `claude-sonnet-4-5-20250929`, `gpt-4o`, or for custom a model id |
| `VISION_MODEL_NAME` | perception-layer model (may equal `MODEL_NAME`); used when screenshots are involved |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `CUSTOM_API_KEY` + `CUSTOM_BASE_URL` | the key matching `MODEL_PROVIDER` |
| `EXECUTOR_BACKEND` | keep `playwright` (local execution; the bench runs locally, no sandbox service needed) |
| `TEMPERATURE` | `0` for reproducible scoring |
| `LLM_DISABLE_THINKING` | `true` recommended (faster, stable) |
| `MAX_AGENT_STEPS` | `10`–`80` depending on model (lower for fast models) |
| `ROW_PARALLELISM` / `MAX_PARALLEL` / `MAX_TOTAL_CHROMIUM_WORKERS` | tune to the machine + API rate limit (start low, e.g. `ROW_PARALLELISM=4`, raise if stable) |
| `LLM_CALLS_PER_SECOND` / `CDP_CONNECTS_PER_SECOND` | set under the provider's rate limit |

Tell the user explicitly: **the benchmark runs locally via Playwright — no sandbox service is required. Do not set any `SANDBOX_*` / `E2B_*` / `REMOTE_SANDBOX_*` / `OPENSANDBOX_*` variables; they are not used.**

### Step 4 — Place the benchmark dataset
The benchmark is distributed on **HuggingFace** (not in this repo). Have the user:
1. Download `benchmark.jsonl` from the HF dataset repo.
2. Place it somewhere, e.g. `./benchmark.jsonl` or `data/benchmark.jsonl`.

Quick peek to confirm format (each row: a `question` + its per-query checklist):
```bash
head -c 300 benchmark.jsonl
wc -l benchmark.jsonl
```

### Step 5 — Sanity smoke test (1 row)
Run a single-row smoke to confirm API + browser + build pipeline end-to-end before the full batch:
```bash
head -1 benchmark.jsonl > /tmp/smoke.jsonl
python scripts/eval_open.py --jsonl /tmp/smoke.jsonl --output /tmp/smoke.json --chunk-size 1 --no-resume
```
Expect: build phase passes, agents produce verdicts, a shard is written. If API/SSL errors → fix `.env` / `SSL_CERT_FILE`. If build fails → check Node version / `playwright install chromium`.

### Step 6 — Hand off
Once smoke passes, tell the user:
> Setup done. To auto-run the full benchmark, invoke the `livebench-run` skill — it will launch eval_open.py, monitor progress, retry failures, and merge results.

If the user wants to run manually instead, give the one-liner:
```bash
nohup python scripts/eval_open.py --jsonl benchmark.jsonl --output runs/report.json \
  --chunk-size 10 --monitor-state runs/monitor.json --log-file runs/run.log --no-resume \
  > runs/nohup.log 2>&1 &
```

## Notes
- All paths are relative to the repo root; do NOT hardcode `/tmp/eval_run` or NFS — use a user-chosen `runs/` dir.
- Do not install cron or background daemons during setup. Long-run self-heal is handled on-demand by `livebench-run`, not by system crons.
- If `.env.example` still lists legacy sandbox vars, leave them blank — local Playwright is the supported public mode.