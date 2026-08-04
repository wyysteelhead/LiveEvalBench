# Tools API Reference

All modules live in `src/tools/`. Import example:

```python
from src.tools.guardrail import check_command
from src.tools.fs import read, write
from src.tools.ports import allocate, release
from src.tools.exec import run, start, plan
from src.tools.runtime import detect, ensure
from src.tools.protocol import write_json_atomic, wait_for_state
```

---

## guardrail.py

Command risk classification and domain allow-listing.

### Constants

| Name | Type | Description |
|------|------|-------------|
| `FORBIDDEN_COMMANDS` | `frozenset[str]` | Always-rejected commands: sudo, apt, apt-get, dnf, brew, docker |
| `COMMAND_POLICY` | `dict[str, str]` | Maps first token → risk level |
| `AUTO_APPROVE` | `str` | `"auto_approve"` |
| `REQUIRE_CONFIRM` | `str` | `"require_confirm"` |
| `REJECT` | `str` | `"reject"` |

### Functions

**`check_command(cmd) -> str`**
- Raises `ValueError` if command is forbidden/rejected
- Returns risk level string otherwise

**`is_domain_allowed(url, mode="benchmark", allowlist=None) -> bool`**
- `mode="benchmark"`: strict allowlist (npm registry, GitHub)
- `mode="open"`: all domains permitted

**`audit_log_entry(cmd, risk, cwd="") -> dict`**
- Returns `{ts, cmd, risk, cwd}` for structured logging

---

## fs.py

File-system helpers with atomic writes.

**`read(path, start=None, end=None) -> str`**
- Reads file; optional 1-based line slice `[start, end]`

**`write(path, content, create=True, overwrite=True) -> None`**
- Atomic write via `tempfile.mkstemp` + `os.replace`
- Creates parent directories when `create=True`

**`apply_patch(patch_text) -> {applied, conflicts}`**
- Delegates to system `patch -p1`
- Returns `{applied: bool, conflicts: list[str]}`

**`snapshot(label, base_dir=".runs/snapshots") -> str`**
- Creates `.runs/snapshots/<timestamp>_<label>/meta.json`
- Returns snapshot directory path

---

## ports.py

SQLite-backed port lease pool at `.runs/ports.db`.

**`allocate(name, holder, pool=(3000,9000), ttl_s=7200, db_path, pid=None) -> dict`**
- Returns `{lease_id, name, port, expires_at}`
- Calls `sweep_stale` internally

**`renew(lease_id, ttl_s=7200, db_path) -> dict`**
- Returns `{lease_id, expires_at}`

**`release(lease_id, db_path) -> bool`**
- Marks lease as `released`; returns True if found

**`list_active(db_path) -> list[dict]`**
- Returns all active leases sorted by port

**`sweep_stale(now=None, db_path, _conn=None) -> int`**
- Marks expired leases as `expired`; returns count updated

---

## exec.py

Command execution with guardrail enforcement.

**`run(cmd, cwd=".", env=None, timeout_s=120, pty=False) -> dict`**
- Returns `{stdout, stderr, exit_code, duration_ms}`
- Writes JSON audit line to stderr

**`start(cmd, cwd=".", env=None, readiness=None) -> dict`**
- Non-blocking `Popen`; returns `{pid, started_at}`
- `readiness = {"url": str, "timeout_s": int, "interval_s": float}`

**`plan(steps) -> dict`**
- Dry-run validates list of `{cmd, cwd}` dicts
- Returns `{normalized_plan, guardrail_report}`; no subprocesses started

---

## runtime.py

Runtime detection and environment preparation.

**`DEFAULT_ENV`** — `dict[str, str]`
- `CI=1`, `NEXT_TELEMETRY_DISABLED=1`, `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`, `PUPPETEER_SKIP_DOWNLOAD=1`

**`detect() -> dict`**
- Returns `{node, npm, pnpm, yarn, package_manager}` (version strings or None)

**`ensure(spec) -> dict`**
- `spec = {"node_version": "18", "package_manager": "pnpm"}`
- Attempts `corepack enable` if package manager missing
- Returns `{ok, detected, env, warnings}`

---

## protocol.py

Manifest/artifacts validation, atomic JSON I/O, state polling.

**`validate_manifest(data) -> list[str]`**
- Returns error strings; empty list = valid

**`validate_artifacts(data) -> list[str]`**
- Extra checks when `state=ready`: requires `ports[].port` and `ports[].url`

**`write_json_atomic(path, data, bump_revision=True)`**
- Increments `revision`, sets `last_updated`, atomically writes

**`read_json(path) -> dict`**
- Returns `{}` if file does not exist

**`wait_for_state(path, desired, timeout_s=120, poll_ms=500) -> dict`**
- Raises `RuntimeError` on `state=failed`
- Raises `TimeoutError` on timeout

**`update_artifacts(path, patch, bump_revision=True) -> dict`**
- Deep-merges `patch` into existing JSON, atomically writes, returns result
