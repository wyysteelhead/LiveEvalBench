"""Build-phase tools for Code Engineer agent.

These tools run locally (no sandbox) and are used during Phase 1 to write
files, run npm commands, start the dev server, and record artifacts.
All tools return JSON strings; errors are returned as {"error": "..."} so
the agent can self-correct without raising exceptions.

Constraints are enforced via session.py — see that module for rules.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import sys
import time
from pathlib import Path

# Ensure src/ is on the path so we can import tools.*
_src_dir = str(Path(__file__).parent.parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from .registry import register_tool
from . import session as _session

_logger = logging.getLogger(__name__)


def _workspace_root(session_id: str = "") -> Path | None:
    workspace_root = _session.get_workspace_root(session_id)
    if not workspace_root:
        return None
    return Path(workspace_root).resolve(strict=False)


def _effective_session_id(session_id: str = "") -> str:
    return _session.get_effective_session_id(session_id)


# Regex captures install commands that add named packages. Bare `npm install`
# / `npm ci` (no package names) carry no new info, so we skip them.
_INSTALL_CMD_RE = re.compile(
    r"^\s*(?:npm\s+(?:install|i|add)|pnpm\s+(?:add|install|i)|yarn\s+add)\b(?P<rest>.*)$",
    re.IGNORECASE,
)


def _parse_install_packages(cmd: str) -> tuple[list[str], bool]:
    """Parse an install command for named package targets.

    Returns (package_names, is_dev). Returns ([], False) for commands that are
    not package-installs or that install no named packages (bare `npm install`).
    """
    match = _INSTALL_CMD_RE.match(cmd)
    if not match:
        return [], False
    try:
        tokens = shlex.split(match.group("rest"))
    except ValueError:
        return [], False
    is_dev = False
    packages: list[str] = []
    for token in tokens:
        if not token:
            continue
        if token.startswith("-"):
            if token in ("--save-dev", "--dev", "-D"):
                is_dev = True
            # Skip all other flags and their values; package names never start with "-"
            continue
        # Skip flag values like `--registry=https://...` (already handled above)
        # and bare positional args that are clearly not packages
        if "=" in token and not token.startswith("@"):
            continue
        packages.append(token)
    return packages, is_dev


def _resolve_pkg_version(cwd_path: Path, pkg_name: str) -> tuple[str, bool]:
    """Look up the installed version of pkg_name from cwd/package.json.

    Returns (version_spec, is_dev). Falls back to ("unknown", False) on any
    parse error so a missing/malformed package.json never breaks the caller.
    """
    try:
        pkg_json = json.loads((cwd_path / "package.json").read_text(encoding="utf-8"))
    except Exception:
        return "unknown", False
    deps = pkg_json.get("dependencies") or {}
    dev_deps = pkg_json.get("devDependencies") or {}
    if pkg_name in dev_deps:
        return str(dev_deps[pkg_name]), True
    if pkg_name in deps:
        return str(deps[pkg_name]), False
    return "unknown", False


def _maybe_record_observed_install(*, cmd: str, cwd: str, session_id: str) -> None:
    """Append a JSON line to <workspace_root>/.observed_installs.jsonl
    whenever an `npm install <pkg>` (or equivalent) succeeds.

    Never raises — observability must not break the eval flow.
    """
    try:
        packages, dev_flag = _parse_install_packages(cmd)
        if not packages:
            return
        workspace_root = _workspace_root(session_id)
        if workspace_root is None:
            return
        cwd_path = Path(cwd)
        records: list[dict] = []
        for name in packages:
            # `pkg@^1.2.3` syntax: strip the spec to look up the resolved version
            bare_name = name
            if bare_name.startswith("@"):
                # scoped: @scope/name or @scope/name@spec
                at_pos = bare_name.find("@", 1)
                if at_pos != -1:
                    bare_name = bare_name[:at_pos]
            else:
                bare_name = bare_name.split("@", 1)[0]
            version, is_dev = _resolve_pkg_version(cwd_path, bare_name)
            records.append({
                "name": bare_name,
                "version": version,
                "dev": is_dev or dev_flag,
            })
        line = json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": session_id,
            "cwd": str(cwd_path),
            "cmd": cmd,
            "packages": records,
        }, ensure_ascii=False)
        jsonl_path = workspace_root / ".observed_installs.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        _logger.warning("Failed to record observed install for cmd=%r", cmd, exc_info=True)


_CHROMIUM_BINS: frozenset[str] = frozenset({"chromium-browser", "chromium", "google-chrome"})

_CHROMIUM_NOISE_FLAGS: tuple[str, ...] = (
    "--disable-logging",
    "--log-level=3",
    "--disable-gpu-sandbox",
    "--disable-software-rasterizer",
)


def _apply_chromium_bin_override(cmd: str) -> str:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return cmd

    if not parts:
        return cmd

    bin_name = os.path.basename(parts[0])
    if bin_name not in _CHROMIUM_BINS:
        return cmd

    chromium_bin = str(os.getenv("CHROMIUM_BIN", "") or "").strip()
    if chromium_bin:
        parts[0] = chromium_bin

    # Inject noise-suppression flags after the binary if not already present
    existing = set(parts[1:])
    extra = [f for f in _CHROMIUM_NOISE_FLAGS if f not in existing]
    if extra:
        parts[1:1] = extra

    return shlex.join(parts)


def _resolve_workspace_path(raw_path: str, session_id: str = "", label: str = "path") -> str:
    root = _workspace_root(session_id)
    path = Path(raw_path).expanduser()
    if root is None:
        return str(path)

    resolved = (path if path.is_absolute() else root / path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay within workspace root {root}: {raw_path}"
        ) from exc
    return str(resolved)


# Matches a stray tool-arg JSON tail leaked into the content field, e.g.
#   ... real content ...\n", "path": "/abs/path"}
#   ... real content ...\n", path: "/abs/path"}        (unquoted key, seen on Gemini 3.1 Flash)
# Trailing whitespace/newlines after the closing brace are tolerated.
_TOOL_ARG_TAIL_TAINT_RE = re.compile(
    r'",\s*"?path"?\s*:\s*"(?P<tail_path>[^"\\]+)"\s*\}\s*\Z'
)


def _strip_tool_arg_tail_taint(content: str, resolved_path: str, original_path: str) -> tuple[str, str | None]:
    """Strip a stray JSON tail like `", "path": "<path>"}` if the model leaked
    its own tool-call envelope into the content field.

    Observed on Gemini 3.1 Flash (and occasionally other models): when content
    is long the model emits a function call whose content string is not
    properly terminated, so the trailing `", "path": "..."}` of the JSON
    envelope ends up appended to content. We only strip when the trailing
    path equals the current call's target — that makes false positives
    effectively impossible (no legitimate source file ends with its own
    absolute path inside a stray JSON closer).
    """
    m = _TOOL_ARG_TAIL_TAINT_RE.search(content)
    if not m:
        return content, None
    tail_path = m.group("tail_path")
    if tail_path not in (resolved_path, original_path):
        return content, None
    return content[: m.start()], tail_path


@register_tool("local_fs_write", "Write content to a local file path (creates parent dirs).")
async def local_fs_write(path: str, content: str, session_id: str = "") -> str:
    try:
        from tools.fs import write as _write
        resolved_path = _resolve_workspace_path(path, session_id=session_id, label="path")
        sanitized, stripped_tail = _strip_tool_arg_tail_taint(content, resolved_path, path)
        _write(resolved_path, sanitized)
        result: dict = {"ok": True, "path": resolved_path}
        if stripped_tail is not None:
            result["sanitized"] = "stripped tool-arg JSON tail leaked into content"
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": f"guardrail_rejected: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


_FENCE_OPEN_RE = re.compile(r"^\s*`{3,}[^\n]*$")
_FENCE_CLOSE_RE = re.compile(r"^\s*`{3,}\s*$")


@register_tool(
    "local_fs_extract_block",
    "Deterministically copy a 1-based inclusive line range from a source file "
    "(typically .model_reply.md) into a destination file in the workspace. "
    "Use this INSTEAD of local_fs_write for any code block longer than ~30 "
    "lines, or whenever the reply is provided as a file pointer. The content "
    "never passes through LLM tool args, so it cannot be truncated or "
    "paraphrased. Pass strip_fence=true to drop a leading ```lang and "
    "trailing ``` line if your range includes them.",
)
async def local_fs_extract_block(
    source_path: str,
    start_line: int,
    end_line: int,
    dest_path: str,
    strip_fence: bool = True,
    session_id: str = "",
) -> str:
    try:
        from tools.fs import read as _read, write as _write
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            return json.dumps({"error": "start_line and end_line must be integers"})
        if start_line < 1 or end_line < start_line:
            return json.dumps({
                "error": f"invalid range: start_line={start_line}, end_line={end_line} "
                         "(both must be >=1 and end_line >= start_line)"
            })

        resolved_source = _resolve_workspace_path(
            source_path, session_id=session_id, label="source_path"
        )
        resolved_dest = _resolve_workspace_path(
            dest_path, session_id=session_id, label="dest_path"
        )
        if resolved_source == resolved_dest:
            return json.dumps({"error": "source_path and dest_path must differ"})

        source_p = Path(resolved_source)
        if not source_p.is_file():
            return json.dumps({"error": f"source_path is not a file: {resolved_source}"})

        body = _read(resolved_source, start=start_line, end=end_line)
        fence_stripped: list[str] = []
        if strip_fence and body:
            # splitlines(keepends=True) preserves trailing newlines so the
            # rejoined body keeps the original line endings.
            lines = body.splitlines(keepends=True)
            if lines and _FENCE_OPEN_RE.match(lines[0].rstrip("\n")):
                fence_stripped.append("leading")
                lines = lines[1:]
            if lines and _FENCE_CLOSE_RE.match(lines[-1].rstrip("\n")):
                fence_stripped.append("trailing")
                lines = lines[:-1]
            body = "".join(lines)

        _write(resolved_dest, body)
        return json.dumps({
            "ok": True,
            "source_path": resolved_source,
            "source_range": [start_line, end_line],
            "dest_path": resolved_dest,
            "bytes_written": len(body.encode("utf-8")),
            "lines_written": body.count("\n") + (0 if body.endswith("\n") or not body else 1),
            "fence_stripped": fence_stripped,
        })
    except ValueError as exc:
        return json.dumps({"error": f"guardrail_rejected: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register_tool(
    "local_fs_read",
    "Read a local file (or list a directory) and return its content as a string. "
    "For long files, pass start_line and end_line (1-based, inclusive) to read a slice; "
    "the response will append a truncation hint with the next start_line when more lines remain.",
)
async def local_fs_read(
    path: str,
    session_id: str = "",
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    try:
        from tools.fs import read as _read
        resolved_path = _resolve_workspace_path(path, session_id=session_id, label="path")
        p = Path(resolved_path)
        if p.is_dir():
            entries = sorted(
            ({"name": e.name, "type": "dir" if e.is_dir() else "file", "size": e.stat().st_size if e.is_file() else 0}
            for e in p.iterdir()),
            key=lambda d: d["name"],
        )
            return json.dumps({"ok": True, "is_dir": True, "path": str(p), "entries": entries})
        start = start_line if start_line and start_line > 0 else None
        end = end_line if end_line and end_line > 0 else None
        if start is None and end is None:
            return _read(str(p))
        body = _read(str(p), start=start, end=end)
        total_lines = sum(1 for _ in p.open("r", encoding="utf-8"))
        effective_end = end if end is not None else total_lines
        remaining = max(0, total_lines - effective_end)
        if remaining > 0:
            next_start = effective_end + 1
            hint = (
                f"\n\n[...truncated. {remaining} more lines remain "
                f"(total {total_lines}). Next call: start_line={next_start}.]"
            )
            return body + hint
        return body
    except ValueError as exc:
        return json.dumps({"error": f"guardrail_rejected: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# Rewrites `npx vitest …` (and `npx -y vitest …` / `npx --pkg=… vitest …`) to
# `./node_modules/.bin/vitest …`.  `npx` triggers an npm-registry metadata
# fetch even when the package is already installed locally; on slow / NFS
# environments this can block the tool call for 60-120s and starve the global
# subtask semaphore.  We rewrite before exec and surface a notice so the LLM
# learns to use the local binary on subsequent calls.
_NPX_VITEST_RE = re.compile(
    r"^(?P<lead>\s*)npx(?P<flags>(?:\s+-{1,2}[A-Za-z][\w-]*(?:=\S+)?)*)\s+vitest(?P<rest>\s.*|\s*$)",
    re.DOTALL,
)


def _rewrite_npx_vitest(cmd: str) -> tuple[str, bool]:
    """Return (possibly-rewritten cmd, was_rewritten)."""
    if not isinstance(cmd, str):
        return cmd, False
    m = _NPX_VITEST_RE.match(cmd)
    if not m:
        return cmd, False
    rewritten = f"{m.group('lead')}./node_modules/.bin/vitest{m.group('rest')}"
    return rewritten, True


@register_tool(
    "local_exec_run",
    "Run a shell command synchronously and return {stdout, stderr, exit_code, duration_ms}. "
    "Pass session_id=<artifacts_path> to enable constraint tracking.",
)
async def local_exec_run(cmd: str, cwd: str, timeout_s: int = 120, session_id: str = "") -> str:
    try:
        from tools.exec import run as _run
        effective_session_id = _effective_session_id(session_id)
        resolved_cwd = _resolve_workspace_path(cwd, session_id=effective_session_id, label="cwd")
        original_cmd = cmd
        cmd, _npx_rewritten = _rewrite_npx_vitest(cmd)
        if _npx_rewritten:
            _logger.warning(
                "local_exec_run: rewrote 'npx vitest' to './node_modules/.bin/vitest' "
                "(session=%s cwd=%s original=%r)",
                effective_session_id or "-",
                resolved_cwd,
                original_cmd,
            )
        # Offload to a thread so subprocess.run() does not block the asyncio event loop.
        result = await asyncio.to_thread(
            _run, cmd, cwd=resolved_cwd, timeout_s=timeout_s,
        )
        if _npx_rewritten:
            result["rewritten_from"] = original_cmd
            result["notice"] = (
                "Your command was rewritten: 'npx vitest …' → './node_modules/.bin/vitest …'. "
                "Reason: `npx` performs an npm-registry metadata lookup that can stall for 60+s "
                "on this network, even though vitest is already installed locally. "
                "Use './node_modules/.bin/vitest' directly in subsequent calls — do NOT use npx."
            )
        # Record successful npm install for session constraints
        if effective_session_id and result.get("exit_code") == 0:
            _maybe_record_observed_install(
                cmd=cmd,
                cwd=resolved_cwd,
                session_id=effective_session_id,
            )
            if "npm install" in cmd.lower():
                _session.record_install_success(effective_session_id)
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": f"guardrail_rejected: {exc}", "exit_code": -1})
    except Exception as exc:
        return json.dumps({"error": str(exc), "exit_code": -1})


@register_tool(
    "local_exec_start",
    "Start a background process. Optionally poll readiness_url until HTTP 200. "
    "Returns {pid, started_at, stdout_path, stderr_path}. Pass session_id=<artifacts_path> to enable constraint enforcement.",
)
async def local_exec_start(
    cmd: str,
    cwd: str,
    readiness_url: str = "",
    timeout_s: int = 60,
    session_id: str = "",
    log_dir: str = "",
) -> str:
    effective_session_id = _effective_session_id(session_id)
    normalized_cmd = _apply_chromium_bin_override(cmd)
    # Enforce session constraints before executing
    if effective_session_id:
        err = _session.check_exec_start(effective_session_id, normalized_cmd)
        if err:
            return json.dumps({"error": err})

    try:
        from tools.exec import start as _start
        resolved_cwd = _resolve_workspace_path(cwd, session_id=effective_session_id, label="cwd")
        readiness = (
            {"url": readiness_url, "timeout_s": timeout_s, "interval_s": 1.0}
            if readiness_url
            else None
        )
        # Offload to a thread so the readiness polling loop (time.sleep)
        # does not block the asyncio event loop.
        result = await asyncio.to_thread(
            _start,
            normalized_cmd,
            cwd=resolved_cwd,
            readiness=readiness,
            log_dir=log_dir or None,
        )
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"error": f"guardrail_rejected: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register_tool(
    "protocol_write_artifacts",
    "Atomically write artifacts JSON to path. Accepts path or artifacts_path. data_json must be a JSON string. "
    "When state=ready, app_url and cdp_url are verified reachable before writing. "
    "If this tool returns an error (e.g. constraint_violation: app_url not reachable), it means the app "
    "or CDP browser is NOT actually serving traffic. Do NOT ignore the error — check whether the dev "
    "server and CDP browser are running (local_fs_read the .process_logs directory), fix the issue, "
    "then retry protocol_write_artifacts. Do NOT call submit_verdict until artifacts are written "
    "successfully.",
)
async def protocol_write_artifacts(
    data_json: str,
    path: str = "",
    artifacts_path: str = "",
    session_id: str = "",
) -> str:
    try:
        from tools.protocol import write_json_atomic as _write_json
        effective_session_id = _effective_session_id(session_id)
        target_path = path or artifacts_path
        if not target_path:
            return json.dumps({"error": "missing required path (path or artifacts_path)"})
        resolved_target_path = _resolve_workspace_path(
            target_path,
            session_id=effective_session_id,
            label="path",
        )
        data = json.loads(data_json)

        # Enforce: state=ready requires HTTP verification
        if data.get("state") == "ready":
            ports = data.get("ports", [])
            app_url = ports[0].get("url", "") if ports else ""
            cdp_url = data.get("cdp_url", "")
            err = _session.check_write_artifacts_ready(app_url, cdp_url, session_id=effective_session_id)
            if err:
                # Write the file with state=error so downstream agents
                # get a meaningful error instead of ENOENT.
                data["state"] = "error"
                data["error"] = err
                _write_json(resolved_target_path, data, bump_revision=True)
                return json.dumps({"error": err})

        _write_json(resolved_target_path, data, bump_revision=True)
        return json.dumps({"ok": True, "path": resolved_target_path})
    except ValueError as exc:
        return json.dumps({"error": f"guardrail_rejected: {exc}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register_tool(
    "runtime_detect",
    "Detect installed Node/npm/pnpm/yarn versions. Returns {node, npm, pnpm, yarn, package_manager}.",
)
async def runtime_detect() -> str:
    try:
        from tools.runtime import detect as _detect
        return json.dumps(_detect())
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register_tool(
    "local_port_allocate",
    "Allocate a free port from the port lease DB. Returns {lease_id, port}. "
    "Must be called before starting any server. Pass session_id=<artifacts_path> to register the port.",
)
async def local_port_allocate(name: str, holder: str, session_id: str = "") -> str:
    try:
        from tools.ports import allocate as _allocate
        result = _allocate(name=name, holder=holder)
        effective_session_id = _effective_session_id(session_id)
        if effective_session_id:
            _session.record_port_allocated(effective_session_id, result["port"])
        return json.dumps({"lease_id": result["lease_id"], "port": result["port"]})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register_tool(
    "local_port_release",
    "Release a port lease back to the pool by lease_id.",
)
async def local_port_release(lease_id: str, session_id: str = "", port: int = 0) -> str:
    try:
        from tools.ports import release as _release
        ok = _release(lease_id)
        effective_session_id = _effective_session_id(session_id)
        if effective_session_id and port:
            _session.record_port_released(effective_session_id, port)
        return json.dumps({"ok": ok, "lease_id": lease_id})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register_tool(
    "protocol_rebuild_workspace",
    "Restart the dev server and CDP browser on fresh ports for a preserved workspace. "
    "Used during sub-task retry to re-establish the runtime environment without "
    "re-running the full build engineer. "
    "Returns {success, app_url, cdp_url, app_port, cdp_port} on success.",
)
async def protocol_rebuild_workspace(
    workspace_root: str,
    session_id: str = "",
    artifacts_path: str = "",
) -> str:
    """Rebuild the workspace runtime environment for a task retry.

    Runs npm install, allocates fresh ports, starts the dev server and CDP
    browser, and writes updated artifacts. Handles both Node.js and static
    (python3 -m http.server) projects.
    """
    try:
        from tools.ports import allocate as _allocate, release as _release
        from tools.exec import run as _exec_run, start as _exec_start
        from tools.runtime import detect as _detect_runtime

        resolved_root = Path(workspace_root).resolve(strict=True)
        effective_session_id = _effective_session_id(session_id)

        # 1. Detect runtime
        runtime_info = _detect_runtime()
        has_node = bool(runtime_info.get("node"))

        # 2. Determine project type
        is_node_project = (resolved_root / "package.json").exists()
        is_static_project = not is_node_project and (resolved_root / "index.html").exists()

        if not is_node_project and not is_static_project:
            return json.dumps({
                "success": False,
                "error": f"No package.json or index.html found in {workspace_root}",
            })

        # 3. npm install for Node.js projects
        if is_node_project and has_node:
            install_result = await asyncio.to_thread(
                _exec_run,
                "npm install",
                cwd=str(resolved_root),
                timeout_s=120,
            )
            if install_result.get("exit_code") != 0:
                return json.dumps({
                    "success": False,
                    "error": f"npm install failed: {install_result.get('stderr', '')[:500]}",
                })
            if effective_session_id:
                _session.record_install_success(effective_session_id)

        # 4. Allocate fresh ports
        app_lease = _allocate(name="app_retry", holder=f"rebuild_{effective_session_id or 'unknown'}")
        app_port = app_lease["port"]

        cdp_lease = _allocate(name="cdp_retry", holder=f"rebuild_{effective_session_id or 'unknown'}")
        cdp_port = cdp_lease["port"]

        if effective_session_id:
            _session.record_port_allocated(effective_session_id, app_port)
            _session.record_port_allocated(effective_session_id, cdp_port)

        # 5. Start dev server
        app_cmd: str
        readiness_url: str
        if is_node_project:
            # Detect dev script
            pkg_json = json.loads((resolved_root / "package.json").read_text(encoding="utf-8"))
            scripts = pkg_json.get("scripts", {})
            dev_script_key = next(
                (k for k in ("dev", "start", "serve") if k in scripts),
                None,
            )
            if dev_script_key:
                app_cmd = f"npm run {dev_script_key} -- --port {app_port}"
            elif "next" in (pkg_json.get("dependencies") or {}):
                app_cmd = f"npx next dev --port {app_port}"
            elif "vite" in (pkg_json.get("dependencies") or {}) or "vite" in (pkg_json.get("devDependencies") or {}):
                app_cmd = f"npx vite --port {app_port}"
            else:
                app_cmd = f"npx vite --port {app_port}"
            readiness_url = f"http://127.0.0.1:{app_port}"
        else:
            app_cmd = f"python3 -m http.server {app_port}"
            readiness_url = f"http://127.0.0.1:{app_port}"

        server_result = await asyncio.to_thread(
            _exec_start,
            app_cmd,
            cwd=str(resolved_root),
            readiness={"url": readiness_url, "timeout_s": 60, "interval_s": 1.0},
            log_dir=str(resolved_root / ".process_logs"),
        )
        if "error" in (server_result or {}):
            return json.dumps({
                "success": False,
                "error": f"Dev server start failed: {server_result.get('error', server_result)}",
            })

        # 6. Start CDP browser
        chromium_bin = str(os.getenv("CHROMIUM_BIN", "") or "").strip() or "chromium-browser"
        browser_cmd = (
            f"{chromium_bin} --headless --no-sandbox "
            f"--remote-debugging-port={cdp_port} "
            f"--remote-debugging-address=0.0.0.0 "
            f"--disable-logging --log-level=3 "
            f"--disable-gpu-sandbox --disable-software-rasterizer "
            # Cap renderer process count: default chromium spawns one renderer per
            # site + zygote/utility/gpu = ~7-10 children. With high row parallelism
            # this triples context-switch overhead. We only ever inspect one app
            # at a time per CDP browser, so 2 renderers is ample headroom.
            f"--renderer-process-limit=2 "
            # Disable site isolation for the same reason — irrelevant for testing,
            # eliminates a class of extra renderer spawns.
            f"--disable-features=site-per-process,IsolateOrigins "
            f"about:blank"
        )
        browser_result = await asyncio.to_thread(
            _exec_start,
            browser_cmd,
            cwd=str(resolved_root),
            readiness={"url": f"http://127.0.0.1:{cdp_port}/json", "timeout_s": 30, "interval_s": 0.5},
            log_dir=str(resolved_root / ".process_logs"),
        )
        if "error" in (browser_result or {}):
            _release(app_lease["lease_id"])
            return json.dumps({
                "success": False,
                "error": f"CDP browser start failed: {browser_result.get('error', browser_result)}",
            })

        # 7. Write updated artifacts
        app_url = f"http://127.0.0.1:{app_port}"
        cdp_url = f"http://127.0.0.1:{cdp_port}"
        artifacts_data = {
            "state": "ready",
            "ports": [{"name": "app", "port": app_port, "url": app_url}],
            "cdp_url": cdp_url,
        }
        target_path = artifacts_path or (str(resolved_root / "artifacts.json"))
        from tools.protocol import write_json_atomic as _write_json
        _write_json(target_path, artifacts_data, bump_revision=True)

        return json.dumps({
            "success": True,
            "app_url": app_url,
            "cdp_url": cdp_url,
            "app_port": app_port,
            "cdp_port": cdp_port,
            "app_lease_id": app_lease["lease_id"],
            "cdp_lease_id": cdp_lease["lease_id"],
        })

    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})
