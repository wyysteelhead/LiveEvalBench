"""Utilities for asyncio subprocess management."""

import asyncio


def close_subprocess_transports(proc: asyncio.subprocess.Process) -> None:
    """Close pipe transports of a completed subprocess.

    After ``proc.communicate()`` the process has exited but pipe transports
    may still be registered with the event loop. If these linger until the
    Process object is garbage-collected after the event loop closes, CPython
    raises ``RuntimeError('Event loop is closed')`` during ``__del__``.
    """
    for pipe_name in ("stdin", "stdout", "stderr"):
        pipe = getattr(proc, pipe_name, None)
        if pipe is None:
            continue
        transport = getattr(pipe, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass