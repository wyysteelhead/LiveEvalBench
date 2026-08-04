"""Row-priority gate for evaluate agent scheduling.

Replaces a plain asyncio.Semaphore so that evaluate agents from rows
that finished building earlier get priority.  Once a row's build phase
completes, all its evaluate agents are dispatched ahead of agents from
rows whose build hasn't finished yet, keeping results per row as close
together as possible and reducing subtask timeouts from sparse scheduling.
"""

from __future__ import annotations

import asyncio


class RowPriorityGate:
    """Async gate that prioritises agents by build-completion order."""

    def __init__(self, max_concurrent: int) -> None:
        self._max = max(1, int(max_concurrent))
        self._running = 0
        self._cond = asyncio.Condition(asyncio.Lock())
        self._queue: list[tuple[str, int, asyncio.Event]] = []
        self._next_seq = 0
        self._row_build_order: dict[str, int] = {}
        self._build_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mark_build_complete(self, row_id: str) -> None:
        """Record that *row_id* has finished building.

        Lower build-order number = earlier completion = higher priority.
        Wakes waiting agents so the priority change takes effect immediately.
        Safe to call multiple times (first call wins).
        """
        async with self._cond:
            if row_id not in self._row_build_order:
                self._row_build_order[row_id] = self._build_counter
                self._build_counter += 1
            self._cond.notify_all()

    async def acquire(self, row_id: str) -> None:
        evt = asyncio.Event()
        async with self._cond:
            seq = self._next_seq
            self._next_seq += 1
            self._queue.append((row_id, seq, evt))
            while True:
                if self._running < self._max:
                    idx = self._pick_next_locked()
                    if idx >= 0 and self._queue[idx][2] is evt:
                        del self._queue[idx]
                        self._running += 1
                        return
                await self._cond.wait()

    async def release(self, row_id: str) -> None:
        async with self._cond:
            self._running = max(0, self._running - 1)
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick_next_locked(self) -> int:
        if not self._queue:
            return -1

        # Build a sort key for each queued item:
        #   group 0 — build-complete rows (ordered by completion time)
        #   group 1 — rows still building (ordered by queue arrival)
        # Within a group, fall back to FIFO order (seq).
        def _sort_key(item: tuple[str, int, asyncio.Event]) -> tuple[int, int, int]:
            rid, seq, _evt = item
            build_order = self._row_build_order.get(rid)
            if build_order is not None:
                return (0, build_order, seq)
            return (1, 0, seq)

        best_idx = 0
        best_key = _sort_key(self._queue[0])
        for i in range(1, len(self._queue)):
            key = _sort_key(self._queue[i])
            if key < best_key:
                best_key = key
                best_idx = i
        return best_idx