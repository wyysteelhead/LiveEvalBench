"""SQLite-backed port lease pool.

Database lives at .runs/ports.db (WAL mode).
Table: leases(lease_id PK, name, port UNIQUE, holder, pid, expires_at, state,
              created_at, updated_at)
"""
from __future__ import annotations

import contextlib
import datetime
import random
import sqlite3
import uuid
from pathlib import Path
from typing import Any

_DEFAULT_DB = ".runs/ports.db"
_DDL = """
CREATE TABLE IF NOT EXISTS leases (
    lease_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    port       INTEGER UNIQUE NOT NULL,
    holder     TEXT NOT NULL,
    pid        INTEGER,
    expires_at TEXT NOT NULL,
    state      TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _expires_iso(ttl_s: int) -> str:
    dt = datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl_s)
    return dt.isoformat() + "Z"


@contextlib.contextmanager
def _conn(db_path: str):
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(_DDL)
    con.commit()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def sweep_stale(
    now: str | None = None,
    db_path: str = _DEFAULT_DB,
    _conn_obj: sqlite3.Connection | None = None,
) -> int:
    """Mark expired active and released leases as 'expired'. Returns count updated."""
    ts = now or _now_iso()

    def _do(con: sqlite3.Connection) -> int:
        count = 0
        cur = con.execute(
            "UPDATE leases SET state='expired', updated_at=? "
            "WHERE state='active' AND expires_at < ?",
            (_now_iso(), ts),
        )
        count += cur.rowcount
        cur = con.execute(
            "UPDATE leases SET state='expired', updated_at=? "
            "WHERE state='released' AND expires_at < ?",
            (_now_iso(), ts),
        )
        count += cur.rowcount
        # Purge expired records older than 1 hour to prevent table / WAL bloat.
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=1)).isoformat() + "Z"
        con.execute("DELETE FROM leases WHERE state='expired' AND updated_at < ?", (cutoff,))
        return count

    if _conn_obj is not None:
        return _do(_conn_obj)
    with _conn(db_path) as con:
        return _do(con)


def allocate(
    name: str,
    holder: str,
    pool: tuple[int, int] = (3000, 9000),
    ttl_s: int = 7200,
    db_path: str = _DEFAULT_DB,
    pid: int | None = None,
    _max_retries: int = 5,
) -> dict[str, Any]:
    """Allocate a free port from *pool* and return a lease dict."""
    for attempt in range(_max_retries):
        with _conn(db_path) as con:
            sweep_stale(_conn_obj=con)
            # Collect ports that are still in use (active or expired but not yet freed)
            taken = {
                row["port"]
                for row in con.execute(
                    "SELECT port FROM leases WHERE state IN ('active', 'expired')"
                )
            }
            candidates = list(range(pool[0], pool[1] + 1))
            random.shuffle(candidates)
            port = next((p for p in candidates if p not in taken), None)
            if port is None:
                raise RuntimeError(f"No free port available in range {pool}")
            now = _now_iso()
            expires = _expires_iso(ttl_s)
            lease_id = str(uuid.uuid4())
            try:
                con.execute(
                    "INSERT INTO leases (lease_id, name, port, holder, pid, expires_at, "
                    "state, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (lease_id, name, port, holder, pid, expires, "active", now, now),
                )
            except sqlite3.IntegrityError:
                # Race: another process grabbed this port between SELECT and INSERT.
                # Roll back (via context manager) and retry with a fresh random port.
                continue
            return {
                "lease_id": lease_id,
                "name": name,
                "port": port,
                "expires_at": expires,
            }
    raise RuntimeError(
        f"Could not allocate a free port in range {pool} "
        f"after {_max_retries} attempts (concurrent contention)"
    )


def renew(
    lease_id: str,
    ttl_s: int = 7200,
    db_path: str = _DEFAULT_DB,
) -> dict[str, Any]:
    """Extend a lease TTL. Returns updated lease info."""
    expires = _expires_iso(ttl_s)
    with _conn(db_path) as con:
        cur = con.execute(
            "UPDATE leases SET expires_at=?, updated_at=?, state='active' "
            "WHERE lease_id=?",
            (expires, _now_iso(), lease_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"lease_id {lease_id!r} not found")
        return {"lease_id": lease_id, "expires_at": expires}


def release(lease_id: str, db_path: str = _DEFAULT_DB) -> bool:
    """Mark a lease as released. Returns True if found."""
    with _conn(db_path) as con:
        cur = con.execute(
            "UPDATE leases SET state='released', updated_at=? WHERE lease_id=?",
            (_now_iso(), lease_id),
        )
        return cur.rowcount > 0


def list_active(db_path: str = _DEFAULT_DB) -> list[dict]:
    """Return all active leases as a list of dicts."""
    with _conn(db_path) as con:
        sweep_stale(_conn_obj=con)
        rows = con.execute(
            "SELECT * FROM leases WHERE state='active' ORDER BY port"
        ).fetchall()
        return [dict(r) for r in rows]
