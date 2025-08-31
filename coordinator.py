# Declaration:
# This program is the result of ChatGPT.


# cocu_base_agents/new_agent/coordinator.py
"""
Coordinator: a tiny, SQLite-backed lock + claim service for multi-agent coordination.

Design goals:
- Non-blocking "try-lock" semantics with TTL (agent retries with backoff in its own loop).
- Safe across separate processes (SQLite WAL + BEGIN IMMEDIATE per write).
- Minimal API: resource locks, order claims, and counter reservations for plate staging.

Lock policy (defaults; tune the constants below):
- DEFAULT_LOCK_TTL_S = 6.0 seconds  → auto-expires if agent dies/stalls.
- CLAIM_TTL_S        = 120.0 seconds → long enough to finish an order.
- RES_TTL_S          = 60.0 seconds  → reasonable time to stage/use a counter.

Fairness:
- First committer wins; we record acquired_at to enable FIFO if you want later.
"""

import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

DB_PATH_DEFAULT = os.path.join("cocu_base_agents", "new_agent", "coordinator.db")

DEFAULT_LOCK_TTL_S = 6.0
CLAIM_TTL_S = 120.0
RES_TTL_S = 60.0

DDL_LOCKS = """
CREATE TABLE IF NOT EXISTS locks (
  resource_id TEXT PRIMARY KEY,
  owner       TEXT NOT NULL,
  kind        TEXT NOT NULL,
  acquired_at REAL NOT NULL,
  expires_at  REAL NOT NULL
);
"""

DDL_CLAIMS = """
CREATE TABLE IF NOT EXISTS claims (
  order_id   TEXT PRIMARY KEY,
  owner      TEXT NOT NULL,
  expires_at REAL NOT NULL
);
"""

DDL_RESERVATIONS = """
CREATE TABLE IF NOT EXISTS reservations (
  counter_id TEXT PRIMARY KEY,
  owner      TEXT NOT NULL,
  expires_at REAL NOT NULL
);
"""

PRAGMAS = [
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA busy_timeout=250;"  # small busy timeout for rare meta-contention
]


class Coordinator:
    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON;")
        for p in PRAGMAS:
            self._conn.execute(p)
        self._conn.execute(DDL_LOCKS)
        self._conn.execute(DDL_CLAIMS)
        self._conn.execute(DDL_RESERVATIONS)
        self._conn.commit()

    @contextmanager
    def _immediate(self):
        """BEGIN IMMEDIATE to obtain a write lock deterministically."""
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE;")
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ---------- helpers ----------
    @staticmethod
    def _now() -> float:
        return time.time()

    def _gc_expired(self, cur: sqlite3.Cursor):
        now = self._now()
        cur.execute("DELETE FROM locks WHERE expires_at < ?;", (now,))
        cur.execute("DELETE FROM claims WHERE expires_at < ?;", (now,))
        cur.execute("DELETE FROM reservations WHERE expires_at < ?;", (now,))

    # ---------- resource locks ----------
    def lock_resource(self, res_id: str, agent_id: str, ttl_s: float = DEFAULT_LOCK_TTL_S, kind: str = "generic") -> bool:
        """
        Try to acquire an exclusive lock on `res_id` for `agent_id`.
        Non-blocking: returns True if succeeded, False if someone else currently holds it.
        Existing owner may extend their own lock by calling this again (idempotent for same owner).
        """
        if not res_id or not agent_id:
            return False
        now = self._now()
        exp = now + float(ttl_s or DEFAULT_LOCK_TTL_S)
        with self._immediate() as cur:
            self._gc_expired(cur)
            # Who owns it?
            cur.execute("SELECT owner FROM locks WHERE resource_id=?;", (res_id,))
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO locks(resource_id, owner, kind, acquired_at, expires_at) VALUES(?,?,?,?,?);",
                    (res_id, agent_id, kind, now, exp)
                )
                return True
            owner = row[0]
            if owner == agent_id:
                # Extend own lock (heartbeat)
                cur.execute("UPDATE locks SET expires_at=? WHERE resource_id=?;", (exp, res_id))
                return True
            # Locked by someone else
            return False

    def unlock_resource(self, res_id: str, agent_id: str) -> bool:
        """Release `res_id` only if `agent_id` owns it. Returns True if released."""
        if not res_id or not agent_id:
            return False
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("DELETE FROM locks WHERE resource_id=? AND owner=?;", (res_id, agent_id))
            return cur.rowcount > 0

    def who_owns(self, res_id: str) -> Optional[str]:
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("SELECT owner FROM locks WHERE resource_id=?;", (res_id,))
            r = cur.fetchone()
            return r[0] if r else None

    def when_acquired(self, res_id: str) -> Optional[float]:
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("SELECT acquired_at FROM locks WHERE resource_id=?;", (res_id,))
            r = cur.fetchone()
            return float(r[0]) if r else None

    def heartbeat(self, agent_id: str, extend_s: float = DEFAULT_LOCK_TTL_S) -> None:
        """Extend TTL for *all* locks owned by agent_id (lightweight safety net)."""
        now = self._now()
        exp = now + float(extend_s or DEFAULT_LOCK_TTL_S)
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("UPDATE locks SET expires_at=? WHERE owner=?;", (exp, agent_id))

    # ---------- order claims ----------
    def claim_order(self, order_id: str, agent_id: str, ttl_s: float = CLAIM_TTL_S) -> bool:
        """Try to claim an order. Returns True if you now own it (or already did)."""
        if not order_id or not agent_id:
            return False
        now = self._now()
        exp = now + float(ttl_s or CLAIM_TTL_S)
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("SELECT owner FROM claims WHERE order_id=?;", (order_id,))
            row = cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO claims(order_id, owner, expires_at) VALUES(?,?,?);",
                            (order_id, agent_id, exp))
                return True
            owner = row[0]
            if owner == agent_id:
                cur.execute("UPDATE claims SET expires_at=? WHERE order_id=?;", (exp, order_id))
                return True
            return False

    def release_order(self, order_id: str, agent_id: str) -> bool:
        if not order_id or not agent_id:
            return False
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("DELETE FROM claims WHERE order_id=? AND owner=?;", (order_id, agent_id))
            return cur.rowcount > 0

    def who_claims(self, order_id: str) -> Optional[str]:
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("SELECT owner FROM claims WHERE order_id=?;", (order_id,))
            row = cur.fetchone()
            return row[0] if row else None

    # ---------- counter reservations (e.g., plate staging) ----------
    def reserve_counter(self, counter_id: str, agent_id: str, ttl_s: float = RES_TTL_S) -> bool:
        """Reserve a plain counter (e.g., for plate staging). Non-blocking."""
        if not counter_id or not agent_id:
            return False
        now = self._now()
        exp = now + float(ttl_s or RES_TTL_S)
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("SELECT owner FROM reservations WHERE counter_id=?;", (counter_id,))
            row = cur.fetchone()
            if row is None:
                cur.execute("INSERT INTO reservations(counter_id, owner, expires_at) VALUES(?,?,?);",
                            (counter_id, agent_id, exp))
                return True
            owner = row[0]
            if owner == agent_id:
                cur.execute("UPDATE reservations SET expires_at=? WHERE counter_id=?;", (exp, counter_id))
                return True
            return False

    def release_counter(self, counter_id: str, agent_id: str) -> bool:
        with self._immediate() as cur:
            self._gc_expired(cur)
            cur.execute("DELETE FROM reservations WHERE counter_id=? AND owner=?;", (counter_id, agent_id))
            return cur.rowcount > 0
