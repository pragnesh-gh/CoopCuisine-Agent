# coordinator_sqlite.py
# Small, robust coordinator using sqlite3 (stdlib).
# - Order claims (with TTL/lease)
# - Resource locks (with TTL)
# - FIFO queues for stations/tools/dispensers (fairness)
# - Plate-spot publishing
# - Minimal agent registry (for stable indices)
#
# Works across processes. Safe to import from multiple agents.

import sqlite3, time, os, threading
from contextlib import contextmanager
from typing import Tuple, Optional, Dict, List

DEFAULT_DB = os.getenv("COORD_DB", "coord.db")

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS order_claims (
  order_id   TEXT PRIMARY KEY,
  agent_id   TEXT NOT NULL,
  expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_locks (
  -- lock over a logical resource name (e.g., "station:<id>", "tool:peel", "disp:<id>")
  name       TEXT PRIMARY KEY,
  agent_id   TEXT NOT NULL,
  expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS resource_queues (
  -- FIFO queue per resource
  name       TEXT NOT NULL,
  agent_id   TEXT NOT NULL,
  enq_ts     REAL NOT NULL,
  PRIMARY KEY (name, agent_id)
);

CREATE TABLE IF NOT EXISTS plate_spots (
  agent_id   TEXT PRIMARY KEY,
  counter_id TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
  agent_id   TEXT PRIMARY KEY,
  seen_at    REAL NOT NULL
);
"""

# namespacing helpers
def _rn_station(station_id: str) -> str: return f"station:{station_id}"
def _rn_tool(tool_name: str)   -> str: return f"tool:{tool_name}"
def _rn_disp(disp_id: str)     -> str: return f"disp:{disp_id}"

class Coordinator:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._conn() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            con = sqlite3.connect(self.db_path, timeout=3.0, isolation_level="IMMEDIATE")
            try:
                yield con
                con.commit()
            finally:
                con.close()

    # ── housekeeping ─────────────────────────────────────────────────────────
    def _purge_expired(self, con):
        now = time.time()
        con.execute("DELETE FROM order_claims   WHERE expires_at < ?", (now,))
        con.execute("DELETE FROM resource_locks WHERE expires_at < ?", (now,))

    # ── agent registry ───────────────────────────────────────────────────────
    def register_agent(self, agent_id: str) -> None:
        now = time.time()
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO agents(agent_id, seen_at) VALUES(?,?)",
                (agent_id, now),
            )

    def list_agents(self) -> List[str]:
        with self._conn() as con:
            rows = con.execute("SELECT agent_id FROM agents ORDER BY agent_id ASC").fetchall()
        return [r[0] for r in rows]

    # ── orders ───────────────────────────────────────────────────────────────
    def claim_order(self, order_id: str, agent_id: str, ttl_s: float = 5.0) -> bool:
        now = time.time(); exp = now + ttl_s
        with self._conn() as con:
            self._purge_expired(con)
            row = con.execute(
                "SELECT agent_id, expires_at FROM order_claims WHERE order_id=?",
                (order_id,),
            ).fetchone()
            if not row:
                con.execute(
                    "INSERT OR REPLACE INTO order_claims(order_id, agent_id, expires_at) VALUES(?,?,?)",
                    (order_id, agent_id, exp),
                )
                return True
            curr_agent, curr_exp = row
            if curr_exp < now or curr_agent == agent_id:
                con.execute(
                    "INSERT OR REPLACE INTO order_claims(order_id, agent_id, expires_at) VALUES(?,?,?)",
                    (order_id, agent_id, exp),
                )
                return True
            return False

    def renew_order(self, order_id: str, agent_id: str, ttl_s: float = 5.0) -> bool:
        now = time.time(); exp = now + ttl_s
        with self._conn() as con:
            self._purge_expired(con)
            row = con.execute("SELECT agent_id FROM order_claims WHERE order_id=?", (order_id,)).fetchone()
            if not row or row[0] != agent_id:
                return False
            con.execute("UPDATE order_claims SET expires_at=? WHERE order_id=?", (exp, order_id))
            return True

    def release_order(self, order_id: str, agent_id: str) -> None:
        with self._conn() as con:
            con.execute("DELETE FROM order_claims WHERE order_id=? AND agent_id=?", (order_id, agent_id))

    def who_claims(self, order_id: str) -> Optional[str]:
        now = time.time()
        with self._conn() as con:
            row = con.execute(
                "SELECT agent_id, expires_at FROM order_claims WHERE order_id=?",
                (order_id,),
            ).fetchone()
            if not row: return None
            agent, exp = row
            return agent if exp >= now else None

    # ── generic queue helpers over resource_queues + resource_locks ──────────
    def _queue_enqueue(self, con, res: str, agent_id: str):
        # idempotent: only one row per (res, agent)
        now = time.time()
        con.execute(
            "INSERT OR IGNORE INTO resource_queues(name, agent_id, enq_ts) VALUES(?,?,?)",
            (res, agent_id, now),
        )

    def _queue_head(self, con, res: str) -> Optional[str]:
        row = con.execute(
            "SELECT agent_id FROM resource_queues WHERE name=? ORDER BY enq_ts ASC LIMIT 1",
            (res,),
        ).fetchone()
        return row[0] if row else None

    def _queue_remove(self, con, res: str, agent_id: str):
        con.execute(
            "DELETE FROM resource_queues WHERE name=? AND agent_id=?",
            (res, agent_id),
        )

    def _queue_position(self, con, res: str, agent_id: str) -> int:
        row = con.execute(
            "SELECT enq_ts FROM resource_queues WHERE name=? AND agent_id=?",
            (res, agent_id),
        ).fetchone()
        if not row:
            return 0
        enq_ts = row[0]
        cnt = con.execute(
            "SELECT COUNT(*) FROM resource_queues WHERE name=? AND enq_ts < ?",
            (res, enq_ts),
        ).fetchone()[0]
        return int(cnt) + 1  # 1-based position

    def _lock_owner(self, con, res: str) -> Optional[str]:
        now = time.time()
        row = con.execute(
            "SELECT agent_id, expires_at FROM resource_locks WHERE name=?",
            (res,),
        ).fetchone()
        if not row: return None
        agent, exp = row
        return agent if exp >= now else None

    def _lock_grant(self, con, res: str, agent_id: str, ttl_s: float):
        exp = time.time() + ttl_s
        con.execute(
            "INSERT OR REPLACE INTO resource_locks(name, agent_id, expires_at) VALUES(?,?,?)",
            (res, agent_id, exp),
        )

    def _lock_release_if_owner(self, con, res: str, agent_id: str):
        con.execute(
            "DELETE FROM resource_locks WHERE name=? AND agent_id=?",
            (res, agent_id),
        )

    # Core request primitive (FIFO): returns ("granted",0) or ("queued",pos)
    def _request(self, res: str, agent_id: str, ttl_s: float = 3.0) -> Tuple[str, int]:
        with self._conn() as con:
            self._purge_expired(con)
            owner = self._lock_owner(con, res)
            head = self._queue_head(con, res)
            if owner is None and (head is None or head == agent_id):
                # grant now
                self._lock_grant(con, res, agent_id, ttl_s)
                # remove from queue if it was queued
                self._queue_remove(con, res, agent_id)
                return ("granted", 0)
            # not granted → enqueue if not already in queue and not owner
            if agent_id != owner:
                self._queue_enqueue(con, res, agent_id)
            pos = self._queue_position(con, res, agent_id)
            return ("queued", pos)

    # Generic release; only owner can release (idempotent)
    def _release(self, res: str, agent_id: str) -> None:
        with self._conn() as con:
            self._lock_release_if_owner(con, res, agent_id)

    # Public helpers to check ownership quickly
    def _who_owns(self, res: str) -> Optional[str]:
        with self._conn() as con:
            return self._lock_owner(con, res)

    # ── stations (oven/stove/fryer) ──────────────────────────────────────────
    def request_station(self, station_id: str, agent_id: str, ttl_s: float = 4.0) -> Tuple[str, int]:
        return self._request(_rn_station(station_id), agent_id, ttl_s)

    def release_station(self, station_id: str, agent_id: str) -> None:
        self._release(_rn_station(station_id), agent_id)

    def who_owns_station(self, station_id: str) -> Optional[str]:
        return self._who_owns(_rn_station(station_id))

    # ── tools (e.g., peel, basket) ───────────────────────────────────────────
    def request_tool(self, tool_name: str, agent_id: str, ttl_s: float = 3.0) -> Tuple[str, int]:
        return self._request(_rn_tool(tool_name), agent_id, ttl_s)

    def release_tool(self, tool_name: str, agent_id: str) -> None:
        self._release(_rn_tool(tool_name), agent_id)

    # ── dispensers (short lock around pickup) ────────────────────────────────
    def request_dispenser(self, disp_id: str, agent_id: str, ttl_s: float = 2.0) -> Tuple[str, int]:
        return self._request(_rn_disp(disp_id), agent_id, ttl_s)

    def release_dispenser(self, disp_id: str, agent_id: str) -> None:
        self._release(_rn_disp(disp_id), agent_id)

    # ── plate spots ──────────────────────────────────────────────────────────
    def set_plate_spot(self, agent_id: str, counter_id: str) -> None:
        now = time.time()
        with self._conn() as con:
            con.execute(
                "INSERT OR REPLACE INTO plate_spots(agent_id, counter_id, updated_at) VALUES(?,?,?)",
                (agent_id, counter_id, now),
            )

    def get_plate_spots(self) -> Dict[str, str]:
        with self._conn() as con:
            rows = con.execute("SELECT agent_id, counter_id FROM plate_spots").fetchall()
        return {a: c for a, c in rows}
    # ── generic resource API (compat layer expected by BT/agent) ─────────────
    def lock_resource(self, name: str, agent_id: str, ttl_s: float = 3.0) -> bool:
        """
        Try to acquire a short lease on an arbitrary logical resource name.
        Returns True iff granted immediately; otherwise queues and returns False.
        """
        status, _pos = self._request(name, agent_id, ttl_s=ttl_s)
        return status == "granted"

    def renew_resource(self, name: str, agent_id: str, ttl_s: float = 3.0) -> bool:
        """
        If agent_id currently owns the resource, extend its lease. Returns True on success.
        """
        with self._conn() as con:
            owner = self._lock_owner(con, name)
            if owner != agent_id:
                return False
            # re-grant with a fresh expiry
            self._lock_grant(con, name, agent_id, ttl_s)
            return True

    def release_resource(self, name: str, agent_id: str) -> None:
        """
        Release the resource if owned by agent_id (idempotent).
        """
        self._release(name, agent_id)

    # Optional: handy for logging/diagnostics
    def who_owns_resource(self, name: str) -> Optional[str]:
        return self._who_owns(name)

    def queue_position(self, name: str, agent_id: str) -> int:
        with self._conn() as con:
            return self._queue_position(con, name, agent_id)
