# cocu_base_agents/new_agent/multi_agent.py
# NOTE: file is a copy of your new_agent.py with minimal additions for multi-agent coordination.
# Goal of this file: run two+ agents safely together, avoid tool-stealing, and serialize recipes that need exclusive gear.

import os
import time
import json
import numpy as np
import py_trees
from py_trees import blackboard

from cooperative_cuisine.base_agent.agent_task import Task, TaskStatus
from cooperative_cuisine.base_agent.base_agent import BaseAgent, run_agent_from_args

# Import our handcrafted builder + shared Blackboard instance (BB)
from cocu_base_agents.new_agent.multiAgent_behavior_tree import BehaviorTreeBuilder, BB  # <- changed import
# Import your explicit stove + dispenser mapping (IDs for counters/equipment)
import id_map as id_map_module

# NEW: Global coordinator for locks/claims (shared small DB under the hood)
from cocu_base_agents.new_agent.coordinator import Coordinator

# ──────────────────────────────────────────────────────────────────────────────
# TaskStatus compatibility shim (different engine versions use different names)
# ──────────────────────────────────────────────────────────────────────────────
try:
    TS_DONE = TaskStatus.DONE
except AttributeError:
    TS_DONE = getattr(
        TaskStatus,
        "FINISHED",
        getattr(TaskStatus, "SUCCESS", getattr(TaskStatus, "COMPLETED", None)),
    )
    if TS_DONE is None:
        TS_DONE = TaskStatus.SCHEDULED  # harmless fallback so code doesn’t break


# ──────────────────────────────────────────────────────────────────────────────
# Multi-agent additions
# ──────────────────────────────────────────────────────────────────────────────
# Agent identity:
# We allow setting an AGENT_ID via env var (e.g., AGENT_ID=B for the second agent).
# Commented example kept for clarity:
# AGENT_ID = os.getenv("AGENT_ID", "A").strip() or "A"
# AGENT_OWNER = f"{AGENT_ID}-{os.getpid()}"

# Single Coordinator instance (handles locks, claims, and heartbeats)
COORD = Coordinator()

# Backoff sequence when a lock is busy: we wait a bit longer each time (capped)
BACKOFF_STEPS_S = (0.2, 0.4, 0.8)

# Session flags track "long-lived" gear ownership (e.g., keep pot while soup cooks)
SESSION = {
    "pot_owned_for_soup": False,
    "stove_pot_id": None,
    # NEW: pizza oven/peel session (prevents peel steal)
    "oven_owned_for_pizza": False,
    "oven_id": None,
    "fryer_owned_for_fry": False,
    "fryer_id": None,
    "pan_owned_for_burger": False,
    "pan_id": None,
}
# Short session for cutting boards (held across chop)
SESSION["board_owned"] = False
SESSION["board_id"] = None
# Recipe token serializes “one cook per meal-type” (e.g., only one pizza cook at a time)
SESSION["recipe_token_owned"] = False
SESSION["recipe_token_id"] = None


def _recipe_token_id(meal_norm: str) -> str:
    """Stable key used to serialize cooking per meal type (e.g., 'recipe::pizza')."""
    return f"recipe::{meal_norm or ''}"


def _norm_name(s):
    """Lowercase, remove underscores/spaces → normalize names for comparisons."""
    if s is None:
        return ""
    return str(s).lower().replace("_", "").replace(" ", "")


def _orders_signature(state):
    """Build a quick signature of the order list to detect changes."""
    orders = state.get("orders") or state.get("order_state") or []
    sig = []
    for i, o in enumerate(orders):
        oid = o.get("id") or o.get("order_id") or o.get("uid")
        meal = (o.get("meal") or o.get("name") or "").lower()
        sig.append(oid or f"{meal}#{i}")
    return tuple(sig)


def _order_time_left(order: dict) -> float:
    """Return remaining order time as float (fallback to 0.0)."""
    try:
        return float(order.get("time_remaining", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _extract_order_id(o: dict) -> str | None:
    """Pull a stable order id from various possible keys."""
    # OrderManager.order_state provides {"id": order.uuid, ...}
    return o.get("id") or o.get("uuid") or o.get("order_id") or o.get("uid")


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────
class BTAgent(BaseAgent):
    """Multi-agent safe behavior-tree-driven agent."""

    def __init__(self, *args, **kwargs):
        """Initialize local state and per-instance identity."""
        super().__init__(*args, **kwargs)
        self._initialized = False
        self.prev_orders_sig = None
        self._last_put_was_serve = False  # toggled when we PUT to the serving window
        self.serving_id = None

        # Backoff tracking per resource id (to avoid spamming locks)
        self._backoff_idx = {}    # res_id -> backoff index
        self._backoff_until = {}  # res_id -> timestamp we can retry after

        # We claim exactly one order at a time
        self._claimed_order_id = None

        # NEW: per-instance identity
        env_id = os.getenv("AGENT_ID", "").strip()
        # Simple in-process auto-id: first instance → "A", second → "B"
        if not hasattr(BTAgent, "_instance_seq"):
            BTAgent._instance_seq = 0
        seq = BTAgent._instance_seq
        BTAgent._instance_seq += 1
        auto_id = "A" if seq == 0 else "B"
        self.AGENT_ID = env_id or auto_id

        # Owner must be unique per object instance (process-safe)
        self.AGENT_OWNER = f"{self.AGENT_ID}-{os.getpid()}-{id(self)}"

    def _pick_leftmost_order_meal(self, state) -> str:
        """Convenience: meal name of the left-most order (normalized)."""
        orders = state.get("orders") or state.get("order_state") or []
        if not isinstance(orders, (list, tuple)) or not orders:
            return ""
        m = (orders[0].get("meal") or orders[0].get("name") or "").lower()
        return m.replace(" ", "")

    def _reset_per_order_state(self):
        """Clear per-order BB flags so the next order starts fresh."""
        for k in ("plate_spot_id", "added_to_plate", "allow_pick_plate_once"):
            if hasattr(BB, k):
                try:
                    delattr(BB, k)
                except Exception:
                    pass

    def initialise(self, state):
        """One-time boot: set identity, map counters, and prep metrics."""
        super().initialise(state)
        if self._initialized:
            return
        self._initialized = True

        # Stable identity (prefer engine-owned id if available)
        self.AGENT_ID = (os.getenv("AGENT_ID", "A").strip() or "A")
        unique_token = (
            getattr(self, "own_player_id", None)
            or state.get("own_player_id")
            or f"local{abs(id(self)) % 100000}"
        )
        self.AGENT_OWNER = f"{self.AGENT_ID}-{unique_token}"

        # Find serving window + cache positions of all counters
        counters = state.get("counters", [])
        service = next((c for c in counters if (c.get("type") or "").lower() == "servingwindow"), None)
        if service is None:
            raise RuntimeError("No serving window found")
        self.serving_id = service["id"]
        self.counter_positions = {c["id"]: np.array(c["pos"]) for c in counters}

        # Keep a snapshot of current orders to detect changes later
        self.prev_orders_sig = _orders_signature(state)

        # Do not build a BT yet; only build after we successfully claim an order
        BB.active_meal = ""
        BB.last_meal = None

        # Track the exclusive gear we own during this order (e.g., oven/fryer/pot/pan)
        self._session_gear_id = None
        BB.gear_wait_id = None

        # Clear any stale schedule/waits on the Blackboard
        if hasattr(BB, "next_task"):
            del BB.next_task
        BB.meta_waiting = None
        BB.dumped_utensil_schema = False

        # Reset long-lived session tokens
        SESSION["pot_owned_for_soup"] = False
        SESSION["stove_pot_id"] = None
        SESSION["recipe_token_owned"] = False
        SESSION["recipe_token_id"] = None

    def _build_tree(self, state):
        """Construct the behavior tree after we claim the order and know the meal."""
        # Expose our identity to the BT (useful for debugging or role-logic)
        BB.agent_id = self.AGENT_ID
        BB.agent_owner = self.AGENT_OWNER
        self.tree = BehaviorTreeBuilder.build(
            state=state,
            counter_positions=self.counter_positions,
            id_map_module=id_map_module,
        )
        self.tree.setup(timeout=10)

        # Episode start markers for simple metrics
        BB.episode_start_tick = getattr(BB, "tick", 0)
        BB.episode_retries_start = getattr(BB, "retries", 0)
        BB.episode_idle_start = getattr(BB, "idle", 0)
        print(f"[METRICS] Episode started for meal='{BB.active_meal}' at tick={BB.episode_start_tick}")
        print(f"[BTAgent:{self.AGENT_ID}] ✅ Built BT for meal='{BB.active_meal}'")

    # ------------------------- Lock helpers -------------------------
    def _now(self):
        """Return current wall-clock time."""
        return time.time()

    def _contested_set(self, state):
        """
        Return all resource ids that should be locked before PUT/INTERACT.
        Why: single-slot items (dispensers, boards, serving, stove tools) need locking.
        """
        ids = set()
        # Prefer explicit static mapping
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}

        # Pull known keys if present in ID_MAP
        for k in (
            "tomato_dispenser",
            "lettuce_dispenser",
            "bun_dispenser",
            "meat_dispenser",
            "onion_dispenser",
            "potato_dispenser",
            "fish_dispenser",
            "dough_dispenser",
            "cheese_dispenser",
            "sausage_dispenser",
            "plate_dispenser",
            "serving_window",
            "stove_pan",
            "stove_pot",
            "deep_fryer",
            "oven",
            "peel",
            "basket",
        ):
            v = im.get(k)
            if isinstance(v, (list, tuple, set)):
                ids.update(v)
            elif v:
                ids.add(v)

        # Cutting boards can be a list
        for b in im.get("cutting_boards", []):
            ids.add(b)

        # Fallback: discover by type names in the state if ID_MAP is incomplete
        by_type = {}
        for c in state.get("counters", []):
            by_type.setdefault((c.get("type") or "").lower(), []).append(c.get("id"))

        for t in (
            "tomatodispenser",
            "lettucedispenser",
            "bundispenser",
            "meatdispenser",
            "oniondispenser",
            "potatodispenser",
            "fishdispenser",
            "doughdispenser",
            "cheesedispenser",
            "sausagedispenser",
            "platedispenser",
            "servingwindow",
            "stove",
            "deepfryer",
            "oven",
            "cuttingboard",
        ):
            for cid in by_type.get(t, []):
                ids.add(cid)

        return ids

    def _is_contested(self, state, target_id: str) -> bool:
        """Quick check if a target needs lock protection."""
        return target_id in self._contested_set(state)

    def _should_session_lock_pot(self) -> bool:
        """We keep the pot locked across the whole soup recipe (prevents steals)."""
        meal = _norm_name(getattr(BB, "active_meal", ""))
        return meal in {"tomatosoup", "onionsoup"}

    def _handle_backoff(self, res_id: str) -> bool:
        """
        True → we should skip this tick because we are still in backoff for this resource.
        Keeps the system from hammering a busy lock every tick.
        """
        until = self._backoff_until.get(res_id, 0.0)
        return self._now() < until

    def _bump_backoff(self, res_id: str):
        """Increase backoff step for this resource (capped) and set next retry time."""
        idx = self._backoff_idx.get(res_id, 0)
        idx = min(idx + 1, len(BACKOFF_STEPS_S) - 1)
        self._backoff_idx[res_id] = idx
        self._backoff_until[res_id] = self._now() + BACKOFF_STEPS_S[idx]

    def _reset_backoff(self, res_id: str):
        """Clear backoff tracking for this resource after a successful lock."""
        self._backoff_idx.pop(res_id, None)
        self._backoff_until.pop(res_id, None)

    # ------------------------- META_COOK polling (unchanged core) -------------------------
    # We poll equipment to see if a timed transform finished (e.g., frying done).
    def _lookup_counter(self, state: dict, counter_id: str) -> dict | None:
        """Find a counter dict by id in the state."""
        cs = state.get("counters")
        if isinstance(cs, dict):
            return cs.get(counter_id)
        if isinstance(cs, list):
            for c in cs:
                if c.get("id") == counter_id:
                    return c
        return None

    def _extract_item_name(self, obj) -> str | None:
        """Try to read the item/equipment name from an object (robust across shapes)."""
        if obj is None:
            return None
        if isinstance(obj, str):
            return None if obj.strip().lower() == "none" else obj
        if isinstance(obj, dict):
            n = obj.get("name") or obj.get("type")
            if n:
                return n
            for k in ("item", "result", "item_info"):
                v = obj.get(k)
                if isinstance(v, dict):
                    n = v.get("name") or v.get("type")
                    if n:
                        return n
        return None

    def _top_name_from_content_list(self, clist) -> str | None:
        """If content_list contains one dict item, return its name for quick checks."""
        if not isinstance(clist, (list, tuple)) or not clist:
            return None
        if isinstance(clist[0], dict):
            return self._extract_item_name(clist[0])
        return None

    def _describe_station_equipment(self, state, station_id: str) -> str:
        """Human-readable snapshot of a station (used only for debug prints)."""
        ctr = self._lookup_counter(state, station_id)
        if not ctr:
            return f"station={station_id} not found"
        occ = ctr.get("occupied_by")
        if not occ:
            return f"station={station_id} (empty)"
        parts = []
        ename = self._extract_item_name(occ)
        if ename:
            parts.append(f"equip={ename}")
        r = occ.get("content_ready")
        rname = self._extract_item_name(r) if isinstance(r, dict) else None
        if rname:
            parts.append(f"ready={rname}")
        cl = occ.get("content_list") or []
        tname = self._top_name_from_content_list(cl)
        parts.append(f"content_top={tname!r}")
        at = occ.get("active_transition")
        if isinstance(at, dict):
            at_name = self._extract_item_name(at.get("result"))
            parts.append(f"active={{sec:{at.get('seconds')}, result:{at_name!r}}}")
        return f"station={station_id} " + " ".join(parts)

    def _handle_meta_cook_poll(self, state) -> TaskStatus:
        """
        If BB.meta_waiting is set, we check whether the expected cooked result is ready.
        Return:
          - IN_PROGRESS while still cooking
          - TS_DONE when ready
          - FAILED if we detect burned/waste
        """
        if not getattr(BB, "meta_waiting", None):
            return TS_DONE

        expected = _norm_name(BB.meta_waiting.get("expected"))
        station_id = BB.meta_waiting.get("station_id")
        if not station_id:
            BB.meta_waiting = None
            return TS_DONE

        ctr = self._lookup_counter(state, station_id)
        if not ctr:
            return TaskStatus.IN_PROGRESS

        occ = ctr.get("occupied_by")
        equip = None
        if isinstance(occ, dict) and occ.get("category") in ("ItemCookingEquipment", "Item"):
            equip = occ
        elif isinstance(occ, list):
            for o in occ:
                if isinstance(o, dict) and o.get("category") in ("ItemCookingEquipment", "Item"):
                    equip = o
                    break
        if equip is None:
            return TaskStatus.IN_PROGRESS

        cl = equip.get("content_list") or []
        cr = equip.get("content_ready")

        def _n(x):
            if isinstance(x, dict):
                return (x.get("type") or x.get("name") or "").lower()
            if isinstance(x, list):
                return [(_i.get("type") or _i.get("name")) for _i in x]
            return None

        # Burn guard — if content top is waste/burnt, abort
        if isinstance(cl, list) and cl and isinstance(cl[0], dict):
            top = _norm_name(self._extract_item_name(cl[0]))
            if top in {"waste", "burnt"}:
                BB.meta_waiting = None
                return TaskStatus.FAILED

        # Instant transitions sometimes put result in content_ready
        rname = _norm_name(self._extract_item_name(cr)) if isinstance(cr, dict) else None
        if rname and rname == expected:
            BB.meta_waiting = None
            return TS_DONE

        # Timed transitions often surface result as the only item in content_list
        if isinstance(cl, list) and len(cl) == 1 and isinstance(cl[0], dict):
            t = _norm_name(self._extract_item_name(cl[0]))
            if t == expected:
                BB.meta_waiting = None
                return TS_DONE

        return TaskStatus.IN_PROGRESS

    # ------------------------- Session locking helpers -------------------------
    SESSION_TTL = 20.0  # seconds; safe TTL while cooking

    def _set_session_flags(self, kind: str, rid: str, on: bool):
        """Toggle the right session flag for the given resource kind."""
        if kind == "stove_pot_session":
            SESSION["pot_owned_for_soup"] = on
            SESSION["stove_pot_id"] = rid if on else None
        elif kind == "oven_session":
            SESSION["oven_owned_for_pizza"] = on
            SESSION["oven_id"] = rid if on else None
        elif kind == "fryer_session":
            SESSION["fryer_owned_for_fry"] = on
            SESSION["fryer_id"] = rid if on else None
        elif kind == "stove_pan_session":
            SESSION["pan_owned_for_burger"] = on
            SESSION["pan_id"] = rid if on else None

    def _try_reserve_gear_for_meal(self, state, meal_norm: str) -> bool:
        """
        Try to lock all exclusive gear needed by this meal up-front.
        If any single lock fails, roll back what we took and return False.
        """
        needs = self._gear_ids_for_meal(state, meal_norm)
        if not needs:
            return True  # nothing exclusive needed

        acquired = []
        for kind, rid in needs.items():
            ok = COORD.lock_resource(rid, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind=kind)
            if ok:
                acquired.append((kind, rid))
            else:
                # Roll back previously acquired locks
                for k2, r2 in acquired:
                    try:
                        COORD.unlock_resource(r2, self.AGENT_OWNER)
                    except Exception:
                        pass
                return False

        # Success → set flags
        for kind, rid in acquired:
            self._set_session_flags(kind, rid, True)
        return True

    def _refresh_session_locks(self, state):
        """
        Extend (refresh) TTL for any held sessions once per tick.
        Prevents lock expiry during long cooking steps.
        """
        if SESSION.get("pot_owned_for_soup") and SESSION.get("stove_pot_id"):
            COORD.lock_resource(
                SESSION["stove_pot_id"],
                self.AGENT_OWNER,
                ttl_s=self.SESSION_TTL,
                kind="stove_pot_session",
            )
        if SESSION.get("oven_owned_for_pizza") and SESSION.get("oven_id"):
            COORD.lock_resource(
                SESSION["oven_id"],
                self.AGENT_OWNER,
                ttl_s=self.SESSION_TTL,
                kind="oven_session",
            )
        if SESSION.get("fryer_owned_for_fry") and SESSION.get("fryer_id"):
            COORD.lock_resource(
                SESSION["fryer_id"],
                self.AGENT_OWNER,
                ttl_s=self.SESSION_TTL,
                kind="fryer_session",
            )
        if SESSION.get("pan_owned_for_burger") and SESSION.get("pan_id"):
            COORD.lock_resource(
                SESSION["pan_id"],
                self.AGENT_OWNER,
                ttl_s=self.SESSION_TTL,
                kind="stove_pan_session",
            )
        if SESSION.get("recipe_token_owned") and SESSION.get("recipe_token_id"):
            COORD.lock_resource(
                SESSION["recipe_token_id"],
                self.AGENT_OWNER,
                ttl_s=self.SESSION_TTL,
                kind="recipe_serial",
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Main scheduler
    # ──────────────────────────────────────────────────────────────────────────
    async def manage_tasks(self, state):
        """
        Runs once per tick:
          1) Keep locks alive (heartbeat + refresh sessions)
          2) Handle META_COOK waits
          3) Claim an order if free (with recipe-token + gear pre-reserve)
          4) Tick the behavior tree
          5) Schedule exactly one engine action via BB.next_task
        """
        # 0) Heartbeat + extend sessions so they don't expire mid-cook
        COORD.heartbeat(self.AGENT_OWNER)
        self._refresh_session_locks(state)

        # Simple per-episode stats on the Blackboard
        BB.tick = getattr(BB, "tick", 0) + 1
        BB.actions = getattr(BB, "actions", 0)
        BB.retries = getattr(BB, "retries", 0)
        BB.idle = getattr(BB, "idle", 0)
        BB.last_state = state

        # Helper to estimate "cost" by number of ingredients (used in next-pick logic)
        def _ingredient_count_for(meal: str, st: dict) -> int:
            m = _norm_name(meal)
            if m == "tomatosoup":
                return 3
            if m == "onionsoup":
                return 3
            if m == "salad":
                return 2
            if m == "burger":
                return 4
            if m == "chips":
                return 1
            if m == "friedfish":
                return 1
            if m == "fishandchips":
                return 2
            if m == "pizza":
                # If sausage exists, pizza has 4 ingredients; else 3
                sausage_present = any(
                    ((c.get("type") or c.get("name") or "").lower() in {"sausage_dispenser", "sausagedispenser", "sausage"})
                    for c in st.get("counters", [])
                )
                return 4 if sausage_present else 3
            return 3

        def _pick_after_serve_with_50pct_rule(st: dict) -> str:
            """
            After serving, pick the next meal:
              - Start with orders that have the max remaining time.
              - If any orders have ≥50% of that time and fewer ingredients than that default,
                prefer those (earlier win with less work).
            """
            orders = st.get("orders") or st.get("order_state") or []
            rows = []
            for idx, o in enumerate(orders):
                meal = _norm_name(o.get("meal") or o.get("name"))
                if not meal:
                    continue
                tleft = _order_time_left(o)
                icost = _ingredient_count_for(meal, st)
                rows.append((tleft, icost, idx, meal, _extract_order_id(o)))
            if not rows:
                return ""
            T_max = max(r[0] for r in rows)
            top = [r for r in rows if r[0] == T_max]
            default = min(top, key=lambda r: r[2])  # left-most among max-time
            threshold = 0.5 * T_max
            eligible = [r for r in rows if r[0] >= threshold and r[1] < default[1]]
            if eligible:
                eligible.sort(key=lambda r: (r[1], -r[0], r[2]))
                return eligible[0][3]
            return default[3]

        # 1) META_COOK has priority → poll until done/fail
        if getattr(BB, "meta_waiting", None):
            status = self._handle_meta_cook_poll(state)
            if status == TaskStatus.IN_PROGRESS:
                return
            # else fall-through to continue

        # Ensure we have a previous orders signature for change detection
        if not hasattr(self, "prev_orders_sig"):
            self.prev_orders_sig = _orders_signature(state)

        # Cache serving window id locally
        serving_id = self.serving_id

        # A) If we served last tick → release sessions and reset for next order
        if self._last_put_was_serve:
            # Episode summary logs (very lightweight metrics)
            ticks = getattr(BB, "tick", 0) - getattr(BB, "episode_start_tick", 0)
            retries = getattr(BB, "retries", 0) - getattr(BB, "episode_retries_start", 0)
            idle = getattr(BB, "idle", 0) - getattr(BB, "episode_idle_start", 0)
            print(f"[METRICS] MeanTicks={ticks} Retries={retries} IdleTicks={idle}")

            self._last_put_was_serve = False

            # Release plate spot if held
            if getattr(BB, "plate_spot_id", None):
                COORD.release_counter(BB.plate_spot_id, self.AGENT_OWNER)

            # Release soup pot session (ownership ends at serve)
            if SESSION["pot_owned_for_soup"] and SESSION["stove_pot_id"]:
                COORD.unlock_resource(SESSION["stove_pot_id"], self.AGENT_OWNER)
            SESSION["pot_owned_for_soup"] = False
            SESSION["stove_pot_id"] = None

            # Release pizza oven/peel session
            if SESSION["oven_owned_for_pizza"] and SESSION["oven_id"]:
                COORD.unlock_resource(SESSION["oven_id"], self.AGENT_OWNER)
            SESSION["oven_owned_for_pizza"] = False
            SESSION["oven_id"] = None

            # Release fryer session
            if SESSION["fryer_owned_for_fry"] and SESSION["fryer_id"]:
                COORD.unlock_resource(SESSION["fryer_id"], self.AGENT_OWNER)
            SESSION["fryer_owned_for_fry"] = False
            SESSION["fryer_id"] = None

            # Release pan session
            if SESSION.get("pan_owned_for_burger") and SESSION.get("pan_id"):
                COORD.unlock_resource(SESSION["pan_id"], self.AGENT_OWNER)
            SESSION["pan_owned_for_burger"] = False
            SESSION["pan_id"] = None

            # Release board session (short-lived but safe to clean)
            if SESSION.get("board_owned") and SESSION.get("board_id"):
                COORD.unlock_resource(SESSION["board_id"], self.AGENT_OWNER)
            SESSION["board_owned"] = False
            SESSION["board_id"] = None

            # Clear any “waiting on recipe/gear” hints
            if hasattr(BB, "recipe_wait"):
                try:
                    delattr(BB, "recipe_wait")
                except Exception:
                    pass

            # Release recipe token (another agent can now cook that meal type)
            if SESSION.get("recipe_token_owned") and SESSION.get("recipe_token_id"):
                COORD.unlock_resource(SESSION["recipe_token_id"], self.AGENT_OWNER)
            SESSION["recipe_token_owned"] = False
            SESSION["recipe_token_id"] = None

            # Clear per-order Blackboard flags
            self._reset_per_order_state()

            # After serve we do not preselect/build; claim loop handles the next order
            BB.active_meal = ""

        # B) If orders changed, update signature (we let the claim loop react)
        curr_sig = _orders_signature(state)
        if curr_sig != self.prev_orders_sig:
            self.prev_orders_sig = curr_sig

        # 1) If we already scheduled a task this tick, do nothing
        if self.current_task is not None:
            return

        # ── CLAIM LOOP (we run this whenever we have no claimed order) ─────────
        if self._claimed_order_id is None:
            orders = state.get("orders") or state.get("order_state") or []

            # If we explicitly want a gear (e.g., were waiting for oven), try to grab it first
            gear_wait_id = getattr(BB, "gear_wait_id", None)
            if gear_wait_id:
                if COORD.lock_resource(gear_wait_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="session"):
                    # Find an order that uses this gear, then try recipe token + claim
                    for o in orders:
                        meal = _norm_name(o.get("meal") or o.get("name"))
                        gear_map = self._gear_ids_for_meal(state, meal)
                        if gear_map and gear_wait_id in gear_map.values():
                            # 1) Take recipe token first (serialize by meal)
                            token_id = _recipe_token_id(meal)
                            if not COORD.lock_resource(
                                token_id,
                                self.AGENT_OWNER,
                                ttl_s=self.SESSION_TTL,
                                kind="recipe_serial",
                            ):
                                BB.recipe_wait = meal  # someone else is cooking this meal-type
                                break
                            SESSION["recipe_token_owned"] = True
                            SESSION["recipe_token_id"] = token_id

                            # 2) Reserve required gear for this meal
                            if not self._try_reserve_gear_for_meal(state, meal):
                                COORD.unlock_resource(token_id, self.AGENT_OWNER)
                                SESSION["recipe_token_owned"] = False
                                SESSION["recipe_token_id"] = None
                                continue

                            # 3) Claim the specific order id
                            oid = _extract_order_id(o)
                            if not oid or not COORD.claim_order(oid, self.AGENT_OWNER):
                                # Couldn’t claim → release gear + recipe token and keep waiting
                                for _, rid in (gear_map or {}).items():
                                    try:
                                        COORD.unlock_resource(rid, self.AGENT_OWNER)
                                    except Exception:
                                        pass
                                COORD.unlock_resource(token_id, self.AGENT_OWNER)
                                SESSION["recipe_token_owned"] = False
                                SESSION["recipe_token_id"] = None
                                break

                            # Success → lock in meal, build the BT, and clear wait flags
                            self._claimed_order_id = oid
                            BB.active_meal = meal or ""
                            self._reset_per_order_state()
                            self._build_tree(state)
                            for _attr in ("gear_wait_id", "recipe_wait"):
                                if hasattr(BB, _attr):
                                    try:
                                        delattr(BB, _attr)
                                    except Exception:
                                        pass
                            print(f"[BTAgent:{self.AGENT_ID}] ✅ claimed order {self._claimed_order_id} (gear wait resolved)")
                            break
                # If that lock failed, we fall through to normal claim pass and then park

            # Pass 1: prefer orders for which we can immediately pre-reserve exclusive gear
            def _try_claim(o):
                oid = _extract_order_id(o)
                if not oid:
                    return False
                meal = _norm_name(o.get("meal") or o.get("name"))

                # 1) Take the recipe token (serialize by meal) — we keep until serve
                token_id = _recipe_token_id(meal)
                if not COORD.lock_resource(token_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="recipe_serial"):
                    # Someone else is already cooking this meal-type → wait
                    BB.recipe_wait = meal
                    return False
                SESSION["recipe_token_owned"] = True
                SESSION["recipe_token_id"] = token_id

                # 2) Pre-reserve required exclusive gear
                if not self._try_reserve_gear_for_meal(state, meal):
                    # Couldn’t grab the gear → release the recipe token and skip
                    COORD.unlock_resource(token_id, self.AGENT_OWNER)
                    SESSION["recipe_token_owned"] = False
                    SESSION["recipe_token_id"] = None
                    return False

                # 3) Claim specific order id
                if not COORD.claim_order(oid, self.AGENT_OWNER):
                    # Couldn’t claim → release gear + token
                    for kind, rid in self._gear_ids_for_meal(state, meal).items():
                        try:
                            COORD.unlock_resource(rid, self.AGENT_OWNER)
                        except Exception:
                            pass
                        self._set_session_flags(kind, rid, False)
                    COORD.unlock_resource(token_id, self.AGENT_OWNER)
                    SESSION["recipe_token_owned"] = False
                    SESSION["recipe_token_id"] = None
                    return False

                # Success → set meal, build BT, and clear waiting hints
                self._claimed_order_id = oid
                BB.active_meal = meal or ""
                self._reset_per_order_state()
                self._build_tree(state)
                for _attr in ("gear_wait_id", "recipe_wait"):
                    if hasattr(BB, _attr):
                        try:
                            delattr(BB, _attr)
                        except Exception:
                            pass
                print(f"[BTAgent:{self.AGENT_ID}] ✅ claimed order {self._claimed_order_id} (recipe token + gear reserved)")
                return True

            claimed = False
            for o in orders:
                if _try_claim(o):
                    claimed = True
                    break

            # Pass 2: nothing was claimable-with-gear → latch the first needed gear and wait
            if not claimed:
                for o in orders:
                    meal = _norm_name(o.get("meal") or o.get("name"))
                    gear_map = self._gear_ids_for_meal(state, meal)
                    if gear_map:
                        BB.gear_wait_id = next(iter(gear_map.values()))
                        break
                # Park near plate or home while waiting (only if we don’t have a plate yet)
                self._park_at_plate(state)
                return

        # If we’re waiting for a recipe token and still don’t own one → park
        if getattr(BB, "recipe_wait", None) and self._claimed_order_id is None and not SESSION.get("recipe_token_owned"):
            self._park_at_plate(state)
            return

        # If we’re waiting for gear and still haven’t claimed → park
        if getattr(BB, "gear_wait_id", None) and self._claimed_order_id is None:
            self._park_at_plate(state)
            return

        # If no BT or no meal, skip (claim loop will try again next tick)
        current_meal = _norm_name(getattr(BB, "active_meal", ""))
        if not getattr(self, "tree", None) or not current_meal:
            return

        # Check which exclusive gear this meal needs
        need = self._gear_ids_for_meal(state, current_meal)

        # Soft gate on recipe token:
        # - Waiters (no token) park if gear is needed.
        # - Cooks (have claimed order) do not park; they anchor near gear and retry.
        if not SESSION.get("recipe_token_owned"):
            if self._claimed_order_id is None:
                if need:
                    self._park_at_plate(state)
                    return
            else:
                if need:
                    anchor_id = next(iter(need.values()))
                    self._anchor_near(state, anchor_id)
                    return
            # If no exclusive gear → proceed even without token

        if need:
            needs_oven = "oven_session" in need
            needs_pot = "stove_pot_session" in need
            needs_fryer = "fryer_session" in need
            needs_pan = "stove_pan_session" in need
            lacking = (
                (needs_oven and not SESSION.get("oven_owned_for_pizza"))
                or (needs_pot and not SESSION.get("pot_owned_for_soup"))
                or (needs_fryer and not SESSION.get("fryer_owned_for_fry"))
                or (needs_pan and not SESSION.get("pan_owned_for_burger"))
            )
            if lacking:
                # If we are NOT the cook (no recipe token), we wait politely at the plate
                if not SESSION.get("recipe_token_owned"):
                    if not getattr(BB, "gear_wait_id", None):
                        BB.gear_wait_id = next(iter(need.values()))
                    self._park_at_plate(state)
                    return
                # We ARE the cook → don't park; anchor near gear and retry next tick
                anchor_id = next(iter(need.values()))
                self._anchor_near(state, anchor_id)
                return

        # 3) Tick the BT (which sets BB.next_task with exactly one engine action)
        self.tree.tick()

        # 4) Schedule emitted task (with smart lock gating below)
        next_task = getattr(BB, "next_task", None)
        if not next_task:
            BB.idle += 1  # no action this tick
            return

        task_type, task_arg = next_task
        target_id = task_arg if isinstance(task_arg, str) else None

        # ─────────────────────────────────────────────────────────────────
        # Debounce: avoid immediate second PUT to the same plate spot / fryer home.
        # This prevents PUT→PUT toggles like "place plate" then "place again".
        # ─────────────────────────────────────────────────────────────────
        if task_type == "PUT":
            target_id = task_arg

            # Plate spot debounce
            if getattr(BB, "plate_spot_id", None) and target_id == BB.plate_spot_id:
                last_tick = getattr(BB, "last_plate_put_tick", -999)
                if (BB.tick - last_tick) <= 1 and not getattr(BB, "allow_pick_plate_once", False):
                    print(f"[BTAgent:{self.AGENT_ID}] ⏭️ debounced: skipping immediate second PUT on plate.")
                    del BB.next_task
                    return
                BB.last_plate_put_tick = BB.tick
                if getattr(BB, "allow_pick_plate_once", False):
                    BB.allow_pick_plate_once = False

            # Fryer basket debounce
            basket_home = getattr(BB, "tool_home", {}).get("basket")
            if basket_home and target_id == basket_home:
                last_tick = getattr(BB, "last_fryer_put_tick", -999)
                if (BB.tick - last_tick) <= 1:
                    print(f"[BTAgent:{self.AGENT_ID}] ⏭️ debounced: skipping immediate second PUT on fryer.")
                    del BB.next_task
                    return
                BB.last_fryer_put_tick = BB.tick

        # META_COOK watcher task → record expectation and return (no engine call here)
        if task_type == "META_COOK":
            required, station_id = task_arg
            BB.meta_waiting = {"expected": _norm_name(required), "station_id": station_id}
            _ = self._handle_meta_cook_poll(state)
            del BB.next_task
            return

        # Mark that we served (we’ll release locks/claims just after scheduling)
        if task_type == "PUT" and serving_id is not None and task_arg == serving_id:
            self._last_put_was_serve = True

        # If we own a board session but target moves elsewhere, release board
        if SESSION.get("board_owned") and SESSION.get("board_id") and target_id != SESSION["board_id"]:
            COORD.unlock_resource(SESSION["board_id"], self.AGENT_OWNER)
            SESSION["board_owned"] = False
            SESSION["board_id"] = None

        # ─────────────────────────────────────────────────────────────────
        # Lock gating strategy:
        # - Boards/Dispensers/Serving → per-action lock (short-lived).
        # - Pot (soups), Oven/Peel (pizza), Fryer/Basket (chips/friedfish/fish&chips),
        #   Pan (burger) → session locks to avoid tool steals mid-recipe.
        # ─────────────────────────────────────────────────────────────────
        held_per_action_lock = False  # track if we need to unlock after scheduling

        if target_id and self._is_contested(state, target_id):
            # 1) Soup pot: hold through the whole soup (session lock)
            if self._should_session_lock_pot() and self._is_stove_pot(state, target_id):
                if not SESSION["pot_owned_for_soup"]:
                    if self._handle_backoff(target_id):
                        # Ask BT to retry exactly this step next tick
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set())
                            inv.add(bad)
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                    if COORD.lock_resource(target_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="stove_pot_session"):
                        SESSION["pot_owned_for_soup"] = True
                        SESSION["stove_pot_id"] = target_id
                        self._reset_backoff(target_id)
                    else:
                        self._bump_backoff(target_id)
                        BB.retries += 1
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set())
                            inv.add(bad)
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            # 2) Pizza oven/peel: session lock to prevent peel steals
            elif _norm_name(getattr(BB, "active_meal", "")) == "pizza" and self._is_oven_or_peel(state, target_id):
                if not SESSION["oven_owned_for_pizza"]:
                    if self._handle_backoff(target_id):
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set())
                            inv.add(bad)
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                    oven_id = self._get_oven_id(state) or target_id
                    if COORD.lock_resource(oven_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="oven_session"):
                        SESSION["oven_owned_for_pizza"] = True
                        SESSION["oven_id"] = oven_id
                        self._reset_backoff(target_id)
                    else:
                        self._bump_backoff(target_id)
                        BB.retries += 1
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set())
                            inv.add(bad)
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            # 3) Fryer/basket: session lock across the frying step
            elif _norm_name(getattr(BB, "active_meal", "")) in {"chips", "friedfish", "fishandchips"} and self._is_fryer_or_basket(state, target_id):
                if not SESSION["fryer_owned_for_fry"]:
                    if self._handle_backoff(target_id):
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set())
                            inv.add(bad)
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                    fryer_id = self._get_fryer_id(state) or target_id
                    if COORD.lock_resource(fryer_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="fryer_session"):
                        SESSION["fryer_owned_for_fry"] = True
                        SESSION["fryer_id"] = fryer_id
                        self._reset_backoff(target_id)
                    else:
                        self._bump_backoff(target_id)
                        BB.retries += 1
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set())
                            inv.add(bad)
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            # 4) Cutting board: short session across the chop (avoid board steals)
            elif self._is_cutting_board(state, target_id):
                if not SESSION["board_owned"]:
                    if self._handle_backoff(target_id):
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set())
                            inv.add(bad)
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                    if COORD.lock_resource(target_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="board_session"):
                        SESSION["board_owned"] = True
                        SESSION["board_id"] = target_id
                        self._reset_backoff(target_id)
                    else:
                        self._bump_backoff(target_id)
                        BB.retries += 1
                        # Switch to a different board if possible by rebuilding the BT
                        try:
                            prev_meal = getattr(BB, "active_meal", "")
                            self._reset_per_order_state()
                            BB.active_meal = prev_meal
                            self._build_tree(state)
                        except Exception:
                            pass
                        # Only waiters park; cooks stay near and retry
                        if not SESSION.get("recipe_token_owned"):
                            self._park_at_plate(state)
                        else:
                            self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            else:
                # Generic per-action lock for everything else (dispensers, serving, etc.)
                if self._handle_backoff(target_id):
                    bad = getattr(BB, "last_oneshot", None)
                    if bad:
                        inv = getattr(BB, "invalidate_oneshots", set())
                        inv.add(bad)
                        BB.invalidate_oneshots = inv
                    self._anchor_near(state, target_id)
                    del BB.next_task
                    return
                if not COORD.lock_resource(target_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="action"):
                    self._bump_backoff(target_id)
                    BB.retries += 1
                    bad = getattr(BB, "last_oneshot", None)
                    if bad:
                        inv = getattr(BB, "invalidate_oneshots", set())
                        inv.add(bad)
                        BB.invalidate_oneshots = inv
                    self._anchor_near(state, target_id)
                    del BB.next_task
                    return
                held_per_action_lock = True  # remember to unlock after scheduling

        # Finally, schedule the chosen action with the engine
        self.set_current_task(Task(task_type, task_args=task_arg, task_status=TaskStatus.SCHEDULED))
        BB.actions += 1
        del BB.next_task

        # Unlock short per-action locks immediately (sessions remain held)
        if held_per_action_lock and target_id:
            COORD.unlock_resource(target_id, self.AGENT_OWNER)
            self._reset_backoff(target_id)

        # If we served, free the claimed order so next cook can pick a new one
        if self._last_put_was_serve and self._claimed_order_id:
            COORD.release_order(self._claimed_order_id, self.AGENT_OWNER)
            self._claimed_order_id = None

        # Short suffix just for cleaner logs when two agents are running
        suffix = self.AGENT_OWNER.split("-")[-1][-4:]
        print(f"[BTAgent:{self.AGENT_ID}/{suffix}] scheduled {task_type} -> {task_arg}")

    # ------------------------- helpers -------------------------
    def _is_stove_pot(self, state, cid: str) -> bool:
        """True if this counter is the stove slot that holds a Pot."""
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        pot_id = im.get("stove_pot") or im.get("POT_STOVE") or im.get("pot_stove")
        if pot_id and cid == pot_id:
            return True
        # Fallback: inspect if this is a stove with occupied_by == Pot
        for c in state.get("counters", []):
            if c.get("id") == cid and (c.get("type") or "").lower() == "stove":
                occ = c.get("occupied_by") or {}
                name = (occ.get("type") or occ.get("name") or "").lower()
                return name == "pot"
        return False

    def _get_oven_id(self, state) -> str | None:
        """Return oven id from ID_MAP or discover from state."""
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        oven = im.get("oven") or im.get("OVEN")
        if oven:
            return oven
        for c in state.get("counters", []):
            if (c.get("type") or "").lower() == "oven":
                return c.get("id")
        return None

    def _is_oven_or_peel(self, state, cid: str) -> bool:
        """True if target id is the oven or its peel (pizza-only session)."""
        if not cid:
            return False
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}

        # Prefer explicit mapping
        oven = im.get("oven") or im.get("OVEN")
        peel = im.get("peel") or im.get("PEEL")

        # Fallback diskoveries
        if not oven:
            for c in state.get("counters", []):
                if (c.get("type") or "").lower() == "oven":
                    oven = c.get("id")
                    break
        if not peel:
            for c in state.get("counters", []):
                occ = c.get("occupied_by")
                name = (occ.get("type") or occ.get("name") or "").lower() if isinstance(occ, dict) else ""
                if name == "peel":
                    peel = c.get("id")
                    break

        return cid in {oven, peel} if cid else False

    def _get_fryer_id(self, state) -> str | None:
        """Return fryer id from ID_MAP or discover from state."""
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        fid = im.get("deep_fryer") or im.get("fryer") or im.get("DEEP_FRYER")
        if fid:
            return fid
        for c in state.get("counters", []):
            if (c.get("type") or "").lower() in {"deepfryer", "fryer"}:
                return c.get("id")
        return None

    def _is_fryer_or_basket(self, state, cid: str) -> bool:
        """True if the id is the fryer or its basket slot."""
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        fryer = im.get("deep_fryer") or im.get("fryer")
        basket = im.get("basket")
        return cid is not None and cid in {fryer, basket}

    def _is_cutting_board(self, state, cid: str) -> bool:
        """True if the id refers to a cutting board (mapped or inferred)."""
        if not cid:
            return False
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        if cid in set(im.get("cutting_boards", [])):
            return True
        for c in state.get("counters", []):
            if c.get("id") == cid and (c.get("type") or "").lower() == "cuttingboard":
                return True
        return False

    # ------------------------- Gear lookup helpers -------------------------
    def _get_pan_id(self, state) -> str | None:
        """Return the stove slot id that currently holds the Pan."""
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        pid = im.get("stove_pan") or im.get("PAN_STOVE") or im.get("pan_stove")
        if pid:
            return pid
        for c in state.get("counters", []):
            if (c.get("type") or "").lower() == "stove":
                occ = c.get("occupied_by") or {}
                name = (occ.get("type") or occ.get("name") or "").lower()
                if name == "pan":
                    return c.get("id")
        return None

    def _is_stove_pan(self, state, cid: str) -> bool:
        """True if the id is a stove slot holding the Pan (robust to naming)."""
        if not cid:
            return False
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        pan_id = im.get("stove_pan") or im.get("PAN_STOVE") or im.get("pan_stove")

        if not pan_id:
            pan_id = self._get_pan_id(state)  # discover by scanning stoves

        if pan_id and cid == pan_id:
            return True

        # Last resort: inspect the concrete counter right now
        ctr = next((c for c in state.get("counters", []) if c.get("id") == cid), None)
        if not ctr:
            return False
        if (ctr.get("type") or "").lower() != "stove":
            return False
        occ = ctr.get("occupied_by") or {}
        name = (occ.get("type") or occ.get("name") or "").lower()
        return name in {"pan", "fryingpan", "skillet"}

    def _gear_ids_for_meal(self, state, meal_norm: str) -> dict:
        """
        Return a dict {session_kind -> resource_id} of exclusive gear needed by this meal.
        Only include gear that we hold across multiple steps.
        """
        m = meal_norm
        needs = {}
        if m == "pizza":
            oven = self._get_oven_id(state)
            if oven:
                needs["oven_session"] = oven
        elif m in {"tomatosoup", "onionsoup"}:
            # Pot stove for soups
            pot = None
            try:
                from importlib import reload
                im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
            except Exception:
                im = {}
            pot = im.get("stove_pot") or im.get("POT_STOVE") or im.get("pot_stove")
            if not pot:
                # Fallback: find stove with Pot
                for c in state.get("counters", []):
                    if (c.get("type") or "").lower() == "stove":
                        occ = c.get("occupied_by") or {}
                        name = (occ.get("type") or occ.get("name") or "").lower()
                        if name == "pot":
                            pot = c.get("id")
                            break
            if pot:
                needs["stove_pot_session"] = pot
        elif m in {"chips", "friedfish", "fishandchips"}:
            fry = self._get_fryer_id(state)
            if fry:
                needs["fryer_session"] = fry
        elif m == "burger":
            pan = self._get_pan_id(state)
            if pan:
                needs["stove_pan_session"] = pan
        # Salad has no exclusive gear
        return needs

    def _park_at_plate(self, state):
        """
        Park strategy:
          - If we are the cook (claimed order), do NOT park — anchor near needed gear.
          - If waiter (no claim), stand at our plate spot or serving window.
        """
        if self._claimed_order_id is not None:
            # Cook: anchor near the current need (or serving window as fallback)
            anchor_id = getattr(BB, "anchor_id", None)
            if not anchor_id:
                meal = _norm_name(getattr(BB, "active_meal", ""))
                need = self._gear_ids_for_meal(state, meal)
                anchor_id = next(iter(need.values())) if need else self.serving_id
            self._anchor_near(state, anchor_id)
            return

        # Waiter: real parking (don’t block gear)
        spot = getattr(BB, "plate_spot_id", None) or self.serving_id
        pos = self.counter_positions.get(spot)
        if pos is not None and self.current_task is None:
            print(f"[BTAgent:{self.AGENT_ID}] parking (waiter) at plate spot {spot}")
            self.set_current_task(Task("GOTO", task_args=pos, task_status=TaskStatus.SCHEDULED))

    def _anchor_near(self, state, target_id: str):
        """Stand near a target counter (sticky proximity without taking any action)."""
        if not target_id:
            return
        pos = self.counter_positions.get(target_id)
        if pos is None:
            # Last chance: read it directly from the state
            ctr = next((c for c in state.get("counters", []) if c.get("id") == target_id), None)
            if ctr:
                pos = np.array(ctr.get("pos"))
        if pos is not None and self.current_task is None:
            BB.anchor_id = target_id
            print(f"[BTAgent:{self.AGENT_ID}] anchoring near {target_id}")
            self.set_current_task(Task("GOTO", task_args=pos, task_status=TaskStatus.SCHEDULED))


if __name__ == "__main__":
    # Entrypoint used by the engine runner
    run_agent_from_args(BTAgent)
