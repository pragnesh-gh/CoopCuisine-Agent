# cocu_base_agents/new_agent/multi_agent.py
# NOTE: file is a copy of your new_agent.py with minimal additions for multi-agent coordination.

import os
import time
import json
import numpy as np
import py_trees
from py_trees import blackboard

from cooperative_cuisine.base_agent.agent_task import Task, TaskStatus
from cooperative_cuisine.base_agent.base_agent import BaseAgent, run_agent_from_args

# Import our handcrafted builder + shared BB
from cocu_base_agents.new_agent.multiAgent_behavior_tree import BehaviorTreeBuilder, BB  # <- changed import
# Import your explicit stove + dispenser mapping
import id_map as id_map_module

# NEW: Coordinator
from cocu_base_agents.new_agent.coordinator import Coordinator

try:
    TS_DONE = TaskStatus.DONE
except AttributeError:
    TS_DONE = getattr(TaskStatus, "FINISHED",
              getattr(TaskStatus, "SUCCESS",
              getattr(TaskStatus, "COMPLETED", None)))
    if TS_DONE is None:
        TS_DONE = TaskStatus.SCHEDULED  # harmless fallback

# ──────────────────────────────────────────────────────────────────────────────
# Multi-agent additions
# ──────────────────────────────────────────────────────────────────────────────
# Agent identity:
# We use an environment variable for clarity in multi-process runs.
# Example: run second agent with AGENT_ID=B
# AGENT_ID = os.getenv("AGENT_ID", "A").strip() or "A"
# AGENT_OWNER = f"{AGENT_ID}-{os.getpid()}"

# Coordinator DB path is fixed in Coordinator; we just instantiate it once.
COORD = Coordinator()

# Backoff on lock contention: 0.2 → 0.4 → 0.8s (cap 0.8)
BACKOFF_STEPS_S = (0.2, 0.4, 0.8)

# Session flags for long-lived resource ownership (soup→pot until served)
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
SESSION["board_owned"] = False
SESSION["board_id"] = None
SESSION["recipe_token_owned"] = False
SESSION["recipe_token_id"] = None

def _recipe_token_id(meal_norm: str) -> str:
    return f"recipe::{meal_norm or ''}"




def _norm_name(s):
    if s is None: return ""
    return str(s).lower().replace("_", "").replace(" ", "")

def _orders_signature(state):
    orders = state.get("orders") or state.get("order_state") or []
    sig = []
    for i,o in enumerate(orders):
        oid = o.get("id") or o.get("order_id") or o.get("uid")
        meal = (o.get("meal") or o.get("name") or "").lower()
        sig.append(oid or f"{meal}#{i}")
    return tuple(sig)

def _order_time_left(order: dict) -> float:
    try: return float(order.get("time_remaining", 0.0))
    except (TypeError, ValueError): return 0.0

def _extract_order_id(o: dict) -> str | None:
    # OrderManager.order_state provides {"id": order.uuid, ...}
    return o.get("id") or o.get("uuid") or o.get("order_id") or o.get("uid")

# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────
class BTAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False
        self.prev_orders_sig = None
        self._last_put_was_serve = False
        self.serving_id = None

        # Track backoff per resource
        self._backoff_idx = {}  # res_id -> idx
        self._backoff_until = {}  # res_id -> timestamp

        # Claim tracking
        self._claimed_order_id = None

        # NEW: per-instance identity
        env_id = os.getenv("AGENT_ID", "").strip()
        # Simple, deterministic in-process assignment: first instance → A, second → B
        if not hasattr(BTAgent, "_instance_seq"):
            BTAgent._instance_seq = 0
        seq = BTAgent._instance_seq
        BTAgent._instance_seq += 1
        auto_id = "A" if seq == 0 else "B"
        self.AGENT_ID = env_id or auto_id

        # Owner must be unique per instance (not per process)
        self.AGENT_OWNER = f"{self.AGENT_ID}-{os.getpid()}-{id(self)}"

    def _pick_leftmost_order_meal(self, state) -> str:
        orders = state.get("orders") or state.get("order_state") or []
        if not isinstance(orders, (list, tuple)) or not orders:
            return ""
        m = (orders[0].get("meal") or orders[0].get("name") or "").lower()
        return m.replace(" ", "")

    def _reset_per_order_state(self):
        for k in ("plate_spot_id", "added_to_plate", "allow_pick_plate_once"):
            if hasattr(BB, k):
                try: delattr(BB, k)
                except Exception: pass

    def initialise(self, state):
        super().initialise(state)
        if self._initialized:
            return
        self._initialized = True
        # Ensure unique identity per agent instance (even in the same process).
        # Prefer the engine's own player id; fall back to a stable local token.
        self.AGENT_ID = (os.getenv("AGENT_ID", "A").strip() or "A")
        unique_token = getattr(self, "own_player_id", None) or state.get("own_player_id") \
                       or f"local{abs(id(self)) % 100000}"
        self.AGENT_OWNER = f"{self.AGENT_ID}-{unique_token}"

        # Build positions map + serving id
        counters = state.get("counters", [])
        service = next((c for c in counters if (c.get("type") or "").lower() == "servingwindow"), None)
        if service is None:
            raise RuntimeError("No serving window found")
        self.serving_id = service["id"]
        self.counter_positions = {c["id"]: np.array(c["pos"]) for c in counters}

        # Cache initial orders signature
        self.prev_orders_sig = _orders_signature(state)

        # Do NOT build a BT yet. We only build after we successfully claim an order.
        # This prevents early planning/drift before we know whether we are the
        # cooking agent or the waiting agent for exclusive-gear recipes.
        BB.active_meal = ""
        BB.last_meal = None
        # Track the single exclusive gear we own for this order (e.g., oven/fryer/pot/pan)
        self._session_gear_id = None
        BB.gear_wait_id = None

        # Clear any stale scheduling / waits
        if hasattr(BB, "next_task"): del BB.next_task
        BB.meta_waiting = None
        BB.dumped_utensil_schema = False

        # Reset session flags
        SESSION["pot_owned_for_soup"] = False
        SESSION["stove_pot_id"] = None
        SESSION["recipe_token_owned"] = False
        SESSION["recipe_token_id"] = None

    def _build_tree(self, state):
        # expose identity to the BT builder
        BB.agent_id = self.AGENT_ID
        BB.agent_owner = self.AGENT_OWNER
        self.tree = BehaviorTreeBuilder.build(
            state=state,
            counter_positions=self.counter_positions,
            id_map_module=id_map_module
        )
        self.tree.setup(timeout=10)
        print(f"[BTAgent:{self.AGENT_ID}] ✅ Built BT for meal='{BB.active_meal}'")

    # ------------------------- Lock helpers -------------------------
    def _now(self): return time.time()

    def _contested_set(self, state):
        """
        Resource ids that should be locked before PUT/INTERACT:
        - All dispensers (single slot)
        - Cutting boards
        - Pan stove, Pot stove
        - Deep fryer, Oven, Serving window, Plate dispenser
        """
        ids = set()
        # From id_map first, fall back to counters by type
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}

        # Explicit IDs if present
        for k in (
            "tomato_dispenser","lettuce_dispenser","bun_dispenser","meat_dispenser",
            "onion_dispenser","potato_dispenser","fish_dispenser","dough_dispenser",
            "cheese_dispenser","sausage_dispenser",
            "plate_dispenser","serving_window","stove_pan","stove_pot",
            "deep_fryer","oven","peel","basket"
        ):
            v = im.get(k)
            if isinstance(v, (list, tuple, set)):
                ids.update(v)
            elif v:
                ids.add(v)

        # Cutting boards can be a list in ID_MAP
        for b in im.get("cutting_boards", []):
            ids.add(b)

        # Fallback by type names from state
        by_type = {}
        for c in state.get("counters", []):
            by_type.setdefault((c.get("type") or "").lower(), []).append(c.get("id"))
        for t in ("tomatodispenser","lettucedispenser","bundispenser","meatdispenser",
                  "oniondispenser","potatodispenser","fishdispenser","doughdispenser",
                  "cheesedispenser","sausagedispenser",
                  "platedispenser","servingwindow","stove","deepfryer","oven","cuttingboard"):
            for cid in by_type.get(t, []):
                ids.add(cid)

        return ids

    def _is_contested(self, state, target_id: str) -> bool:
        return target_id in self._contested_set(state)

    def _should_session_lock_pot(self) -> bool:
        """Session-level pot ownership ONLY for soups, as per your rule."""
        meal = _norm_name(getattr(BB, "active_meal", ""))
        return meal in {"tomatosoup", "onionsoup"}

    def _handle_backoff(self, res_id: str) -> bool:
        """Return True if we should skip this tick due to backoff on this resource."""
        until = self._backoff_until.get(res_id, 0.0)
        if self._now() < until:
            return True
        return False

    def _bump_backoff(self, res_id: str):
        idx = self._backoff_idx.get(res_id, 0)
        idx = min(idx + 1, len(BACKOFF_STEPS_S) - 1)
        self._backoff_idx[res_id] = idx
        self._backoff_until[res_id] = self._now() + BACKOFF_STEPS_S[idx]

    def _reset_backoff(self, res_id: str):
        self._backoff_idx.pop(res_id, None)
        self._backoff_until.pop(res_id, None)

    # ------------------------- META_COOK polling (unchanged) -------------------------
    # (same as your single-agent version)
    def _lookup_counter(self, state: dict, counter_id: str) -> dict | None:
        cs = state.get("counters")
        if isinstance(cs, dict):
            return cs.get(counter_id)
        if isinstance(cs, list):
            for c in cs:
                if c.get("id") == counter_id:
                    return c
        return None

    def _extract_item_name(self, obj) -> str | None:
        if obj is None: return None
        if isinstance(obj, str): return None if obj.strip().lower() == "none" else obj
        if isinstance(obj, dict):
            n = obj.get("name") or obj.get("type")
            if n: return n
            for k in ("item", "result", "item_info"):
                v = obj.get(k)
                if isinstance(v, dict):
                    n = v.get("name") or v.get("type")
                    if n: return n
        return None

    def _top_name_from_content_list(self, clist) -> str | None:
        if not isinstance(clist, (list, tuple)) or not clist: return None
        if isinstance(clist[0], dict):
            return self._extract_item_name(clist[0])
        return None

    def _describe_station_equipment(self, state, station_id: str) -> str:
        ctr = self._lookup_counter(state, station_id)
        if not ctr: return f"station={station_id} not found"
        occ = ctr.get("occupied_by")
        if not occ: return f"station={station_id} (empty)"
        parts = []
        ename = self._extract_item_name(occ)
        if ename: parts.append(f"equip={ename}")
        r = occ.get("content_ready")
        rname = self._extract_item_name(r) if isinstance(r, dict) else None
        if rname: parts.append(f"ready={rname}")
        cl = occ.get("content_list") or []
        tname = self._top_name_from_content_list(cl)
        parts.append(f"content_top={tname!r}")
        at = occ.get("active_transition")
        if isinstance(at, dict):
            at_name = self._extract_item_name(at.get("result"))
            parts.append(f"active={{sec:{at.get('seconds')}, result:{at_name!r}}}")
        return f"station={station_id} " + " ".join(parts)

    def _handle_meta_cook_poll(self, state) -> TaskStatus:
        # unchanged from your single-agent file...
        if not getattr(BB, "meta_waiting", None):
            return TS_DONE
        expected = _norm_name(BB.meta_waiting.get("expected"))
        station_id = BB.meta_waiting.get("station_id")
        if not station_id:
            BB.meta_waiting = None; return TS_DONE
        ctr = self._lookup_counter(state, station_id)
        if not ctr: return TaskStatus.IN_PROGRESS
        occ = ctr.get("occupied_by")
        equip = None
        if isinstance(occ, dict) and occ.get("category") in ("ItemCookingEquipment", "Item"):
            equip = occ
        elif isinstance(occ, list):
            for o in occ:
                if isinstance(o, dict) and o.get("category") in ("ItemCookingEquipment", "Item"):
                    equip = o; break
        if equip is None:
            return TaskStatus.IN_PROGRESS
        cl = equip.get("content_list") or []
        cr = equip.get("content_ready")
        def _n(x):
            if isinstance(x, dict): return (x.get("type") or x.get("name") or "").lower()
            if isinstance(x, list): return [(_i.get("type") or _i.get("name")) for _i in x]
            return None
        # Burn guard
        if isinstance(cl, list) and cl and isinstance(cl[0], dict):
            top = _norm_name(self._extract_item_name(cl[0]))
            if top in {"waste", "burnt"}:
                BB.meta_waiting = None
                return TaskStatus.FAILED
        # Instant transitions
        rname = _norm_name(self._extract_item_name(cr)) if isinstance(cr, dict) else None
        if rname and rname == expected:
            BB.meta_waiting = None
            return TS_DONE
        # Timed transitions: result in content_list
        if isinstance(cl, list) and len(cl) == 1 and isinstance(cl[0], dict):
            t = _norm_name(self._extract_item_name(cl[0]))
            if t == expected:
                BB.meta_waiting = None
                return TS_DONE
        return TaskStatus.IN_PROGRESS
    # ------------------------- Session locking helpers -------------------------
    SESSION_TTL = 20.0  # seconds; safe upper bound for cook phases

    def _set_session_flags(self, kind: str, rid: str, on: bool):
        if kind == "stove_pot_session":
            SESSION["pot_owned_for_soup"] = on;
            SESSION["stove_pot_id"] = rid if on else None
        elif kind == "oven_session":
            SESSION["oven_owned_for_pizza"] = on;
            SESSION["oven_id"] = rid if on else None
        elif kind == "fryer_session":
            SESSION["fryer_owned_for_fry"] = on;
            SESSION["fryer_id"] = rid if on else None
        elif kind == "stove_pan_session":
            SESSION["pan_owned_for_burger"] = on;
            SESSION["pan_id"] = rid if on else None

    def _try_reserve_gear_for_meal(self, state, meal_norm: str) -> bool:
        """
        Attempt to pre-reserve ALL exclusive gear needed for this meal.
        If any lock fails, roll back and return False.
        """
        needs = self._gear_ids_for_meal(state, meal_norm)
        if not needs:
            return True  # nothing exclusive required
        acquired = []
        for kind, rid in needs.items():
            ok = COORD.lock_resource(rid, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind=kind)
            if ok:
                acquired.append((kind, rid))
            else:
                # rollback
                for k2, r2 in acquired:
                    try:
                        COORD.unlock_resource(r2, self.AGENT_OWNER)
                    except Exception:
                        pass
                return False
        # success: set flags
        for kind, rid in acquired:
            self._set_session_flags(kind, rid, True)
        return True

    def _refresh_session_locks(self, state):
        """
        Refresh (extend) TTL for any held sessions each tick by re-locking them.
        Assumes Coordinator.lock_resource is idempotent for the same owner.
        """
        if SESSION.get("pot_owned_for_soup") and SESSION.get("stove_pot_id"):
            COORD.lock_resource(SESSION["stove_pot_id"], self.AGENT_OWNER, ttl_s=self.SESSION_TTL,
                                kind="stove_pot_session")
        if SESSION.get("oven_owned_for_pizza") and SESSION.get("oven_id"):
            COORD.lock_resource(SESSION["oven_id"], self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="oven_session")
        if SESSION.get("fryer_owned_for_fry") and SESSION.get("fryer_id"):
            COORD.lock_resource(SESSION["fryer_id"], self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="fryer_session")
        if SESSION.get("pan_owned_for_burger") and SESSION.get("pan_id"):
            COORD.lock_resource(SESSION["pan_id"], self.AGENT_OWNER, ttl_s=self.SESSION_TTL,
                                kind="stove_pan_session")
        if SESSION.get("recipe_token_owned") and SESSION.get("recipe_token_id"):
            COORD.lock_resource(SESSION["recipe_token_id"], self.AGENT_OWNER, ttl_s=self.SESSION_TTL,
                                kind="recipe_serial")
    # ──────────────────────────────────────────────────────────────────────────
    # Main scheduler
    # ──────────────────────────────────────────────────────────────────────────
    async def manage_tasks(self, state):
        # Heartbeat once per tick (extends owned locks)
        COORD.heartbeat(self.AGENT_OWNER)
        # Extend our held session locks (oven/fryer/pot/pan) to avoid TTL expiry mid-cook
        self._refresh_session_locks(state)

        BB.tick = getattr(BB, "tick", 0) + 1
        BB.last_state = state

        # Helper to pick next meal after serve (unchanged rule)
        def _ingredient_count_for(meal: str, st: dict) -> int:
            m = _norm_name(meal)
            if m == "tomatosoup":   return 3
            if m == "onionsoup":    return 3
            if m == "salad":        return 2
            if m == "burger":       return 4
            if m == "chips":        return 1
            if m == "friedfish":    return 1
            if m == "fishandchips": return 2
            if m == "pizza":
                sausage_present = any(((c.get("type") or c.get("name") or "").lower() in
                                       {"sausage_dispenser","sausagedispenser","sausage"})
                                       for c in st.get("counters", []))
                return 4 if sausage_present else 3
            return 3

        def _pick_after_serve_with_50pct_rule(st: dict) -> str:
            orders = st.get("orders") or st.get("order_state") or []
            rows = []
            for idx, o in enumerate(orders):
                meal = _norm_name(o.get("meal") or o.get("name"))
                if not meal: continue
                tleft = _order_time_left(o)
                icost = _ingredient_count_for(meal, st)
                rows.append((tleft, icost, idx, meal, _extract_order_id(o)))
            if not rows: return ""
            T_max = max(r[0] for r in rows)
            top = [r for r in rows if r[0] == T_max]
            default = min(top, key=lambda r: r[2])  # left-most among max-time
            threshold = 0.5 * T_max
            eligible = [r for r in rows if r[0] >= threshold and r[1] < default[1]]
            if eligible:
                eligible.sort(key=lambda r: (r[1], -r[0], r[2]))
                return eligible[0][3]
            return default[3]

        # 0) META_COOK priority
        if getattr(BB, "meta_waiting", None):
            status = self._handle_meta_cook_poll(state)
            if status == TaskStatus.IN_PROGRESS:
                return
            # else fall-through

        # Init prev_orders_sig
        if not hasattr(self, "prev_orders_sig"):
            self.prev_orders_sig = _orders_signature(state)

        # Find serving window id
        serving_id = self.serving_id

        # A) If we served last tick → reset & rebuild (+ soup pot session release)
        if self._last_put_was_serve:
            self._last_put_was_serve = False
            if getattr(BB, "plate_spot_id", None):
                COORD.release_counter(BB.plate_spot_id, self.AGENT_OWNER)
            # Release soup pot session (session-level ownership ends at serve)
            if SESSION["pot_owned_for_soup"] and SESSION["stove_pot_id"]:
                COORD.unlock_resource(SESSION["stove_pot_id"], self.AGENT_OWNER)
            SESSION["pot_owned_for_soup"] = False
            SESSION["stove_pot_id"] = None
            # Release pizza oven/peel session at serve (keeps it simple and safe)
            if SESSION["oven_owned_for_pizza"] and SESSION["oven_id"]:
                COORD.unlock_resource(SESSION["oven_id"], self.AGENT_OWNER)
            SESSION["oven_owned_for_pizza"] = False
            SESSION["oven_id"] = None
            #Release fryer
            if SESSION["fryer_owned_for_fry"] and SESSION["fryer_id"]:
                COORD.unlock_resource(SESSION["fryer_id"], self.AGENT_OWNER)
            SESSION["fryer_owned_for_fry"] = False
            SESSION["fryer_id"] = None
            # Release pan session
            if SESSION.get("pan_owned_for_burger") and SESSION.get("pan_id"):
                COORD.unlock_resource(SESSION["pan_id"], self.AGENT_OWNER)
            SESSION["pan_owned_for_burger"] = False
            SESSION["pan_id"] = None
            #Release board
            if SESSION.get("board_owned") and SESSION.get("board_id"):
                COORD.unlock_resource(SESSION["board_id"], self.AGENT_OWNER)
            SESSION["board_owned"] = False
            SESSION["board_id"] = None
            # Clear any gear-wait intent
            if hasattr(BB, "recipe_wait"):
                try:
                    delattr(BB, "recipe_wait")
                except Exception:
                    pass
            # Release recipe token
            if SESSION.get("recipe_token_owned") and SESSION.get("recipe_token_id"):
                COORD.unlock_resource(SESSION["recipe_token_id"], self.AGENT_OWNER)
            SESSION["recipe_token_owned"] = False
            SESSION["recipe_token_id"] = None


            self._reset_per_order_state()
            # After serve: do NOT pre-select or build. Let the claim loop decide next.
            BB.active_meal = ""

        # B) Orders list changed → if no active meal, (re)select
        curr_sig = _orders_signature(state)
        if curr_sig != self.prev_orders_sig:
            self.prev_orders_sig = curr_sig

        # 1) Wait in-flight
        if self.current_task is not None:
            return

        # ── CLAIM LOOP (runs whenever we have no claimed order) ─────────────
        if self._claimed_order_id is None:
            orders = state.get("orders") or state.get("order_state") or []

            # If we're explicitly waiting for a particular gear, try to grab it first.
            gear_wait_id = getattr(BB, "gear_wait_id", None)
            if gear_wait_id:
                if COORD.lock_resource(gear_wait_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="session"):
                    # Find an order that needs this gear; reserve rest & claim.
                    for o in orders:
                        meal = _norm_name(o.get("meal") or o.get("name"))
                        gear_map = self._gear_ids_for_meal(state, meal)
                        if gear_map and gear_wait_id in gear_map.values():
                            # Take recipe token first
                            token_id = _recipe_token_id(meal)
                            if not COORD.lock_resource(token_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL,
                                                       kind="recipe_serial"):
                                BB.recipe_wait = meal
                                break
                            SESSION["recipe_token_owned"] = True
                            SESSION["recipe_token_id"] = token_id
                            # Now reserve gear
                            if not self._try_reserve_gear_for_meal(state, meal):
                                COORD.unlock_resource(token_id, self.AGENT_OWNER)
                                SESSION["recipe_token_owned"] = False
                                SESSION["recipe_token_id"] = None
                                continue
                            oid = _extract_order_id(o)
                            if not oid or not COORD.claim_order(oid, self.AGENT_OWNER):
                                # Release reserved gear and keep waiting
                                for _, rid in (gear_map or {}).items():
                                    try:
                                        COORD.unlock_resource(rid, self.AGENT_OWNER)
                                    except Exception:
                                        pass
                                # release recipe token
                                COORD.unlock_resource(token_id, self.AGENT_OWNER)
                                SESSION["recipe_token_owned"] = False
                                SESSION["recipe_token_id"] = None
                                break
                            # success → set meal, build, clear wait
                            self._claimed_order_id = oid
                            BB.active_meal = meal or ""
                            self._reset_per_order_state()
                            self._build_tree(state)
                            # clear wait intents
                            for _attr in ("gear_wait_id", "recipe_wait"):
                                if hasattr(BB, _attr):
                                    try:
                                        delattr(BB, _attr)
                                    except Exception:
                                        pass
                            print(
                                f"[BTAgent:{self.AGENT_ID}] ✅ claimed order {self._claimed_order_id} (gear wait resolved)")
                            break
                # If lock failed, fall through to normal pass; we’ll park below.

            # Pass 1: prefer orders whose exclusive gear we can pre-reserve now
            def _try_claim(o):
                oid = _extract_order_id(o)
                if not oid:
                    return False
                meal = _norm_name(o.get("meal") or o.get("name"))
                # 1) Take the recipe token (serialize by meal) – wait-until-serve policy
                token_id = _recipe_token_id(meal)
                if not COORD.lock_resource(token_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="recipe_serial"):
                    # Someone else is cooking this meal end-to-end → latch and wait
                    BB.recipe_wait = meal
                    return False
                # Mark token owned (temporarily; may rollback on failure next steps)
                SESSION["recipe_token_owned"] = True
                SESSION["recipe_token_id"] = token_id

                # 2) Pre-reserve required exclusive gear for this meal
                if not self._try_reserve_gear_for_meal(state, meal):
                    # rollback recipe token
                    COORD.unlock_resource(token_id, self.AGENT_OWNER)
                    SESSION["recipe_token_owned"] = False
                    SESSION["recipe_token_id"] = None
                    return False

                # 3) Claim a specific order id
                if not COORD.claim_order(oid, self.AGENT_OWNER):
                    # release gear & recipe token since claim failed
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
                # success → set meal, build
                self._claimed_order_id = oid
                BB.active_meal = meal or ""
                self._reset_per_order_state()
                self._build_tree(state)
                # clear any wait intents now that we own the meal
                for _attr in ("gear_wait_id", "recipe_wait"):
                    if hasattr(BB, _attr):
                        try:
                            delattr(BB, _attr)
                        except Exception:
                            pass
                print(
                    f"[BTAgent:{self.AGENT_ID}] ✅ claimed order {self._claimed_order_id} (recipe token + gear reserved)")
                return True

            claimed = False
            for o in orders:
                if _try_claim(o):
                    claimed = True
                    break

            # Pass 2: if nothing claimable-with-gear → latch left-most gear and wait
            if not claimed:
                for o in orders:
                    meal = _norm_name(o.get("meal") or o.get("name"))
                    gear_map = self._gear_ids_for_meal(state, meal)
                    if gear_map:
                        BB.gear_wait_id = next(iter(gear_map.values()))
                        break
                # No plate yet? park fallback will use home position.
                self._park_at_plate(state)
                return

        # If we latched recipe to wait for and still have no claim → park & return
        if getattr(BB, "recipe_wait", None) and self._claimed_order_id is None and not SESSION.get(
                "recipe_token_owned"):
            self._park_at_plate(state)
            return

        # If we latched gear to wait for and still have no claim → park & return
        if getattr(BB, "gear_wait_id", None) and self._claimed_order_id is None:
            self._park_at_plate(state)
            return

        # We have (or just built) a BT only after claim
        current_meal = _norm_name(getattr(BB, "active_meal", ""))

        # Safety: if tree missing or no meal, bail (claim loop runs again next tick)
        if not getattr(self, "tree", None) or not current_meal:
            return
        # Compute exclusive gear needs once
        need = self._gear_ids_for_meal(state, current_meal)
        # Recipe-token gate (soft): only makes waiters park; cooks never park mid-recipe
        if not SESSION.get("recipe_token_owned"):
            if self._claimed_order_id is None:
                # Waiter: only park if this meal requires exclusive gear (serialized)
                if need:
                    self._park_at_plate(state)
                    return
            else:
                # Cook: do NOT park; if meal needs gear, anchor near it and retry
                if need:
                    anchor_id = next(iter(need.values()))
                    self._anchor_near(state, anchor_id)
                    return
                # No exclusive gear → proceed without token
        if need:
            needs_oven = "oven_session" in need
            needs_pot = "stove_pot_session" in need
            needs_fryer = "fryer_session" in need
            needs_pan = "stove_pan_session" in need
            lacking = (
                    (needs_oven and not SESSION.get("oven_owned_for_pizza")) or
                    (needs_pot and not SESSION.get("pot_owned_for_soup")) or
                    (needs_fryer and not SESSION.get("fryer_owned_for_fry")) or
                    (needs_pan and not SESSION.get("pan_owned_for_burger"))
            )
            if lacking:
                # If we are NOT the cook (no recipe token), we’re a waiter → park.
                if not SESSION.get("recipe_token_owned"):
                    if not getattr(BB, "gear_wait_id", None):
                        BB.gear_wait_id = next(iter(need.values()))
                    self._park_at_plate(state)
                    return
                # We ARE the cook → do NOT park. Just retry next tick (stay anchored).
                # Optionally, anchor near the first needed gear to keep proximity sticky.
                anchor_id = next(iter(need.values()))
                self._anchor_near(state, anchor_id)
                return

        # 3) Drive BT
        self.tree.tick()

        # 4) Schedule emitted task (with lock gating)
        next_task = getattr(BB, "next_task", None)
        if not next_task:
            return

        task_type, task_arg = next_task
        target_id = task_arg if isinstance(task_arg, str) else None

        # ─────────────────────────────────────────────────────────────────
        # Debounce: prevent immediate second PUT to the plate spot (and fryer)
        # Rationale:
        # - Many recipes pour to the plate, then soon after pick the plate to serve.
        # - Back-to-back PUTs to the same plate spot can cause a "place → pick" toggle.
        # - We skip the second PUT if it's within a short window; BT will try next tick.
        # Tuning: widen to <=2 ticks if your env is noisy.
        # ─────────────────────────────────────────────────────────────────
        if task_type == "PUT":
            target_id = task_arg

            # Plate spot debounce
            if getattr(BB, "plate_spot_id", None) and target_id == BB.plate_spot_id:
                last_tick = getattr(BB, "last_plate_put_tick", -999)
                # Use <= 2 ticks for extra stability (change to 1 if you prefer tighter pacing)
                if (BB.tick - last_tick) <= 1 and not getattr(BB, "allow_pick_plate_once", False):
                    print(f"[BTAgent:{self.AGENT_ID}] ⏭️ debounced: skipping immediate second PUT on plate.")
                    del BB.next_task
                    return
                BB.last_plate_put_tick = BB.tick
                if getattr(BB, "allow_pick_plate_once", False):
                    BB.allow_pick_plate_once = False

            # Fryer basket debounce (prevents PUT-PUT flipping on the fryer home)
            basket_home = getattr(BB, "tool_home", {}).get("basket")
            if basket_home and target_id == basket_home:
                last_tick = getattr(BB, "last_fryer_put_tick", -999)
                if (BB.tick - last_tick) <= 1:
                    print(f"[BTAgent:{self.AGENT_ID}] ⏭️ debounced: skipping immediate second PUT on fryer.")
                    del BB.next_task
                    return
                BB.last_fryer_put_tick = BB.tick


        # Handle META_COOK watcher
        if task_type == "META_COOK":
            required, station_id = task_arg
            BB.meta_waiting = {"expected": _norm_name(required), "station_id": station_id}
            _ = self._handle_meta_cook_poll(state)
            del BB.next_task
            return

        # Final-serve edge detection
        if task_type == "PUT" and serving_id is not None and task_arg == serving_id:
            self._last_put_was_serve = True


        if SESSION.get("board_owned") and SESSION.get("board_id") and target_id != SESSION["board_id"]:
            COORD.unlock_resource(SESSION["board_id"], self.AGENT_OWNER)
            SESSION["board_owned"] = False
            SESSION["board_id"] = None
        # ─────────────────────────────────────────────────────────────────
        # Lock gating
        # - Boards/Dispensers/Serving: per-action lock (short, safe).
        # - Pot (soups), Oven/Peel (pizza), Fryer/Basket (chips/friedfish/fish&chips):
        #   session-level locks to avoid tool steals.
        # ─────────────────────────────────────────────────────────────────
        held_per_action_lock = False  # track if we must unlock after scheduling

        if target_id and self._is_contested(state, target_id):

            # 1) Soup pot session
            if self._should_session_lock_pot() and self._is_stove_pot(state, target_id):
                if not SESSION["pot_owned_for_soup"]:
                    if self._handle_backoff(target_id):
                        # tell the BT to retry this exact step next tick
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set());
                            inv.add(bad);
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
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set());
                            inv.add(bad);
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            # 2) Pizza oven/peel session (prevents peel steal)
            elif _norm_name(getattr(BB, "active_meal", "")) == "pizza" and self._is_oven_or_peel(state, target_id):
                if not SESSION["oven_owned_for_pizza"]:
                    if self._handle_backoff(target_id):
                        # tell the BT to retry this exact step next tick
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set());
                            inv.add(bad);
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
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set());
                            inv.add(bad);
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            # 3) Fryer session (chips / friedfish / fishandchips)
            elif _norm_name(getattr(BB, "active_meal", "")) in {"chips", "friedfish",
                                                                "fishandchips"} and self._is_fryer_or_basket(state,
                                                                                                             target_id):
                if not SESSION["fryer_owned_for_fry"]:
                    if self._handle_backoff(target_id):
                        # tell the BT to retry this exact step next tick
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set());
                            inv.add(bad);
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
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set());
                            inv.add(bad);
                            BB.invalidate_oneshots = inv
                        self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            # 4) Cutting board short session (hold through chop)
            elif self._is_cutting_board(state, target_id):
                if not SESSION["board_owned"]:
                    if self._handle_backoff(target_id):
                        # tell the BT to retry this exact step next tick
                        bad = getattr(BB, "last_oneshot", None)
                        if bad:
                            inv = getattr(BB, "invalidate_oneshots", set());
                            inv.add(bad);
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
                        # Rebuild BT to re-pick an alternate cutting board (the other one should be free/owned by someone else)
                        try:
                            prev_meal = getattr(BB, "active_meal", "")
                            self._reset_per_order_state()
                            BB.active_meal = prev_meal
                            self._build_tree(state)
                        except Exception:
                            pass
                        # Only waiters park; cooks anchor and retry
                        if not SESSION.get("recipe_token_owned"):
                            self._park_at_plate(state)
                        else:
                            self._anchor_near(state, target_id)
                        del BB.next_task
                        return
                # already own it → proceed

            else:
                # Per-action lock for everything else
                if self._handle_backoff(target_id):
                    # tell the BT to retry this exact step next tick
                    bad = getattr(BB, "last_oneshot", None)
                    if bad:
                        inv = getattr(BB, "invalidate_oneshots", set());
                        inv.add(bad);
                        BB.invalidate_oneshots = inv
                    self._anchor_near(state, target_id)
                    del BB.next_task
                    return
                if not COORD.lock_resource(target_id, self.AGENT_OWNER, ttl_s=self.SESSION_TTL, kind="action"):
                    self._bump_backoff(target_id)
                    bad = getattr(BB, "last_oneshot", None)
                    if bad:
                        inv = getattr(BB, "invalidate_oneshots", set());
                        inv.add(bad);
                        BB.invalidate_oneshots = inv
                    self._anchor_near(state, target_id)
                    del BB.next_task
                    return
                held_per_action_lock = True  # mark to unlock right after scheduling

        # Schedule the task
        self.set_current_task(Task(task_type, task_args=task_arg, task_status=TaskStatus.SCHEDULED))
        del BB.next_task

        # Immediately unlock per-action locks (not session locks)
        if held_per_action_lock and target_id:
            COORD.unlock_resource(target_id, self.AGENT_OWNER)
            self._reset_backoff(target_id)

        # Release claim on serve
        if self._last_put_was_serve and self._claimed_order_id:
            COORD.release_order(self._claimed_order_id, self.AGENT_OWNER)
            self._claimed_order_id = None

        suffix = self.AGENT_OWNER.split("-")[-1][-4:]
        print(f"[BTAgent:{self.AGENT_ID}/{suffix}] scheduled {task_type} -> {task_arg}")

    # ------------------------- helpers -------------------------
    def _is_stove_pot(self, state, cid: str) -> bool:
        # Prefer explicit id_map; fall back to type lookup
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        pot_id = im.get("stove_pot") or im.get("POT_STOVE") or im.get("pot_stove")
        if pot_id and cid == pot_id:
            return True
        # fallback: if target is a stove and holds a Pot
        for c in state.get("counters", []):
            if c.get("id") == cid and (c.get("type") or "").lower() == "stove":
                occ = c.get("occupied_by") or {}
                name = (occ.get("type") or occ.get("name") or "").lower()
                return name == "pot"
        return False

    def _get_oven_id(self, state) -> str | None:
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

        # Fallback: discover from state if id_map is missing
        if not oven:
            for c in state.get("counters", []):
                if (c.get("type") or "").lower() == "oven":
                    oven = c.get("id");
                    break
        if not peel:
            for c in state.get("counters", []):
                occ = c.get("occupied_by")
                name = (occ.get("type") or occ.get("name") or "").lower() if isinstance(occ, dict) else ""
                if name == "peel":
                    peel = c.get("id");
                    break

        return cid in {oven, peel} if cid else False

    def _get_fryer_id(self, state) -> str | None:
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        fid = im.get("deep_fryer") or im.get("fryer") or im.get("DEEP_FRYER")
        if fid: return fid
        for c in state.get("counters", []):
            if (c.get("type") or "").lower() in {"deepfryer", "fryer"}:
                return c.get("id")
        return None

    def _is_fryer_or_basket(self, state, cid: str) -> bool:
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        fryer = im.get("deep_fryer") or im.get("fryer")
        basket = im.get("basket")
        return cid is not None and cid in {fryer, basket}

    def _is_cutting_board(self, state, cid: str) -> bool:
        if not cid: return False
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
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        pid = im.get("stove_pan") or im.get("PAN_STOVE") or im.get("pan_stove")
        if pid: return pid
        for c in state.get("counters", []):
            if (c.get("type") or "").lower() == "stove":
                occ = c.get("occupied_by") or {}
                name = (occ.get("type") or occ.get("name") or "").lower()
                if name == "pan":
                    return c.get("id")
        return None

    def _is_stove_pan(self, state, cid: str) -> bool:
        if not cid:
            return False
        # Prefer explicit mapping from id_map if present
        try:
            from importlib import reload
            im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
        except Exception:
            im = {}
        pan_id = im.get("stove_pan") or im.get("PAN_STOVE") or im.get("pan_stove")

        # Robust fallback: reuse our discovery helper (kept in one place)
        if not pan_id:
            pan_id = self._get_pan_id(state)  # scans stoves for occupied_by == Pan

        # If we’ve got a concrete pan stove id, use it
        if pan_id and cid == pan_id:
            return True

        # Last resort: inspect the counter referred to by cid right now
        ctr = next((c for c in state.get("counters", []) if c.get("id") == cid), None)
        if not ctr:
            return False
        if (ctr.get("type") or "").lower() != "stove":
            return False
        occ = ctr.get("occupied_by") or {}
        name = (occ.get("type") or occ.get("name") or "").lower()
        # Accept common aliases just in case item_info uses a different label
        return name in {"pan", "fryingpan", "skillet"}

    def _gear_ids_for_meal(self, state, meal_norm: str) -> dict:
        """
        Return a dict of session 'kinds' -> resource_id needed by this meal.
        Only include single-session anchors (we lock fryer for basket too).
        """
        m = meal_norm
        needs = {}
        if m == "pizza":
            oven = self._get_oven_id(state)
            if oven: needs["oven_session"] = oven
        elif m in {"tomatosoup", "onionsoup"}:
            # pot stove
            pot = None
            try:
                from importlib import reload
                im = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}
            except Exception:
                im = {}
            pot = im.get("stove_pot") or im.get("POT_STOVE") or im.get("pot_stove")
            if not pot:
                # fallback: find stove holding a pot
                for c in state.get("counters", []):
                    if (c.get("type") or "").lower() == "stove":
                        occ = c.get("occupied_by") or {}
                        name = (occ.get("type") or occ.get("name") or "").lower()
                        if name == "pot":
                            pot = c.get("id");
                            break
            if pot: needs["stove_pot_session"] = pot
        elif m in {"chips", "friedfish", "fishandchips"}:
            fry = self._get_fryer_id(state)
            if fry: needs["fryer_session"] = fry
        elif m == "burger":
            pan = self._get_pan_id(state)
            if pan: needs["stove_pan_session"] = pan
        # salad has no exclusive gear
        return needs

    def _park_at_plate(self, state):
        """
        Park at our staged plate spot (if any), otherwise at a neutral home.
        BUT: if we are the COOK (we have a claimed order), never park — anchor & retry instead.
        This globally prevents the cook from being 'demoted' into waiter behavior mid-recipe.
        """
        # If we have a claimed order, we are the cook. Do NOT park; anchor near the current need.
        if self._claimed_order_id is not None:
            # Prefer the last anchor target if present; else the first exclusive gear for this meal; else serving window.
            anchor_id = getattr(BB, "anchor_id", None)
            if not anchor_id:
                meal = _norm_name(getattr(BB, "active_meal", ""))
                need = self._gear_ids_for_meal(state, meal)
                if need:
                    anchor_id = next(iter(need.values()))
                else:
                    anchor_id = self.serving_id
            self._anchor_near(state, anchor_id)
            return

        # Waiter: real parking
        spot = getattr(BB, "plate_spot_id", None) or self.serving_id
        pos = self.counter_positions.get(spot)
        if pos is not None and self.current_task is None:
            print(f"[BTAgent:{self.AGENT_ID}] parking (waiter) at plate spot {spot}")
            self.set_current_task(Task("GOTO", task_args=pos, task_status=TaskStatus.SCHEDULED))

    def _anchor_near(self, state, target_id: str):
        if not target_id:
            return
        pos = self.counter_positions.get(target_id)
        if pos is None:
            # fallback: ask state for the counter position
            ctr = next((c for c in state.get("counters", []) if c.get("id") == target_id), None)
            if ctr:
                pos = np.array(ctr.get("pos"))
        if pos is not None and self.current_task is None:
            BB.anchor_id = target_id
            print(f"[BTAgent:{self.AGENT_ID}] anchoring near {target_id}")
            self.set_current_task(Task("GOTO", task_args=pos, task_status=TaskStatus.SCHEDULED))

if __name__ == "__main__":
    run_agent_from_args(BTAgent)
