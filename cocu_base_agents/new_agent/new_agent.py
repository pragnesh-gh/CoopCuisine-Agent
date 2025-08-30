# new_agent.py — handcrafted-BT executor with left-most order selection + continuous loop
import json
import numpy as np
import py_trees
from py_trees import blackboard

from cooperative_cuisine.base_agent.agent_task import Task, TaskStatus
from cooperative_cuisine.base_agent.base_agent import BaseAgent, run_agent_from_args

# Import our handcrafted builder + shared BB
from cocu_base_agents.new_agent.behavior_tree import BehaviorTreeBuilder, BB
# Import your explicit stove + dispenser mapping
import id_map as id_map_module

# Compatibility for DONE status across envs
try:
    TS_DONE = TaskStatus.DONE
except AttributeError:
    TS_DONE = getattr(TaskStatus, "FINISHED",
              getattr(TaskStatus, "SUCCESS",
              getattr(TaskStatus, "COMPLETED", None)))
    if TS_DONE is None:
        TS_DONE = TaskStatus.SCHEDULED  # harmless fallback

# ──────────────────────────────────────────────────────────────────────────────
# META_COOK helpers (robust across payload shapes)
# ──────────────────────────────────────────────────────────────────────────────
def _norm_name(s: str | None) -> str:
    if s is None: return ""
    if isinstance(s, str) and s.strip().lower() == "none": return ""
    return str(s).lower().replace("_", "")

def _extract_item_name(obj) -> str | None:
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

def _lookup_counter(state: dict, counter_id: str) -> dict | None:
    cs = state.get("counters")
    if isinstance(cs, dict):
        return cs.get(counter_id)
    if isinstance(cs, list):
        for c in cs:
            if c.get("id") == counter_id:
                return c
    return None

def _top_name_from_content_list(clist) -> str | None:
    if not isinstance(clist, (list, tuple)) or not clist: return None
    if isinstance(clist[0], dict):
        return _extract_item_name(clist[0])
    return None
def _orders_signature(state):
    """
    Build a stable signature of the active orders, left→right.
    Prefer an order id if present; fall back to meal+index.
    """
    orders = state.get("orders") or state.get("order_state") or []
    sig = []
    for i, o in enumerate(orders):
        oid = o.get("id") or o.get("order_id") or o.get("uid")
        meal = (o.get("meal") or o.get("name") or "").lower()
        sig.append(oid or f"{meal}#{i}")
    return tuple(sig)
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

    def _pick_leftmost_order_meal(self, state) -> str:
        """
        Returns the meal name for the *left-most* active order.
        Assumes state["orders"] is left→right. Falls back to '' if none.
        """
        orders = state.get("orders") or state.get("order_state") or []
        if not isinstance(orders, (list, tuple)) or not orders:
            return ""
        # common fields: `meal` or `name`
        m = (orders[0].get("meal") or orders[0].get("name") or "").lower()
        # normalize a couple common spellings
        m = m.replace(" ", "")
        return m

    def _reset_per_order_state(self):
        # Clear per-order placements/flags so next order starts clean
        for k in ("plate_spot_id", "added_to_plate", "allow_pick_plate_once"):
            if hasattr(BB, k):
                try: delattr(BB, k)
                except Exception: pass

    def initialise(self, state):
        super().initialise(state)
        if self._initialized:
            return
        self._initialized = True

        # Log recipes once for sanity
        try:
            print("\n[BTAgent] Loaded recipe_graph:")
            print(json.dumps(self.recipe_graph, indent=2))
            print("[BTAgent] End of recipe_graph\n")
        except Exception:
            pass

        # Build positions map
        counters = state.get("counters", [])
        service = next((c for c in counters if (c.get("type") or "").lower() == "servingwindow"), None)
        if service is None:
            raise RuntimeError("No serving window found")
        self.serving_id = service["id"]
        self.counter_positions = {c["id"]: np.array(c["pos"]) for c in counters}

        # Cache initial orders signature
        self.prev_orders_sig = _orders_signature(state)

        # Set initial active meal
        BB.active_meal = self._pick_leftmost_order_meal(state)
        BB.last_meal = None  # force initial build
        self._build_tree(state)

        # clear any stale scheduling / waits
        if hasattr(BB, "next_task"): del BB.next_task
        BB.meta_waiting = None
        BB.dumped_utensil_schema = False

    def _build_tree(self, state):
        self.tree = BehaviorTreeBuilder.build(
            state               = state,
            counter_positions   = self.counter_positions,
            id_map_module       = id_map_module
        )
        self.tree.setup(timeout=10)
        print(f"[BTAgent] ✅ Built BT for meal='{BB.active_meal}'")

    # ──────────────────────────────────────────────────────────────────────────
    # META_COOK polling (same robust logic as your auto version)
    # ──────────────────────────────────────────────────────────────────────────
    def _describe_station_equipment(self, state, station_id: str) -> str:
        ctr = _lookup_counter(state, station_id)
        if not ctr: return f"station={station_id} not found"
        occ = ctr.get("occupied_by")
        if not occ: return f"station={station_id} (empty)"
        parts = []
        ename = _extract_item_name(occ)
        if ename: parts.append(f"equip={ename}")
        r = occ.get("content_ready")
        rname = _extract_item_name(r) if isinstance(r, dict) else None
        if rname: parts.append(f"ready={rname}")
        cl = occ.get("content_list") or []
        tname = _top_name_from_content_list(cl)
        parts.append(f"content_top={tname!r}")
        at = occ.get("active_transition")
        if isinstance(at, dict):
            at_name = _extract_item_name(at.get("result"))
            parts.append(f"active={{sec:{at.get('seconds')}, result:{at_name!r}}}")
        return f"station={station_id} " + " ".join(parts)

    def _handle_meta_cook_poll(self, state) -> TaskStatus:
        if not getattr(BB, "meta_waiting", None):
            return TS_DONE
        expected = _norm_name(BB.meta_waiting.get("expected"))
        station_id = BB.meta_waiting.get("station_id")
        if not station_id:
            print("[BTAgent][META_COOK] ⚠️ missing station_id; clearing")
            BB.meta_waiting = None
            return TS_DONE

        ctr = _lookup_counter(state, station_id)
        if not ctr:
            print(f"[BTAgent][META_COOK] waiting… (station not found) :: {station_id}")
            return TaskStatus.IN_PROGRESS

        occ = ctr.get("occupied_by")
        if not occ:
            print(f"[BTAgent][META_COOK] waiting… (utensil absent) :: {station_id}")
            return TaskStatus.IN_PROGRESS

        equip = None
        if isinstance(occ, dict) and occ.get("category") in ("ItemCookingEquipment", "Item"):
            equip = occ
        elif isinstance(occ, list):
            for o in occ:
                if isinstance(o, dict) and o.get("category") in ("ItemCookingEquipment", "Item"):
                    equip = o; break
        if equip is None:
            print(f"[META_COOK][DEBUG] station={station_id} occupied_by={type(occ).__name__} -> {occ}")
            return TaskStatus.IN_PROGRESS

        cl = equip.get("content_list") or []
        cr = equip.get("content_ready")
        def _types_of(x):
            if isinstance(x, dict): return x.get("type") or x.get("name")
            if isinstance(x, list): return [(_i.get("type") or _i.get("name")) for _i in x]
            return None
        print(f"[META_COOK][DEBUG] station={station_id} equip.type={equip.get('type')} "
              f"content_list={_types_of(cl)} content_ready={_types_of(cr)}")
        print(f"[BTAgent][META_COOK] poll -> {self._describe_station_equipment(state, station_id)}")

        # burn/invalid guard
        if isinstance(cl, list) and cl and isinstance(cl[0], dict):
            top = _norm_name(_extract_item_name(cl[0]))
            if top in {"waste", "burnt"}:
                print("[BTAgent][META_COOK] ❌ item burned → FAILED")
                BB.meta_waiting = None
                return TaskStatus.FAILED

        # instant transitions
        rname = _norm_name(_extract_item_name(cr)) if isinstance(cr, dict) else None
        if rname and rname == expected:
            print(f"[BTAgent][META_COOK] ✅ content_ready == {expected} → DONE")
            BB.meta_waiting = None
            return TS_DONE

        # timed transitions: single result in content_list
        if isinstance(cl, list) and len(cl) == 1 and isinstance(cl[0], dict):
            t = _norm_name(_extract_item_name(cl[0]))
            if t == expected:
                print(f"[BTAgent][META_COOK] ✅ cooked via content_list -> {expected} → DONE")
                BB.meta_waiting = None
                return TS_DONE

        return TaskStatus.IN_PROGRESS

    # ──────────────────────────────────────────────────────────────────────────
    # Main scheduler
    # ──────────────────────────────────────────────────────────────────────────
    async def manage_tasks(self, state):
        # tick counter for debounce
        BB.tick = getattr(BB, "tick", 0) + 1
        BB.last_state = state

        # ─────────────────────────────────────────────────────────────────────────
        # Helper (local) : stable signature of orders left→right
        # ─────────────────────────────────────────────────────────────────────────
        def _orders_signature(st):
            orders = st.get("orders") or st.get("order_state") or []
            sig = []
            for i, o in enumerate(orders):
                oid = o.get("id") or o.get("order_id") or o.get("uid")
                meal = (o.get("meal") or o.get("name") or "").lower()
                sig.append(oid or f"{meal}#{i}")
            return tuple(sig)

        def _order_time_left(order: dict) -> float:
            """Use OrderManager.order_state(..) 'time_remaining' (seconds)."""
            try:
                return float(order.get("time_remaining", 0.0))
            except (TypeError, ValueError):
                return 0.0

        def _ingredient_count_for(meal: str, state: dict) -> int:
            """
            Ingredient counts used for tie-breaking.
            pizza = 3 baseline; 4 if a sausage dispenser exists in the current map.
            """
            m = _norm_name(meal)
            if m == "tomatosoup":   return 3
            if m == "onionsoup":    return 3
            if m == "salad":        return 2
            if m == "burger":       return 4
            if m == "chips":        return 1
            if m == "friedfish":    return 1
            if m == "fishandchips": return 2
            if m == "pizza":
                sausage_present = any(
                    ((c.get("type") or c.get("name") or "").lower() in
                     {"sausage_dispenser", "sausagedispenser", "sausage"})
                    for c in state.get("counters", [])
                )
                return 4 if sausage_present else 3
            return 3  # default for unknown/new meals

        def _pick_after_serve_with_50pct_rule(state: dict) -> str:
            """
            After serving: pick max time_remaining unless a candidate with >=50% of that time
            has strictly fewer ingredients; among those, choose fewest ingredients, then higher time,
            then left-most index.
            Returns normalized meal name or "" if none.
            """
            orders = state.get("orders") or state.get("order_state") or []
            if not orders:
                return ""

            # Collect (time, ingredients, idx, meal)
            rows = []
            for idx, o in enumerate(orders):
                meal = _norm_name(o.get("meal") or o.get("name"))
                if not meal:
                    continue
                tleft = _order_time_left(o)
                icost = _ingredient_count_for(meal, state)
                rows.append((tleft, icost, idx, meal))
            if not rows:
                return ""

            # 1) Find T_max and default (top-time, left-most)
            T_max = max(r[0] for r in rows)
            top_candidates = [r for r in rows if r[0] == T_max]
            # If multiple share T_max, choose the left-most as default
            default_t, default_ic, default_idx, default_meal = min(
                top_candidates, key=lambda r: r[2]
            )

            # 2) Find “fast-enough & fewer-ingredients” candidates
            threshold = 0.5 * T_max
            eligible = [
                r for r in rows
                if r[0] >= threshold and r[1] < default_ic
            ]

            if eligible:
                # fewest ingredients → highest time → left-most index
                eligible.sort(key=lambda r: (r[1], -r[0], r[2]))
                return eligible[0][3]

            # 3) Otherwise keep the default top-time choice
            return default_meal

        # ─────────────────────────────────────────────────────────────────────────
        # 0) META_COOK has absolute priority
        # ─────────────────────────────────────────────────────────────────────────
        if getattr(BB, "meta_waiting", None):
            status = self._handle_meta_cook_poll(state)
            if status == TaskStatus.IN_PROGRESS:
                return
            # if DONE/FAILED, allow new scheduling below

        # ─────────────────────────────────────────────────────────────────────────
        # Init on first call of new fields we use for multi-order continuity
        # ─────────────────────────────────────────────────────────────────────────
        if not hasattr(self, "prev_orders_sig"):
            self.prev_orders_sig = _orders_signature(state)
        if not hasattr(self, "_last_put_was_serve"):
            self._last_put_was_serve = False

        # Find serving window id (for final-serve edge detection)
        serving_id = None
        counters = state.get("counters", [])
        if isinstance(counters, list):
            for c in counters:
                if (c.get("type") or "").lower() == "servingwindow":
                    serving_id = c.get("id");
                    break

        # ─────────────────────────────────────────────────────────────────────────
        # A) If we served on the previous tick but orders UI hasn't updated yet,
        #    proactively reset & rebuild once (keeps agent from idling).
        #    Do this BEFORE inflight check so we can immediately start next order.
        # ─────────────────────────────────────────────────────────────────────────
        if self._last_put_was_serve:
            self._last_put_was_serve = False
            print("[BTAgent] ✅ Served last tick → proactive reset & rebuild")
            self._reset_per_order_state()
            BB.active_meal = _pick_after_serve_with_50pct_rule(state)
            self._build_tree(state)
            if not BB.active_meal:
                print("[BTAgent]    no active order after serve; idling")
                return

        # ─────────────────────────────────────────────────────────────────────────
        # B) If the orders list changed, update signature. Do NOT switch mid-order.
        #    Only (re)select if we currently have no active meal (just finished).
        # ─────────────────────────────────────────────────────────────────────────
        curr_sig = _orders_signature(state)
        if curr_sig != self.prev_orders_sig:
            self.prev_orders_sig = curr_sig
            if not getattr(BB, "active_meal", ""):
                candidate = _pick_after_serve_with_50pct_rule(state)
                if candidate:
                    BB.active_meal = candidate
                    self._build_tree(state)
                    return

        # ─────────────────────────────────────────────────────────────────────────
        # 1) If a task is inflight, wait
        # ─────────────────────────────────────────────────────────────────────────
        if self.current_task is not None:
            print("[BTAgent]    skipping, waiting on in-flight task")
            return

        # ─────────────────────────────────────────────────────────────────────────
        # 2) Recompute active meal from left-most order each tick (still keep this
        #    for the simple "meal switched" case; complements the signature check)
        # ─────────────────────────────────────────────────────────────────────────
        current_meal = self._pick_leftmost_order_meal(state)
        if current_meal != getattr(BB, "active_meal", ""):
            # Either new order arrived, or old one finished → rebuild
            print(f"[BTAgent] 🔁 Meal switched: {getattr(BB, 'active_meal', '')} → {current_meal}")
            BB.active_meal = current_meal
            self._reset_per_order_state()
            self._build_tree(state)

        # ─────────────────────────────────────────────────────────────────────────
        # 3) If no tree or no active orders, idle
        # ─────────────────────────────────────────────────────────────────────────
        if not getattr(self, "tree", None) or not current_meal:
            print("[BTAgent]    no active order; idling")
            return

        # ─────────────────────────────────────────────────────────────────────────
        # 4) Drive BT
        # ─────────────────────────────────────────────────────────────────────────
        print("[BTAgent]    ticking behavior tree")
        self.tree.tick()

        # ─────────────────────────────────────────────────────────────────────────
        # 5) Schedule emitted task
        # ─────────────────────────────────────────────────────────────────────────
        next_task = getattr(BB, "next_task", None)
        if not next_task:
            print("[BTAgent]    no new task, idling")
            return

        task_type, task_arg = next_task
        print(f"[BTAgent]    scheduling task: {task_type} -> {task_arg}")

        # Handle META_COOK as a *watcher*, not a scheduled Task
        if task_type == "META_COOK":
            required, station_id = task_arg
            BB.meta_waiting = {"expected": _norm_name(required), "station_id": station_id}
            print(f"[BTAgent][META_COOK] ⏳ waiting for '{_norm_name(required)}' at station {station_id}")
            # Optional: immediate debug poll
            _ = self._handle_meta_cook_poll(state)
            del BB.next_task
            return

        # Final-serve edge detection: remember if we are about to PUT to serving window
        if task_type == "PUT" and serving_id is not None and task_arg == serving_id:
            self._last_put_was_serve = True

        # Plate PUT debounce (prevents pick-back immediately after place)
        if task_type == "PUT":
            target_id = task_arg

            if getattr(BB, "plate_spot_id", None) and target_id == BB.plate_spot_id:
                last_tick = getattr(BB, "last_plate_put_tick", -999)
                if (BB.tick - last_tick) <= 1 and not getattr(BB, "allow_pick_plate_once", False):
                    print("[BTAgent] ⏭️ debounced: skipping immediate second PUT on plate.")
                    del BB.next_task
                    return
                BB.last_plate_put_tick = BB.tick
                if getattr(BB, "allow_pick_plate_once", False):
                    BB.allow_pick_plate_once = False
            basket_home = getattr(BB, "tool_home", {}).get("basket")  # where the basket lives (the fryer id)
            if basket_home and target_id == basket_home:
                last_tick = getattr(BB, "last_fryer_put_tick", -999)
                if (BB.tick - last_tick) <= 1:
                    print("[BTAgent] ⏭️ debounced: skipping immediate second PUT on fryer.")
                    del BB.next_task
                    return
                BB.last_fryer_put_tick = BB.tick

        self.set_current_task(Task(task_type, task_args=task_arg, task_status=TaskStatus.SCHEDULED))
        del BB.next_task
        print("[BTAgent] ← manage_tasks done\n")


if __name__ == "__main__":
    run_agent_from_args(BTAgent)