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

# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────
class BTAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False

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
        self.counter_positions = {c["id"]: np.array(c["pos"]) for c in counters}

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

        # 0) META_COOK has absolute priority
        if getattr(BB, "meta_waiting", None):
            status = self._handle_meta_cook_poll(state)
            if status == TaskStatus.IN_PROGRESS:
                return
            # if DONE/FAILED, allow new scheduling below

        # 1) If a task is inflight, wait
        if self.current_task is not None:
            print("[BTAgent]    skipping, waiting on in-flight task")
            return

        # 2) Recompute active meal from left-most order each tick
        current_meal = self._pick_leftmost_order_meal(state)
        if current_meal != getattr(BB, "active_meal", ""):
            # Either new order arrived, or old one finished → rebuild
            print(f"[BTAgent] 🔁 Meal switched: {getattr(BB,'active_meal','')} → {current_meal}")
            BB.active_meal = current_meal
            self._reset_per_order_state()
            self._build_tree(state)

        # 3) If no tree or no active orders, idle
        if not getattr(self, "tree", None) or not current_meal:
            print("[BTAgent]    no active order; idling")
            return

        # 4) Drive BT
        print("[BTAgent]    ticking behavior tree")
        self.tree.tick()

        # 5) Schedule emitted task
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

        self.set_current_task(Task(task_type, task_args=task_arg, task_status=TaskStatus.SCHEDULED))
        del BB.next_task
        print("[BTAgent] ← manage_tasks done\n")


if __name__ == "__main__":
    run_agent_from_args(BTAgent)
