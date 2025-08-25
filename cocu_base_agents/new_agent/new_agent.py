# new_agent.py
import json
import numpy as np
import py_trees
from py_trees import blackboard

from cooperative_cuisine.base_agent.agent_task import Task, TaskStatus
from cooperative_cuisine.base_agent.base_agent import BaseAgent, run_agent_from_args
from cocu_base_agents.new_agent.behavior_tree import BehaviorTreeBuilder, BB

# compatibility shim for differing enum names
try:
    TS_DONE = TaskStatus.DONE
except AttributeError:
    TS_DONE = getattr(TaskStatus, "FINISHED",
              getattr(TaskStatus, "SUCCESS",
              getattr(TaskStatus, "COMPLETED", None)))
    if TS_DONE is None:
        # last resort: treat any non-IN_PROGRESS as "done" upstream; here we pick FINISHED-like fallback
        TS_DONE = TaskStatus.SCHEDULED  # harmless placeholder; manage_tasks only checks IN_PROGRESS explicitly

# ──────────────────────────────────────────────────────────────────────────────
# <<< NEW: normalizers & extractors used by META_COOK poll
# ──────────────────────────────────────────────────────────────────────────────
def _norm_name(s: str | None) -> str:
    """Normalize things like 'CookedPatty', 'cooked_patty', 'cookedpatty' -> 'cookedpatty'."""
    if s is None:
        return ""
    if isinstance(s, str) and s.strip().lower() == "none":
        return ""  # tolerate 'None' (string) from some serializers
    return str(s).lower().replace("_", "")

def _extract_item_name(obj) -> str | None:
    """Best-effort item name from various shapes (dicts/strings)."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return None if obj.strip().lower() == "none" else obj
    if isinstance(obj, dict):
        n = obj.get("name") or obj.get("type")
        if n:
            return n
        it = obj.get("item")
        if isinstance(it, dict):
            return it.get("name") or it.get("type")
        rs = obj.get("result")
        if isinstance(rs, dict):
            return rs.get("name") or rs.get("type")
        ii = obj.get("item_info")
        if isinstance(ii, dict):
            return ii.get("name") or ii.get("type")
    return None

def _extract_top_name_from_content_list(clist) -> str | None:
    """Return name for topmost content_list entry, if any."""
    if not isinstance(clist, (list, tuple)) or not clist:
        return None
    return _extract_item_name(clist[0])

def _lookup_counter(state: dict, counter_id: str) -> dict | None:
    """Find a counter dict by id in the current state payload."""
    counters = state.get("counters")
    if isinstance(counters, dict):
        return counters.get(counter_id)
    if isinstance(counters, list):
        for c in counters:
            if c.get("id") == counter_id:
                return c
    return None
# ──────────────────────────────────────────────────────────────────────────────


class BTAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False

    def initialise(self, state):
        super().initialise(state)
        if self._initialized:
            return
        self._initialized = True

        # 1) Print recipes
        print("\n[BTAgent] Loaded recipe_graph:")
        print(json.dumps(self.recipe_graph, indent=2))
        print("[BTAgent] End of recipe_graph\n")

        # 2) Locate service counter
        counters = state.get("counters", [])
        print(f"[BTAgent] Available counters: {[c['id'] for c in counters]}")
        service = next(
            (c for c in counters if c.get("type", "").lower() == "servingwindow"),
            None
        )
        if service is None:
            raise RuntimeError(
                f"[BTAgent] ❌ Could not find a service counter among: "
                f"{[c.get('type') for c in counters]}"
            )
        service_id = service["id"]
        print(f"[BTAgent] ✅ Using service counter: {service_id}")

        # 2b) Build counter_positions map
        counter_positions = {c["id"]: np.array(c["pos"]) for c in counters}

        # 3) Build & setup the behavior tree
        self.tree = BehaviorTreeBuilder.build(
            recipe_graphs       = self.recipe_graph,
            service_counter_id  = service_id,
            counters            = counters,
            counter_positions   = counter_positions,
            player_key          = self.own_player_id
        )
        self.tree.setup(timeout=15)

        # clear any stale task and meta wait
        if hasattr(BB, "next_task"):
            del BB.next_task
        # <<< NEW: clear meta-wait state on (re)initialise
        BB.meta_waiting = None
        BB.dumped_utensil_schema = False

    # ──────────────────────────────────────────────────────────────────────────
    # <<< NEW: richer station snapshot for logs used by META_COOK poller
    # ──────────────────────────────────────────────────────────────────────────
    def _describe_station_equipment(self, state, station_id: str) -> str:
        ctr = _lookup_counter(state, station_id)
        if not ctr:
            return f"station={station_id} not found"
        equip = ctr.get("occupied_by")
        if not equip:
            return f"station={station_id} (empty)"
        parts = []
        ename = _extract_item_name(equip)
        if ename:
            parts.append(f"equip={ename}")
        ready = equip.get("content_ready")
        rname = _extract_item_name(ready)
        if rname:
            parts.append(f"ready={rname}")
        cl = equip.get("content_list") or []
        tname = _extract_top_name_from_content_list(cl)
        parts.append(f"content_top={tname!r}")
        at = equip.get("active_transition")
        if isinstance(at, dict):
            at_name = _extract_item_name(at.get("result"))
            parts.append(f"active={{sec:{at.get('seconds')}, result:{at_name!r}}}")
        return f"station={station_id} " + " ".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # <<< NEW: META_COOK poller — robust completion checks + one-time schema dump
    # ──────────────────────────────────────────────────────────────────────────
    def _handle_meta_cook_poll(self, state) -> TaskStatus:
        if not getattr(BB, "meta_waiting", None):
            return TS_DONE

        expected = _norm_name(BB.meta_waiting.get("expected"))
        station_id = BB.meta_waiting.get("station_id")
        if not station_id:
            print("[BTAgent][META_COOK] ⚠️ missing station_id in meta_waiting; clearing")
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

        # ---------- Resolve equipment dict robustly (dict or list) ----------
        equip_dict = None
        if isinstance(occ, dict):
            # Accept both ItemCookingEquipment and Item (some servers tag equipment as Item)
            if occ.get("category") in ("ItemCookingEquipment", "Item"):
                equip_dict = occ
        elif isinstance(occ, list):
            for _o in occ:
                if isinstance(_o, dict) and _o.get("category") in ("ItemCookingEquipment", "Item"):
                    equip_dict = _o
                    break

        if equip_dict is None:
            # Unknown shape — dump once and keep waiting
            print(f"[META_COOK][DEBUG] station={station_id} occupied_by={type(occ).__name__} -> {occ}")
            return TaskStatus.IN_PROGRESS

        # ---------- Compact DEBUG probe ----------
        def _types_of(x):
            if isinstance(x, dict):
                return x.get("type") or x.get("name")
            if isinstance(x, list):
                return [(_i.get("type") or _i.get("name")) for _i in x]
            return None

        cl = equip_dict.get("content_list") or []
        cr = equip_dict.get("content_ready")
        print(
            f"[META_COOK][DEBUG] station={station_id} equip.type={equip_dict.get('type')} "
            f"content_list={_types_of(cl)} content_ready={_types_of(cr)}"
        )

        # Snapshot line (optional but handy)
        print(f"[BTAgent][META_COOK] poll -> {self._describe_station_equipment(state, station_id)}")

        # ---------- Burn/invalid guard by name (JSON doesn't expose item_category) ----------
        top_name = None
        if isinstance(cl, list) and cl and isinstance(cl[0], dict):
            top_name = _norm_name(_extract_item_name(cl[0]))
            if top_name in {"waste", "burnt"}:
                print("[BTAgent][META_COOK] ❌ item burned (waste/burnt) → FAILED")
                BB.meta_waiting = None
                return TaskStatus.FAILED

        # ---------- Instant transitions: content_ready ----------
        rname = _norm_name(_extract_item_name(cr)) if isinstance(cr, dict) else None
        if rname and rname == expected:
            print(f"[BTAgent][META_COOK] ✅ content_ready == {expected} → DONE")
            BB.meta_waiting = None
            return TS_DONE

        # ---------- Timed transitions: content_list replaced by [result] ----------
        if isinstance(cl, list) and len(cl) == 1 and isinstance(cl[0], dict):
            if top_name == expected:
                print(f"[BTAgent][META_COOK] ✅ cooked via content_list[0]='{top_name}' → DONE")
                BB.meta_waiting = None
                return TS_DONE

        # ---------- Unknown/empty shape once ----------
        if not top_name and not rname and not getattr(BB, "dumped_utensil_schema", False):
            print("[BTAgent][META_COOK][DEBUG] Unknown utensil payload shape, dumping full structure once:")
            try:
                import pprint
                pprint.pprint(equip_dict, width=120, compact=True)
            except Exception:
                print(str(equip_dict))
            BB.dumped_utensil_schema = True

        return TaskStatus.IN_PROGRESS

    async def manage_tasks(self, state):
        # --- tick counter for simple debouncing ---
        BB.tick = getattr(BB, "tick", 0) + 1

        # 1) META_COOK has absolute priority: poll first and short-circuit if still waiting
        if getattr(BB, "meta_waiting", None):
            status = self._handle_meta_cook_poll(state)
            if status == TaskStatus.IN_PROGRESS:
                # we're still waiting; do NOT tick the tree or schedule anything else
                return
        print(f"[BTAgent] → manage_tasks tick; current_task={self.current_task}")
        # If we’re in a META_COOK wait, poll it first (non-blocking)
        if getattr(BB, "meta_waiting", None):
            status = self._handle_meta_cook_poll(state)  # <<< NEW
            print(f"[BTAgent]    META_COOK wait status = {status}")
            if status == TaskStatus.DONE or status == TaskStatus.FAILED:
                # Clear any current_task related to META_COOK, if one exists
                self.set_current_task(None)
            print("[BTAgent] ← manage_tasks done (meta-wait)")
            return

        if self.current_task is not None:
            print("[BTAgent]    skipping, waiting on in-flight task")
            return
        if not getattr(self, "tree", None):
            print("[BTAgent]    tree not ready yet")
            return

        print("[BTAgent]    ticking behavior tree")
        self.tree.tick()

        next_task = getattr(BB, "next_task", None)
        if next_task:
            task_type, task_arg = next_task
            print(f"[BTAgent]    scheduling task: {task_type} -> {task_arg}")

            # Arm META_COOK watcher and stop scheduling anything else this tick
            if task_type == "META_COOK":
                required, station_id = task_arg
                # normalize once here
                BB.meta_waiting = {"expected": _norm_name(required), "station_id": station_id}
                print(f"[BTAgent][META_COOK] ⏳ waiting for '{_norm_name(required)}' at station {station_id}")

                # (optional) start a soft timer if you added one elsewhere
                try:
                    import time
                    BB.meta_started_at = time.time()
                except Exception:
                    pass

                # do one immediate poll so you see the DEBUG line on this tick
                _ = self._handle_meta_cook_poll(state)
                # IMPORTANT: do NOT schedule this as a Task, and do NOT fall through
                return
            elif task_type == "PUT":
                target_id = task_arg

                # track last plate PUT
                if getattr(BB, "plate_spot_id", None) and target_id == BB.plate_spot_id:
                    last_tick = getattr(BB, "last_plate_put_tick", -999)
                    # if we just touched the plate last tick and we're NOT explicitly allowed to pick it, skip
                    if (BB.tick - last_tick) <= 1 and not getattr(BB, "allow_pick_plate_once", False):
                        print("[BTAgent] ⏭️ debounced: skipping immediate second PUT on plate to avoid pick-back.")
                        return
                    # record this plate PUT; we'll use it to detect accidental double-taps
                    BB.last_plate_put_tick = BB.tick
                    # if we *were* allowed to pick plate, consume the one-shot permission
                    if getattr(BB, "allow_pick_plate_once", False):
                        BB.allow_pick_plate_once = False
                self.set_current_task(Task(task_type, task_args=task_arg, task_status=TaskStatus.SCHEDULED))
            else:
                self.set_current_task(Task(task_type, task_args=task_arg, task_status=TaskStatus.SCHEDULED))

            del BB.next_task
        else:
            print("[BTAgent]    no new task, idling")

        print("[BTAgent] ← manage_tasks done\n")


if __name__ == "__main__":
    run_agent_from_args(BTAgent)
