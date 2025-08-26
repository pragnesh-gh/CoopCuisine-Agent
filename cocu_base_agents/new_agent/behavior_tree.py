import numpy as np
import py_trees
from py_trees.decorators import Retry
from py_trees import blackboard

# ──────────────────────────────────────────────────────────────────────────────
# Blackboard (shared scratchpad between leaves and the agent loop)
# ──────────────────────────────────────────────────────────────────────────────
BB = blackboard.Blackboard()
# BB fields we use:
# - next_task:        ("GOTO"/"PUT"/"INTERACT"/"META_COOK", payload)
# - plate_spot_id:    counter id where we staged the plate
# - cb_index:         round-robin index for cutting boards
# - tool_home:        { "pan": stove_id, "pot": stove_id }
# - added_to_plate:   set of tokens already placed (for gating cold additions)
# - active_meal:      current target meal name (e.g., "burgers", "tomatosoup")
# - last_meal:        last built meal name (used by the agent to rebuild)
# - allow_pick_plate_once: bool one-shot gate to avoid immediate double PUT

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _norm(s: str | None) -> str:
    return (s or "").lower().replace("_", "")

def _lookup_counter(state: dict, counter_id: str) -> dict | None:
    cs = state.get("counters")
    if isinstance(cs, dict):
        return cs.get(counter_id)
    if isinstance(cs, list):
        for c in cs:
            if c.get("id") == counter_id:
                return c
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Leaf behaviours — emit exactly one Task into BB.next_task
# ──────────────────────────────────────────────────────────────────────────────
class GoTo(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, target_pos):
        super().__init__(name); self.target_pos = target_pos
    def update(self):
        print(f"[BT][GoTo] → scheduling GOTO {self.target_pos}")
        BB.next_task = ("GOTO", np.array(self.target_pos))
        return py_trees.common.Status.SUCCESS

class Put(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, target_id: str):
        super().__init__(name); self.target_id = target_id
    def update(self):
        print(f"[BT][Put] → scheduling PUT {self.target_id}")
        BB.next_task = ("PUT", self.target_id)
        return py_trees.common.Status.SUCCESS

class Interact(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, counter_id: str):
        super().__init__(name); self.counter_id = counter_id
    def update(self):
        print(f"[BT][Interact] → scheduling INTERACT {self.counter_id}")
        BB.next_task = ("INTERACT", self.counter_id)
        return py_trees.common.Status.SUCCESS

class MetaCook(py_trees.behaviour.Behaviour):
    """Ask BaseAgent to wait for a cooked product to appear at station_id."""
    def __init__(self, name: str, required_product: str, station_id: str):
        super().__init__(name); self.required_product = required_product; self.station_id = station_id
    def update(self):
        print(f"[BT][MetaCook] → scheduling META_COOK station={self.station_id} wait_for={self.required_product}")
        BB.next_task = ("META_COOK", (_norm(self.required_product), self.station_id))
        return py_trees.common.Status.SUCCESS

class MarkPlateAddition(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = _norm(token)
    def update(self):
        if not hasattr(BB, "added_to_plate"):
            BB.added_to_plate = set()
        BB.added_to_plate.add(self.token)
        print(f"[BT][Mark] → recorded on plate: {self.token}")
        return py_trees.common.Status.SUCCESS

class IfTokenMissing(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = _norm(token)
    def update(self):
        present = self.token in getattr(BB, "added_to_plate", set())
        return py_trees.common.Status.FAILURE if present else py_trees.common.Status.SUCCESS

# OneShot wrapper: on first SUCCESS, report RUNNING to consume a tick
class OneShot(py_trees.decorators.Decorator):
    def __init__(self, name: str, child: py_trees.behaviour.Behaviour):
        super().__init__(name=name, child=child)
        self.triggered = False
    def update(self):
        self.decorated.tick()
        status = self.decorated.status
        if not self.triggered and status == py_trees.common.Status.SUCCESS:
            self.triggered = True
            return py_trees.common.Status.RUNNING
        return status

# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────
class BehaviorTreeBuilder:
    """
    Hand-crafted BT with 3 branches: TomatoSoup / Salad / Burger.

    We assume:
    - `id_map.ID_MAP` provides explicit IDs for dispensers, boards, serving window,
      and the *specific* stove used for pan vs pot.
    - `counter_positions` is a mapping id -> np.array([x,y]).
    - `state["orders"]` lists active orders left→right; we take the left-most.
    """
    @staticmethod
    def build(state: dict,
              counter_positions: dict,
              id_map_module) -> py_trees.trees.BehaviourTree:
        from importlib import reload
        id_map = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}

        def pos_of(cid: str):
            return counter_positions.get(cid)

        # index counters by type for fallbacks (keeps your original “free plate spot” behavior)
        counters = state.get("counters", [])
        by_type = {}
        for c in counters:
            ctype = (c.get("type") or "").lower()
            by_type.setdefault(ctype, []).append(c)

        def first_id_of(t: str) -> str | None:
            xs = by_type.get(t.lower(), [])
            return xs[0]["id"] if xs else None

        # ——— Pull important IDs (prefer id_map, fallback to type search) ———
        plate_src = id_map.get("plate_dispenser") or first_id_of("platedispenser") or first_id_of("plate")
        serving   = id_map.get("serving_window") or first_id_of("servingwindow")
        cutting_boards = list(id_map.get("cutting_boards", [])) or [c["id"] for c in by_type.get("cuttingboard", [])]

        tomato_disp  = id_map.get("tomato_dispenser")  or first_id_of("tomatodispenser")  or first_id_of("tomato_dispenser")
        lettuce_disp = id_map.get("lettuce_dispenser") or first_id_of("lettucedispenser") or first_id_of("lettuce_dispenser")
        bun_disp     = id_map.get("bun_dispenser")     or first_id_of("bundispenser")     or first_id_of("bun_dispenser")
        meat_disp    = id_map.get("meat_dispenser")    or first_id_of("meatdispenser")    or first_id_of("meat_dispenser")

        stove_pan = id_map.get("stove_pan")  # REQUIRED for burger patty
        stove_pot = id_map.get("stove_pot")  # REQUIRED for tomato soup

        # fallback: if specific stove not provided, use first stove
        stove_pan = stove_pan or (by_type.get("stove", [{}])[0].get("id") if by_type.get("stove") else None)
        stove_pot = stove_pot or stove_pan

        # maintain round-robin over cutting boards
        if not cutting_boards:
            print("[BTBuilder] ⚠️ No cutting boards found")
        BB.cb_index = 0
        def pick_board():
            if not cutting_boards: return None
            cid = cutting_boards[BB.cb_index % len(cutting_boards)]
            BB.cb_index += 1
            return cid

        # ——— stage a plate on a free Counter spot (preserve your original behavior) ———
        def ensure_plate_staged(seq) -> bool:
            if getattr(BB, "plate_spot_id", None):
                return True
            # pick any plain counter as staging spot
            counter_spots = [c["id"] for c in by_type.get("counter", [])]
            spot = (id_map.get("plate_staging_counter") or (counter_spots[0] if counter_spots else None))
            if not plate_src or not spot:
                print(f"[BTBuilder] ⚠️ Missing plate source ({plate_src}) or staging spot ({spot})")
                return False
            BB.plate_spot_id = spot
            print(f"[BTBuilder] Staging plate at counter: {BB.plate_spot_id}")
            # pick plate → drop on spot
            _add_goto(seq, plate_src); _add_put(seq, plate_src)
            _add_goto(seq, spot);      _add_put(seq, spot)
            return True

        # ——— tiny DSL for adding leaves with Retry + OneShot ———
        def _add_goto(seq, cid):
            p = pos_of(cid)
            if p is None:
                print(f"[BTBuilder] ⚠️ Missing position for {cid}; skipping GOTO"); return
            seq.add_child(OneShot(f"Action_Goto_{cid}",
                                  Retry(f"Retry_Goto_{cid}", GoTo(f"Goto_{cid}", p), 3)))
        def _add_put(seq, cid):
            seq.add_child(OneShot(f"Action_Put_{cid}",
                                  Retry(f"Retry_Put_{cid}", Put(f"Put_{cid}", cid), 3)))
        def _add_interact(seq, cid):
            seq.add_child(OneShot(f"Action_Interact_{cid}",
                                  Retry(f"Retry_Interact_{cid}", Interact(f"Interact_{cid}", cid), 3)))
        def _add_cook_wait(seq, required, station_id):
            seq.add_child(OneShot(f"Action_MetaCook_{station_id}",
                                  Retry(f"Retry_MetaCook_{station_id}", MetaCook(f"Cook_{station_id}", required, station_id), 3)))
        def _mark(seq, token):
            seq.add_child(OneShot(f"Action_Mark_{token}", MarkPlateAddition(f"Mark_{token}", token)))

        # ——— selectors for meals: we’ll add *one* sequence depending on active order ———
        root = py_trees.composites.Selector(name="Root", memory=False)

        meal = _norm(getattr(BB, "active_meal", ""))  # set by agent based on left-most order
        print(f"[BTBuilder] Building for meal: {meal!r}")

        # Always ensure we track plate additions per order
        BB.added_to_plate = set()

        # =============== TOMATO SOUP BRANCH ===============
        if meal == "tomatosoup":
            seq = py_trees.composites.Sequence(name="TomatoSoup", memory=True)
            ensure_plate_staged(seq)

            # 3× (tomato → board → chop → pick → pot at stove_pot)
            for i in range(3):
                board = pick_board()
                if not (tomato_disp and board and stove_pot):
                    print("[BTBuilder] ⚠️ Soup missing tomato/board/stove_pot id")
                    break
                _add_goto(seq, tomato_disp);
                _add_put(seq, tomato_disp)  # pick tomato from dispenser
                _add_goto(seq, board);
                _add_put(seq, board)  # place tomato on cutting board
                _add_interact(seq, board)  # chop tomato
                _add_put(seq, board)  # pick up chopped tomato
                _add_goto(seq, stove_pot);
                _add_put(seq, stove_pot)  # drop chopped tomato into pot (on stove)

            # Start cooking and wait for soup
            if stove_pot:
                _add_interact(seq, stove_pot)  # start cooking (interact stove)
                _add_cook_wait(seq, "tomatosoup", stove_pot)  # wait until soup ready (META_COOK)
                # Pour to plate
                _add_put(seq, stove_pot)  # pick pot (with soup)
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # pour onto plate
                    _mark(seq, "tomatosoup")  # mark soup added
                _add_goto(seq, stove_pot);
                _add_put(seq, stove_pot)  # return empty pot

            # Serve the plate
            if getattr(BB, "plate_spot_id", None) and serving in counter_positions:
                _add_goto(seq, BB.plate_spot_id);
                _add_put(seq, BB.plate_spot_id)  # pick plate
                _add_goto(seq, serving);
                _add_put(seq, serving)  # deliver to window
            root.add_child(seq)


        # =============== SALAD BRANCH ===============
        elif meal == "salad":
            seq = py_trees.composites.Sequence(name="Salad", memory=True)
            ensure_plate_staged(seq)

            # Tomato → chop → pick → plate
            board_t = pick_board()
            if tomato_disp and board_t:
                _add_goto(seq, tomato_disp);
                _add_put(seq, tomato_disp)  # pick tomato from dispenser
                _add_goto(seq, board_t);
                _add_put(seq, board_t)  # place tomato on cutting board
                _add_interact(seq, board_t)  # chop tomato
                _add_put(seq, board_t)  # pick up chopped tomato
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # place on plate
                    _mark(seq, "choppedtomato")  # mark recorded

            # Lettuce → chop → pick → plate
            board_l = pick_board()
            if lettuce_disp and board_l:
                _add_goto(seq, lettuce_disp);
                _add_put(seq, lettuce_disp)  # pick lettuce from dispenser
                _add_goto(seq, board_l);
                _add_put(seq, board_l)  # place lettuce on cutting board
                _add_interact(seq, board_l)  # chop lettuce
                _add_put(seq, board_l)  # pick up chopped lettuce
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # place on plate
                    _mark(seq, "choppedlettuce")  # mark recorded

            # serve
            if getattr(BB, "plate_spot_id", None) and serving in counter_positions:
                _add_goto(seq, BB.plate_spot_id); _add_put(seq, BB.plate_spot_id)
                _add_goto(seq, serving);          _add_put(seq, serving)
            root.add_child(seq)

        # =============== BURGER BRANCH ===============
        elif meal == "burger":
            seq = py_trees.composites.Sequence(name="Burger", memory=True)
            ensure_plate_staged(seq)

            # Lettuce → board → chop → pick → plate
            board_l = pick_board()
            if lettuce_disp and board_l:
                _add_goto(seq, lettuce_disp);
                _add_put(seq, lettuce_disp)  # pick lettuce
                _add_goto(seq, board_l);
                _add_put(seq, board_l)  # place on board
                _add_interact(seq, board_l)  # chop lettuce
                _add_put(seq, board_l)  # pick chopped lettuce
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # place on plate
                    _mark(seq, "choppedlettuce")

            # Tomato → board → chop → pick → plate
            board_t = pick_board()
            if tomato_disp and board_t:
                _add_goto(seq, tomato_disp);
                _add_put(seq, tomato_disp)  # pick tomato
                _add_goto(seq, board_t);
                _add_put(seq, board_t)  # place on board
                _add_interact(seq, board_t)  # chop tomato
                _add_put(seq, board_t)  # pick chopped tomato
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # place on plate
                    _mark(seq, "choppedtomato")

            # Meat → board → chop → pick → pan → cook → pour → return pan
            board_m = pick_board()
            if meat_disp and board_m and stove_pan:
                _add_goto(seq, meat_disp);
                _add_put(seq, meat_disp)  # pick meat
                _add_goto(seq, board_m);
                _add_put(seq, board_m)  # place on board
                _add_interact(seq, board_m)  # chop → raw patty
                _add_put(seq, board_m)  # pick raw patty
                _add_goto(seq, stove_pan);
                _add_put(seq, stove_pan)  # put patty into pan on stove
                _add_interact(seq, stove_pan)  # start cooking
                _add_cook_wait(seq, "cookedpatty", stove_pan)  # wait till cooked
                _add_put(seq, stove_pan)  # pick pan (with patty)
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # pour patty on plate
                    _mark(seq, "cookedpatty")
                _add_goto(seq, stove_pan);
                _add_put(seq, stove_pan)  # return empty pan

            # Bun LAST → pick → plate
            if bun_disp and getattr(BB, "plate_spot_id", None):
                # NOTE: bun step stays INSIDE the same sequence, AFTER all other toppings.
                _add_goto(seq, bun_disp);
                _add_put(seq, bun_disp)  # pick bun
                _add_goto(seq, BB.plate_spot_id);
                _add_put(seq, BB.plate_spot_id)  # place bun on plate
                _mark(seq, "bun")  # mark bun added

            # Serve the plate
            if getattr(BB, "plate_spot_id", None) and serving in counter_positions:
                _add_goto(seq, BB.plate_spot_id);
                _add_put(seq, BB.plate_spot_id)  # pick plate
                _add_goto(seq, serving);
                _add_put(seq, serving)  # deliver to window

            # Attach ONLY the main seq to root (no separate bun child!)
            root.add_child(seq)



        else:
            # No active meal → idle leaf so we don't spam logs
            class Idle(py_trees.behaviour.Behaviour):
                def update(self): return py_trees.common.Status.RUNNING
            root.add_child(Idle("Idle_NoActiveMeal"))

        return py_trees.trees.BehaviourTree(root)
