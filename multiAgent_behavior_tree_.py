# Declaration:
# This program is the result of combined work between a human developer 
# and ChatGPT. The AI contributed helpers, initial structures, and 
# formatting improvements. The human developer contributed the 
# high-level design, behavior tree logic, and integration of ideas. 
# The final implementation is a product of human–AI collaboration.

import numpy as np
import py_trees
from py_trees.decorators import Retry
from py_trees import blackboard
import os
try:
    from cocu_base_agents.new_agent.coordinator import Coordinator
    _COORD = Coordinator()
except Exception:
    _COORD = None


# ──────────────────────────────────────────────────────────────────────────────
# Blackboard (shared scratchpad between leaves and the agent loop)
# Think of this like a simple shared note that all leaves can read/write.
# ──────────────────────────────────────────────────────────────────────────────
BB = blackboard.Blackboard()
# BB fields we use:
# - next_task:        ("GOTO"/"PUT"/"INTERACT"/"META_COOK", payload) → exactly one per tick
# - plate_spot_id:    counter id where we staged the plate
# - cb_index:         round-robin index for cutting boards
# - tool_home:        { "pan": stove_id, "pot": stove_id }
# - added_to_plate:   set of tokens already placed (for gating cold additions)
# - active_meal:      current target meal name (e.g., "burger", "tomatosoup")
# - last_meal:        last built meal name (used by the agent to rebuild)
# - allow_pick_plate_once: bool one-shot gate to avoid immediate double PUT


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (small utility functions; no engine calls here)
# ──────────────────────────────────────────────────────────────────────────────
def _norm(s: str | None) -> str:
    """Normalize name: lowercase and remove underscores so comparisons are stable."""
    return (s or "").lower().replace("_", "")

def _lookup_counter(state: dict, counter_id: str) -> dict | None:
    """Find a counter dict by its id in state (works for list or dict layouts)."""
    cs = state.get("counters")
    if isinstance(cs, dict):
        return cs.get(counter_id)
    if isinstance(cs, list):
        for c in cs:
            if c.get("id") == counter_id:
                return c
    return None

def _yield_tick(seq, tag: str):
    """Consume exactly one game tick to let the environment settle (no engine task)."""
    seq.add_child(OneShot(f"Action_Yield_{tag}",
                          Retry(f"Retry_Yield_{tag}", TickYield(f"Yield_{tag}"), 1)))

def _has_token(name: str) -> bool:
    """
    Check if we've already marked this token as on the plate.
    Useful to guard serving until we truly plated the item.
    """
    s = getattr(BB, "added_to_plate", set())
    return _norm(name) in s


def _wait_empty(seq, counter_id: str, tag: str = ""):
    """Add a short wait that completes when a counter is empty (nothing on it)."""
    name = f"empty_{tag}" if tag else "empty"
    seq.add_child(
        OneShot(
            f"Action_WaitEmpty_{name}",
            Retry(f"Retry_WaitEmpty_{name}", WaitCounterEmpty(counter_id), 1),
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# Leaf behaviours — each leaf schedules at most ONE engine task via BB.next_task
# (GOTO, PUT, INTERACT, META_COOK) or consumes a tick with no task.
# ──────────────────────────────────────────────────────────────────────────────
class GoTo(py_trees.behaviour.Behaviour):
    """Schedule a GOTO to a specific (x, y) position."""
    def __init__(self, name: str, target_pos):
        super().__init__(name); self.target_pos = target_pos
    def update(self):
        print(f"[BT][GoTo] → scheduling GOTO {self.target_pos}")
        BB.next_task = ("GOTO", np.array(self.target_pos))
        return py_trees.common.Status.SUCCESS

class Put(py_trees.behaviour.Behaviour):
    """Schedule a PUT to a target counter id (place/pick context-sensitive)."""
    def __init__(self, name: str, target_id: str):
        super().__init__(name); self.target_id = target_id
    def update(self):
        print(f"[BT][Put] → scheduling PUT {self.target_id}")
        BB.next_task = ("PUT", self.target_id)
        return py_trees.common.Status.SUCCESS

class Interact(py_trees.behaviour.Behaviour):
    """Schedule an INTERACT at a counter id (e.g., chop/start cook)."""
    def __init__(self, name: str, counter_id: str):
        super().__init__(name); self.counter_id = counter_id
    def update(self):
        print(f"[BT][Interact] → scheduling INTERACT {self.counter_id}")
        BB.next_task = ("INTERACT", self.counter_id)
        return py_trees.common.Status.SUCCESS

class MetaCook(py_trees.behaviour.Behaviour):
    """Ask BaseAgent to watch a station until a specific cooked product appears."""
    def __init__(self, name: str, required_product: str, station_id: str):
        super().__init__(name); self.required_product = required_product; self.station_id = station_id
    def update(self):
        print(f"[BT][MetaCook] → scheduling META_COOK station={self.station_id} wait_for={self.required_product}")
        BB.next_task = ("META_COOK", (_norm(self.required_product), self.station_id))
        return py_trees.common.Status.SUCCESS

class MarkPlateAddition(py_trees.behaviour.Behaviour):
    """Record that an item/token is now on the current plate."""
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = _norm(token)
    def update(self):
        if not hasattr(BB, "added_to_plate"):
            BB.added_to_plate = set()
        BB.added_to_plate.add(self.token)
        print(f"[BT][Mark] → recorded on plate: {self.token}")
        return py_trees.common.Status.SUCCESS

class IfTokenMissing(py_trees.behaviour.Behaviour):
    """Branch guard: succeed only if the token is NOT on the plate yet."""
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = _norm(token)
    def update(self):
        present = self.token in getattr(BB, "added_to_plate", set())
        return py_trees.common.Status.FAILURE if present else py_trees.common.Status.SUCCESS

class TickYield(py_trees.behaviour.Behaviour):
    """Consume one tick to let the environment update (no engine task is sent)."""
    def update(self):
        # No task to schedule; just succeed.
        return py_trees.common.Status.SUCCESS

class WaitCounterHas(py_trees.behaviour.Behaviour):
    """
    Passive wait: succeeds only when the given counter/equipment
    contains an item whose name includes `expected_substr` (case-insensitive).
    Returns RUNNING while waiting. Schedules NO engine task.
    """
    def __init__(self, counter_id: str, expected_substr: str):
        super().__init__(name=f"WaitHas[{expected_substr}]@{counter_id}")
        self.counter_id = counter_id
        self.expected = expected_substr.lower()

    def update(self):
        st = getattr(BB, "last_state", None)
        if not st:
            # fail-open if we somehow don't have state
            return py_trees.common.Status.SUCCESS

        # find the counter
        ctr = _lookup_counter(st, self.counter_id)
        if not ctr:
            return py_trees.common.Status.RUNNING

        occ = ctr.get("occupied_by")
        names = []

        # If the counter holds a tool (board/pot/pan/basket/peel), check its content_list
        if isinstance(occ, dict):
            tool_type = (occ.get("type") or occ.get("name") or "").lower()
            content = occ.get("content_list") or []
            if isinstance(content, list):
                for it in content:
                    if isinstance(it, dict):
                        nm = (it.get("name") or it.get("type") or "").lower()
                        if nm:
                            names.append(nm)

        # Optional: could also check a plain "top" item if the state exposes it.

        # Succeed only when we see the expected item name
        ok = any(self.expected in nm for nm in names)
        return py_trees.common.Status.SUCCESS if ok else py_trees.common.Status.RUNNING

class WaitCounterEmpty(py_trees.behaviour.Behaviour):
    """
    Passive wait: succeed when the target counter has no occupied_by (truly empty).
    Returns RUNNING otherwise. No engine task is scheduled.
    """
    def __init__(self, counter_id: str):
        super().__init__(name=f"WaitEmpty@{counter_id}")
        self.counter_id = counter_id

    def update(self):
        st = getattr(BB, "last_state", None)
        if not st:
            return py_trees.common.Status.RUNNING
        ctr = _lookup_counter(st, self.counter_id)
        if not ctr:
            return py_trees.common.Status.RUNNING
        occ = ctr.get("occupied_by")
        return py_trees.common.Status.SUCCESS if not occ else py_trees.common.Status.RUNNING

class ReturnBasket(py_trees.behaviour.Behaviour):
    """
    PUT the basket back into the deep fryer exactly once, then wait until the fryer
    reports it has a Basket inside. Never issues PUT twice (prevents drop/pick loop).
    """
    def __init__(self, fryer_id: str):
        super().__init__(name=f"ReturnBasket@{fryer_id}")
        self.fryer_id = fryer_id
        self.issued_put = False

    def update(self):
        st = getattr(BB, "last_state", None)
        if not st:
            return py_trees.common.Status.RUNNING

        # Helper: find fryer counter + check if its occupied_by is Basket
        ctr = _lookup_counter(st, self.fryer_id)
        if not ctr:
            return py_trees.common.Status.RUNNING
        occ = ctr.get("occupied_by")
        occ_name = (occ.get("type") or occ.get("name") or "").lower() if isinstance(occ, dict) else ""

        # If we already see Basket inside fryer, we're done.
        if occ_name == "basket":
            return py_trees.common.Status.SUCCESS

        # Otherwise, if we haven't issued the PUT yet, do it once and then just poll.
        if not self.issued_put:
            # schedule one engine action: PUT(fryer)
            BB.next_task = ("PUT", self.fryer_id)
            self.issued_put = True
            # Return RUNNING so the BT will come back next tick to confirm
            return py_trees.common.Status.RUNNING

        # PUT already issued once; keep polling until fryer shows Basket inside.
        return py_trees.common.Status.RUNNING


def _add_return_basket(seq, fryer_id: str):
    """Convenience: add a single-shot 'return basket' step (PUT once + confirm)."""
    seq.add_child(ReturnBasket(fryer_id))

class WaitToken(py_trees.behaviour.Behaviour):
    """Succeed only when BB.added_to_plate contains the token."""
    def __init__(self, token: str):
        super().__init__(name=f"WaitToken[{token}]")
        self.token = token

    def update(self):
        return (py_trees.common.Status.SUCCESS
                if _has_token(self.token)
                else py_trees.common.Status.RUNNING)

def _wait_token(seq, token: str):
    """Inline helper to add a WaitToken leaf to a sequence."""
    seq.add_child(WaitToken(token))


# OneShot wrapper: on first SUCCESS, report RUNNING to consume a tick
class OneShot(py_trees.decorators.Decorator):
    """
    Tick the child until it first returns SUCCESS, then:
      - on that tick, return RUNNING (this paces the BT to one engine action per tick)
      - on all later ticks, return SUCCESS without ticking the child again.
    This prevents re-scheduling the same engine task (e.g., PUT) on the next tick.
    """
    def __init__(self, name: str, child: py_trees.behaviour.Behaviour):
        super().__init__(name=name, child=child)
        self.triggered = False

    def update(self):
        # If a previous schedule attempt for this node was dropped by the agent
        # (e.g., lock contention), allow this OneShot to re-fire.
        inv = getattr(BB, "invalidate_oneshots", set())

        if self.triggered and self.name in inv:
                    inv.discard(self.name)
                    BB.invalidate_oneshots = inv
                    self.triggered = False  # re-arm

          # If already triggered and not invalidated, don't re-tick the child.
        if self.triggered:
            return py_trees.common.Status.SUCCESS

        # First time: tick child. If SUCCESS, mark as triggered and
        # return RUNNING once to pace. Next tick we'll return SUCCESS.
        self.decorated.tick()
        status = self.decorated.status
        if status == py_trees.common.Status.SUCCESS:
            self.triggered = True
            # Let the agent know which OneShot scheduled this tick so it can
            # invalidate us if lock gating drops the task.
            BB.last_oneshot = self.name
            return py_trees.common.Status.RUNNING
        return status


# ──────────────────────────────────────────────────────────────────────────────
# Builder (creates a full Behavior Tree for the current active meal)
# ──────────────────────────────────────────────────────────────────────────────
class BehaviorTreeBuilder:
    """
    Hand-crafted BT with multiple recipe branches:
      - TomatoSoup, OnionSoup, Salad, Burger, FriedFish, Chips, Pizza, FishAndChips

    Assumptions:
    - id_map.ID_MAP provides explicit IDs for dispensers, boards, serving window,
      and the stove slots used for pan vs pot (when available).
    - counter_positions is a mapping id -> np.array([x, y]).
    - state["orders"] lists active orders left→right; the agent picks a meal and sets BB.active_meal.
    """
    @staticmethod
    def build(state: dict,
              counter_positions: dict,
              id_map_module) -> py_trees.trees.BehaviourTree:
        from importlib import reload
        id_map = reload(id_map_module).ID_MAP if hasattr(id_map_module, "ID_MAP") else {}

        def pos_of(cid: str):
            """Lookup position vector for a counter id."""
            return counter_positions.get(cid)

        # Index counters by type for quick fallbacks (keeps original “free plate spot” behavior)
        counters = state.get("counters", [])
        by_type = {}
        for c in counters:
            ctype = (c.get("type") or "").lower()
            by_type.setdefault(ctype, []).append(c)

        def first_id_of(t: str) -> str | None:
            """Return the id of the first counter of given type, if any."""
            xs = by_type.get(t.lower(), [])
            return xs[0]["id"] if xs else None

        def _name_of(d):
            if not isinstance(d, dict): return ""
            return (d.get("type") or d.get("name") or "").lower()

        def _detect_fryer_and_basket(state):
            """Auto-detect fryer id and basket id (if basket is parked inside fryer)."""
            fryer_id, basket_id = None, None
            for c in state.get("counters", []):
                if _name_of(c) == "deepfryer":
                    fryer_id = c.get("id")
                    occ = c.get("occupied_by")
                    if _name_of(occ) == "basket":
                        basket_id = occ.get("id") or basket_id
            return fryer_id, basket_id

        def _detect_oven_and_peel(state):
            """Auto-detect oven id and where the peel is parked."""
            oven_id, peel_home_id = None, None
            for c in state.get("counters", []):
                if _name_of(c) == "oven" and oven_id is None:
                    oven_id = c.get("id")
                # Peel might be on a normal counter with occupied_by.type == "Peel"
                occ = c.get("occupied_by")
                if _name_of(occ) == "peel" and peel_home_id is None:
                    peel_home_id = c.get("id")
            return oven_id, peel_home_id

        def _detect_stoves_from_state(st):
            """
            Find (stove_pan_id, stove_pot_id) by reading stoves:
              - if occupied_by == 'Pan' → stove_pan
              - if occupied_by == 'Pot' → stove_pot
            """
            def _name_of(x):
                if not isinstance(x, dict): return ""
                return (x.get("type") or x.get("name") or "").lower()

            stove_pan_id, stove_pot_id = None, None
            for c in st.get("counters", []):
                if (c.get("type") or "").lower() != "stove":
                    continue
                occ = c.get("occupied_by")
                n = _name_of(occ)
                if n == "pan" and stove_pan_id is None:
                    stove_pan_id = c.get("id")
                if n == "pot" and stove_pot_id is None:
                    stove_pot_id = c.get("id")
            return stove_pan_id, stove_pot_id

        # ——— Pull important IDs (prefer id_map, fallback to type search) ———
        plate_src = id_map.get("plate_dispenser") or first_id_of("platedispenser") or first_id_of("plate")
        serving   = id_map.get("serving_window") or first_id_of("servingwindow")
        cutting_boards = list(id_map.get("cutting_boards", [])) or [c["id"] for c in by_type.get("cuttingboard", [])]

        tomato_disp  = id_map.get("tomato_dispenser")  or first_id_of("tomatodispenser")  or first_id_of("tomato_dispenser")
        lettuce_disp = id_map.get("lettuce_dispenser") or first_id_of("lettucedispenser") or first_id_of("lettuce_dispenser")
        bun_disp     = id_map.get("bun_dispenser")     or first_id_of("bundispenser")     or first_id_of("bun_dispenser")
        meat_disp    = id_map.get("meat_dispenser")    or first_id_of("meatdispenser")    or first_id_of("meat_dispenser")
        onion_disp = id_map.get("onion_dispenser") or id_map.get("ONION")  or first_id_of("oniondispenser") or first_id_of("onion_dispenser")
        potato_disp = id_map.get("potato_dispenser") or id_map.get("POTATO") or first_id_of("potatodispenser") or first_id_of("potato_dispenser")
        fish_disp = id_map.get("fish_dispenser") or id_map.get("FISH") or first_id_of("fishdispenser") or first_id_of("fish_dispenser")
        dough_disp = id_map.get("dough_dispenser") or id_map.get("DOUGH") or first_id_of("doughdispenser") or first_id_of("dough_dispenser")
        cheese_disp = id_map.get("cheese_dispenser") or id_map.get("CHEESE") or first_id_of("cheesedispenser") or first_id_of("cheese_dispenser")
        sausage_disp = id_map.get("sausage_dispenser") or id_map.get("SAUSAGE") or first_id_of("sausagedispenser") or first_id_of("sausage_dispenser")

        # Stoves
        stove_pan = id_map.get("stove_pan") or id_map.get("PAN_STOVE") or id_map.get("pan_stove")  # BURGER
        stove_pot = id_map.get("stove_pot") or id_map.get("POT_STOVE") or id_map.get("pot_stove")  # TOMATO/ONION SOUP

        det_pan, det_pot = _detect_stoves_from_state(state)
        if not stove_pan: stove_pan = det_pan
        if not stove_pot: stove_pot = det_pot

        if not stove_pan:
            stove_pan = (by_type.get("stove", [{}])[0].get("id") if by_type.get("stove") else None)
            if stove_pan:
                print(f"[BTBuilder] ⚠️ stove_pan not provided; falling back to {stove_pan}")
        if not stove_pot:
            stoves = by_type.get("stove", [])
            stove_pot = (stoves[1]["id"] if len(stoves) > 1 else stove_pan)
            print(f"[BTBuilder] ⚠️ stove_pot not provided; using {stove_pot}")

        print(f"[BTBuilder] Using stoves: pan={stove_pan}, pot={stove_pot}")

        # --- Deep fryer & basket (auto-detect if not in id_map) ---
        deep_fryer = id_map.get("deep_fryer") or id_map.get("DEEP_FRYER")
        basket = id_map.get("basket") or id_map.get("BASKET")
        det_fryer, det_basket = _detect_fryer_and_basket(state)
        if not deep_fryer: deep_fryer = det_fryer
        if not basket:     basket = det_basket
        if not deep_fryer:
            fry = next((c for c in counters if _name_of(c) == "deepfryer"), None)
            deep_fryer = fry.get("id") if fry else None
        print(f"[BTBuilder] Using fryer={deep_fryer}, basket={basket}")

        # --- Oven & Peel (peel 'home' counter where we park it) ---
        oven = id_map.get("oven") or id_map.get("OVEN")
        peel = id_map.get("peel") or id_map.get("PEEL")
        det_oven, det_peel = _detect_oven_and_peel(state)
        if not oven: oven = det_oven
        if not peel: peel = det_peel
        if not oven:
            ov = next((c for c in counters if _name_of(c) == "oven"), None)
            oven = ov.get("id") if ov else None
        print(f"[BTBuilder] Using oven={oven}, peel={peel}")

        # maintain round-robin over cutting boards (plus simple ownership checks)
        if not cutting_boards:
            print("[BTBuilder] ⚠️ No cutting boards found")
        BB.cb_index = 0

        def pick_board():
            """
            Choose a cutting board smartly:
              - Prefer free boards
              - Prefer near our staged plate (agents naturally diverge)
              - Respect Coordinator locks if available (avoid contention)
              - As a last resort, take the preferred board; agent lock will park/wait
            """
            if not cutting_boards:
                return None
            owner = getattr(BB, "agent_owner", None) or f"{os.getenv('AGENT_ID', 'A')}-{os.getpid()}"
            # Prefer the board closest to our plate staging spot
            ref_id = getattr(BB, "plate_spot_id", None)

            def _is_busy(board_id: str) -> bool:
                ctr = next((c for c in counters if c.get("id") == board_id), None)
                occ = bool(ctr and ctr.get("occupied_by"))
                locker = None
                if _COORD:
                    try:
                        locker = _COORD.who_owns(board_id)
                    except Exception:
                        locker = None
                return occ or (locker and locker != owner)

            if ref_id:
                ref_pos = pos_of(ref_id)
                ordered = sorted(cutting_boards, key=lambda b: np.linalg.norm(pos_of(b) - ref_pos))
            else:
                # fallback: A/B preference or stable hash parity (for very early calls)
                aid = getattr(BB, "agent_id", None) or ("A" if (hash(owner) & 1) == 0 else "B")
                pref_idx = 0 if str(aid).upper().startswith("A") else 1
                ordered = [cutting_boards[pref_idx % len(cutting_boards)]]
                if len(cutting_boards) > 1:
                    ordered.append(cutting_boards[(pref_idx + 1) % len(cutting_boards)])
            for b in ordered:
                if not _is_busy(b):
                    return b
            return ordered[0]  # both busy → stick to preferred; agent-side lock will park/wait

        # ——— Stage a plate on a free Counter spot (also try to reserve it) ———
        def ensure_plate_staged(seq) -> bool:
            """
            What this does:
              1) If we already have a plate spot and it's still free, reuse it.
              2) Else, pick a free plain counter (not equipment, not dispensers).
              3) Try to reserve that spot via Coordinator (so agents don’t clash).
              4) Fetch a plate from dispenser and place it at the chosen spot.
            Returns True if a spot is staged; False if we can’t find/reserve a free tile.
            """
            def _is_free_plain_counter(c):
                ctype = (c.get("type") or "").lower()
                if ctype != "counter":
                    return False
                if c.get("occupied_by"):
                    return False
                return True

            # If we already have a spot, verify it's still free (for THIS order)
            if getattr(BB, "plate_spot_id", None):
                ctr = _lookup_counter(state, BB.plate_spot_id)
                if ctr and _is_free_plain_counter(ctr):
                    # still good
                    pass
                else:
                    BB.plate_spot_id = None

            # If no valid spot, pick a fresh one
            if not getattr(BB, "plate_spot_id", None):
                exclude_ids = {
                                  serving, plate_src, stove_pan, stove_pot, tomato_disp, lettuce_disp, bun_disp,
                                  meat_disp
                              } | set(cutting_boards)

                free_spots = [
                    c for c in counters
                    if _is_free_plain_counter(c) and c.get("id") not in exclude_ids
                ]
                if not free_spots:
                    print("[BTBuilder] ❌ No free plain Counter found to stage a plate")
                    return False

                src_pos = pos_of(plate_src) if plate_src else None
                if src_pos is not None:
                    free_spots.sort(key=lambda c: np.linalg.norm(pos_of(c["id"]) - src_pos))

                # TRY to reserve a spot; if reservation fails, try next
                chosen = None

                for cand in free_spots:
                    cid = cand["id"]
                    owner = getattr(BB, "agent_owner", None) or f"{os.getenv('AGENT_ID', 'A')}-{os.getpid()}"
                    if _COORD is None or _COORD.reserve_counter(cid, owner):
                        chosen = cid
                        break
                if chosen is None:
                    print("[BTBuilder] ❌ Could not reserve any plain Counter for plate staging")
                    return False

                BB.plate_spot_id = chosen
                print(f"[BTBuilder] Staging plate at reserved counter: {BB.plate_spot_id}")

                if not plate_src:
                    print("[BTBuilder] ❌ plate_dispenser not found; cannot stage plate")
                    return False
                _add_goto(seq, plate_src); _add_put(seq, plate_src)   # get a plate
                _add_goto(seq, chosen);    _add_put(seq, chosen)      # place at spot

            return True


        # ——— tiny DSL for adding leaves with Retry + OneShot (keeps code compact) ———
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

        def _has_token(name: str) -> bool:
            """Return True if we've marked this token as placed on the plate this order."""
            s = getattr(BB, "added_to_plate", set())
            return _norm(name) in s

        def _wait_has(seq, counter_id: str, expected_substr: str, tag: str = ""):
            """Add a short wait that succeeds when a counter shows an expected item name."""
            name = expected_substr if not tag else f"{expected_substr}_{tag}"
            seq.add_child(
                OneShot(
                    f"Action_WaitHas_{name}",
                    Retry(f"Retry_WaitHas_{name}", WaitCounterHas(counter_id, expected_substr), 1),
                )
            )
        def _mark(seq, token):
            """Record that a token (ingredient/dish) was placed on the plate."""
            seq.add_child(OneShot(f"Action_Mark_{token}", MarkPlateAddition(f"Mark_{token}", token)))

        # Root: we attach exactly ONE recipe sequence based on BB.active_meal
        root = py_trees.composites.Selector(name="Root", memory=False)

        meal = _norm(getattr(BB, "active_meal", ""))  # set by agent based on claimed order
        print(f"[BTBuilder] Building for meal: {meal!r}")

        # Always reset plate-additions per order
        BB.added_to_plate = set()

        # =============== TOMATO SOUP BRANCH ===============
        # Step-by-step (TomatoSoup):
        # 1) Stage a plate at a free counter.
        # 2) Do 3x loop:
        #    - Pick tomato → place on board → chop → pick chopped tomato → drop into pot on stove.
        # 3) Start cooking at pot, wait until 'tomatosoup' is ready (META_COOK).
        # 4) Pick pot → pour soup onto staged plate → mark 'tomatosoup'.
        # 5) Return empty pot to stove.
        # 6) Pick plated soup → deliver to serving window.
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
                _yield_tick(seq, f"tomato_{board}")  # let board register chopped item
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


        # =============== ONION SOUP BRANCH ===============
        # Step-by-step (OnionSoup):
        # 1) Stage a plate.
        # 2) Do 3x loop:
        #    - Pick onion → board → chop → pick chopped onion → drop into pot.
        # 3) Start cooking, wait for 'onionsoup'.
        # 4) Pick pot → pour onto plate → mark 'onionsoup' → return pot.
        # 5) Serve the plated soup.
        elif meal == "onionsoup":
            seq = py_trees.composites.Sequence(name="OnionSoup", memory=True)
            ensure_plate_staged(seq)

            # 3× (onion → board → chop → pick → pot at stove_pot)
            for i in range(3):
                board = pick_board()
                if not (onion_disp and board and stove_pot):
                    print("[BTBuilder] ⚠️ Soup missing onion/board/stove_pot id")
                    break
                _add_goto(seq, onion_disp)
                _add_put(seq, onion_disp)  # pick onion from dispenser
                _add_goto(seq, board);
                _add_put(seq, board)  # place onion on cutting board
                _add_interact(seq, board)  # chop onion
                _yield_tick(seq, f"onion_{board}")
                _add_put(seq, board)  # pick up chopped onion
                _add_goto(seq, stove_pot);
                _add_put(seq, stove_pot)  # drop chopped onion into pot (on stove)

            # Start cooking and wait for soup
            if stove_pot:
                _add_interact(seq, stove_pot)  # start cooking (interact stove)
                _add_cook_wait(seq, "onionsoup", stove_pot)  # wait until onion soup ready (META_COOK)
                # Pour to plate
                _add_put(seq, stove_pot)  # pick pot (with onion soup)
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # pour onto plate
                    _mark(seq, "onionsoup")  # mark onion soup added
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
        # Step-by-step (Salad):
        # 1) Stage a plate.
        # 2) Tomato: pick → board → chop → pick chopped → place on plate → mark 'choppedtomato'.
        # 3) Lettuce: pick → board → chop → pick chopped → place on plate → mark 'choppedlettuce'.
        # 4) Pick the plate → deliver to serving window.
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
                _yield_tick(seq, f"tomato_{board_t}")  # let board register item
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
                _yield_tick(seq, f"lettuce_{board_l}")
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
        # Step-by-step (Burger):
        # 1) Stage a plate.
        # 2) Lettuce: pick → board → chop → pick chopped → place on plate → mark.
        # 3) Tomato: pick → board → chop → pick chopped → place on plate → mark.
        # 4) Meat: pick → board → chop to raw patty → pick patty → pan on stove → start cooking →
        #          wait cooked → pick pan → pour patty on plate → mark → return pan.
        # 5) Bun last: pick bun → place on plate → mark 'bun'.
        # 6) Pick the finished burger plate → deliver to serving window.
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
                _yield_tick(seq, f"lettuce_{board_l}")
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
                _yield_tick(seq, f"tomato_{board_t}")
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
                _yield_tick(seq, f"meat_{board_m}")
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


        # =============== FRIED FISH BRANCH ===============
        # Step-by-step (FriedFish):
        # 1) Stage a plate.
        # 2) Fish: pick → board → wait 'fish' appears → chop → wait 'choppedfish' →
        #          pick chopped → go to fryer → PUT (drop into basket).
        # 3) Start frying → wait until 'friedfish' → pick basket from fryer.
        # 4) Pour onto plate → mark 'friedfish'.
        # 5) Return basket once (PUT once, then confirm basket is inside) → no toggle loops.
        # 6) Pick plate → deliver to serving window.
        elif meal == "friedfish":
            seq = py_trees.composites.Sequence(name="FriedFish", memory=True)
            ensure_plate_staged(seq)

            # Fish → board → (wait raw) → chop → (wait chopped) → pick → fryer(basket) → cook → pour → return basket
            board_f = pick_board()
            if fish_disp and board_f and deep_fryer:
                _add_goto(seq, fish_disp);
                _add_put(seq, fish_disp)  # pick fish
                _add_goto(seq, board_f);
                _add_put(seq, board_f)  # place on cutting board

                # Ensure raw fish present before chopping (prevents "Interaction not progressing")
                if "_wait_has" in globals():
                    _wait_has(seq, board_f, "fish", tag=f"{board_f}")

                _add_interact(seq, board_f)  # chop → ChoppedFish

                # Ensure ChoppedFish appears before pickup (prevents retry-looking double PUT)
                if "_wait_has" in globals():
                    _wait_has(seq, board_f, "choppedfish", tag=f"{board_f}")

                _add_put(seq, board_f)  # pick chopped fish

                # IMPORTANT: basket usually has no map position—use the FRYER counter.
                _add_goto(seq, deep_fryer);
                _add_put(seq, deep_fryer)  # drop chopped fish into basket (via fryer)

                # Fry → wait → pick → pour
                _add_interact(seq, deep_fryer)  # start frying
                _add_cook_wait(seq, "friedfish", deep_fryer)  # wait until fried fish
                _add_put(seq, deep_fryer)  # take basket (with fried fish)

                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # pour fried fish onto plate
                    _mark(seq, "friedfish")

                # Go back to fryer and return basket EXACTLY ONCE (no toggle loop)
                _add_goto(seq, deep_fryer)
                _add_return_basket(seq, deep_fryer)  # PUT once + poll until basket is inside

            else:
                print("[BTBuilder][FriedFish] ⚠️ skipping:",
                      "fish_disp=", fish_disp, "board=", board_f, "fryer=", deep_fryer)

            # Serve the plate — runtime guard so it isn't stripped at build time
            if getattr(BB, "plate_spot_id", None) and serving in counter_positions:
                if "_wait_token" in globals():
                    _wait_token(seq, "friedfish")
                _add_goto(seq, BB.plate_spot_id);
                _add_put(seq, BB.plate_spot_id)  # pick plate
                _add_goto(seq, serving);
                _add_put(seq, serving)  # deliver to window

            root.add_child(seq)


        # =============== CHIPS BRANCH ===============
        # Step-by-step (Chips):
        # 1) Stage a plate.
        # 2) Potato: pick → board → wait 'potato' appears → chop to 'rawchips' → wait 'rawchips' →
        #            pick → go to fryer → PUT (into basket).
        # 3) Start frying → wait until 'chips' → pick basket.
        # 4) Pour onto plate → mark 'chips'.
        # 5) Return basket once (no toggle loop).
        # 6) Wait for 'chips' token → pick plate → deliver.
        elif meal == "chips":
            seq = py_trees.composites.Sequence(name="Chips", memory=True)
            ensure_plate_staged(seq)

            board_p = pick_board()
            if potato_disp and board_p and deep_fryer:
                # Potato → board
                _add_goto(seq, potato_disp);
                _add_put(seq, potato_disp)  # pick potato
                _add_goto(seq, board_p);
                _add_put(seq, board_p)  # place on board

                # Ensure raw potato present before chopping (prevents "Interaction not progressing")
                if "_wait_has" in globals():
                    _wait_has(seq, board_p, "potato", tag=f"{board_p}")

                _add_interact(seq, board_p)  # chop → RawChips

                # Ensure RawChips exists before pickup (prevents retry-looking double PUT)
                if "_wait_has" in globals():
                    _wait_has(seq, board_p, "rawchips", tag=f"{board_p}")

                _add_put(seq, board_p)  # pick RawChips

                # IMPORTANT: Basket usually has no map position. Drop into basket via the FRYER counter.
                _add_goto(seq, deep_fryer);
                _add_put(seq, deep_fryer)  # put RawChips into fryer (basket)

                # Fry → wait → pick → pour
                _add_interact(seq, deep_fryer)  # start frying
                _add_cook_wait(seq, "chips", deep_fryer)  # wait until 'Chips'
                _add_put(seq, deep_fryer)  # pick basket (with chips)
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # pour onto plate
                    _mark(seq, "chips")

                # Go back to fryer and return basket EXACTLY ONCE (no toggle loop)
                _add_goto(seq, deep_fryer)
                _add_return_basket(seq, deep_fryer)  # PUT once + poll until basket is inside

            else:
                print("[BTBuilder][Chips] ⚠️ skipping: potato_disp:", potato_disp,
                      "board:", board_p, "fryer:", deep_fryer)

            # Serve ONLY if we actually plated 'chips'
            if getattr(BB, "plate_spot_id", None) and serving in counter_positions:
                _wait_token(seq, "chips")
                _add_goto(seq, BB.plate_spot_id);
                _add_put(seq, BB.plate_spot_id)
                _add_goto(seq, serving);
                _add_put(seq, serving)

            root.add_child(seq)


        # =============== PIZZA BRANCH ===============
        # Step-by-step (Pizza):
        # 1) Stage a plate.
        # 2) Dough: pick → board → wait 'dough' → interact to 'pizzabase' → wait 'pizzabase' → pick → put on PEEL.
        # 3) Tomato: pick → board → wait 'tomato' → chop → wait 'choppedtomato' → pick → add to PEEL.
        # 4) Cheese: pick → board → wait 'cheese' → grate → wait 'gratedcheese' → pick → add to PEEL.
        # 5) (Optional) Sausage: pick → board → wait 'sausage' → chop → wait 'choppedsausage' → pick → add to PEEL.
        # 6) Load pizza into oven:
        #    - If peel is already at the oven tile, PUT once to insert; else pick peel → go oven → PUT to insert.
        #    - Wait until oven shows 'peel' inserted (safety).
        # 7) Start baking → wait until 'pizza' is ready (META_COOK).
        # 8) Take peel (with pizza) → slide onto plate → mark 'pizza'.
        # 9) Return empty peel to oven, confirm it's back.
        # 10) Wait for 'pizza' token → pick plate → deliver to serving window.
        elif meal == "pizza":
            seq = py_trees.composites.Sequence(name="Pizza", memory=True)
            ensure_plate_staged(seq)

            # Dough → board → (wait raw) → interact → (wait base) → pick → PEEL
            board_d = pick_board()
            if dough_disp and board_d and peel and oven:
                _add_goto(seq, dough_disp);
                _add_put(seq, dough_disp)  # pick dough
                _add_goto(seq, board_d);
                _add_put(seq, board_d)  # place on board

                if "_wait_has" in globals():
                    _wait_has(seq, board_d, "dough", tag=f"{board_d}")  # ensure raw dough is on board

                _add_interact(seq, board_d)  # process → PizzaBase

                if "_wait_has" in globals():
                    _wait_has(seq, board_d, "pizzabase", tag=f"{board_d}")  # wait until base exists

                _add_put(seq, board_d)  # pick PizzaBase
                _add_goto(seq, peel);
                _add_put(seq, peel)  # place base on peel

                # Tomato → board → (wait raw) → chop → (wait chopped) → pick → PEEL
                board_t = pick_board()
                if tomato_disp and board_t:
                    _add_goto(seq, tomato_disp);
                    _add_put(seq, tomato_disp)
                    _add_goto(seq, board_t);
                    _add_put(seq, board_t)

                    if "_wait_has" in globals():
                        _wait_has(seq, board_t, "tomato", tag=f"{board_t}")

                    _add_interact(seq, board_t)  # chop → ChoppedTomato

                    if "_wait_has" in globals():
                        _wait_has(seq, board_t, "choppedtomato", tag=f"{board_t}")

                    _add_put(seq, board_t)
                    _add_goto(seq, peel);
                    _add_put(seq, peel)  # add to peel

                # Cheese → board → (wait raw) → grate → (wait grated) → pick → PEEL
                board_c = pick_board()
                if cheese_disp and board_c:
                    _add_goto(seq, cheese_disp);
                    _add_put(seq, cheese_disp)
                    _add_goto(seq, board_c);
                    _add_put(seq, board_c)

                    if "_wait_has" in globals():
                        _wait_has(seq, board_c, "cheese", tag=f"{board_c}")

                    _add_interact(seq, board_c)  # grate → GratedCheese

                    if "_wait_has" in globals():
                        _wait_has(seq, board_c, "gratedcheese", tag=f"{board_c}")

                    _add_put(seq, board_c)
                    _add_goto(seq, peel);
                    _add_put(seq, peel)  # add to peel

                # Sausage → board → (wait raw) → chop → (wait chopped) → pick → PEEL
                board_s = pick_board()
                if sausage_disp and board_s:
                    _add_goto(seq, sausage_disp);
                    _add_put(seq, sausage_disp)
                    _add_goto(seq, board_s);
                    _add_put(seq, board_s)

                    if "_wait_has" in globals():
                        _wait_has(seq, board_s, "sausage", tag=f"{board_s}")

                    _add_interact(seq, board_s)  # chop → ChoppedSausage

                    if "_wait_has" in globals():
                        _wait_has(seq, board_s, "choppedsausage", tag=f"{board_s}")

                    _add_put(seq, board_s)
                    _add_goto(seq, peel);
                    _add_put(seq, peel)  # add to peel

                    _yield_tick(seq, "peel_debounce")

                # Insert into oven (single PUT when peel is already at oven tile)
                if peel == oven:
                    _yield_tick(seq, "peel_at_oven_debounce")
                    _add_goto(seq, oven)
                    _add_put(seq, oven)  # single PUT to load pizza into oven
                    if "_wait_has" in globals():
                        _wait_has(seq, oven, "peel", tag="inserted")
                else:
                    _add_goto(seq, peel)
                    _add_put(seq, peel)  # pick peel (with pizza)
                    _add_goto(seq, oven)
                    _add_put(seq, oven)  # insert peel into oven
                    if "_wait_has" in globals():
                        _wait_has(seq, oven, "peel", tag="inserted")

                _add_interact(seq, oven)  # start baking
                _add_cook_wait(seq, "pizza", oven)  # wait until pizza ready

                _add_put(seq, oven)  # take peel (with pizza)
                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # slide to plate
                    _mark(seq, "pizza")

                # Return the peel once and confirm it's back at the oven
                _add_goto(seq, oven)
                _add_put(seq, oven)  # return empty peel
                if "_wait_has" in globals():
                    _wait_has(seq, oven, "peel", tag="returned")

            else:
                print("[BTBuilder][Pizza] ⚠️ skipping: dough_disp:", dough_disp,
                      "board:", board_d, "peel:", peel, "oven:", oven)

            # Serve — runtime guarded (added at build time, gated by token at run time)
            if getattr(BB, "plate_spot_id", None) and serving in counter_positions:
                if "_wait_token" in globals():
                    _wait_token(seq, "pizza")
                _add_goto(seq, BB.plate_spot_id);
                _add_put(seq, BB.plate_spot_id)
                _add_goto(seq, serving);
                _add_put(seq, serving)

            root.add_child(seq)


        # =============== FISH & CHIPS BRANCH ===============
        # Step-by-step (FishAndChips):
        # 1) Stage a plate.
        # 2) Fried Fish sub-pipeline (same robustness as FriedFish):
        #    - Fish: pick → board → wait raw → chop → wait chopped → pick → fryer PUT.
        #    - Start frying → wait 'friedfish' → pick basket → pour to plate → mark 'friedfish' → return basket once.
        # 3) Chips sub-pipeline (same robustness as Chips):
        #    - Potato: pick → board → wait raw → chop to 'rawchips' → wait → pick → fryer PUT.
        #    - Start frying → wait 'chips' → pick basket → pour to plate → mark 'chips' → return basket once.
        # 4) Wait until BOTH tokens ('friedfish' and 'chips') are marked.
        # 5) Pick plate → deliver to serving window.
        elif meal == "fishandchips":
            seq = py_trees.composites.Sequence(name="FishAndChips", memory=True)
            ensure_plate_staged(seq)

            # --- Fried Fish pipeline (same as FriedFish robust) ---
            board_f = pick_board()
            if fish_disp and board_f and deep_fryer:
                _add_goto(seq, fish_disp);
                _add_put(seq, fish_disp)  # pick fish
                _add_goto(seq, board_f);
                _add_put(seq, board_f)  # place on board

                # Wait until the raw fish is actually on the board
                if "_wait_has" in globals():
                    _wait_has(seq, board_f, "fish", tag=f"{board_f}")

                _add_interact(seq, board_f)  # chop → ChoppedFish

                # Wait until chopped fish is present before picking it up
                if "_wait_has" in globals():
                    _wait_has(seq, board_f, "choppedfish", tag=f"{board_f}")

                _add_put(seq, board_f)  # pick chopped fish

                # Use the FRYER counter (basket typically has no map position)
                _add_goto(seq, deep_fryer);
                _add_put(seq, deep_fryer)  # drop chopped fish into basket
                _add_interact(seq, deep_fryer)  # start frying
                _add_cook_wait(seq, "friedfish", deep_fryer)  # wait until fried
                _add_put(seq, deep_fryer)  # take basket (with fried fish)

                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # pour fried fish
                    _mark(seq, "friedfish")

                # Return basket exactly once (no toggle loop)
                _add_goto(seq, deep_fryer)
                _add_return_basket(seq, deep_fryer)

            else:
                print("[BTBuilder][Fish&Chips] ⚠️ skipping fish:",
                      "fish_disp=", fish_disp, "board=", board_f, "fryer=", deep_fryer)

            # --- Chips pipeline (same as Chips robust) ---
            board_p = pick_board()
            if potato_disp and board_p and deep_fryer:
                _add_goto(seq, potato_disp);
                _add_put(seq, potato_disp)  # pick potato
                _add_goto(seq, board_p);
                _add_put(seq, board_p)  # place on board

                # Wait until raw potato is on board
                if "_wait_has" in globals():
                    _wait_has(seq, board_p, "potato", tag=f"{board_p}")

                _add_interact(seq, board_p)  # chop → RawChips

                # Wait until raw chips appear before pickup
                if "_wait_has" in globals():
                    _wait_has(seq, board_p, "rawchips", tag=f"{board_p}")

                _add_put(seq, board_p)  # pick RawChips

                _add_goto(seq, deep_fryer);
                _add_put(seq, deep_fryer)  # drop into basket via fryer
                _add_interact(seq, deep_fryer)  # start frying
                _add_cook_wait(seq, "chips", deep_fryer)  # wait until chips
                _add_put(seq, deep_fryer)  # take basket (with chips)

                if getattr(BB, "plate_spot_id", None):
                    _add_goto(seq, BB.plate_spot_id);
                    _add_put(seq, BB.plate_spot_id)  # pour chips
                    _mark(seq, "chips")

                # Return basket exactly once (no toggle loop)
                _add_goto(seq, deep_fryer)
                _add_return_basket(seq, deep_fryer)

            else:
                print("[BTBuilder][Fish&Chips] ⚠️ skipping chips:",
                      "potato_disp=", potato_disp, "board=", board_p, "fryer=", deep_fryer)

            # Serve — runtime guard so it isn't stripped at build time.
            if getattr(BB, "plate_spot_id", None) and serving in counter_positions:
                # Wait until BOTH items were plated
                if "_wait_token" in globals():
                    _wait_token(seq, "friedfish")
                    _wait_token(seq, "chips")
                _add_goto(seq, BB.plate_spot_id);
                _add_put(seq, BB.plate_spot_id)  # pick plate
                _add_goto(seq, serving);
                _add_put(seq, serving)  # deliver

            root.add_child(seq)


        else:
            # No active meal → gentle idle leaf so we don't spam logs
            class Idle(py_trees.behaviour.Behaviour):
                """Do nothing while there is no active meal (agent handles claiming)."""
                def update(self): return py_trees.common.Status.RUNNING
            root.add_child(Idle("Idle_NoActiveMeal"))

        # Return the built behaviour tree (root is a Selector; we added one child seq)
        return py_trees.trees.BehaviourTree(root)
