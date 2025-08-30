import numpy as np
import py_trees
from py_trees.decorators import Retry
from py_trees import blackboard


# ────────────────────────────────────────────────
# Blackboard (shared scratchpad between BT and agent)
# ────────────────────────────────────────────────
BB = blackboard.Blackboard()
# Keys used:
# - next_task:        ("GOTO"/"PUT"/"INTERACT"/"META_COOK", payload)
# - plate_spot_id:    counter ID where we staged the plate
# - cb_index:         round-robin index for cutting boards
# - tool_home:        {"pan": stove_id, "pot": stove_id}
# - added_to_plate:   set of items placed on plate (for gating)
# - active_meal:      current meal (e.g., "burger")
# - last_meal:        last built meal (to detect changes)
# - allow_pick_plate_once: one-shot gate to prevent double PUT


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def _norm(s: str | None) -> str:
    """Normalize item names: lowercase, remove underscores."""
    return (s or "").lower().replace("_", "")

def _lookup_counter(state: dict, counter_id: str) -> dict | None:
    """Find a counter by id in the state."""
    cs = state.get("counters")
    if isinstance(cs, dict):
        return cs.get(counter_id)
    if isinstance(cs, list):
        for c in cs:
            if c.get("id") == counter_id:
                return c
    return None

def _yield_tick(seq, tag: str):
    """Consume one tick so the environment can update after an action."""
    seq.add_child(OneShot(
        f"Action_Yield_{tag}",
        Retry(f"Retry_Yield_{tag}", TickYield(f"Yield_{tag}"), 1)
    ))

def _has_token(name: str) -> bool:
    """Check if a token (like 'chips') was marked as placed on the plate."""
    s = getattr(BB, "added_to_plate", set())
    return _norm(name) in s

def _wait_empty(seq, counter_id: str, tag: str = ""):
    """Add a wait until a counter becomes empty."""
    name = f"empty_{tag}" if tag else "empty"
    seq.add_child(
        OneShot(
            f"Action_WaitEmpty_{name}",
            Retry(f"Retry_WaitEmpty_{name}", WaitCounterEmpty(counter_id), 1)
        )
    )

def _wait_token(seq, token: str):
    """Add a wait until a specific token is recorded on plate."""
    seq.add_child(WaitToken(token))


# ────────────────────────────────────────────────
# Leaf behaviours — low-level actions
# ────────────────────────────────────────────────
class GoTo(py_trees.behaviour.Behaviour):
    """Move agent to a target position."""
    def __init__(self, name: str, target_pos):
        super().__init__(name); self.target_pos = target_pos
    def update(self):
        print(f"[BT][GoTo] → scheduling GOTO {self.target_pos}")
        BB.next_task = ("GOTO", np.array(self.target_pos))
        return py_trees.common.Status.SUCCESS

class Put(py_trees.behaviour.Behaviour):
    """Put down or pick up an object at a counter."""
    def __init__(self, name: str, target_id: str):
        super().__init__(name); self.target_id = target_id
    def update(self):
        print(f"[BT][Put] → scheduling PUT {self.target_id}")
        BB.next_task = ("PUT", self.target_id)
        return py_trees.common.Status.SUCCESS

class Interact(py_trees.behaviour.Behaviour):
    """Interact with a counter (chop, start cook, etc.)."""
    def __init__(self, name: str, counter_id: str):
        super().__init__(name); self.counter_id = counter_id
    def update(self):
        print(f"[BT][Interact] → scheduling INTERACT {self.counter_id}")
        BB.next_task = ("INTERACT", self.counter_id)
        return py_trees.common.Status.SUCCESS

class MetaCook(py_trees.behaviour.Behaviour):
    """Ask agent to wait until cooking produces the required item."""
    def __init__(self, name: str, required_product: str, station_id: str):
        super().__init__(name); self.required_product = required_product; self.station_id = station_id
    def update(self):
        print(f"[BT][MetaCook] → scheduling META_COOK station={self.station_id} wait_for={self.required_product}")
        BB.next_task = ("META_COOK", (_norm(self.required_product), self.station_id))
        return py_trees.common.Status.SUCCESS

class MarkPlateAddition(py_trees.behaviour.Behaviour):
    """Record that we’ve placed a specific token on the plate."""
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = _norm(token)
    def update(self):
        if not hasattr(BB, "added_to_plate"):
            BB.added_to_plate = set()
        BB.added_to_plate.add(self.token)
        print(f"[BT][Mark] → recorded on plate: {self.token}")
        return py_trees.common.Status.SUCCESS

class IfTokenMissing(py_trees.behaviour.Behaviour):
    """Condition: only succeed if token not already on plate."""
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = _norm(token)
    def update(self):
        present = self.token in getattr(BB, "added_to_plate", set())
        return py_trees.common.Status.FAILURE if present else py_trees.common.Status.SUCCESS

class TickYield(py_trees.behaviour.Behaviour):
    """Consume one tick; no engine task issued."""
    def update(self):
        return py_trees.common.Status.SUCCESS

class WaitCounterHas(py_trees.behaviour.Behaviour):
    """Poll until a counter contains an item matching expected_substr."""
    def __init__(self, counter_id: str, expected_substr: str):
        super().__init__(name=f"WaitHas[{expected_substr}]@{counter_id}")
        self.counter_id = counter_id
        self.expected = expected_substr.lower()
    def update(self):
        st = getattr(BB, "last_state", None)
        if not st: return py_trees.common.Status.SUCCESS
        ctr = _lookup_counter(st, self.counter_id)
        if not ctr: return py_trees.common.Status.RUNNING
        occ = ctr.get("occupied_by")
        names = []
        if isinstance(occ, dict):
            content = occ.get("content_list") or []
            for it in content:
                if isinstance(it, dict):
                    nm = (it.get("name") or it.get("type") or "").lower()
                    if nm: names.append(nm)
        return (py_trees.common.Status.SUCCESS
                if any(self.expected in nm for nm in names)
                else py_trees.common.Status.RUNNING)

class WaitCounterEmpty(py_trees.behaviour.Behaviour):
    """Succeed when the counter has no item on it."""
    def __init__(self, counter_id: str):
        super().__init__(name=f"WaitEmpty@{counter_id}")
        self.counter_id = counter_id
    def update(self):
        st = getattr(BB, "last_state", None)
        if not st: return py_trees.common.Status.RUNNING
        ctr = _lookup_counter(st, self.counter_id)
        if not ctr: return py_trees.common.Status.RUNNING
        occ = ctr.get("occupied_by")
        return py_trees.common.Status.SUCCESS if not occ else py_trees.common.Status.RUNNING

class ReturnBasket(py_trees.behaviour.Behaviour):
    """Return the fryer basket exactly once, then poll until it’s back in fryer."""
    def __init__(self, fryer_id: str):
        super().__init__(name=f"ReturnBasket@{fryer_id}")
        self.fryer_id = fryer_id
        self.issued_put = False
    def update(self):
        st = getattr(BB, "last_state", None)
        if not st: return py_trees.common.Status.RUNNING
        ctr = _lookup_counter(st, self.fryer_id)
        if not ctr: return py_trees.common.Status.RUNNING
        occ = ctr.get("occupied_by")
        occ_name = (occ.get("type") or occ.get("name") or "").lower() if isinstance(occ, dict) else ""
        if occ_name == "basket": return py_trees.common.Status.SUCCESS
        if not self.issued_put:
            BB.next_task = ("PUT", self.fryer_id)
            self.issued_put = True
        return py_trees.common.Status.RUNNING

def _add_return_basket(seq, fryer_id: str):
    seq.add_child(ReturnBasket(fryer_id))

class WaitToken(py_trees.behaviour.Behaviour):
    """Succeed only when a token is recorded on the plate."""
    def __init__(self, token: str):
        super().__init__(name=f"WaitToken[{token}]")
        self.token = token
    def update(self):
        return py_trees.common.Status.SUCCESS if _has_token(self.token) else py_trees.common.Status.RUNNING


# ────────────────────────────────────────────────
# OneShot decorator: ensures one task per tick
# ────────────────────────────────────────────────
class OneShot(py_trees.decorators.Decorator):
    """
    Tick child until SUCCESS, then:
      - First SUCCESS → report RUNNING (pace tick)
      - Later ticks → report SUCCESS without reticking
    Prevents repeating the same engine action twice.
    """
    def __init__(self, name: str, child: py_trees.behaviour.Behaviour):
        super().__init__(name=name, child=child)
        self.triggered = False
    def update(self):
        if self.triggered:
            return py_trees.common.Status.SUCCESS
        self.decorated.tick()
        status = self.decorated.status
        if status == py_trees.common.Status.SUCCESS:
            self.triggered = True
            return py_trees.common.Status.RUNNING
        return status
