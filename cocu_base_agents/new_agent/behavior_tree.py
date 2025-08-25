import numpy as np
import py_trees
from py_trees.decorators import Retry
from py_trees import blackboard

# ──────────────────────────────────────────────────────────────────────────────
# Blackboard (shared scratchpad between leaves and the agent loop)
# ──────────────────────────────────────────────────────────────────────────────
BB = blackboard.Blackboard()
# We’ll store:
# - BB.next_task:    ("GOTO"/"PUT"/"INTERACT"/"META_COOK", payload)
# - BB.plate_spot_id: counter id where we staged the plate
# - BB.cb_index:     round‑robin index for cutting boards
# - BB.tool_home:    { "pan": stove_id, "pot": stove_id, "basket": fryer_id, "peel": oven_id }

# // NEW: helpers used by the (agent-side) META_COOK logic; harmless to keep here too
def _norm_name(s: str | None) -> str:
    """Normalize names like 'CookedPatty', 'cooked_patty', 'cookedpatty' -> 'cookedpatty'."""
    return (s or "").lower().replace("_", "")

def _lookup_counter(state: dict, counter_id: str) -> dict | None:
    """
    Find a counter dict by id in the current state payload.
    Assumes state["counters"] is either a dict[id]->counter or a list[ {id:..., ...}, ... ].
    """
    counters = state.get("counters")
    if isinstance(counters, dict):
        return counters.get(counter_id)
    if isinstance(counters, list):
        for c in counters:
            if c.get("id") == counter_id:
                return c
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Leaf behaviours (each leaf emits exactly one Task into BB.next_task)
# ──────────────────────────────────────────────────────────────────────────────
class GoToBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, target_pos):
        super().__init__(name); self.target_pos = target_pos
    def update(self):
        print(f"[BT][GoTo] → scheduling GOTO {self.target_pos}")
        BB.next_task = ("GOTO", np.array(self.target_pos))
        return py_trees.common.Status.SUCCESS


class PutBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, target_id: str):
        super().__init__(name); self.target_id = target_id
    def update(self):
        print(f"[BT][Put] → scheduling PUT {self.target_id}")
        BB.next_task = ("PUT", self.target_id)
        return py_trees.common.Status.SUCCESS


class InteractBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, counter_id: str):
        super().__init__(name); self.counter_id = counter_id
    def update(self):
        print(f"[BT][Interact] → scheduling INTERACT {self.counter_id}")
        BB.next_task = ("INTERACT", self.counter_id)
        return py_trees.common.Status.SUCCESS


class MetaCookBehaviour(py_trees.behaviour.Behaviour):
    """
    Ask the BaseAgent to wait for a particular cooked product to appear at `station_id`.
    The agent’s META_COOK handler should interpret the payload as (required_product_base, station_id).
    """
    def __init__(self, name: str, required_product: str, station_id: str):
        super().__init__(name); self.required_product = required_product; self.station_id = station_id
    def update(self):
        print(f"[BT][MetaCook] → scheduling META_COOK station={self.station_id} wait_for={self.required_product}")
        BB.next_task = ("META_COOK", (self.required_product, self.station_id))
        return py_trees.common.Status.SUCCESS


# NEW ↓↓↓ — record “this ingredient is now on the plate”
class MarkPlateAddition(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = token
    def update(self):
        if not hasattr(BB, "added_to_plate"):
            BB.added_to_plate = set()
        BB.added_to_plate.add(_norm_name(self.token))
        print(f"[BT][Mark] → recorded on plate: {self.token}")
        return py_trees.common.Status.SUCCESS


# NEW ↓↓↓ — gate: run child only if token missing from plate
class IfTokenMissing(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, token: str):
        super().__init__(name); self.token = token
    def update(self):
        present = _norm_name(self.token) in getattr(BB, "added_to_plate", set())
        return py_trees.common.Status.FAILURE if present else py_trees.common.Status.SUCCESS

class AllowPickPlateOnce(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "AllowPickPlateOnce"):
        super().__init__(name)
    def update(self):
        BB.allow_pick_plate_once = True
        print("[BT][Serve] → allow one pick from plate spot")
        return py_trees.common.Status.SUCCESS

# ──────────────────────────────────────────────────────────────────────────────
# OneShotAction: fire child once; on first SUCCESS, return RUNNING to consume a tick
# ──────────────────────────────────────────────────────────────────────────────
class OneShotAction(py_trees.decorators.Decorator):
    def __init__(self, name: str, child: py_trees.behaviour.Behaviour):
        super().__init__(name=name, child=child); self.triggered = False
    def update(self):
        self.decorated.tick(); status = self.decorated.status
        if not self.triggered and status == py_trees.common.Status.SUCCESS:
            self.triggered = True; return py_trees.common.Status.RUNNING
        return status


# ──────────────────────────────────────────────────────────────────────────────
# BehaviorTreeBuilder — utensil-aware cooking, multi-board chopping, soup gating
# ──────────────────────────────────────────────────────────────────────────────
class BehaviorTreeBuilder:
    @staticmethod
    def build(recipe_graphs,
              service_counter_id: str,
              counters: list,
              counter_positions: dict,
              player_key: str):

        root = py_trees.composites.Selector(name="RootSelector", memory=False)

        # ---------- Index the world ----------
        by_type = {}
        for c in counters:
            ctype = (c.get("type") or "").lower()
            by_type.setdefault(ctype, []).append(c)
        print(f"[BTBuilder] COUNTER_TYPES: {sorted(by_type.keys())}")

        def ids_of(type_key: str):
            ids = [c["id"] for c in by_type.get(type_key.lower(), [])]
            print(f"[BTBuilder] ids_of('{type_key}') -> {ids}")
            return ids

        def first_or_none(seq):
            return seq[0] if seq else None

        def pos_of(cid: str):
            return counter_positions.get(cid)

        # add helpers to a sequence
        def add_goto(seq, cid):
            pos = pos_of(cid)
            if pos is None:
                print(f"[BTBuilder] ⚠️ Missing position for {cid}; skipping GOTO"); return
            seq.add_child(OneShotAction(f"Action_Goto_{cid}",
                         Retry(f"Retry_Goto_{cid}", GoToBehaviour(f"Goto_{cid}", pos), 3)))

        def add_put(seq, cid):
            seq.add_child(OneShotAction(f"Action_Put_{cid}",
                         Retry(f"Retry_Put_{cid}", PutBehaviour(f"Put_{cid}", cid), 3)))

        def add_interact(seq, cid):
            seq.add_child(OneShotAction(f"Action_Interact_{cid}",
                         Retry(f"Retry_Interact_{cid}", InteractBehaviour(f"Interact_{cid}", cid), 3)))

        def add_meta_cook(seq, required, station):
            seq.add_child(OneShotAction(f"Action_MetaCook_{station}",
                         Retry(f"Retry_MetaCook_{station}", MetaCookBehaviour(f"Cook_{station}", required, station), 3)))

        def add_mark(seq, token):
            seq.add_child(OneShotAction(f"Action_Mark_{token}",
                                        MarkPlateAddition(f"Mark_{token}", token)))
        # ---------- Recipe parsing helpers ----------
        def base(node_name: str) -> str:
            return node_name.split("_", 1)[0]  # "Tomato_abc" -> "Tomato"

        def type_for_node(node: str) -> str | None:
            """
            Map a recipe node to a counter 'type' string reported by the env.
            Tries both 'x_dispenser' and 'xdispenser'. Special-cases plate/board/stations.
            """
            b = base(node).lower()
            if b == "cuttingboard":
                return "cuttingboard"
            if b == "plate":
                if "platedispenser" in by_type: return "platedispenser"
                if "plate" in by_type:          return "plate"
                return None
            if b in ("stove", "oven", "deepfryer"):
                return b
            # ingredients (dispenser names have no underscore in Co-op Cuisine)
            underscored = f"{b}_dispenser"
            squashed    = f"{b}dispenser"
            if underscored in by_type: return underscored
            if squashed    in by_type: return squashed
            return None

        # choose a cutting board (round‑robin)
        cuttingboard_ids_all = ids_of("cuttingboard")
        BB.cb_index = 0
        def pick_board() -> str | None:
            if not cuttingboard_ids_all: return None
            cid = cuttingboard_ids_all[BB.cb_index % len(cuttingboard_ids_all)]
            BB.cb_index += 1
            return cid

        # stage a plate on a plain counter (once per order sequence)
        plate_spot_candidates = ids_of("counter")
        def ensure_plate_staged(seq) -> bool:
            if getattr(BB, "plate_spot_id", None):
                return True
            plate_src = first_or_none(ids_of("platedispenser")) or first_or_none(ids_of("plate"))
            spot      = first_or_none(plate_spot_candidates)
            if not plate_src or not spot:
                print(f"[BTBuilder] ⚠️ Missing plate source ({plate_src}) or counter spot ({spot})"); return False
            BB.plate_spot_id = spot
            print(f"[BTBuilder] Staging plate at counter spot: {BB.plate_spot_id}")
            add_goto(seq, plate_src); add_put(seq, plate_src)   # pick plate
            add_goto(seq, spot);      add_put(seq, spot)        # drop plate
            return True

        # Tool ↔ Station mapping + “home”
        TOOL_TO_STATION = {
            "pan": "stove",
            "pot": "stove",
            "basket": "deepfryer",
            "peel": "oven",
        }
        if not hasattr(BB, "tool_home"):
            BB.tool_home = {}  # tool_type -> station_id it lives on

        # Given a tool keyword (pan/pot/basket/peel), pick a station id to use (and remember as home)
        def pick_station_for_tool(tool_key: str) -> str | None:
            station_type = TOOL_TO_STATION.get(tool_key)
            if not station_type: return None
            sts = ids_of(station_type)
            if not sts: return None
            station_id = BB.tool_home.get(tool_key) or sts[0]
            BB.tool_home[tool_key] = station_id
            return station_id

        # Tiny edge helpers
        def next_dst(edges, node):
            # first edge whose src == node
            for s, d in edges:
                if s == node:
                    return d
            return None

        # ---------- Build one sequence per meal ----------
        for graph in recipe_graphs:
            meal = graph.get("meal", "Meal")
            raw_edges = list(graph.get("edges", []))

            # topological order by edge dependency (edge u before v if u.dst == v.src)
            n = len(raw_edges)
            adj, indeg = {i: [] for i in range(n)}, {i: 0 for i in range(n)}
            for i, (s1, d1) in enumerate(raw_edges):
                for j, (s2, _) in enumerate(raw_edges):
                    if d1 == s2: adj[i].append(j); indeg[j] += 1
            q, order = [i for i in range(n) if indeg[i] == 0], []
            while q:
                u = q.pop(0); order.append(u)
                for v in adj[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0: q.append(v)
            if len(order) != n: order = list(range(n))
            edges = [raw_edges[i] for i in order]

            # // CHANGED: Keep only process edges (boards/tools/stations) to avoid double PUTs on plates
            PROC = ("cuttingboard", "pan", "pot", "basket", "peel", "stove", "oven", "deepfryer")
            process_edges = [e for e in edges if base(e[1]).lower() in PROC]
            edges_ord     = process_edges  # ← CHANGED

            seq = py_trees.composites.Sequence(name=f"Order_{meal}_UTENSILS", memory=True)
            print(f"[BTBuilder] BUILDING {meal}: edges = {edges_ord}")

            ensure_plate_staged(seq)  # stage plate once

            # NEW ↓↓↓ — cold (direct-to-plate) additions like Bun, Cheese, etc.
            if not hasattr(BB, "added_to_plate"):
                BB.added_to_plate = set()

            cold_items = []
            for s, d in edges:  # use full graph, not filtered edges_ord
                if base(d).lower() == "plate":
                    src_base = base(s)
                    low = src_base.lower()
                    if low.startswith("chopped") or low.startswith("cooked"):
                        continue  # handled elsewhere
                    dtype = type_for_node(s)  # e.g., 'bundispenser'
                    if dtype and dtype.endswith("dispenser"):
                        cold_items.append((_norm_name(src_base), dtype))

            if cold_items:
                cold_selector = py_trees.composites.Selector(name=f"{meal}_ColdAdditions", memory=False)
                for token, disp_type in cold_items:
                    disp_id = first_or_none(ids_of(disp_type))
                    if not disp_id:
                        print(f"[BTBuilder] ⚠️ No ids for dispenser type {disp_type}; skipping cold item '{token}'")
                        continue
                    if token == "bun":
                        continue
                    cold_seq = py_trees.composites.Sequence(name=f"Cold_{token}", memory=True)
                    # gate: only run if token not on plate yet
                    cold_seq.add_child(IfTokenMissing(f"Need_{token}", token))
                    # make sure plate is staged (no-op if already done)
                    ensure_plate_staged(cold_seq)
                    # fetch → plate → mark
                    add_goto(cold_seq, disp_id);            add_put(cold_seq, disp_id)
                    if getattr(BB, "plate_spot_id", None):
                        add_goto(cold_seq, BB.plate_spot_id); add_put(cold_seq, BB.plate_spot_id)
                        add_mark(cold_seq, token)
                    cold_selector.add_child(cold_seq)

                # Put cold additions BEFORE the main utensil sequence
                root.add_child(cold_selector)


            for src, dst in edges_ord:
                src_b, dst_b = base(src).lower(), base(dst).lower()

                # ── 1) Ingredient -> CuttingBoard (chop, free board) ──────────────────────
                if dst_b == "cuttingboard":
                    ing_type = type_for_node(src)
                    board_id = pick_board()
                    if ing_type and board_id:
                        ing_id = first_or_none(ids_of(ing_type))
                        if not ing_id:
                            print(f"[BTBuilder] ⚠️ No ids for dispenser type {ing_type}; skipping")
                            continue
                        # pick ingredient → place on board → chop → pick chopped (frees board)
                        add_goto(seq, ing_id);
                        add_put(seq, ing_id)
                        add_goto(seq, board_id);
                        add_put(seq, board_id)
                        add_interact(seq, board_id)
                        add_put(seq, board_id)

                        # look-ahead: produced node after board, then next action target
                        produced_node = next_dst(edges, dst)  # e.g., RawPatty_* or ChoppedTomato_*
                        nxt = next_dst(edges, produced_node) if produced_node else None
                        nxt_b = base(nxt).lower() if nxt else None

                        # If cooking is next (tool/station), branch into cooking pipeline now.
                        if nxt_b in ("pan", "pot", "basket", "peel", "stove", "oven", "deepfryer"):
                            # resolve final station (skip tool layer if present)
                            tool_or_station = nxt
                            tool_b = nxt_b
                            station_node = (tool_or_station if tool_b in ("stove", "oven", "deepfryer")
                                            else next_dst(edges, tool_or_station))
                            station_type = type_for_node(station_node) if station_node else None
                            if not station_type:
                                print(f"[BTBuilder] ⚠️ Could not resolve station for {station_node}; skipping")
                            else:
                                # Decide which physical station to use (and treat it as tool home)
                                if tool_b in TOOL_TO_STATION:
                                    station_id = pick_station_for_tool(tool_b)
                                else:
                                    st_ids = ids_of(station_type)
                                    station_id = st_ids[0] if st_ids else None
                                if station_id:
                                    # drop prepped item into utensil at station → cook → wait → pour → return utensil
                                    add_goto(seq, station_id);
                                    add_put(seq, station_id)  # place item into utensil
                                    add_interact(seq, station_id)  # start cooking
                                    cooked_node = next_dst(edges, station_node) if station_node else None
                                    required = base(cooked_node).lower() if cooked_node else "cooked"
                                    add_meta_cook(seq, required, station_id)  # wait for done
                                    add_put(seq, station_id)  # pick utensil (with food)
                                    if getattr(BB, "plate_spot_id", None):
                                        add_goto(seq, BB.plate_spot_id)
                                        add_put(seq, BB.plate_spot_id)  # pour to plate
                                        # ✅ mark the **cooked** item as added (e.g., 'cookedpatty')
                                        add_mark(seq, _norm_name(required))
                                    add_goto(seq, station_id)
                                    add_put(seq, station_id)  # return empty utensil
                                    continue  # handled fully

                        # else (no cooking next): assemble directly on the staged plate spot
                        if getattr(BB, "plate_spot_id", None):
                            add_goto(seq, BB.plate_spot_id);
                            add_put(seq, BB.plate_spot_id)
                            # ✅ mark the **chopped** result as added (e.g., 'choppedtomato'/'choppedlettuce')
                            if produced_node:
                                add_mark(seq, _norm_name(base(produced_node)))
                    else:
                        print(f"[BTBuilder] ⚠️ Missing ing_type or board for {src}->{dst}; skipping")
                    continue

                # ── 2) Tool/Prepped -> Station (safety path if graphs give these earlier) ──
                if dst_b in ("pan", "pot", "basket", "peel", "stove", "oven", "deepfryer"):
                    # infer final station, and which tool (if any) this implies
                    tool_b = dst_b if dst_b in TOOL_TO_STATION else None
                    station_node = dst if dst_b in ("stove", "oven", "deepfryer") else next_dst(edges, dst)
                    station_type = type_for_node(station_node) if station_node else None
                    if not station_type:
                        print(f"[BTBuilder] ⚠️ Could not resolve station for {station_node}; skipping")
                        continue
                    station_id = (pick_station_for_tool(tool_b) if tool_b
                                  else (ids_of(station_type)[0] if ids_of(station_type) else None))
                    if not station_id:
                        print(f"[BTBuilder] ⚠️ No station id available for {station_type}; skipping")
                        continue
                    add_goto(seq, station_id); add_put(seq, station_id)     # place item into utensil
                    add_interact(seq, station_id)                           # start cooking
                    cooked_node = next_dst(edges, station_node) if station_node else None
                    required = base(cooked_node).lower() if cooked_node else "cooked"
                    add_meta_cook(seq, required, station_id)
                    add_put(seq, station_id)                                # pick utensil with food
                    if getattr(BB, "plate_spot_id", None):
                        add_goto(seq, BB.plate_spot_id); add_put(seq, BB.plate_spot_id)  # pour
                    add_goto(seq, station_id); add_put(seq, station_id)     # return utensil
                    continue

                # (Plate edges deliberately ignored)       # // CHANGED: removed Sections 3 & 4 to stop double PUTs

                # else: ignore unhandled edges (e.g., decorations)

            # ── Serve: pick plate from its spot → serving window → drop ────────────────────
            if getattr(BB, "plate_spot_id", None) and service_counter_id in counter_positions:
                add_goto(seq, BB.plate_spot_id);   add_put(seq, BB.plate_spot_id)    # pick final plate
                add_goto(seq, service_counter_id); add_put(seq, service_counter_id)  # serve

            root.add_child(seq)
            # === Add bun LAST (after all chopped/cooked items are on the plate) ===
            has_bun_edge = any(base(s).lower() == "bun" and base(d).lower() == "plate" for s, d in edges)
            if has_bun_edge:
                bun_id = first_or_none(ids_of("bundispenser"))
                if not bun_id:
                    print("[BTBuilder] ⚠️ No 'bundispenser' found; cannot add bun last.")
                elif not getattr(BB, "plate_spot_id", None):
                    print("[BTBuilder] ⚠️ No plate_spot_id; cannot add bun last.")
                else:
                    bun_last = py_trees.composites.Sequence(name=f"{meal}_AddBunLast", memory=True)
                    bun_last.add_child(IfTokenMissing("Need_bun", "bun"))  # skip if bun already on plate
                    add_goto(bun_last, bun_id);
                    add_put(bun_last, bun_id)
                    add_goto(bun_last, BB.plate_spot_id);
                    add_put(bun_last, BB.plate_spot_id)
                    add_mark(bun_last, "bun")
                    root.add_child(bun_last)

        return py_trees.trees.BehaviourTree(root)



