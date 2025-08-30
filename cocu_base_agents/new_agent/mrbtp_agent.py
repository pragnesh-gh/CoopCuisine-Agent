
from cooperative_cuisine.base_agent.base_agent import BaseAgent, run_agent_from_args
from cooperative_cuisine.base_agent.agent_task import Task, TaskStatus
# from intention_utils import save_intentions
from constants import TASK_POSITIONS
import numpy as np
from cooperative_cuisine.action import ActionType


class FullBurgerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_step = "GET_PLATE"
        self.plate_counter_pos = None
        self.just_arrived = False
        self.returning_pan = False

    def parse_state(self, state):
        self.state_counters = state["counters"]
        for player in state["players"]:
            if player["id"] == self.own_player_id:
                self.current_agent_pos = np.array(player["pos"])
                self.held_item = player["holding"]
                if player["current_nearest_counter_id"]:
                    for counter in self.state_counters:
                        if counter["id"] == player["current_nearest_counter_id"]:
                            self.nearest_counter = counter
                            return
        self.nearest_counter = None

    async def handle_task(self, state):
        t = self.current_task.task_type.upper()
        if t == Task.GOTO:
            await self.handle_task_goto(state)
        elif t == Task.INTERACT:
            await self.handle_task_interact(state)
        elif t in (Task.PUT, "PICKUP", "PUTDOWN", "DROPOFF"):
            await self.handle_task_put(state)
        else:
            self.finalize_current_task(TaskStatus.FAILED, f"Unknown task type: {t}")

    async def handle_task_put(self, state):
        if self.nearest_counter is None:
            self.finalize_current_task(TaskStatus.FAILED, "No counter nearby")
        else:
            await self._execute_action(action_type=ActionType.PICK_UP_DROP)
            self.finalize_current_task(TaskStatus.SUCCESS, "Picked up or dropped off")

    async def manage_tasks(self, state):
        if self.current_task:
            return

        # save_intentions({self.own_player_id: {"step": self.pipeline_step}})

        # 1) Get plate
        if self.pipeline_step == "GET_PLATE":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["PLATE_DISPENSER"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "PLACE_PLATE"
                self.just_arrived = False
            return

        # 2) Place plate
        if self.pipeline_step == "PLACE_PLATE":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                self.plate_counter_pos = np.array(self.nearest_counter["pos"])
                print(f"Plate placed at {self.plate_counter_pos}")
                self.pipeline_step = "GET_BUN"
                self.just_arrived = False
            else:
                # Free counter
                free_counter_pos = None
                for c in self.state_counters:
                    if (
                        c.get("occupied_by") is None
                        and c.get("type") not in ("CuttingBoard", "Pan")
                    ):
                        free_counter_pos = np.array(c["pos"])
                        break
                if free_counter_pos is None:
                    free_counter_pos = np.array(TASK_POSITIONS["CUTTING_BOARD_1"])
                self.set_current_task(Task(Task.GOTO, task_args=free_counter_pos))
                self.just_arrived = True
            return

        # 3) Get bun
        if self.pipeline_step == "GET_BUN":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    print("Picked bun.")
                    self.pipeline_step = "PLACE_BUN"
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["GET_BUN"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "PLACE_BUN"
                self.just_arrived = False
            return

        # 4) Place bun
        if self.pipeline_step == "PLACE_BUN":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                print("Bun placed.")
                self.pipeline_step = "GET_LETTUCE"
                self.just_arrived = False
            else:
                self.set_current_task(Task(Task.GOTO, task_args=self.plate_counter_pos))
                self.just_arrived = True
            return

        # 5) Get lettuce
        if self.pipeline_step == "GET_LETTUCE":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["GET_LETTUCE"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "CUT_LETTUCE"
                self.just_arrived = False
            return

        # 6) Cut lettuce
        if self.pipeline_step == "CUT_LETTUCE":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                self.pipeline_step = "CHOPPING_LETTUCE"
                self.just_arrived = False
            else:
                pos = TASK_POSITIONS["CUTTING_BOARD_1"]
                self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                self.just_arrived = True
            return

        if self.pipeline_step == "CHOPPING_LETTUCE":
            if self.held_item:
                self.pipeline_step = "CUT_LETTUCE"
            else:
                self.set_current_task(Task(Task.INTERACT))
                self.pipeline_step = "PICK_CHOPPED_LETTUCE"
            return

        if self.pipeline_step == "PICK_CHOPPED_LETTUCE":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["CUTTING_BOARD_1"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "PLACE_LETTUCE"
                self.just_arrived = False
            return

        if self.pipeline_step == "PLACE_LETTUCE":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                print("Lettuce placed.")
                self.pipeline_step = "GET_TOMATO"
                self.just_arrived = False
            else:
                self.set_current_task(Task(Task.GOTO, task_args=self.plate_counter_pos))
                self.just_arrived = True
            return

        # 7) Get tomato
        if self.pipeline_step == "GET_TOMATO":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["GET_TOMATO"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "CUT_TOMATO"
                self.just_arrived = False
            return

        # 8) Cut tomato
        if self.pipeline_step == "CUT_TOMATO":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                self.pipeline_step = "CHOPPING_TOMATO"
                self.just_arrived = False
            else:
                pos = TASK_POSITIONS["CUTTING_BOARD_1"]
                self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                self.just_arrived = True
            return

        if self.pipeline_step == "CHOPPING_TOMATO":
            if self.held_item:
                self.pipeline_step = "CUT_TOMATO"
            else:
                self.set_current_task(Task(Task.INTERACT))
                self.pipeline_step = "PICK_CHOPPED_TOMATO"
            return

        if self.pipeline_step == "PICK_CHOPPED_TOMATO":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["CUTTING_BOARD_1"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "PLACE_TOMATO"
                self.just_arrived = False
            return

        if self.pipeline_step == "PLACE_TOMATO":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                print(" Tomato placed.")
                self.pipeline_step = "GET_MEAT"
                self.just_arrived = False
            else:
                self.set_current_task(Task(Task.GOTO, task_args=self.plate_counter_pos))
                self.just_arrived = True
            return

        # 9) Get meat
        if self.pipeline_step == "GET_MEAT":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["GET_MEAT"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "CUT_MEAT"
                self.just_arrived = False
            return

        if self.pipeline_step == "CUT_MEAT":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                self.pipeline_step = "CHOPPING_MEAT"
                self.just_arrived = False
            else:
                pos = TASK_POSITIONS["CUTTING_BOARD_1"]
                self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                self.just_arrived = True
            return

        if self.pipeline_step == "CHOPPING_MEAT":
            if self.held_item:
                self.pipeline_step = "CUT_MEAT"
            else:
                self.set_current_task(Task(Task.INTERACT))
                self.pipeline_step = "PICK_CHOPPED_MEAT"
            return

        if self.pipeline_step == "PICK_CHOPPED_MEAT":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["CUTTING_BOARD_1"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.pipeline_step = "COOK_MEAT"
                self.just_arrived = False
            return

        if self.pipeline_step == "COOK_MEAT":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                self.pipeline_step = "COOKING_MEAT"
                self.just_arrived = False
            else:
                pos = TASK_POSITIONS["PAN"]
                self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                self.just_arrived = True
            return

        if self.pipeline_step == "COOKING_MEAT":
            if self.held_item:
                self.pipeline_step = "COOK_MEAT"
            else:
                self.set_current_task(Task(Task.INTERACT))
                self.pipeline_step = "PICK_COOKED_MEAT"
            return

        if self.pipeline_step == "PICK_COOKED_MEAT":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["PAN"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            else:
                self.returning_pan = True
                self.pipeline_step = "PLACE_MEAT"
                self.just_arrived = False
            return

        if self.pipeline_step == "PLACE_MEAT":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                print("Meat placed.")
                self.pipeline_step = "RETURN_PAN"
                self.just_arrived = False
            else:
                self.set_current_task(Task(Task.GOTO, task_args=self.plate_counter_pos))
                self.just_arrived = True
            return

        if self.pipeline_step == "RETURN_PAN":
            if self.just_arrived:
                self.set_current_task(Task(Task.PUT))
                self.pipeline_step = "SERVE"
                self.just_arrived = False
            else:
                pos = TASK_POSITIONS["PAN"]
                self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                self.just_arrived = True
            return

        # Serve
        if self.pipeline_step == "SERVE":
            if not self.held_item:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.just_arrived = False
                else:
                    self.set_current_task(Task(Task.GOTO, task_args=self.plate_counter_pos))
                    self.just_arrived = True
            else:
                if self.just_arrived:
                    self.set_current_task(Task(Task.PUT))
                    self.pipeline_step = "DONE"
                    self.just_arrived = False
                else:
                    pos = TASK_POSITIONS["SERVING_WINDOW"]
                    self.set_current_task(Task(Task.GOTO, task_args=np.array(pos)))
                    self.just_arrived = True
            return

        if self.pipeline_step == "DONE":
            print("[DEBUG] All done. Idling.")


if __name__ == "__main__":
    run_agent_from_args(FullBurgerAgent)



