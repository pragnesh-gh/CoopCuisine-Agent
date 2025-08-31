"""
The _Cooperative Cuisine_ environment. It is configured via three configs:
- [`environment_config.yml`](https://gitlab.ub.uni-bielefeld.de/scs/cocosy/cooperative-cuisine/-/blob/main/cooperative_cuisine/configs/environment_config.yaml?ref_type=heads)
- [`xyz.layout`](https://gitlab.ub.uni-bielefeld.de/scs/cocosy/cooperative-cuisine/-/blob/main/cooperative_cuisine/configs/layouts/basic.layout?ref_type=heads)
- [`item_info.yml`](https://gitlab.ub.uni-bielefeld.de/scs/cocosy/cooperative-cuisine/-/blob/main/cooperative_cuisine/configs/item_info.yaml?ref_type=heads)

You can pass either the file path or the file content (str) to the `Environment`class of the config files. Set the `as_files` parameter accordingly.

"""
from __future__ import annotations

import inspect
import json
import logging
import sys
import time
from collections import defaultdict
from copy import copy
from datetime import timedelta, datetime
from pathlib import Path
from random import Random
from typing import Literal, TypedDict, Callable, Set, Any, Tuple

import numpy as np
import numpy.typing as npt
import yaml

from cooperative_cuisine.action import ActionType, InterActionData, Action
from cooperative_cuisine.counter_factory import (
    CounterFactory,
)
from cooperative_cuisine.counters import (
    PlateConfig,
    Counter,
)
from cooperative_cuisine.effects import EffectManager
from cooperative_cuisine.hooks import (
    ITEM_INFO_LOADED,
    LAYOUT_FILE_PARSED,
    ENV_INITIALIZED,
    PRE_PERFORM_ACTION,
    POST_PERFORM_ACTION,
    PLAYER_ADDED,
    GAME_ENDED_STEP,
    PRE_STATE,
    STATE_DICT,
    JSON_STATE,
    PRE_RESET_ENV_TIME,
    POST_RESET_ENV_TIME,
    Hooks,
    ACTION_ON_NOT_REACHABLE_COUNTER,
    ACTION_PUT,
    ACTION_INTERACT_START,
    ITEM_INFO_CONFIG,
    POST_STEP,
    hooks_via_callback_class,
    ADDITIONAL_STATE_UPDATE,
    SCORE_CHANGED, POST_PLAYER_MOVEMENT,
)
from cooperative_cuisine.items import (
    ItemInfo,
    ItemType,
)
from cooperative_cuisine.movement import Movement
from cooperative_cuisine.orders import (
    OrderManager,
    OrderConfig,
)
from cooperative_cuisine.player import Player, PlayerConfig
from cooperative_cuisine.state_representation import InfoMsg
from cooperative_cuisine.utils import (
    create_init_env_time,
    get_closest,
)
from cooperative_cuisine.validation import Validation

log = logging.getLogger(__name__)
"""The logger for this module."""


class EnvironmentConfig(TypedDict):
    """Configuration dict for an environment."""
    
    plates: PlateConfig
    """Config for plate behavior."""
    game: dict[
        Literal["time_limit_seconds"] | Literal["undo_dispenser_pickup"],
        Literal["validate_recipes"],
        int | bool,
    ]
    """Overall game settings."""
    orders: OrderConfig
    """Config about the generation of orders for the players to complete."""
    player_config: PlayerConfig
    """Configuration about the player characters."""
    layout_chars: dict[str, str]
    """Definition of which characters in the layout file correspond to which kitchen counter."""
    hook_callbacks: dict[str, dict]
    """Configuration of callbacks via HookCallbackClass."""
    effect_manager: dict[str, dict]
    """Config of different effects in the environment, which control for example fire behavior."""


class Environment:
    """Environment class which handles the game logic for the overcooked-inspired environment.

    Handles player movement, collision-detection, counters, cooking processes, recipes, incoming orders, time.
    """
    
    # pdoc does not detect attributes in the tuple assignment in init
    counters: list[Counter]
    """Counters of the environment."""
    designated_player_positions: list[npt.NDArray[float]]
    """The positions new players can spawn based on the layout (`A` char)."""
    free_positions: list[Tuple[npt.NDArray[float], str]]
    """list of 2D points of free (no counters) positions in the environment and the name of the ground types."""
    kitchen_height: int
    """Grid height of the kitchen."""
    kitchen_width: int
    """Grid width of the kitchen."""
    layout_extra_info: dict[str, Any]
    """Extra info stored in the layout file after `;` in one-liner yaml format."""
    
    def __init__(
        self,
        env_config: Path | str,
        layout_config: Path | str,
        item_info: Path | str,
        as_files: bool = True,
        env_name: str = "cooperative_cuisine_1",
        seed: int = 56789223842348,
        yaml_already_loaded: bool = False
    ):
        """Constructor of the Environment.

        Args:
            env_config: The path or string representation of the environment configuration file.
            layout_config: The path or string representation of the layout configuration file.
            item_info: The path or string representation of the item info file.
            as_files: (optional) A flag indicating whether the configuration parameters are file paths.
                      If True, the method will read the contents of the files. Defaults to True.
            env_name: (optional) The name of the environment. Defaults to "overcooked_sim".
            seed: (optional) The seed for generating random numbers. Defaults to 56789223842348.
        """
        self.env_name: str = env_name
        """Reference to the run. E.g, the env id."""
        self.env_time: datetime = create_init_env_time()
        """the internal time of the environment. An environment starts always with the time from 
        `create_init_env_time`."""
        
        self.random: Random = Random(seed)
        """Random instance."""
        self.hook: Hooks = Hooks(self)
        """Hook manager. Register callbacks and create hook points with additional kwargs."""
        
        self.score: float = 0.0
        """The current score of the environment."""
        
        self.players: dict[str, Player] = {}
        """the player, keyed by their id/name."""
        
        self.as_files: bool = as_files
        """Are the configs just the path to the files."""
        if self.as_files:
            with open(env_config, "r", encoding="utf-8") as file:
                env_config = file.read()
            with open(layout_config, "r", encoding="utf-8") as layout_file:
                layout_config = layout_file.read()
            with open(item_info, "r", encoding="utf-8") as file:
                item_info = file.read()
        if not yaml_already_loaded:
            self.environment_config: EnvironmentConfig = yaml.load(
                env_config, Loader=yaml.Loader
            )
        else:
            self.environment_config = env_config
        
        """The config of the environment. All environment specific attributes is configured here."""
        self.environment_config["player_config"] = PlayerConfig(
            **(
                self.environment_config["player_config"]
                if "player_config" in self.environment_config
                else {}
            )
        )
        self.player_view_restricted: bool = self.environment_config[
            "player_config"
        ].restricted_view
        """If field-of-view of players is restricted in this environment."""
        if self.player_view_restricted:
            self.player_view_angle: float = self.environment_config[
                "player_config"
            ].view_angle
            self.player_view_range: float = self.environment_config[
                "player_config"
            ].view_range
        
        self.hook_callbacks()
        
        self.layout_config: str = layout_config
        """The layout config for the environment"""
        # self.counter_side_length = 1  # -> this changed! is 1 now
        
        self.item_info: dict[str, ItemInfo] = self.load_item_info(item_info)
        """The loaded item info dict. Keys are the item names."""
        self.hook(
            ITEM_INFO_LOADED, item_info=item_info, parsed_item_info=self.item_info
        )
        
        if self.environment_config["orders"]["meals"]["all"]:
            self.allowed_meal_names: Set[str] = set(
                [
                    item
                    for item, info in self.item_info.items()
                    if info.type == ItemType.Meal
                ]
            )
        else:
            self.allowed_meal_names: Set[str] = set(
                self.environment_config["orders"]["meals"]["list"]
            )
            """The allowed meals depend on the `environment_config.yml` configured behaviour. Either all meals that 
            are possible or only a limited subset."""
        self.order_manager: OrderManager = OrderManager(
            order_config=self.environment_config["orders"],
            hook=self.hook,
            random=Random(seed),
        )
        """The manager for the orders and score update."""
        
        do_validation = (
            self.environment_config["game"]["validate_recipes"]
            if "validate_recipes" in self.environment_config["game"].keys()
            else True
        )
        
        self.recipe_validation: Validation = Validation(
            meals=[m for m in self.item_info.values() if m.type == ItemType.Meal]
            if self.environment_config["orders"]["meals"]["all"]
            else [
                self.item_info[m]
                for m in self.environment_config["orders"]["meals"]["list"]
                if self.item_info[m].type == ItemType.Meal
            ],
            item_info=self.item_info,
            order_manager=self.order_manager,
            do_validation=do_validation,
        )
        """Validates configs and creates recipe graphs."""
        
        plate_config = PlateConfig(
            **(
                self.environment_config["plates"]
                if "plates" in self.environment_config
                else {}
            )
        )
        
        self.recipe_validation.update_plate_config(plate_config, self.environment_config["layout_chars"],
                                                   self.layout_config)
        
        self.counter_factory: CounterFactory = CounterFactory(
            layout_chars_config=self.environment_config["layout_chars"],
            item_info=self.item_info,
            serving_window_additional_kwargs={
                "meals": self.allowed_meal_names,
                "env_time_func": self.get_env_time,
            },
            plate_config=plate_config,
            order_manager=self.order_manager,
            effect_manager_config=self.environment_config["effect_manager"],
            undo_dispenser_pickup=self.environment_config["game"][
                "undo_dispenser_pickup"
            ]
            if "game" in self.environment_config
               and "undo_dispenser_pickup" in self.environment_config["game"]
            else False,
            hook=self.hook,
            random=self.random,
        )
        """Handles the creation of counters based on their config."""
        
        (
            self.counters,
            self.designated_player_positions,
            self.free_positions,
            self.kitchen_width,
            self.kitchen_height,
            self.layout_extra_info,
        ) = self.counter_factory.parse_layout_file(self.layout_config)
        self.original_designated_player_positions = self.designated_player_positions.copy()
        self.hook(LAYOUT_FILE_PARSED)
        self.free_spawn_positions: list[Tuple[npt.NDArray[float], str]] = copy(self.free_positions)
        self.free_positions.extend([(p, "Free") for p in self.designated_player_positions])
        self.movement: Movement = Movement(
            counter_positions=np.array([c.pos for c in self.counters]),
            player_config=self.environment_config["player_config"],
            world_borders=np.array(
                [[-0.5, self.kitchen_width - 0.5], [-0.5, self.kitchen_height - 0.5]],
                dtype=float,
            ),
            hook=self.hook,
        )
        """Does the movement of players in each step."""
        
        self.progressing_counters: list[Counter] = []
        """Counters that needs to be called in the step function via the `progress` method."""
        
        self.effect_manager: dict[
            str, EffectManager
        ] = self.counter_factory.setup_effect_manger()
        """Dict of effect managers. Currently only the fire effect manager."""
        
        self.overwrite_counters(self.counters)
        
        meals_to_be_ordered = self.recipe_validation.validate_environment(self.counters)
        # assert meals_to_be_ordered, "Need possible meals for order generation."
        
        available_meals = {meal: self.item_info[meal] for meal in meals_to_be_ordered}
        self.order_manager.set_available_meals(available_meals)
        self.order_manager.create_init_orders(self.env_time)
        self.start_time: datetime = self.env_time
        """The relative env time when it started."""
        self.env_time_end: datetime = self.env_time + timedelta(
            seconds=self.environment_config["game"]["time_limit_seconds"]
        )
        """The relative env time when it will stop/end"""
        log.debug(f"End time: {self.env_time_end}")
        
        self.info_msgs_per_player: dict[str, list[InfoMsg]] = defaultdict(list)
        """Cache of info messages per player which should be showed in the visualization of each player."""
        
        self.additional_state_content: dict[str, Any] = {}
        """The environment will extend the content of each state with this dictionary. Adapt it with the setter 
        function."""
        
        self.force_game_end = False
        """Hooks can set the variable to True when the env should end immediately."""
        self.hook(
            ENV_INITIALIZED,
            environment_config=env_config,
            layout_config=self.layout_config,
            seed=seed,
            env_start_time_worldtime=datetime.now(),
        )
    
    @property
    def game_ended(self) -> bool:
        """Whether the game is over or not based on the calculated `Environment.env_time_end`"""
        return self.env_time >= self.env_time_end or self.force_game_end
    
    def overwrite_counters(self, counters):
        """Resets counters.

        Args:
            counters: A list of counter objects representing the counters in the system.

        This method takes a list of counter objects representing the counters in the system and updates the counters
        attribute of the current object to the provided list. It also updates the counter_positions attribute of the
        movement object with the positions of the counters.

        Additionally, it assigns the counter classes with a "progress" attribute to the variable
        progress_counter_classes. It does this by filtering the classes in the cooperative_cuisine.counters module
        using the inspect module to only keep the classes that have a "progress" attribute.

        Next, it filters the counters based on whether their class is in the progress_counter_classes list and
        assigns the filtered counters to the progressing_counters attribute of the current object.

        Finally, it sets the counters for each effect manager in the effect_manager dictionary to the provided counters.
        """
        self.counters = counters
        self.movement.counter_positions = np.array([c.pos for c in self.counters])
        
        progress_counter_classes = list(
            filter(
                lambda cl: hasattr(cl, "progress"),
                dict(
                    inspect.getmembers(
                        sys.modules["cooperative_cuisine.counters"], inspect.isclass
                    )
                ).values(),
            )
        )
        self.progressing_counters = list(
            filter(
                lambda c: c.__class__ in progress_counter_classes,
                self.counters,
            )
        )
        for manager in self.effect_manager.values():
            manager.set_counters(counters)
    
    def get_env_time(self) -> datetime:
        """the internal time of the environment. An environment starts always with the time from `create_init_env_time`.

        Utility method to pass a reference to the serving window."""
        return self.env_time
    
    def _nearest_free_position(self, target_xy: np.ndarray) -> np.ndarray:
        """ Find the nearest available ground/designated position to the target.
        Does NOT consume (pop) from the env's lists; we only use it to choose a safe coordinate.
        """
        # Candidate pool: all known free ground tiles + designated spawn markers
        candidates = [p for (p, _) in self.free_positions] + list(self.original_designated_player_positions)

        if not candidates:
            # As a last resort, just clamp to world bounds (may collide, but better than None)
            x = float(np.clip(target_xy[0], 0.0, self.kitchen_width - 1.0))
            y = float(np.clip(target_xy[1], 0.0, self.kitchen_height - 1.0))
            return np.array([x, y], dtype=float)

        # Euclidean nearest neighbor
        d2 = [np.sum((c - target_xy) ** 2) for c in candidates]
        best_idx = int(np.argmin(d2))
        return candidates[best_idx].astype(float)


    def _formation_spawn_points(self) -> list[np.ndarray]:
        """
        Returns up to 3 spawn points:
        [ left-center, right-center, bottom-center ]
        Computed from kitchen width/height, then snapped to nearest free tile.
        """
        w, h = float(self.kitchen_width), float(self.kitchen_height)

        # Raw targets in continuous space
        left_center   = np.array([1.0,          h / 2.0], dtype=float)
        right_center  = np.array([w - 2.0,      h / 2.0], dtype=float)
        bottom_center = np.array([w / 2.0,      h - 2.0], dtype=float)

        # Snap each to nearest valid free position
        pts = [
            self._nearest_free_position(left_center),
            self._nearest_free_position(right_center),
            self._nearest_free_position(bottom_center),
        ]
        return pts

    
    def load_item_info(self, item_info: str | dict[str, ItemInfo]) -> dict[str, ItemInfo]:
        """Load `item_info.yml`, create ItemInfo classes and replace equipment strings with item infos."""
        self.hook(ITEM_INFO_CONFIG, item_info_config=item_info)
        if isinstance(item_info, str):
            item_info = yaml.safe_load(item_info)
        for item_name in item_info:
            item_info[item_name] = ItemInfo(name=item_name, **item_info[item_name])
        for item_name, single_item_info in item_info.items():
            if single_item_info.equipment:
                single_item_info.equipment = item_info[single_item_info.equipment]
        return item_info
    
    def perform_action(self, action: Action):
        """Performs an action of a player in the environment. Maps different types of action inputs to the
        correct execution of the players.
        Possible action types are movement, pickup and interact actions.

        Args:
            action: The action to be performed
        """
        assert action.player in self.players.keys(), "Unknown player."
        self.hook(PRE_PERFORM_ACTION, action=action)
        player = self.players[action.player]
        
        if action.action_type == ActionType.MOVEMENT:
            player.set_movement(
                action.action_data,
                self.env_time + timedelta(seconds=action.duration),
            )
        else:
            counter = get_closest(player.facing_point, self.counters)
            if player.can_reach(counter):
                if action.action_type == ActionType.PICK_UP_DROP:
                    player.put_action(counter)
                    self.hook(ACTION_PUT, action=action, counter=counter, player=action.player)
                elif action.action_type == ActionType.INTERACT:
                    if action.action_data == InterActionData.START:
                        player.perform_interact_start(counter)
                        self.hook(ACTION_INTERACT_START, action=action, counter=counter, player=action.player)
            else:
                self.hook(
                    ACTION_ON_NOT_REACHABLE_COUNTER, action=action, counter=counter, player=action.player
                )
            if action.action_data == InterActionData.STOP:
                player.perform_interact_stop()
        
        self.hook(POST_PERFORM_ACTION, action=action)
    
    def add_player(self, player_name: str, pos: npt.NDArray = None):
        """Add a player to the environment.

        Args:
            player_name: The id/name of the player to reference actions and in the state.
            pos: The optional init position of the player.

        Raises:
            ValueError: if the player_name already exists.
        """
        if player_name in self.players:
            raise ValueError(f"Player {player_name} already exists.")
        log.debug(f"Add player {player_name} to the game")
        player = Player(
            player_name,
            player_config=self.environment_config["player_config"],
            hook=self.hook,
            pos=pos,
        )
        self.players[player.name] = player
        #Assigning Position
        if player.pos is None:
            # Preferred: formation-based spawn for the first three players
            formation = self._formation_spawn_points()
            current_n = len(self.players)  # this includes the player we just inserted into self.players above

            # We want deterministic slots based on join order:
            #  - 1st player -> index 0 (left-center)
            #  - 2nd player -> index 1 (right-center)
            #  - 3rd player -> index 2 (bottom-center)
            if current_n <= 3:
                player.move_abs(formation[current_n - 1])
            else:
                # Fallback to original behavior for player 4+
                if len(self.designated_player_positions) > 0:
                    free_idx = self.random.randint(0, len(self.designated_player_positions) - 1)
                    player.move_abs(self.designated_player_positions[free_idx])
                    del self.designated_player_positions[free_idx]
                elif len(self.free_spawn_positions) > 0:
                    free_idx = self.random.randint(0, len(self.free_spawn_positions) - 1)
                    player.move_abs(self.free_spawn_positions[free_idx][0])
                    del self.free_spawn_positions[free_idx]
                else:
                    log.debug("No free positions left in kitchens")

            player.update_facing_point()
        
        self.movement.set_collision_arrays(len(self.players))
        self.hook(PLAYER_ADDED, player=player, pos=pos)
    
    def step(self, passed_time: timedelta):
        """Performs a step of the environment. Affects time based events such as cooking or cutting things, orders
        and time limits.
        """
        # self.hook(PRE_STEP, passed_time=passed_time)
        self.env_time += passed_time
        
        if self.game_ended:
            self.hook(GAME_ENDED_STEP, served_meals=self.order_manager.served_meals)
        else:
            for player in self.players.values():
                player.progress(passed_time, self.env_time)
            
            self.movement.perform_movement(
                passed_time, self.env_time, self.players, self.counters
            )
            self.hook(POST_PLAYER_MOVEMENT, player_positions=[p.pos.tolist() for p in self.players.values()],
                      player_facing_direction=[p.facing_direction.tolist() for p in self.players.values()],
                      player_ids=[p.name for p in self.players.values()])
            
            for counter in self.progressing_counters:
                counter.progress(passed_time=passed_time, now=self.env_time)
            self.order_manager.progress(passed_time=passed_time, now=self.env_time)
            for effect_manager in self.effect_manager.values():
                effect_manager.progress(passed_time=passed_time, now=self.env_time)
        self.hook(POST_STEP, passed_time=passed_time)
    
    def get_state(self, player_id: str = None) -> dict:
        """Get the current state of the game environment. The state here is accessible by the current python objects.

        Args:
            player_id: The player for which to get the state.

        Returns:
            The state of the game as a dict.
        """
        if player_id in self.players:
            self.hook(PRE_STATE, player_id=player_id)
            # add ground (self.free_space) too?
            state = {
                "players": [p.to_dict() for p in self.players.values()],
                "counters": [c.to_dict() for c in self.counters],
                "kitchen": {"width": self.kitchen_width, "height": self.kitchen_height},
                "score": self.score,
                "orders": self.order_manager.order_state(self.env_time),
                "ended": self.game_ended,
                "env_time": self.env_time.isoformat(),
                "remaining_time": max(
                    self.env_time_end - self.env_time, timedelta(0)
                ).total_seconds(),
                "view_restrictions": [
                    {
                        "direction": player.facing_direction.tolist(),
                        "position": player.pos.tolist(),
                        "angle": self.player_view_angle,
                        "counter_mask": None,
                        "range": self.player_view_range,
                    }
                    for player in self.players.values()
                ]
                if self.player_view_restricted
                else None,
                "served_meals": [
                    (player, str(meal))
                    for (meal, _, player) in self.order_manager.served_meals
                ],
                "info_msg": [
                    (msg["msg"], msg["level"])
                    for msg in self.info_msgs_per_player[player_id]
                    if msg["start_time"] < self.env_time < msg["end_time"]
                ],
                **self.additional_state_content,
            }
            self.hook(STATE_DICT, state=state, player_id=player_id)
            return state
        raise ValueError(f"No valid {player_id=}")
    
    def update_player_info(self, player_id: str, player_info_update: dict):
        """Update the player info of a player."""
        self.players[player_id].update_player_info(player_info_update)
    
    def get_json_state(self, player_id: str = None) -> str:
        """Return the current state of the game formatted in json dict.

        Args:
            player_id: The player for which to get the state.

        Returns:
            The state of the game formatted as a json-string

        """
        state = self.get_state(player_id)
        json_data = json.dumps(state)
        self.hook(JSON_STATE, json_data=json_data, player_id=player_id)
        # assert StateRepresentation.model_validate_json(json_data=json_data)
        return json_data
    
    def reset_env_time(self):
        """Reset the env time to the initial time, defined by `create_init_env_time`."""
        self.hook(PRE_RESET_ENV_TIME, env_time=self.env_time)
        self.env_time = create_init_env_time()
        self.hook(POST_RESET_ENV_TIME, env_time=self.env_time)
        log.debug(f"Reset env time to {self.env_time}")
    
    def register_callback_for_hook(self, hook_ref: str | list[str], callback: Callable):
        """Registers a callback function for a given hook reference.

        Args:
            hook_ref (str | list[str]): The reference to the hook or hooks for which the callback should be registered.
            It can be a single string or a list of strings.
            callback (Callable): The callback function to be registered for the specified hook(s).
            The function should accept the necessary parameters and perform the desired actions.

        """
        self.hook.register_callback(hook_ref, callback)
    
    def hook_callbacks(self):
        """Executes extra setup functions specified in the environment configuration."""
        if self.environment_config["hook_callbacks"]:
            for callback_name, setup_kwargs in self.environment_config[
                "hook_callbacks"
            ].items():
                # log.info(f"Setup hook callback {callback_name}")
                hooks_via_callback_class(name=callback_name, env=self, **setup_kwargs)
    
    def increment_score(self, score: int | float, info: str = ""):
        """Add a value to the current score and log it."""
        self.score += score
        self.hook(SCORE_CHANGED, score=self.score, increase=score, info=info)
        log.debug(f"Score: {self.score} ({score:+g}) - {info}")
    
    def update_additional_state_content(self, **kwargs):
        """
        Update the additional state content with the given key-value pairs.

        Args:
            **kwargs: The key-value pairs to update the additional state content with.

        """
        self.hook(ADDITIONAL_STATE_UPDATE, update=kwargs)
        self.additional_state_content.update(kwargs)
