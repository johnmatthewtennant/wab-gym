import gym
from gym import wrappers, logger
from gym import spaces
from gym.utils import seeding

import numpy as np
import pandas as pd
import itertools
from PIL import Image, ImageDraw

default_game_options = {
    # GYM OPTIONS
    "reward_per_turn": 0,
    "reward_for_being_killed": -1,
    "reward_for_starving": -1,
    "reward_for_finishing": 1,
    "reward_for_eating": 0.1,
    "gatherer_only": False,  # Allows gatherers to see wolves. Disables lookout mode
    "lookout_only": True,
    "restrict_view": False, # True means apply the view mask, false means don't
    "starting_role": 1,  # None values will be assigned randomly
    # GAME
    "max_turns": 80,
    "num_ostriches": 1,
    "height": 11,  # Viewport height
    "width": 11,  # Viewport width
    "bush_power": 100,  # Power-law distribution: Higher values produce fewer bushes
    "max_berries_per_bush": 200,
    # FOOD
    "turns_to_fill_food": 8,  # How many turns of gathering does it take to fill food?
    "turns_to_empty_food": 40,  # How many turns of not gathering does it take to starve?
    "starting_food": 1,  # 0 to 1 float. None values will be assigned randomly
    # WOLVES
    "wolf_spawn_margin": 1,  # How many squares out from the player's view should wolves spawn?, 1 seems reasonable
    "chance_wolf_on_square": 0.001,  # Chance for wolf to spawn on each square of the perimeter, .001 seems reasonable
    "wolf_chance_to_despawn": 0.05,  # Chance for a wolf to despawn each turn, .05 seems reasonable
    "wolves": True,  # Spawn wolves?
    "wolves_can_move": True,  # Can wolves move?
}


def get_distances_to_everything(ostriches, master_df):
    ostriches["key"] = 0
    ostriches = ostriches.rename(
        columns={
            "x": "ostrich_x",
            "y": "ostrich_y",
            "role": "ostrich_role",
            "alive_starved_killed": "ostrich_alive_starved_killed",
        }
    )
    ostriches["ostrich_id"] = ostriches.index
    master_df["key"] = 0
    master_df["object_id"] = master_df.index
    master_df = master_df.rename(columns={"x": "object_x", "y": "object_y", "type": "object_type"})

    both = ostriches.merge(master_df, how="left", on="key")
    both.drop("key", 1, inplace=True)
    both["delta_x"] = both["ostrich_x"] - both["object_x"]
    both["delta_y"] = both["ostrich_y"] - both["object_y"]
    both["taxicab_distance"] = (abs(both["delta_x"]) + abs(both["delta_y"])).astype(int)
    # both["point_distance"] = (both["delta_x"] ** 2 + both["delta_y"] ** 2) ** 0.5
    return both


def assemble_master_df(ostriches, bushes, wolves):
    master_df = pd.concat([wolves, bushes, ostriches])
    return master_df


def generate_potential_actions(player=None):
    if player is None:
        player = {"x": 0, "y": 0}

    # up, right, down, left, stay (clockwise)
    potential_coords = [
        [player["x"], player["y"] + 1],
        [player["x"] + 1, player["y"]],
        [player["x"], player["y"] - 1],
        [player["x"] - 1, player["y"]],
        [player["x"], player["y"]],
    ]
    potential_actions = pd.DataFrame(potential_coords, columns=["x", "y"])
    return potential_actions


class DummySpec:
    # TODO don't use this dummy spec
    def __init__(
        self,
        id,
        reward_threshold=None,
        nondeterministic=False,
        max_episode_steps=None,
        kwargs=None,
    ):
        self.id = id
        self.reward_threshold = reward_threshold
        self.nondeterministic = nondeterministic
        self.max_episode_steps = max_episode_steps


class WolvesAndBushesEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 12}

    def __init__(self, game_options=default_game_options, render=False):
        self.game_options = game_options

        self.lookout_tile_mask = np.array(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ]
        )

        self.gatherer_tile_mask = np.asarray(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        self.spec = DummySpec(
            id="WolvesAndBushes-v0",
            max_episode_steps=game_options["max_turns"],
            # reward_threshold=(game_options["max_turns"] - 2) * game_options["reward_per_turn"]
            # + game_options["reward_for_finishing"],
            reward_threshold=80,  # TODO make this less arbitrary
        )
        if self.game_options["width"] % 2 == 0 or self.game_options["height"] % 2 == 0:
            raise ValueError("width and height must be odd numbers")
        if self.game_options["gatherer_only"]:
            self.action_definitions = pd.DataFrame(
                [
                    {"x": 0, "y": 1, "role": None},  # up
                    {"x": 1, "y": 0, "role": None},  # right
                    {"x": 0, "y": -1, "role": None},  # down
                    {"x": -1, "y": 0, "role": None},  # left
                    {"x": 0, "y": 0, "role": 1},  # don't move. Be gatherer
                ],
                dtype=int,
            )
        elif self.game_options["lookout_only"]:
            self.action_definitions = pd.DataFrame(
                [
                    {"x": 0, "y": 1, "role": None},  # up
                    {"x": 1, "y": 0, "role": None},  # right
                    {"x": 0, "y": -1, "role": None},  # down
                    {"x": -1, "y": 0, "role": None},  # left
                    {"x": 0, "y": 0, "role": 0},  # don't move. Be lookout
                ],
                dtype=int,
            )
        else:
            self.action_definitions = pd.DataFrame(
                [
                    {"x": 0, "y": 1, "role": None},  # up
                    {"x": 1, "y": 0, "role": None},  # right
                    {"x": 0, "y": -1, "role": None},  # down
                    {"x": -1, "y": 0, "role": None},  # left
                    {"x": 0, "y": 0, "role": 1},  # don't move. Be gatherer
                    {"x": 0, "y": 0, "role": 0},  # don't move. Be lookout
                ],
                dtype=int,
            )
        # TODO action space will need to be tuple for multiple agents
        self.initialize_action_space()
        self.initialize_observation_space()
        self.reset()

    def initialize_action_space(self):
        self.action_space = spaces.Discrete(
            len(self.action_definitions)
        )  # up, right, down, left, stay & gather, stay & lookout

    def initialize_observation_space(self):
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    0,
                    1,
                    shape=(
                        self.game_options["width"],
                        self.game_options["height"],
                    ),
                    dtype=int,
                ),  # wolves
                spaces.Box(
                    0,
                    1,
                    shape=(
                        self.game_options["width"],
                        self.game_options["height"],
                    ),
                    dtype=int,
                ),  # bushes
                spaces.Box(
                    0,
                    1,
                    shape=(
                        self.game_options["width"],
                        self.game_options["height"],
                    ),
                    dtype=int,
                ),  # ostriches
                spaces.Discrete(
                    self.game_options["turns_to_empty_food"] + 1
                ),  # turns until starvation (food)
                spaces.Discrete(2),  # current role
                spaces.Discrete(3),  # alive, starved, killed
            )
        )

    def reset(self):
        self.current_turn = 0
        # TODO random seed
        self.ostriches = pd.DataFrame(
            columns=["type", "x", "y", "food", "role", "alive_starved_killed"]
        )
        self.bushes = pd.DataFrame(columns=["type", "x", "y", "food"])
        self.wolves = pd.DataFrame(columns=["type", "x", "y"])

        self.spawn_ostriches(
            starting_food=self.game_options["starting_food"],
            starting_role=self.game_options["starting_role"],
        )
        self.generate_bushes()
        if self.game_options["wolves"]:
            self.initialize_wolves()
        self.update_master_df_and_distances()
        return self._get_obs()

    def step(self, actions):
        reward = 0
        self.current_turn += 1
        action_details = self.action_definitions.iloc[actions]
        # TODO applying the action will have to be totally rewritten for multiplayer
        self.ostriches.x = self.ostriches.x + action_details["x"]
        self.ostriches.y = self.ostriches.y + action_details["y"]
        if not np.isnan(action_details["role"]):
            self.ostriches.role = action_details["role"]
        self.generate_bushes()

        # wolves probabilistically despawn, remove wolves that fail check
        self.wolves = self.wolves[
            np.random.random(self.wolves.shape[0]) > self.game_options["wolf_chance_to_despawn"]
        ]

        self.update_master_df_and_distances()
        if self.game_options["wolves_can_move"]:
            # wolf move
            ostrich_wolf_pairs = self.distances.loc[
                (
                    self.distances[(self.distances.object_type == "wolf")]
                    .groupby(["object_id"])
                    .agg({"taxicab_distance": "idxmin"})
                ).taxicab_distance
            ]
            if not ostrich_wolf_pairs.empty:
                # TODO randomize when equal
                move_x = (
                    abs(ostrich_wolf_pairs.delta_x) >= abs(ostrich_wolf_pairs.delta_y)
                ) * np.sign(ostrich_wolf_pairs.delta_x)
                move_y = (
                    abs(ostrich_wolf_pairs.delta_x) < abs(ostrich_wolf_pairs.delta_y)
                ) * np.sign(ostrich_wolf_pairs.delta_y)
                # TODO this doesn't work because I'm not using object id as index properly. Currently hacking it and just ignore the index for this sum. Should probably make the dfs use uuid indices
                self.wolves.loc[ostrich_wolf_pairs.object_id, "x"] += list(move_x)
                self.wolves.loc[ostrich_wolf_pairs.object_id, "y"] += list(move_y)

            # TODO is it necessary to do this again?
            self.update_master_df_and_distances()

        # wolf kill
        if not self.game_options.get("god_mode"):
            ostrich_wolf_kills = self.distances[
                (self.distances.taxicab_distance == 0) & (self.distances.object_type == "wolf")
            ]
            if not ostrich_wolf_kills.empty:
                self.ostriches.loc[ostrich_wolf_kills.ostrich_id, "alive_starved_killed"] = 2

        # ostrich eat
        ostrich_bush_pairs = self.distances[
            (self.distances.taxicab_distance == 0)
            & ((self.distances.ostrich_role == 1) | self.game_options["lookout_only"])  # gatherer
            & (self.distances.object_type == "bush")
            & (self.distances.ostrich_alive_starved_killed == 0)
        ]
        if not ostrich_bush_pairs.empty:
            self.ostriches.loc[ostrich_bush_pairs.ostrich_id, "food"] += (
                1 / self.game_options["turns_to_fill_food"]
            )
            self.ostriches.food.clip(0, 1, inplace=True)
            # TODO will this work if two ostriches are on the same bush? bush should go down by 2
            self.bushes.loc[ostrich_bush_pairs.object_id, "food"] -= 1
            reward += self.game_options["reward_for_eating"]

        # ostrich get hungry
        self.ostriches.food -= 1 / self.game_options["turns_to_empty_food"]

        # ostrich starve
        self.ostriches.loc[self.ostriches.food <= 0, ["alive_starved_killed", "food"]] = [
            1,
            0,
        ]

        # wolf spawn
        if self.game_options["wolves"]:
            self.spawn_wolves()

        if self.ostriches.iloc[0].alive_starved_killed == 0:
            if self.current_turn >= self.game_options["max_turns"]:
                reward += self.game_options["reward_for_finishing"]
                done = True
            else:
                reward += self.game_options["reward_per_turn"]
                done = False
        elif self.ostriches.iloc[0].alive_starved_killed == 1:
            reward += self.game_options["reward_for_starving"]
            done = True
        else:  # self.ostriches.iloc[0].alive_starved_killed == 2:
            reward += self.game_options["reward_for_being_killed"]
            done = True

        return self._get_obs(), reward, done, {}

    def mask_grid(self, grid, role):
        if role == 1:
            # gatherer
            view_mask = self.gatherer_tile_mask
        else:
            view_mask = self.lookout_tile_mask
        #added this to test lookout only
        if not self.game_options["restrict_view"]:
            view_mask = np.zeros((11,11))

        blindspots = np.where(view_mask == 1)
        grid[blindspots] = 0

        return grid

    def _get_obs(self):
        role = self._get_role_obs()
        if role == 1:
            # gatherer
            view_mask = self.gatherer_tile_mask
        else:
            view_mask = self.lookout_tile_mask

        if not self.game_options["restrict_view"]:
            view_mask = np.zeros((11,11))

        wolves = self._get_wolf_grid()
        bushes = self._get_bush_grid()
        ostriches = self._get_ostrich_grid()

        return (
            wolves,
            bushes,
            ostriches,
            # here are the three ways of representing food, Not sure which is best
            # self._get_one_hot_food(),
            # self._get_food_quantity(),
            self._get_turns_until_starve(),
            role,
            self._get_alive_starved_killed_obs(),
            view_mask,
        )

    def _get_alive_starved_killed_obs(self):
        return self.ostriches.iloc[0].alive_starved_killed

    def _get_role_obs(self):
        return int(self.ostriches.iloc[0].role)

    def _get_ostrich_grid(self):
        ostrich_grid = np.zeros((self.game_options["width"], self.game_options["height"]))
        # vision_distance = self.game_options["vision_distance"]
        # TODO combine these next two lines
        visible_objects = self.distances[
            (self.distances.ostrich_id == 0)
            & (abs(self.distances.delta_x) < self.game_options["width"] / 2)
            & (abs(self.distances.delta_y) < self.game_options["height"] / 2)
        ]
        visible_ostriches = visible_objects[(visible_objects.object_type == "ostrich")]
        ostrich_grid[
            np.array(visible_ostriches.delta_x + self.game_options["width"] // 2, int),
            np.array(
                visible_ostriches.delta_y + self.game_options["height"] // 2,
                int,
            ),
        ] = 1
        return self.mask_grid(ostrich_grid, self._get_role_obs())

    def _get_wolf_grid(self):
        # TODO instead of masking, filter visible objects down to those within vision distance
        wolf_grid = np.zeros((self.game_options["width"], self.game_options["height"]))
        # TODO combine these next two lines
        visible_objects = self.distances[
            (self.distances.ostrich_id == 0)
            & (abs(self.distances.delta_x) < self.game_options["width"] / 2)
            & (abs(self.distances.delta_y) < self.game_options["height"] / 2)
        ]

        visible_wolves = visible_objects[visible_objects.object_type == "wolf"]

        wolf_grid[
            np.array(visible_wolves.delta_x + self.game_options["width"] // 2, int),
            np.array(visible_wolves.delta_y + self.game_options["height"] // 2, int),
        ] = 1
        return self.mask_grid(wolf_grid, self._get_role_obs())

    def _get_bush_grid(self):
        bush_grid = np.zeros((self.game_options["width"], self.game_options["height"]))
        # TODO combine these next two lines
        visible_objects = self.distances[
            (self.distances.ostrich_id == 0)
            & (abs(self.distances.delta_x) < self.game_options["width"] / 2)
            & (abs(self.distances.delta_y) < self.game_options["height"] / 2)
        ]
        visible_bushes = visible_objects[visible_objects.object_type == "bush"]
        bush_grid[
            np.array(visible_bushes.delta_x + self.game_options["width"] // 2, int),
            np.array(visible_bushes.delta_y + self.game_options["height"] // 2, int),
        ] = 1

        return self.mask_grid(bush_grid, self._get_role_obs())

    def _get_food_quantity(self) -> float:
        # Returns food value as a float between 0 and 1
        return self.ostriches.iloc[0].food

    def _get_turns_until_starve(self) -> int:
        # returns an integer denoting the number of turns until starvation
        return int(np.ceil(self.ostriches.iloc[0].food * self.game_options["turns_to_empty_food"]))

    def _get_one_hot_food(self) -> np.ndarray:
        food = np.zeros(
            (int(np.ceil(np.log2(self.game_options["turns_to_empty_food"]))),),
            dtype=int,
        )
        turns_to_death = np.maximum(
            self.ostriches.iloc[0].food * self.game_options["turns_to_empty_food"],
            1,
        )
        transformed_food = int(np.ceil(np.log2(turns_to_death)))
        if transformed_food < len(food):
            food[transformed_food] = 1
        return food

    def render(self, mode="rgb_array", scale=32, draw_health=True):
        # This is for rendering video. Bushes are green, wolves red, ostriches blue
        wolves = self._get_wolf_grid()
        bushes = self._get_bush_grid()
        ostriches = self._get_ostrich_grid()
        food = self._get_turns_until_starve()
        role = self.ostriches.iloc[0].role
        alive_starved_killed = self.ostriches.iloc[0].alive_starved_killed

        # wolves, bushes, ostriches, food, role, alive_starved_killed = self._get_obs()
        image = np.zeros(
            (self.game_options["width"], self.game_options["height"], 3),
            dtype="uint8",
        )

        image[:, :, 0] = 255 * wolves
        image[:, :, 1] = 255 * bushes
        image[:, :, 2] = 255 * ostriches
        if alive_starved_killed == 2:
            empty = (image[:, :, 0] == 0) * (image[:, :, 1] == 0) * (image[:, :, 2] == 0)
            image[empty] = 127
        else:
            empty = (image[:, :, 0] == 0) * (image[:, :, 1] == 0) * (image[:, :, 2] == 0)
            image[empty] = 255
            image[:, :, 0] = self.mask_grid(image[:, :, 0], role)
            image[:, :, 1] = self.mask_grid(image[:, :, 1], role)
            image[:, :, 2] = self.mask_grid(image[:, :, 2], role)
        image = image.repeat(scale, axis=0).repeat(scale, axis=1)
        if draw_health:
            image_from_array = Image.fromarray(image)
            imd = ImageDraw.Draw(image_from_array)
            imd.text((0, 0), str(food), fill="blue")
            return np.array(image_from_array)
        else:
            return image

    def update_master_df_and_distances(self):
        self.master_df = assemble_master_df(
            self.ostriches, self.wolves, self.bushes[self.bushes.food > 0]
        )
        self.distances = get_distances_to_everything(self.ostriches, self.master_df)

    def visible_coords(self):
        visible_coords = set()
        for _, ostrich in self.ostriches.iterrows():
            visible_coords = visible_coords | set(
                itertools.product(
                    range(
                        int(ostrich.x - (self.game_options["width"] // 2)),
                        int(ostrich.x + (self.game_options["width"] // 2)) + 1,
                    ),
                    range(
                        int(ostrich.y - self.game_options["height"] // 2),
                        int(ostrich.y + self.game_options["height"] // 2) + 1,
                    ),
                )
            )
        return visible_coords

    def spawn_wolves(self):
        if self.ostriches.empty:
            return

        # get candidate wolf spawn positions
        candidate_spawn_coords = set()
        for _, ostrich in self.ostriches.iterrows():
            candidate_spawn_coords = candidate_spawn_coords | set(
                itertools.product(
                    range(
                        int(
                            ostrich.x
                            - self.game_options["width"] // 2
                            - self.game_options["wolf_spawn_margin"]
                        ),
                        int(
                            ostrich.x
                            + self.game_options["width"] // 2
                            + self.game_options["wolf_spawn_margin"]
                            + 1
                        ),
                    ),
                    range(
                        int(
                            ostrich.y
                            - self.game_options["height"] // 2
                            - self.game_options["wolf_spawn_margin"]
                        ),
                        int(
                            ostrich.y
                            + self.game_options["height"] // 2
                            + self.game_options["wolf_spawn_margin"]
                            + 1
                        ),
                    ),
                )
            )

        # get visible coordinates without bush values
        new_coords = candidate_spawn_coords - self.visible_coords()
        new_wolves = pd.DataFrame(new_coords, columns=["x", "y"])
        new_wolves["type"] = "wolf"
        # TODO decide if it should be chance_wolf_on_square/2
        self.wolves = self.wolves.append(
            new_wolves[
                np.random.random(new_wolves.shape[0])
                < self.game_options["chance_wolf_on_square"] / 2
            ],
            ignore_index=True,
        )

    def initialize_wolves(self):
        if self.ostriches.empty:
            return

        # get all coordinates where a wolf could be
        new_coords = self.visible_coords()
        new_wolves = pd.DataFrame(new_coords, columns=["x", "y"])
        new_wolves["type"] = "wolf"
        # TODO decide if it should be chance_wolf_on_square/2
        self.wolves = self.wolves.append(
            new_wolves[
                np.random.random(new_wolves.shape[0])
                < self.game_options["chance_wolf_on_square"] / 2
            ],
            ignore_index=True,
        )

    def spawn_ostriches(self, starting_food=None, starting_role=None):
        if starting_food is None:
            starting_food = np.random.random()
        if starting_role is None:
            starting_role = np.random.randint(2)
        # TODO multiple playersp
        self.ostriches = self.ostriches.append(
            {
                "type": "ostrich",
                "x": 0,
                "y": 0,
                "food": starting_food,
                "role": starting_role,
                "alive_starved_killed": 0,
            },
            ignore_index=True,
        )

    def generate_bushes(self):
        if self.ostriches.empty:
            return

        # get coordinates of all current bushes
        if not self.bushes.empty:
            bush_coords = set(zip(self.bushes.x, self.bushes.y))
        else:
            bush_coords = set()

        # get visible coordinates without bush values
        new_coords = self.visible_coords() - bush_coords
        new_bushes = pd.DataFrame(new_coords, columns=["x", "y"])
        new_bushes["type"] = "bush"
        new_bushes["food"] = self.generate_n_bush_values(len(new_bushes))

        self.bushes = self.bushes.append(new_bushes, ignore_index=True)

    def generate_n_bush_values(self, n):
        return np.round(
            np.random.random(n) ** self.game_options["bush_power"]
            * self.game_options["max_berries_per_bush"]
        )

    def _get_wolf_proximities(self):
        if not self.wolves.empty:
            potential_actions = generate_potential_actions(self.ostriches.iloc[0])
            distances = get_distances_to_everything(potential_actions, self.wolves.copy())
            wolf_distance = (
                distances.groupby(
                    ["ostrich_id"]
                ).agg(  # TODO ostrich_id here is actually action_id and should be renamed as such
                    {"taxicab_distance": "min"}
                )
            )["taxicab_distance"]
        else:
            wolf_distance = pd.Series([0] * 5)
        return np.clip(self.max_distance - wolf_distance, 0, self.max_distance)

    def _get_bush_proximities(self):
        if not self.bushes[self.bushes.food > 0].empty:
            potential_actions = generate_potential_actions(self.ostriches.iloc[0])
            bushes = self.bushes.loc[self.bushes.food > 0].copy()
            distances = get_distances_to_everything(potential_actions, bushes)
            bush_distance = (
                distances.groupby(
                    ["ostrich_id"]
                ).agg(  # TODO ostrich_id here is actually action_id and should be renamed as such
                    {"taxicab_distance": "min"}
                )
            )["taxicab_distance"]
        else:
            bush_distance = pd.Series([0] * 5)

        return np.clip(self.max_distance - bush_distance, 0, self.max_distance)


class PragmaticObsWrapper(gym.ObservationWrapper):
    """A wrapper class that outputs the following obs:
    nearest_wolf:
        A list of four quantities ([up, right, down, left]), denoting the
        distance in each direction from the closest wolf. if the player
        exists at [5][5], and the nearest wolf is located at [7, 9]
        (both given from the top-left corner), then
        nearest_wolf would be: [0, 4, 0, 2]
    num_wolves:
        A list of four quantities, each describing the number of wolves in
        all four directions ([up, right, down, left]) relative to the player.
        For example, for a grid like:
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, w, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        num_wolves would be: [4, 3, 3, 3] (wolves can count in more than one zone)
    nearest_bush:
        same as nearest_wolf but for bushes
    num_bushes:
        same as num_wolves but for bushes
    food:
        an integer representing the number of turns until starving
    role and alive_starved_killed are the same as always.
    view_mask is an 11x11 grid of tiles. The 0s are visible tiles, and the 1s
    are out of the field of view of the agent. use for calculations accordingly
    here.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.max_distance = self.game_options["width"] // 2 + self.game_options["height"] // 2 + 1
        self.observation_space = spaces.Tuple(
            (
                spaces.Tuple([spaces.Discrete(self.max_distance + 1)] * 4),  # nearest wolf
                spaces.Tuple([spaces.Discrete(self.max_distance + 1)] * 4),  # second nearest wolf
                spaces.Tuple([spaces.Discrete(11)] * 4),  # num wolves (up to a max of 10)
                spaces.Tuple([spaces.Discrete(self.max_distance + 1)] * 4),  # nearest bush
                spaces.Tuple([spaces.Discrete(self.max_distance + 1)] * 4),  # second nearest bush
                spaces.Tuple([spaces.Discrete(11)] * 4),  # num bushes (up to a max of 10)
                spaces.Discrete(2),  # standing on bush
                self.env.observation_space[3],  # food
                self.env.observation_space[4],  # role
                self.env.observation_space[5],  # alive_killed_starved
                spaces.Box(0, 1, shape=(121,), dtype=int),  # view mask
            )
        )

    def observation(self, obs):

        view_mask = obs[6]

        wolves = obs[0]
        bushes = obs[1]

        nearest_wolf, second_nearest_wolf = self._get_nearest_things(wolves)
        wolves_in_each_direction = np.minimum(self._get_num_things_each_direction(wolves), 10)

        nearest_bush, second_nearest_bush = self._get_nearest_things(bushes)
        bushes_in_each_direction = np.minimum(self._get_num_things_each_direction(bushes), 10)

        # note: this was obs[2][self.max_distance//2, self.max_distance//2] but
        #   obs[2] currently corresponds to the ostrich grid. I changed it to
        #   bushes to work properly.
        standing_on_bush = int(bushes[self.max_distance // 2, self.max_distance // 2])

        # assuming food has int output (turns until starving)
        food = obs[3]
        role = obs[4]
        alive_starved_killed = obs[5]

        return (
            nearest_wolf,
            second_nearest_wolf,
            wolves_in_each_direction,
            nearest_bush,
            second_nearest_bush,
            bushes_in_each_direction,
            standing_on_bush,
            food,
            role,
            alive_starved_killed,
            view_mask,
        )

    def _get_nearest_things(self, binary_map: np.ndarray):

        # abstracted so this method can be used with wolf maps, bush maps,
        #   and possibly ostrich maps in the future

        # indexes is a tuple of two np.ndarrays, the first denoting the rows
        #   and the second denoting the columns of each wolf (.
        indexes = np.where(binary_map == 1)
        if len(indexes[0]) == 0:
            return [0, 0, 0, 0], [0, 0, 0, 0]

        shortest_taxicab = self.max_distance

        second_shortest_taxicab = self.max_distance
        shortest_taxicab_indexes = [0, 0]
        for j in range(len(indexes[0])):
            relative_row = indexes[0][j] - self.game_options["height"] // 2
            relative_column = indexes[1][j] - self.game_options["width"] // 2
            taxicab = abs(relative_row) + abs(relative_column)
            if taxicab <= shortest_taxicab:
                second_shortest_taxicab = shortest_taxicab
                second_shortest_taxicab_indexes = shortest_taxicab_indexes[:]
                shortest_taxicab = taxicab
                shortest_taxicab_indexes[0] = relative_row
                shortest_taxicab_indexes[1] = relative_column
            elif (taxicab <= second_shortest_taxicab):
                second_shortest_taxicab = taxicab
                second_shortest_taxicab_indexes[0] = relative_row
                second_shortest_taxicab_indexes[1] = relative_column
        up = abs(min(shortest_taxicab_indexes[0], 0))
        up = bool(up) * (self.max_distance - up)
        right = max(shortest_taxicab_indexes[1], 0)
        right = bool(right) * (self.max_distance - right)
        down = max(shortest_taxicab_indexes[0], 0) # negative indexes mean up
        down = bool(down) * (self.max_distance - down)
        left = abs(min(shortest_taxicab_indexes[1], 0))
        left = bool(left) * (self.max_distance - left)

        up2 = abs(min(second_shortest_taxicab_indexes[0], 0))
        up2 = bool(up2) * (self.max_distance - up2)
        right2 = max(second_shortest_taxicab_indexes[1], 0)
        right2 = bool(right2) * (self.max_distance - right2)
        down2 = max(second_shortest_taxicab_indexes[0], 0)
        down2 = bool(down2) * (self.max_distance - down2)
        left2 = abs(min(second_shortest_taxicab_indexes[1], 0))
        left2 = bool(left2) * (self.max_distance - left2)

        return ([up, right, down, left], [up2, right2, down2, left2])

    def _get_num_things_each_direction(self, binary_map):
        # also abstracted for the same reason as _get_nearest_thing
        half_row_index = self.game_options["height"] // 2
        half_column_index = self.game_options["width"] // 2

        # asserting that these values are 0 on the view_mask as well in order
        #   to include them.
        up = np.count_nonzero(binary_map[0:half_row_index, :] == 1)
        right = np.count_nonzero(binary_map[:, half_column_index + 1 :] == 1)
        down = np.count_nonzero(binary_map[half_row_index + 1 :, :] == 1)
        left = np.count_nonzero(binary_map[:, 0:half_column_index] == 1)

        return [up, right, down, left]


class NNFriendlyObsWrapper(gym.ObservationWrapper):
    """A wrapper class used to convert observation space data to
    a generalized format.
    """

    def __init__(self, env: gym.Env):
        """Initialize a new wrapper instance
        :param env: The environment from which the observations space data
        will be collected.
        """
        super().__init__(env)
        self.env = env
        self.max_distance = self.game_options["width"] // 2 + self.game_options["height"] // 2 + 1

    def _get_condensed_taxicabs(self, ob):
        # assuming when the input is a np.ndarray it is a 0/1 map
        if isinstance(ob, np.ndarray):
            indexes = np.where(ob == 1)
            ob = np.empty(len(indexes[0]))

            for j in range(len(indexes[0])):

                taxicab = abs(indexes[0][j] - self.game_options["height"] // 2) + abs(
                    indexes[1][j] - self.game_options["width"] // 2
                )
                # we use * 2 - 1 here to adjust the value for [-1, 1]

                # we use * 2 - 1 here to adjust the value for [-1, 1]
                ob.append(((self.max_distance - taxicab) / self.max_distance * 2) - 1)

        return ob

    def observation(self, obs):
        """Adjust all observations values in obs to be neural net friendly
        (i.e. 1D array where all values are between -1 and 1).
        """

        wolves = self._get_condensed_taxicabs(obs[0])

        bushes = self._get_condensed_taxicabs(obs[1])

        # Comment below if not using ostriches as part of the observations. Also
        #   must adjust indexes accordingly if this is the case.
        # ostriches = self._get_condensed_taxicabs(obs[2])

        food = obs[2]

        if isintance(food, int) or isinstance(food, float):
            if isinstance(food, float):
                food = int(np.maximum(food * self.game_options["turns_to_empty_food"], 1))
            # here food is an int
            one_hot = np.zeros(food)
            transformed_food = int(np.ceil(np.log2(food)))
            if transformed_food < len(food):
                one_hot[transformed_food] = 1
            food = one_hot

        # assuming role is either 0 or 1
        role = obs[3]

        # subtracting 1 for NN friendlyness ([0, 2] -> [-1, 1])
        alive_starved_killed = obs[4] - 1
        return np.concatenate(
            (
                wolves,
                bushes,
                food,
                np.array([role]),
                np.array([alive_starved_killed]),
            )
        )


class SuperBasicObservationWrapper(PragmaticObsWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.max_distance = self.game_options["width"] // 2 + self.game_options["height"] // 2 + 1
        self.observation_space = spaces.Tuple(
            (
                spaces.Tuple([spaces.Discrete(self.max_distance)] * 4),  # nearest bush
                self.env.observation_space[3],  # food
                self.env.observation_space[4],  # role
                self.env.observation_space[5],  # alive_killed_starved
            )
        )

    def observation(self, obs):

        bushes_in_each_direction = self._get_num_things_each_direction(obs[1])
        nearest_bush, second_nearest_bush = self._get_nearest_things(obs[1])
        food = obs[3]
        role = obs[4]
        alive_starved_killed = obs[5]

        return (
            nearest_bush,
            food,
            role,
            alive_starved_killed,
        )


class WolvesAndBushesEnvEgoCentric(WolvesAndBushesEnv):
    def initialize_observation_space(self):
        self.max_distance = (
            self.game_options["width"] // 2 + self.game_options["height"] // 2 + 1
        )  # this is the maximum taxicab distance any object on screen can be from any of the squares the ostrich can reach on the next turn
        self.observation_space = spaces.Tuple(
            (
                # spaces.Tuple(
                #     [spaces.Discrete(self.max_distance + 1)] * 5
                # ),  # proximity to nearest wolves
                spaces.Tuple(
                    [spaces.Discrete(self.max_distance + 1)] * 5
                ),  # proximities to nearest bushes
                spaces.Discrete(self.game_options["turns_to_empty_food"] + 1),  # current food
                spaces.Discrete(2),  # current role
                spaces.Discrete(3),  # alive, starved, killed
            )
        )

    def _get_obs(self):
        wolves = self._get_wolf_proximities()
        bushes = self._get_bush_proximities()
        food = self._get_turns_until_starve()
        role = self._get_role_obs()
        alive_starved_killed = self._get_alive_starved_killed_obs()

        return (bushes, food, role, alive_starved_killed)  # wolves,

    def _get_raw_obs(self):
        return super()._get_obs(self)


class WolvesAndBushesEnvEgocentricJustBushes(WolvesAndBushesEnvEgoCentric):
    def initialize_observation_space(self):
        self.max_distance = (
            self.game_options["width"] // 2 + self.game_options["height"] // 2 + 1
        )  # this is the maximum taxicab distance any object on screen can be from any of the squares the ostrich can reach on the next turn
        self.observation_space = spaces.Tuple([spaces.Discrete(self.max_distance + 1)] * 5)
        # proximities to nearest bushes

    def initialize_action_space(self):
        self.action_space = spaces.Discrete(5)

    def _get_obs(self):
        # wolves = self._get_wolf_proximities()
        bushes = self._get_bush_proximities()
        # food = self._get_turns_until_starve()
        # role = self._get_role_obs()
        # alive_starved_killed = self._get_alive_starved_killed_obs()
        return bushes  # wolves,


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()



if __name__ == "__main__":
    # NOTE FROM JOHN: This code is just for testing the env with a random agent.
    # To use the env, just import it from wab-env import WolvesAndBushesEnv
    # then env = WolvesAndBushesEnv() or env = WolvesAndBushesEnv(game_options)
    # THE REST OF THE CODE AND COMMENTS ARE NOT MINE

    # OPEN AI GYM ----
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)
    env = WolvesAndBushesEnv()
    # env = PragmaticObsWrapper(WolvesAndBushesEnv())
    # env = WolvesAndBushesEnvEgocentricJustBushes()
    # env = NNFriendlyObsWrapper(WolvesAndBushesEnvEgocentric())

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = "/tmp/random-agent-results"
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

