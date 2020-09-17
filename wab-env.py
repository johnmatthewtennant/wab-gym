import gym
from gym import wrappers, logger
from gym import spaces
from gym.utils import seeding

import numpy as np
import pandas as pd
import itertools
from PIL import Image, ImageDraw

default_game_options = {
    "num_ostriches": 1,
    "height": 11,
    "width": 11,
    "chance_wolf_on_square": 0.01,
    "bush_power": 50,
    "max_food": 40,
    "food_gathered_per_turn": 5,
    "food_consumed_per_turn": 1,
    "max_berries_per_bush": 200,
    "wolf_spawn_distance": 12,
    "max_turns": 200,
    "wolf_chance_to_despawn": 0.05,
    "start_with_wolves": True,
    "starting_food": None,  # None values will be assigned randomly
    # "starting_food": 40,
    "starting_role": None,  # None values will be assigned randomly
    "wolves_can_move": False,
}

action_definitions = pd.DataFrame(
    [
        {"x": 0, "y": 1, "role": 0},  # up
        {"x": 1, "y": 0, "role": 0},  # right
        {"x": 0, "y": -1, "role": 0},  # down
        {"x": -1, "y": 0, "role": 0},  # left
        {"x": 0, "y": 0, "role": 0},  # stay
        {"x": 0, "y": 0, "role": 1},  # switch role
    ]
)


def get_distances_to_everything(ostriches, master_df):
    ostriches["key"] = 0
    ostriches = ostriches.rename(
        columns={
            "x": "ostrich_x",
            "y": "ostrich_y",
            "role": "ostrich_role",
            "alive_killed_starved": "ostrich_alive_killed_starved",
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
    both["point_distance"] = (both["delta_x"] ** 2 + both["delta_y"] ** 2) ** 0.5
    return both


def assemble_master_df(ostriches, bushes, wolves):
    master_df = pd.concat([wolves, bushes, ostriches])
    return master_df


class WolvesAndBushesEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, game_options=default_game_options, render=False):

        self.game_options = game_options
        if self.game_options["width"] % 2 == 0 or self.game_options["height"] % 2 == 0:
            raise ValueError("width and height must be odd numbers")
        # TODO action space will need to be tuple for multiple agents
        self.action_space = spaces.Discrete(6)  # up, right, down, left, stay, switch roles
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    0, 1, shape=(self.game_options["width"], self.game_options["height"]), dtype=int
                ),  # wolves
                spaces.Box(
                    0, 1, shape=(self.game_options["width"], self.game_options["height"]), dtype=int
                ),  # bushes
                spaces.Box(
                    0, 1, shape=(self.game_options["width"], self.game_options["height"]), dtype=int
                ),  # ostriches
                spaces.Box(0, 1, shape=(1, 1)),  # current food
                spaces.Discrete(2),  # current role
                spaces.Discrete(3),  # alive, killed, starved
            )
        )
        self.reset()

    def reset(self):
        self.current_turn = 0
        # TODO random seed
        self.ostriches = pd.DataFrame(
            columns=["type", "x", "y", "food", "role", "alive_killed_starved"]
        )
        self.bushes = pd.DataFrame(columns=["type", "x", "y", "food"])
        self.wolves = pd.DataFrame(columns=["type", "x", "y"])

        self.spawn_ostriches(
            starting_food=self.game_options["starting_food"],
            starting_role=self.game_options["starting_role"],
        )
        self.generate_bushes()
        if self.game_options["start_with_wolves"]:
            self.initialize_wolves()
        self.update_master_df_and_distances()
        return self._get_obs()

    def update_master_df_and_distances(self):
        self.master_df = assemble_master_df(
            self.ostriches, self.wolves, self.bushes[self.bushes.food > 0]
        )
        self.distances = get_distances_to_everything(self.ostriches, self.master_df)

    def step(self, actions):
        self.current_turn += 1
        action_details = action_definitions.iloc[actions]
        self.ostriches[["x", "y", "role"]] = (
            self.ostriches[["x", "y", "role"]] + action_details[["x", "y", "role"]]
        )
        self.ostriches.role = self.ostriches.role % 2
        self.generate_bushes()

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
        ostrich_wolf_kills = self.distances[
            (self.distances.taxicab_distance == 0) & (self.distances.object_type == "wolf")
        ]
        if not ostrich_wolf_kills.empty:
            self.ostriches.loc[ostrich_wolf_kills.ostrich_id, "alive_killed_starved"] = 1

        # ostrich eat
        ostrich_bush_pairs = self.distances[
            (self.distances.taxicab_distance == 0)
            & (self.distances.ostrich_role == 1)  # gatherer
            & (self.distances.object_type == "bush")
            & (self.distances.ostrich_alive_killed_starved == 0)
        ]
        if not ostrich_bush_pairs.empty:
            self.ostriches.loc[ostrich_bush_pairs.ostrich_id, "food"] += self.game_options[
                "food_gathered_per_turn"
            ]
            # TODO will this work if two ostriches are on the same bush? bush should go down by 2
            self.bushes.loc[ostrich_bush_pairs.object_id, "food"] -= self.game_options[
                "food_gathered_per_turn"
            ]

        # ostrich get hungry
        self.ostriches.food -= self.game_options["food_consumed_per_turn"]

        # ostrich starve
        self.ostriches.loc[self.ostriches.food < 0, "alive_killed_starved"] = 2

        if self.ostriches.iloc[0].alive_killed_starved == 0:
            reward = 0.05
        else:
            reward = -1
        done = (
            self.current_turn >= self.game_options["max_turns"] or reward == -1
        )  # or self.all_dead()
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        wolf_grid = np.zeros(self.observation_space[0].shape)
        bush_grid = np.zeros(self.observation_space[1].shape)
        ostrich_grid = np.zeros(self.observation_space[2].shape)

        # TODO fix ostrich_id, especially for multi agent
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
        visible_wolves = visible_objects[
            (visible_objects.object_type == "wolf") & (visible_objects.ostrich_role == 0)  # lookout
        ]
        wolf_grid[
            np.array(visible_wolves.delta_x + self.game_options["width"] // 2, int),
            np.array(visible_wolves.delta_y + self.game_options["height"] // 2, int),
        ] = 1
        visible_ostriches = visible_objects[(visible_objects.object_type == "ostrich")]
        ostrich_grid[
            np.array(visible_ostriches.delta_x + self.game_options["width"] // 2, int),
            np.array(visible_ostriches.delta_y + self.game_options["height"] // 2, int),
        ] = 1
        # print(bush_grid)
        # print(wolf_grid)
        food = [self.ostriches.iloc[0].food]
        role = self.ostriches.iloc[0].role
        alive_killed_starved = self.ostriches.iloc[0].alive_killed_starved
        return (wolf_grid, bush_grid, ostrich_grid, food, role, alive_killed_starved)

    def spawn_wolf_maybe(self, x, y, chance_to_spawn_wolf=0.01):
        if np.random.random() <= chance_to_spawn_wolf:
            self.wolves = self.wolves.append({"x": x, "y": y, "type": "wolf"}, ignore_index=True)

    def initialize_wolves(self):
        if self.ostriches.empty:
            return

        # get all coordinates where a wolf could be
        coords = set(
            itertools.product(
                range(
                    int(self.ostriches["x"].min() - self.game_options["wolf_spawn_distance"]),
                    int(self.ostriches["x"].max() + 1 + self.game_options["wolf_spawn_distance"]),
                ),
                range(
                    int(self.ostriches["y"].min() - self.game_options["wolf_spawn_distance"]),
                    int(self.ostriches["y"].max() + 1 + self.game_options["wolf_spawn_distance"]),
                ),
            )
        )

        for index, (x, y) in enumerate(coords):
            self.spawn_wolf_maybe(x, y, self.game_options["chance_wolf_on_square"])

    def spawn_ostriches(self, starting_food=None, starting_role=None):
        if starting_food is None:
            # starting_food = np.random.random()
            starting_food = np.random.randint(1, self.game_options["max_food"] + 1)
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
                "alive_killed_starved": 0,
            },
            ignore_index=True,
        )

    def generate_bushes(self):
        if self.ostriches.empty:
            return

        # get all coordinates that an ostrich can currently see
        ostrich_coords = set(
            itertools.product(
                range(
                    int(self.ostriches["x"].min() - self.game_options["width"]),
                    int(self.ostriches["x"].max() + 1 + self.game_options["width"]),
                ),
                range(
                    int(self.ostriches["y"].min() - self.game_options["height"]),
                    int(self.ostriches["y"].max() + 1 + self.game_options["height"]),
                ),
            )
        )

        # get coordinates of all current bushes
        if not self.bushes.empty:
            bush_coords = set(zip(self.bushes.x, self.bushes.y))
        else:
            bush_coords = set()

        # get visible coordinates without bush values
        new_coords = ostrich_coords - bush_coords
        new_bushes = pd.DataFrame(new_coords, columns=["x", "y"])
        new_bushes["type"] = "bush"
        new_bushes["food"] = self.generate_n_bush_values(len(new_bushes))

        self.bushes = self.bushes.append(new_bushes, ignore_index=True)

    def generate_n_bush_values(self, n):
        return np.round(
            np.random.random(n) ** self.game_options["bush_power"]
            * self.game_options["max_berries_per_bush"]
        )

    def render(self, mode="rgb_array", scale=32, draw_health=True):
        # This is for rendering video. Bushes are green, wolves red, ostriches blue
        wolves, bushes, ostriches, food, role, _ = self._get_obs()
        image = np.zeros(
            (self.game_options["width"], self.game_options["height"], 3), dtype="uint8"
        )

        image[:, :, 0] = 255 * wolves
        image[:, :, 1] = 255 * bushes
        image[:, :, 2] = 255 * ostriches
        if role == 0:
            empty = (image[:, :, 0] == 0) * (image[:, :, 1] == 0) * (image[:, :, 2] == 0)
            image[empty] = 255
        image = image.repeat(scale, axis=0).repeat(scale, axis=1)
        image_from_array = Image.fromarray(image)
        imd = ImageDraw.Draw(image_from_array)
        imd.text((0, 0), str(food[0]), fill="blue")
        return np.array(image_from_array)


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
