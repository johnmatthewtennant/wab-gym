import gym
import numpy as np
import pandas as pd
from World import World
import random


class WAB_Environment2_Single(gym.Env):
    """This is our second environment. The main extensions this environment
    provides are the dynamic wolves, the globe-like and extant world, and
    the ability to simulate multiple agents at once.

    One thing to note is that we have decided to keep the wolves on a finer
    grid space than the ostriches and bushes. This allows them to move at a
    fraction (e.g. 1.5) of the speed of the ostriches, which gives us better
    tools to balance the environment. In fact, we can actually calculate the
    dimensions of this grid based on the ratio of the ostrich speed to the
    wolf's running speed, saving us the hassle of having to change the
    wolf grid dimensions manually. This should be done in __init__.

    When we return ostrich-specific information, we should do it index-wise,
    i.e. when we return our observations for the ostrich that is 3rd in the
    list of ostriches, we should return these observations as 3rd in the list of
    observations.
    """

    def __init__(self, world, type, starting_position_x, starting_position_y):
        """Initialize an environment with the specified width, height, and
        game_options.
        """
        self.world = world
        self.id = self.world.create_entity(type, starting_position_x,
                                           starting_position_y)
        self.current_turn = 0

    def reset(self, new_x=-1, new_y=-1):
        self.current_turn = 0
        # when we reset the entity, we randomize its position
        if new_x < 0 or new_y < 0:
            new_x, new_y = self._get_random_spawn_indices()
        self.world.reset_entity(self.id, new_x, new_y)

    def _get_random_spawn_indices(self):
        """Return a random position on the world space."""
        x = random.randint(0, self.world.get_width())
        y = random.randint(0, self.world.get_height())

        return x, y

    def step(self, action):
        """Steps through one turn of the epoch. This involves:
        1. Performing each action specified for each ostrich.
        2. Calling a wolf policy method that determines the each wolves' actions
            performing that action on each wolf.
        3. Updating the master globe dataframe to reflect the new state.
        4. Collecting observations about the world by calling _get_obs()
        5. Calculating the reward of the new state for each ostrich.
        6. returning the reward of the one ostrich as "reward" output, returning
            the rest as part of the "observations" output.
        7. determining whether all the ostriches are dead, if so done=True.
        8. Return all these values.
        """
        reward = self.world.perform_entity_action(self.id, action, self.current_turn)

        self.current_turn += 1

        done = self.world.is_entity_done(self.id)

        return reward, done

    def get_obs(self):
        """Return RAW observations. This means we're going to have a massive
        list of raw observations for each ostrich. Each ostrich should have:
        1. A grid showing the nearby ostriches
        2. A grid showing the nearby wolves
        3. A grid showing the nearby bushes
        4. A view mask, showing what the ostrich can actually see
        5. An integer representing the state (alive_starved_killed)
        6. An integer representing the role (gatherer or lookout)
        7. An integer representing the food somehow (one hot encoded or
            something similar)

        This method will end up being very computationally intensive, and will
        have to be called on each turn.

        Anything that "softens" this data to make learning easier should be
        done through wrappers.
        """
        return self.world.get_observations(self.id, self.current_turn)


