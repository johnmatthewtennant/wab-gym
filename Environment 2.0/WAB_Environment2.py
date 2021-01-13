from World import World
from WAB_Environment2_Single import WAB_Environment2_Single
import numpy as np
import pandas as pd
import gym
import random


default_game_options = {
    # GYM OPTIONS
    "ostrich_mode_or_wolf_mode": 0, #ostrich mode is 0, wolf mode is 1
    "reward_per_turn": 0,
    "reward_for_being_killed": -1,
    "reward_for_starving": -1,
    "reward_for_finishing": 1,
    "reward_for_eating": 0,
    "gatherer_only": False,  # Allows gatherers to see wolves. Disables lookout mode
    "lookout_only": True,
    "restrict_view": False, # True means apply the view mask, false means don't
    "starting_role": 1,  # None values will be assigned randomly
    # GAME
    "max_turns": 80,
    "num_ostriches": 20,
    "height": 11,  # Viewport height
    "width": 11,  # Viewport width
    "bush_power": 100,  # Power-law distribution: Higher values produce fewer bushes
    "max_berries_per_bush": 200,
    # BUSHES
    "food_per_bush": 20,
    "food_given_per_turn": 5,
    # OSTRICHES
    "ostrich_starting_food": 40.0,
    "ostrich_food_eaten_per_turn": 1.0,
    "ostrich_move_speed": 1.0,
    "lookout_view_radius": 9, #specifies the range that the ostrich can see in lookout
    "gatherer_view_radius": 5,# "" same for gatherer
    # WOLVES
    "num_wolves": 20,
    "wolf_spawn_margin": 1,  # How many squares out from the player's view should wolves spawn?, 1 seems reasonable
    "chance_wolf_on_square": 0.001,  # Chance for wolf to spawn on each square of the perimeter, .001 seems reasonable
    "wolves": True,  # Spawn wolves?
    "wolf_starting_food": 20,
    "wolf_food_for_eating_ostrich": 10,
    "wolves_can_move": True,  # Can wolves move?
    "wolf_walk_speed": 1.0, # speed that a wolf can move while not sprinting.
    "wolf_walk_cost": 0.1, # cost of walking for a wolf per unit moved
    "wolf_run_speed": 2.0, # speed of a sprinting wolf
    "wolf_run_cost": 0.2, # cost of running for a wolf per unit moved
    "wolf_view_radius": 6
}


class WAB_Environment2():

    def __init__(self, world_width, world_height,
                 game_options=default_game_options):
        self._world = World(world_width, world_height, game_options)
        self._environments = []
        self.num_entities_acted_this_turn = 0

    def create_ostriches(self, num_ostriches, spawn_positions=[]):
        if spawn_positions == []:
            #generate random coordinates
            spawn_positions = [(random.randint(0, self._world.get_width()-1),
                                random.randint(0, self._world.get_height()-1))
                               for _ in range(num_ostriches)]
        if len(spawn_positions) != num_ostriches:
            print("Did not specify all the positions of the ostriches, "
                  "generating random positions for the rest.")
            spawn_positions.extend([(random.randint(0, self._world.get_width()-1),
                                     random.randint(0, self._world.get_height()-1))
                                    for _ in range(num_ostriches - len(spawn_positions))])
        for i in range(num_ostriches):
            self._environments.append(WAB_Environment2_Single(self._world, "Ostrich",
                                                             spawn_positions[i][0],
                                                             spawn_positions[i][1]))

    def create_wolves(self, num_wolves, spawn_positions=[]):
        if spawn_positions == []:
            #generate random coordinates
            spawn_positions = [(random.randint(0, self._world.get_width()-1),
                                random.randint(0, self._world.get_height()-1))
                               for _ in range(num_wolves)]
        if len(spawn_positions) !=  num_wolves:
            print("Did not specify all the positions of the wolves, "
                  "generating random positions for the rest.")
            spawn_positions.extend([(random.randint(0, self._world.get_width()-1),
                                     random.randint(0, self._world.get_height()-1))
                                    for _ in range(num_wolves - len(spawn_positions))])
        for i in range(num_wolves):
            self._environments.append(WAB_Environment2_Single(self._world, "Wolf",
                                                             spawn_positions[i][0],
                                                             spawn_positions[i][1]))

    def create_bushes(self, num_bushes, spawn_positions=[]):
        if spawn_positions == []:
            #generate random coordinates
            spawn_positions = [(random.randint(0, self._world.get_width()-1),
                                random.randint(0, self._world.get_height()-1))
                               for _ in range(num_bushes)]
        if len(spawn_positions) !=  num_bushes:
            print("Did not specify all the positions of the bushes, "
                  "generating random positions for the rest.")
            spawn_positions.extend([(random.randint(0, self._world.get_width()-1),
                                     random.randint(0, self._world.get_height()-1))
                                    for _ in range(num_bushes - len(spawn_positions))])
        for i in range(num_bushes):
            self._environments.append(WAB_Environment2_Single(self._world, "Bush",
                                                             spawn_positions[i][0],
                                                             spawn_positions[i][1]))


    def reset_environment(self):
        """Resets each entity, currently randomizing their positions."""
        for env in self._environments:
            env.reset()
        self.num_entities_acted_this_turn = 0
        self._world.reset_world()

    def get_obs(self, entity_id):
        """Return the observations of the specified entity, which should be done
        right before the entity sends their action."""
        return self._environments[entity_id].get_obs()

    def take_action(self, entity_id, action):
        """ Performs the specified action for the specified entity. returns
        the reward, and a boolean representing whether or not the entity is
        done acting this epoch. """
        self.num_entities_acted_this_turn += 1
        reward, done = self._environments[entity_id].step(action)
        if self.num_entities_acted_this_turn == len(self._environments):
            self._world.increment_turn()
            self.num_entities_acted_this_turn = 0
        return reward, done

    def set_entity_act_function(self, entity_id, act_function):
        """Change how the entity performs its action according to the passed-in
        function."""
        self.world.set_entity_act_function(entity_id, act_function)

    def set_entity_internal_obs_function(self, entity_id, internal_obs_function):
        """Change how the entity's observations are collected using the passed-in
        function"""
        self.world.set_entity_internal_obs_function(entity_id,
                                                    internal_obs_function)

    def set_entity_external_obs_function(self, entity_id, external_obs_function):
        """Change how the entity performs its action according to the passed-in
        function."""
        self.world.set_entity_external_obs_function(entity_id,
                                                    external_obs_function)

    def set_update_function(self, update_function):
        """Change how the update step of the environment functions."""
        self.world.set_update_function(update_function)

    def set_reward_function(self, entity_id, reward_function):
        """Change the reward structure of the specified entity according to
        the passed-in reward function"""
        self.word.set_reward_function(entity_id, reward_function)
