import pandas as pd
import Entity
import Wolf
import Bush
import Ostrich
import asyncio


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
    # WOLVES
    "num_wolves": 20,
    "wolf_spawn_margin": 1,  # How many squares out from the player's view should wolves spawn?, 1 seems reasonable
    "chance_wolf_on_square": 0.001,  # Chance for wolf to spawn on each square of the perimeter, .001 seems reasonable
    "wolves": True,  # Spawn wolves?
    "wolf_starting_food": 20,
    "wolves_can_move": True,  # Can wolves move?
    "wolf_walk_speed": 1.0, # speed that a wolf can move while not sprinting.
    "wolf_walk_cost": 0.1, # cost of walking for a wolf per unit moved
    "wolf_run_speed": 2.0, # speed of a sprinting wolf
    "wolf_run_cost": 0.2, # cost of running for a wolf per unit moved
}


def default_bush_obs(self):
    return [self.x, self.y, self.food]

def default_bush_visible_data(self):
    return [self.food]

def default_ostrich_visible_data(self):
    return

def default_ostrich_obs(self):
    return [self.x, self.y, self.food, self.role, self.status]

def default_wolf_visible_data(self):
    return

def default_wolf_obs(self):
    return [self.x, self.y, self.food, self.is_running, self.status]


class World:
    """A class representing the world of the environment.

    This class should store all entities in the world space and each entity
    should have a unique ID associated with it.
    """

    def __init__(self, world_width, world_height):
        """Initialize the World"""
        self._id_generator = -1
        self._current_turn = 0
        self._world_width = world_width
        self._world_height = world_height
        # currently this is the setup for storing entities.
        self._entities = pd.DataFrame(columns=["Id", "Type", "Entity Object",
                                               "X", "Y"])

    def create_wolf(self, starting_position_x, starting_position_y,
                    act_function=None, obs_function=default_wolf_obs,
                    visible_data_function=default_wolf_visible_data):
        """Create a new wolf and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1
        self._entities.loc[self._id_generator] = [self._id_generator,
             "Wolf",
             Wolf.Wolf(self._id_generator,
                                   starting_position_x, starting_position_y,
                                   act_function, obs_function,
                                   visible_data_function,
                                   default_game_options["wolf_starting_food"],
                                   default_game_options["wolf_walk_cost"],
                                   default_game_options["wolf_run_cost"],
                                   default_game_options["wolf_walk_speed"],
                                   default_game_options["wolf_run_speed"]),
             starting_position_x,
             starting_position_y]

        return self._id_generator

    def create_bush(self, starting_position_x, starting_position_y,
                    act_function=None, obs_function=default_bush_obs,
                    visible_data_function=default_bush_visible_data):
        """Create a new bush and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1

        self._entities.loc[self._id_generator] = [self._id_generator,
             "Bush",
             Bush.Bush(self._id_generator,
                                   starting_position_x, starting_position_y,
                                   act_function, obs_function,
                                   visible_data_function,
                                   default_game_options["food_per_bush"],
                                   default_game_options["food_given_per_turn"]),
             starting_position_x,
             starting_position_y]

        return self._id_generator

    def create_ostrich(self, starting_position_x, starting_position_y,
                       act_function=None, obs_function=default_ostrich_obs,
                       visible_data_function=default_ostrich_visible_data):
        """Create a new ostrich and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1
        self._entities.loc[self._id_generator] = [self._id_generator,
             "Ostrich",
             Ostrich.Ostrich(self._id_generator,
                                      starting_position_x, starting_position_y,
                                      act_function, obs_function,
                                      visible_data_function,
                                      default_game_options[
                                          "ostrich_starting_food"],
                                      default_game_options[
                                          "ostrich_food_eaten_per_turn"],
                                      default_game_options[
                                          "ostrich_move_speed"],
                                      default_game_options["starting_role"]),
             starting_position_x,
             starting_position_y]

        return self._id_generator

    def _get_visible_objects(self, entity_id: int, viewradius: float):
        """Return all objects visible to the entity with specified id and vision
        cone specified by viewradius."""
        entity_x = self._entities.loc[self._entities['Id'] == entity_id]['X'].iloc[0]
        entity_y = self._entities.loc[self._entities['Id'] == entity_id]['Y'].iloc[0]

        # this collects all objects that should be on the grid.
        nearby_objects_df = self._entities.copy(deep=False)

        nearby_objects_df["Delta_X"] = nearby_objects_df["X"] - entity_x
        nearby_objects_df["Delta_Y"] = nearby_objects_df["Y"] - entity_y

        if entity_x < viewradius:
            nearby_objects_df.loc[self._world_width - (viewradius - entity_x) <=
                              nearby_objects_df["X"], "Wrap_around_X"] = \
                -entity_x - (self._world_width - nearby_objects_df["X"])

            for i in range(len(nearby_objects_df)):
                nearby_objects_df.loc[i, "Delta_X"] = min(nearby_objects_df.loc[i, "Delta_X"],
                                                          nearby_objects_df.loc[i, "Wrap_around_X"], key=abs)

        elif self._world_width < entity_x + viewradius:
            # this long-winded line is essentially assigning the minimum Delta_X
            #   to be a wrap-around if appropriate
            nearby_objects_df.loc[nearby_objects_df["X"] <= viewradius - self._world_width -
                              entity_x, "Wrap_around_X"] = \
                nearby_objects_df["X"] + self._world_width - entity_x

            for i in range(len(nearby_objects_df)):
                nearby_objects_df.loc[i, "Delta_X"] = min(nearby_objects_df.loc[i, "Delta_X"],
                                                          nearby_objects_df.loc[i, "Wrap_around_X"], key=abs)


        if entity_y < viewradius:
            nearby_objects_df.loc[self._world_height - (viewradius - entity_y) <=
                              nearby_objects_df["Y"], "Wrap_around_Y"] = -entity_y - (self._world_height - nearby_objects_df["Y"])

            for i in range(len(nearby_objects_df)):
                nearby_objects_df.loc[i, "Delta_Y"] = min(nearby_objects_df.loc[i, "Delta_Y"],
                                                          nearby_objects_df.loc[i, "Wrap_around_Y"], key=abs)

        elif self._world_width < entity_y + viewradius:
            nearby_objects_df.loc[nearby_objects_df["Y"] <= viewradius - self._world_width -
                              entity_y, "Delta_Y"] = \
                nearby_objects_df["Y"] + self._world_width - entity_y

            for i in range(len(nearby_objects_df)):
                nearby_objects_df.loc[i, "Delta_Y"] = min(nearby_objects_df.loc[i, "Delta_Y"],
                                                          nearby_objects_df.loc[i, "Wrap_around_Y"], key=abs)


        # currently collecting all objects in a circle around the entity
        nearby_objects_df = nearby_objects_df[
            (nearby_objects_df["Delta_X"] ** 2 +
             nearby_objects_df["Delta_Y"] ** 2) ** 0.5 <= viewradius]

        visible_data = []
        for obj in nearby_objects_df["Entity Object"]:
            visible_data.append(obj.visible_data(obj))

        nearby_objects_df = pd.DataFrame(
            data={"Delta_X": nearby_objects_df["Delta_X"],
                  "Delta_Y": nearby_objects_df["Delta_Y"],
                  "Type": nearby_objects_df["Type"],
                  "Additional_Data": visible_data}
        )

        return nearby_objects_df

    def _get_additional_obs(self, entity_id):
        """Return all additional observations that are relevant, such as food
        levels."""

        return self._entities["Id" == entity_id].obs()

    def perform_entity_action(self, entity_id, action, current_turn):
        """Performs the specified action for the specified entity according to
        that entity's act_function."""
        assert self._current_turn == current_turn
        entity_object = self._entities.iloc[entity_id]["Entity Object"]
        entity_object.act(action)
        self._entities.iloc[entity_id]["X"] = entity_object.getX()
        self._entities.iloc[entity_id]["Y"] = entity_object.getY()

    def get_observations(self, entity_id, viewradius, current_turn):
        """Return all raw observations for the entity specified."""
        assert self._current_turn == current_turn
        return [self._get_visible_objects(entity_id, viewradius)] + \
               self._get_additional_obs(entity_id)
