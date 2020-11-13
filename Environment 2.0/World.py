import pandas as pd
import Entity
import Wolf
import Bush
import Ostrich
import asyncio


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
                    act_function=None, obs_function=None,
                    visible_data_function=None):
        """Create a new wolf and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1
        self._entities = self._entities.append(
            {"Id": self._id_generator,
             "Type": "Wolf",
             "Entity Object": Wolf(self._id_generator,
                                   starting_position_x, starting_position_y,
                                   act_function, obs_function,
                                   visible_data_function,
                                   default_game_options["wolf_starting_food"],
                                   default_game_options["wolf_walk_cost"],
                                   default_game_options["wolf_run_cost"],
                                   default_game_options["wolf_walk_speed"],
                                   default_game_options["wolf_run_speed"]),
             "X": starting_position_x,
             "Y": starting_position_y})

        return self._id_generator

    def create_bush(self, starting_position_x, starting_position_y,
                    act_function=None, obs_function=None,
                    visible_data_function=None):
        """Create a new bush and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1

        self._entities = self._entities.append(
            {"Id": self._id_generator,
             "Type": "Bush",
             "Entity Object": Bush(self._id_generator,
                                   starting_position_x, starting_position_y,
                                   act_function, obs_function,
                                   visible_data_function,
                                   default_game_options["food_per_bush"],
                                   default_game_options["food_given_per_turn"]),
             "X": starting_position_x,
             "Y": starting_position_y})

        return self._id_generator

    def create_ostrich(self, starting_position_x, starting_position_y,
                       act_function=None, obs_function=None,
                       visible_data_function=None):
        """Create a new ostrich and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1
        self._entities = self._entities.append(
            {"Id": self._id_generator,
             "Type": "Ostrich",
             "Entity Object": Ostrich(self._id_generator,
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
             "X": starting_position_x,
             "Y": starting_position_y})

        return self._id_generator

    def _get_visible_objects(self, entity_id: int, viewradius: float):
        """Return all objects visible to the entity with specified id and vision
        cone specified by viewradius."""
        entity_x = self._entities['Id' == entity_id]['X']
        entity_y = self._entities['Id' == entity_id]['Y']

        # this collects all objects that should be on the grid.
        nearby_objects_df = self._entities.copy(deep=False)

        nearby_objects_df["Delta_X"] = nearby_objects_df["X"] - entity_x
        nearby_objects_df["Delta_Y"] = nearby_objects_df["Y"] - entity_y

        if (entity_x < viewradius):
            indices_to_add = nearby_objects_df.index(self._world_width -
                                                     (viewradius - entity_x) <=
                                                     nearby_objects_df["X"])

            for i in indices_to_add:
                nearby_objects_df.iloc[i]["Delta_X"] = entity_x + \
                                                       (self._world_width - nearby_objects_df.iloc[i]["X"])

        if(self.world_width - viewradius < entity_x):
            indices_to_add = nearby_objects_df.index(nearby_objects_df["X"] <=
                                                     viewradius -
                                                     (self._world_width - entity_x))

            for i in indices_to_add:
                nearby_objects_df.iloc[i]["Delta_X"] = nearby_objects_df.iloc[i]["X"] + \
                                                       (self._world_width - entity_x)

        if (entity_y < viewradius):
            indices_to_add = nearby_objects_df.index(self._world_height -
                                                     (viewradius - entity_y) <=
                                                     nearby_objects_df["Y"])

            for i in indices_to_add:
                nearby_objects_df.iloc[i]["Delta_Y"] = entity_y + \
                                                       (self._world_height - nearby_objects_df.iloc[i]["Y"])

        if(self.world_width - viewradius < entity_y):
            indices_to_add = nearby_objects_df.index(nearby_objects_df["Y"] <=
                                                     viewradius -
                                                     (self._world_width - entity_y))

            for i in indices_to_add:
                nearby_objects_df.iloc[i]["Delta_Y"] = nearby_objects_df.iloc[i]["Y"] + \
                                                       (self._world_width - entity_y)


        # currently collecting all objects in a circle around the entity
        nearby_objects_df = nearby_objects_df[
            (nearby_objects_df["Delta_X"] ** 2 +
             nearby_objects_df["Delta_Y"] ** 2) ** 0.5 <= viewradius]



        nearby_objects_df = pd.DataFrame(
            data={"Delta_X": nearby_objects_df["Delta_X"],
                  "Delta_Y": nearby_objects_df["Delta_Y"],
                  "Type": nearby_objects_df["Type"],
                  "Additional_Data": [x.visible_data() for x in
                                      nearby_objects_df["Entity_Object"]]}
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
