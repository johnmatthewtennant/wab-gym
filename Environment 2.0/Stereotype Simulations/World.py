import pandas as pd
import Entity
import Wolf
import Bush
import Ostrich
import random


def default_bush_act(self, action):
    return


def default_bush_external_obs(self):
    return [self.food]


def default_bush_internal_obs(self):
    return [self.x, self.y, self.food]


def default_bush_compute_reward(self):
    return 0


def default_ostrich_act(self, action):
    if action == 0:
        #up
        self.y += 1
    elif action == 1:
        #right
        self.x += 1
    elif action == 2:
        #down
        self.y -= 1
    elif action == 3:
        #left
        self.x -= 1
    elif action == 4:
        #don't move, be lookout
        self.role = 0
    elif action == 5:
        # don't move, be gatherer
        self.role = 1


def default_ostrich_external_obs(self):
    return []


def default_ostrich_internal_obs(self):
    return [self.x, self.y, self.food, self.role, self.status]


def default_ostrich_compute_reward(self):
    if not self.is_done():
        return 1
    else:
        return 0


def default_wolf_act(self, action):
    if action == 0:
        #up
        self.y += 1
    elif action == 1:
        #right
        self.x += 1
    elif action == 2:
        #down
        self.y -= 1
    elif action == 3:
        #left
        self.x -= 1


def default_wolf_external_obs(self):
    return []


def default_wolf_internal_obs(self):
    return [self.x, self.y, self.food, self.is_running, self.status]


def default_wolf_compute_reward(self):
    return self.get_food() > 10

#This is the default update function. can be overridden to apply new game rules
#   this essentially handles all update rules that require interaction between
#   different agents, internal agent state updates are done inside the agent
#   class


def default_game_update(self, entity_id):
    i = entity_id
    copy = self._entities.copy(deep=False)
    if copy.iloc[i]["Type"] == "Bush":
        return
    else:
        entity = copy.iloc[i]
        copy = copy[copy["Visible"]]
        copy = copy[copy["X"] == entity["X"]]
        copy = copy[copy["Y"] == entity["Y"]]

        #### BELOW WE MAKE RULES FOR WHAT HAPPENS TO EACH OBJECT ON THE SAME SQUARE AS ANNOTHER OBJECT ####
        #have randomness in the eating priority, use pandas sample function to get random row
        #wolf can eat two ostriches here on the same turn
        if entity["Type"] == "Wolf":
            copy = copy[copy["Type"] == "Ostrich"]
            if len(copy) == 0:
                #no ostriches on the tile, so return
                return
            j = random.randint(0, len(copy) - 1) #take a random index of an ostrich to eat
            entity["Entity_Object"].increment_food(self._game_options["wolf_food_for_eating_ostrich"])
            copy.iloc[j]["Entity_Object"].set_status(2) # killed
            self._entities.loc[j, "Visible"] = False
            return #return at the end so we only eat one ostrich per turn

        elif entity["Type"] == "Ostrich":
            # at every turn, an ostrich automatically eats from the bush.
            copy = copy[copy["Type"] == "Bush"]
            # take a random index of a bush to eat from
            if len(copy) == 0:
                # there are no bushes on that tile, so return
                return
            j = random.randint(0, len(copy) - 1)
            bush = copy.iloc[j]["Entity_Object"]
            entity["Entity_Object"].increment_food(bush.take_food())

            has_food = bush.get_has_food()
            if not has_food:
                self._entities.iloc[j]["Visible"] = False
            return # return so the ostrich only eats one bush


class World:
    """A class representing the world of the environment.

    This class should store all entities in the world space and each entity
    should have a unique ID associated with it.
    """
    def __init__(self, world_width, world_height, game_options,
                 game_update_function=default_game_update):
        """Initialize the World"""
        self._id_generator = -1
        self._current_turn = 0
        self._world_width = world_width
        self._world_height = world_height
        self.update = game_update_function
        # currently this is the setup for storing entities.
        self._entities = pd.DataFrame(columns=["Id", "Type", "Entity_Object",
                                               "X", "Y", "Visible"])
        self._game_options = game_options

    def increment_turn(self):
        self._current_turn += 1

    def create_wolf(self, starting_position_x, starting_position_y,
                    compute_reward_function=default_wolf_compute_reward,
                    act_function=default_wolf_act,
                    internal_obs_function=default_wolf_internal_obs,
                    external_obs_function=default_wolf_external_obs):
        """Create a new wolf and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1
        self._entities.loc[self._id_generator] = [self._id_generator,
             "Wolf",
             Wolf.Wolf(self._id_generator,
                       starting_position_x, starting_position_y,
                       compute_reward_function,
                       action_function=act_function,
                       internal_obs_function=internal_obs_function,
                       external_obs_function=external_obs_function,
                       starting_food=self._game_options["wolf_starting_food"],
                       walking_food_cost=self._game_options["wolf_walk_cost"],
                       running_food_cost=self._game_options["wolf_run_cost"],
                       walking_speed=self._game_options["wolf_walk_speed"],
                       running_speed=self._game_options["wolf_run_speed"]),
             starting_position_x,
             starting_position_y,
             True]

        return self._id_generator

    def create_bush(self, starting_position_x, starting_position_y,
                    compute_reward_function=default_bush_compute_reward,
                    act_function=default_bush_act,
                    internal_obs_function=default_bush_internal_obs,
                    external_obs_function=default_bush_external_obs):
        """Create a new bush and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1

        self._entities.loc[self._id_generator] = [self._id_generator,
             "Bush",
             Bush.Bush(self._id_generator,
                       starting_position_x, starting_position_y,
                       compute_reward_function,
                       act_function, internal_obs_function,
                       external_obs_function,
                       self._game_options["food_per_bush"],
                       self._game_options["food_given_per_turn"]),
             starting_position_x,
             starting_position_y,
             True]

        return self._id_generator

    def create_ostrich(self, starting_position_x, starting_position_y,
                       compute_reward_function=default_ostrich_compute_reward,
                       act_function=default_ostrich_act,
                       internal_obs_function=default_ostrich_internal_obs,
                       external_obs_function=default_ostrich_external_obs):
        """Create a new ostrich and add it to the world. Return the id of the
        created object."""
        self._id_generator += 1
        self._entities.loc[self._id_generator] = [self._id_generator,
             "Ostrich",
             Ostrich.Ostrich(self._id_generator,
                             starting_position_x, starting_position_y,
                             compute_reward_function,
                             act_function, internal_obs_function,
                             external_obs_function,
                             self._game_options["ostrich_starting_food"],
                             self._game_options["ostrich_food_eaten_per_turn"],
                             self._game_options["ostrich_move_speed"],
                             self._game_options["starting_role"]),
             starting_position_x,
             starting_position_y,
             True]

        return self._id_generator

    def create_entity(self, type, starting_position_x, starting_position_y):
        if type == "Bush":
            return self.create_bush(starting_position_x, starting_position_y)
        elif type == "Ostrich":
            return self.create_ostrich(starting_position_x, starting_position_y)
        elif type == "Wolf":
            return self.create_wolf(starting_position_x, starting_position_y)
        else:
            print("Could not create entity! " + type + " is not a valid entity type.")

    def _get_visible_objects(self, entity_id: int, viewradius: float):
        """Return all objects visible to the entity with specified id and vision
        cone specified by viewradius."""
        entity_x = self._entities.loc[self._entities['Id'] == entity_id]['X'].iloc[0]
        entity_y = self._entities.loc[self._entities['Id'] == entity_id]['Y'].iloc[0]

        # this collects all objects that should be on the grid.
        nearby_objects_df = self._entities.copy(deep=True)

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
            nearby_objects_df.loc[nearby_objects_df["X"] <= viewradius - self._world_width +
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

        elif self._world_height < entity_y + viewradius:
            nearby_objects_df.loc[nearby_objects_df["Y"] <= viewradius - self._world_height +
                                  entity_y, "Wrap_around_Y"] = \
                nearby_objects_df["Y"] + self._world_height - entity_y

            for i in range(len(nearby_objects_df)):
                nearby_objects_df.loc[i, "Delta_Y"] = min(nearby_objects_df.loc[i, "Delta_Y"],
                                                          nearby_objects_df.loc[i, "Wrap_around_Y"], key=abs)


        # currently collecting all objects in a circle around the entity
        nearby_objects_df = nearby_objects_df[
            (nearby_objects_df["Delta_X"] ** 2 +
             nearby_objects_df["Delta_Y"] ** 2) ** 0.5 <= viewradius]

        #removes all dead ostriches/bushes from the space
        nearby_objects_df = nearby_objects_df[nearby_objects_df["Visible"] == True]


        visible_data = []
        for obj in nearby_objects_df["Entity_Object"]:
            visible_data.append(obj.external_obs(obj))

        nearby_objects_df = pd.DataFrame(
            data={"Delta_X": nearby_objects_df["Delta_X"],
                  "Delta_Y": nearby_objects_df["Delta_Y"],
                  "Type": nearby_objects_df["Type"],
                  "Additional_Data": visible_data}
        ).reset_index()
        # above we reset the index to handle values that were removed from the
        #   dataframe.

        return nearby_objects_df

    def _get_additional_obs(self, entity_id):
        """Return all additional observations that are relevant, such as food
        levels."""
        entity_object = self._entities.loc[self._entities['Id'] ==
                                           entity_id]["Entity_Object"].iloc[0]
        return entity_object.internal_obs(entity_object)

    def perform_entity_action(self, entity_id, action, current_turn):
        """Performs the specified action for the specified entity according to
        that entity's act_function."""
        assert self._current_turn == current_turn
        entity_object = self._entities.iloc[entity_id]["Entity_Object"]
        entity_object.act(entity_object, action)
        self._entities.at[entity_id, "X"] = entity_object.getX() % self._world_width
        self._entities.at[entity_id, "Y"] = entity_object.getY() % self._world_height
        self.update(self, entity_id)
        return entity_object.compute_reward(entity_object)

    def increment_turn_count(self):
        self._current_turn += 1

    def is_entity_done(self, entity_id):
        """Return True iff the entity is done for the current epoch (i.e. dead
        or cannot act).
        """
        return self._entities.iloc[entity_id]["Entity_Object"].is_done()


    def reset_entity(self, entity_id, new_x, new_y):
        self._entities.iloc[entity_id]["Entity_Object"].reset(new_x, new_y)
    #TODO: Call this from the class where world is created

    def reset_world(self):
        """Resets the world to begin a new epoch. ONLY CALL THIS AFTER
        CALLING reset() ON EVERY ENTITY THROUGH ITS RESPECTIVE GYM."""
        for i in range(len(self._entities)):
            self._entities.loc[i, "Visible"] = True
            self._entities.iloc[i]["X"] = self._entities.iloc[i]["Entity_Object"].getX() % self._world_width
            self._entities.iloc[i]["Y"] = self._entities.iloc[i]["Entity_Object"].getY() % self._world_height

        self._current_turn = 0

    def get_observations(self, entity_id, current_turn):
        """Return all raw observations for the entity specified."""
        assert self._current_turn == current_turn, "Not all entities have acted yet"
        entity_row = self._entities.iloc[entity_id]

        #if ostrich, specify radius based on role
        if entity_row["Type"] == "Ostrich":
            if entity_row["Entity_Object"].get_role() == 1: #1 is gatherer
                viewradius = self._game_options["gatherer_view_radius"]
            elif entity_row["Entity_Object"].get_role() == 0: #0 is lookout
                viewradius = self._game_options["lookout_view_radius"]
        elif entity_row["Type"] == "Wolf":
            viewradius = self._game_options["wolf_view_radius"]
        else:
            viewradius = 0

        return [self._get_visible_objects(entity_id, viewradius)] + \
               [self._get_additional_obs(entity_id)]

    def set_entity_internal_obs_function(self, entity_id, internal_obs_function):
        self._entities.iloc[entity_id].internal_obs = internal_obs_function

    def set_entity_external_obs_functions(self, entity_id, external_obs_function):
        self._entities.iloc[entity_id].external_obs = external_obs_function

    def set_entity_act_function(self, entity_id, act_function):
        self._entities.iloc[entity_id].act = act_function

    def set_entity_reward_function(self, entity_id, reward_function):
        self._entities.iloc[entity_id].compute_reward = reward_function

    def get_width(self):
        return self._world_width

    def get_height(self):
        return self._world_height
