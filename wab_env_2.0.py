import gym
import numpy as np
import pandas as pd

default_game_options = {
    # GYM OPTIONS
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
    # FOOD
    "turns_to_fill_food": 8,  # How many turns of gathering does it take to fill food?
    "turns_to_empty_food": 40,  # How many turns of not gathering does it take to starve?
    "starting_food": 1,  # 0 to 1 float. None values will be assigned randomly
    # WOLVES
    "num_wolves": 20,
    "wolf_spawn_margin": 1,  # How many squares out from the player's view should wolves spawn?, 1 seems reasonable
    "chance_wolf_on_square": 0.001,  # Chance for wolf to spawn on each square of the perimeter, .001 seems reasonable
    "wolves": True,  # Spawn wolves?
    "wolves_can_move": True,  # Can wolves move?
    "wolf_walk_speed": 1.0, # speed that a wolf can move while not sprinting.
    "wolf_walk_cost": 0.1, # cost of walking for a wolf per unit moved
    "wolf_run_speed": 2.0, # speed of a sprinting wolf
    "wolf_run_cost": 0.2, # cost of running for a wolf per unit moved
}


# TODO: might want to put this in a sperate file. The last env got very messy
#   because it was all in one script.
class GlobeWorld:
    """A class representing the world of the environment. Essentially just
    a dataframe which gets indexed with modulo arithmetic so if the width of the
    dataframe is 20, and I index in at 25, it will return the item at index
    6. (25 % 19 is 6).

    CLARIFICATION: THIS CLASS PROVIDES THE INTERFACE OF A GRID, BUT ACTUALLY
    STORES THE FLOATING-POINT LOCATION OF EACH ITEM IN THE WORLD. WE SHOULD
    DISCUSS HOW WE WANT TO IMPLEMENT THIS.
    """

    def __init__(self):
        """Initialize the object"""

    def __getitem__(self, index):
        """This method overrides how items are indexed.
        """
        return self._world[index % len(self)]

    def populate_world(self, wolves, bushes, ostriches):
        """Adds the specified wolves, bushes, and ostriches based on the
        indexes provided by the arguments.
        """

    def update_world(self, wolf_moves, ostrich_moves):
        """Updates the world based on the moves of the wolves and ostriches.
        TODO: Should we have this logic here or in the env. I'm leaning towards
        the env honestly.
        """


class WolvesAndBushesEnvV2(gym.env):
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

    def __init__(self, game_options = default_game_options):
        """Initialize an environment with the specified options.
        This method needs to:
        1. Initialize action and observation spaces
            a. To do this, we need to calculate the dimensions of the wolf
                grid by using the ratio of the wolf's top speed to the ostrich
                speed, and the dimensions of the ostrich grid.
            b. We'll also need to generate a globe-based world. A seperate
                class should be used for this which overrides the indexing
                method to work on a modulo-adjusted basis.
        2. Call reset()
        """

    def _update_world(self):
        """Updates the world to reflect the current state of the wolves, bushes,
        and ostriches.
        """

    def reset(self):
        """Resets the environment to a starting state. This involves:
        1. Creating a new world
            a. Spawning wolves
            b. Spawning bushes
            c. Spawning ostriches
            (note, the above should be done by calling individual methods which
            we can change later without affecting this method.)
            d. Calling GlobeWorld.populate_world
        2. setting current_turn to 0
        3. Returning the observations of the starting state. Described in more
            detail in _get_obs()
        """

    def _get_wolf_spawn_indices(self):
        """Returns the indexes of all spawn locations of the wolves.
        """

    def _get_ostrich_spawn_indices(self):
        """Returns the indexes of all spawn locations of the wolves.
        """

    def _get_bush_spawn_indices(self):
        """Returns the indexes of all spawn locations of the wolves.
        """

    def step(self, actions):
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

    def _move_ostriches(self, ostrich_moves):
        """Performs all actions that ostriches are taking. Update the ostriches'
        internal states to reflect if they're dead, changed roles, etc...
        """

    def _generate_wolf_actions(self):
        """Generates the actions of the wolves for this turn."""

    def _move_wolves(self, wolf_moves):
        """Performs all actions that wolves are taking. Update the wolves'
        """

    def _compute_reward(self):
        """Computes and returns the rewards for each ostrich given the current
        world state.
        """

    def _all_ostriches_dead(self) -> bool:
        """Return true iff all ostriches are dead.
        """

    def _get_obs(self):
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

    def _generate_ostrich_obs(self, ostrich_index):
        """Return the observations of the ostrich given by the index provided.
        """

    def _generate_grids(self, ostrich_index):
        """Return grids showing the positions of nearby wolves, ostriches, and
        bushes. Apply their view masks based on the role by calling
        _apply_view_mask.
        """

    def _apply_view_mask(self, grid, grid_width, grid_height, ostrich_role):
        """Return the masked grid based on the role of the ostrich (lookout or
        gatherer).
        """
