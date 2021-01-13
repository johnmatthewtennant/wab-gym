
def default_act(self, action):
    """This is the default action function. Entities that don't act (such as
    bushes) should use this by not passing in a function for action_function in
    the Entity constructor.
    """
    return


def default_external_obs(self):
    return


def default_internal_obs(self):
    """Default function for formatting and returning observations"""
    # this may not work. Might need to put this in the function.
    return [self.x, self.y]


class Entity:
    """A class describing an entity in our environment. This should be
    subclassed by each agent, such as a wolf, bush or ostrich.

    Each entity has a position in the world (float, float) and a unique id.

    Each entity also has a act() function, which returns the action given the
    current world space. The act() function is a member of the class, passed
    in via the __init__ method. This allows us to have flexible policies for
    each agent, and allows us to change the policy function of the agent while
    we're running our model.

    The default value for the action function is default_action, as defined
    above.
    """

    def __init__(self, id: int, _x, _y, compute_reward_function,
                 act_function=default_act,
                 internal_obs_function=default_internal_obs,
                 external_obs_function=default_external_obs):
        """Initialize a new Entity with a starting position and id.
        """
        self._id = id
        self.act = act_function
        self.compute_reward = compute_reward_function
        self.internal_obs = internal_obs_function
        self.external_obs = external_obs_function
        self.x = _x
        self.y = _y

    def get_id(self):
        return self._id

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def is_done(self):
        return

    def reset(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
