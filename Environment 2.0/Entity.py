
def default_act(self, action):
    """This is the default action function. Entities that don't act (such as
    bushes) should use this by not passing in a function for action_function in
    the Entity constructor.
    """
    return

def default_visible_data(self):
    return

def default_obs(self):
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

    def __init__(self, id: int, _x, _y, act_function=default_act,
                 obs_function=default_obs,
                 visible_data_function=default_visible_data):
        """Initialize a new Entity with a starting position and id.
        """
        self._id = id
        self.act = act_function
        self.obs = obs_function
        self.visible_data = visible_data_function
        self.x = _x
        self.y = _y

    def get_id(self):
        return self._id

    def getX(self):
        return self.x

    def getY(self):
        return self.y
