
def default_action(*args):
    """This is the default action function. Entities that don't act (such as
    bushes) should use this by not passing in a function for action_function in
    the Entity constructor.
    """
    return None


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

    def __init__(self, id: int, starting_x, starting_y,
                 action_function=default_action):
        """Initialize a new Entity with a starting position and id.
        """
        self._id = id
        self._x = starting_x
        self._y = starting_y
        self.act = action_function

    def set_position(self, newX: float, newY: float):
        """Updates the current position of the entity to be (newX, newY)."""
        self._x = newX
        self._y = newY

    def get_position(self):
        return (self._x, self._y)

    def get_id(self):
        return this._id
