import Entity


def default_bush_action(self):
    return


def default_bush_obs(self):
    return [self.x, self.y, self.food]


def default_bush_visible_data(self):
    return [self.food]


class Bush(Entity):
    """An instance of a bush. On top of the other Entity members, bushes have a
    current food value, and an amount of food that they give up per time that
    they are eaten from, and a status (has food, doesn't have food)
    """

    def __init__(self, id: int, _x, _y, action_function = None,
                 obs_function=default_bush_obs,
                 visible_data_function=default_bush_visible_data,
                 initial_food=10, food_given_when_eaten=3):
        if action_function is None:
            super.__init__(id, _x, _y, obs_function=obs_function,
                           visible_data_function=visible_data_function)
        else:
            super.__init__(id, _x, _y, action_function,
                           obs_function=obs_function,
                           visible_data_function=visible_data_function)

        self.food = initial_food
        self.food_given_when_eaten = food_given_when_eaten
        self.has_food = self.food > 0

    def take_food(self):
        if self.food >= self.food_given_when_eaten:
            self.food -= self.food_given_when_eaten
            return self.food_given_when_eaten
        else:
            food_before = self.food
            self.food = 0
            self.has_food = False
            return food_before

    def has_food(self):
        return self.has_food

    def get_food(self):
        return self.food

