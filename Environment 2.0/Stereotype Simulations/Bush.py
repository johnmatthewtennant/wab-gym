import Entity


class Bush(Entity.Entity):
    """An instance of a bush. On top of the other Entity members, bushes have a
    current food value, and an amount of food that they give up per time that
    they are eaten from, and a status (has food, doesn't have food)
    """

    def __init__(self, id: int, _x, _y, compute_reward_function=None,
                 action_function=None,
                 internal_obs_function=None,
                 external_obs_function=None,
                 initial_food=10, food_given_when_eaten=3):
        if action_function is None:
            Entity.Entity.__init__(self, id, _x, _y,
                                   compute_reward_function=compute_reward_function,
                                   internal_obs_function=internal_obs_function,
                                   external_obs_function=external_obs_function)
        else:
            Entity.Entity.__init__(self, id, _x, _y, compute_reward_function,
                                   action_function,
                                   internal_obs_function=internal_obs_function,
                                   external_obs_function=external_obs_function)

        self.initial_food = initial_food
        self.food = self.initial_food
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

    def get_has_food(self):
        return self.has_food

    def get_food(self):
        return self.food

    # always return true for bushes because they never act, so they're always
    #   done
    def is_done(self):
        return True

    def reset(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
        self.food = self.initial_food
        self.has_food = self.food > 0
