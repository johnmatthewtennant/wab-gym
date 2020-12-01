import Entity


class Wolf(Entity.Entity):
    """An entity class describing a wolf. On top of the existing Entity members,
    a wolf has:
    1. A walking speed
    2. A running speed
    3. A food value (wolves can starve)
    4. A food cost per turn for walking
    5. A food cost per turn for running
    6. a boolean representing whether or not the wolf is currently running.
    7. a status (alive/starved)
    """

    def __init__(self, id: int, _x, _y,
                 action_function=None, obs_function=None,
                 visible_data_function=None,
                 starting_food=20., walking_food_cost=1., running_food_cost=2.,
                 walking_speed=1., running_speed=1.5):
        if action_function is None:
            Entity.Entity.__init__(self, id, _x, _y, obs_function=obs_function,
                                   visible_data_function=visible_data_function)
        else:
            Entity.Entity.__init__(self, id, _x, _y, action_function,
                                   obs_function, visible_data_function)

        self.food = starting_food
        self.walking_food_cost = walking_food_cost
        self.running_food_cost = running_food_cost
        self.walking_speed = walking_speed
        self.running_speed = running_speed
        self.is_running = False
        self.status = 0 # alive, 1 means dead

    def increment_food(self, food_gained):
        self.food += food_gained

    def get_food(self):
        return self.food

    def toggle_running(self):
        if self.is_running:
            self.is_running = False
        else:
            self.is_running = True

    def get_is_running(self):
        return self.is_running

    def get_walking_speed(self):
        return self.walking_speed

    def get_running_speed(self):
        return self.running_speed

    def get_status(self):
        return self.status

    def change_status(self, new_status):
        self.status = new_status
