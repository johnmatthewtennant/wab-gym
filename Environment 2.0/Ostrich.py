import Entity


class Ostrich(Entity.Entity):
    """A class representing an ostrich in the environment.

    Ostriches, along with the other things entities store, have a current food
    level, a food cost per turn, a current speed, a role, and a status
    (alive/starved/killed)"""

    def __init__(self, id: int, _x, _y, action_function,
                 obs_function,
                 visible_data_function,
                 starting_food=1, food_cost_per_turn=0.01, speed=1.,
                 starting_role=0, alive_starved_killed=0):
        if action_function is None:
            super().__init__(self, id, _x, _y, obs_function=obs_function,
                           visible_data_function=visible_data_function)
        else:
            super().__init__(self, id, _x, _y, action_function, obs_function,
                           visible_data_function)

        self.food = starting_food
        self.food_cost = food_cost_per_turn
        self.speed = speed
        self.role = starting_role
        self.status = alive_starved_killed

    def increment_food(self, food_eaten):
        self.food += food_eaten

    def get_food(self):
        return self.food

    def flip_role(self):
        if self.role == 0:
            self.role = 1
        else:
            self.role = 0

    def get_role(self):
        return self.role

    def get_speed(self):
        return self.speed

    def set_speed(self, new_speed):
        self.speed = new_speed

    def get_status(self):
        return self.status

    def set_status(self, new_status):
        self.status = new_status
