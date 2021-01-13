import Entity


class Ostrich(Entity.Entity):
    """A class representing an ostrich in the environment.

    Ostriches, along with the other things entities store, have a current food
    level, a food cost per turn, a current speed, a role, and a status
    (alive/starved/killed)"""

    def __init__(self, id: int, _x, _y, compute_reward_function,
                 action_function,
                 internal_obs_function,
                 external_obs_function,
                 starting_food=1, food_cost_per_turn=0.01, speed=1.,
                 starting_role=0, alive_starved_killed=0):
        if action_function is None:
            Entity.Entity.__init__(self, id, _x, _y,
                                   compute_reward_function=compute_reward_function,
                                   internal_obs_function=internal_obs_function,
                                   external_obs_function=external_obs_function)
        else:
            Entity.Entity.__init__(self, id, _x, _y, compute_reward_function,
                                   action_function, internal_obs_function,
                                   external_obs_function)

        self.starting_food = starting_food
        self.food = self.starting_food
        self.food_cost = food_cost_per_turn
        self.speed = speed
        self.starting_role = starting_role
        self.role = self.starting_role
        self.starting_status = alive_starved_killed
        self.status = self.starting_status

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

    def is_done(self):
        return self.status != 0

    def reset(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
        self.food = self.starting_food
        self.role = self.starting_role
        self.status = self.starting_status

