from World import World
from WAB_Environment2_Single import WAB_Environment2_Single


class WAB_Environment2:

    def __init__(self, world_width, world_height):
        self.world = World(world_width, world_height)


    def set_entity_act_function(self, act_function):
        """Change how the entity performs its action according to the passed-in
        function."""
        self.world.set_entity_act_function(self.id, act_function)

    def set_entity_internal_obs_function(self, internal_obs_function):
        """Change how the entity's observations are collected using the passed-in
        function"""
        self.world.set_entity_internal_obs_function(self.id, internal_obs_function)

    def set_entity_external_obs_function(self, external_obs_function):
        """Change how the entity performs its action according to the passed-in
        function."""
        self.world.set_entity_external_obs_function(self.id, external_obs_function)

    def set_update_function(self, update_function):
        self.world.set_update_function(update_function)
