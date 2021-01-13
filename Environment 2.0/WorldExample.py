from World import World

def new_ostrich_act(self, action):

    if action == 0:
        self.y += 1


def new_update_function(self):
    #move the first entity in the df 1 to the right

    if self._entities.iloc[1]["X"] == self._entities.iloc[0]["X"] and \
        self._entities.iloc[1]["Y"] == self._entities.iloc[0]["Y"]:
        self._entities.at[0, "Entity_Object"].set_status(2) # set killed
        self._entities.at[0, "Visible"] = False #set invisible now

    self._current_turn +=1


if __name__ == "__main__":

    world = World(20, 20, game_update_function=new_update_function)

    ostrich_id = world.create_ostrich(10, 9, act_function=new_ostrich_act)

    wolf_id = world.create_wolf(10, 10)

    current_turn = 0

    action = 0 #changing the action makes the assertions at the bottom false, as the ostrich no longer moves
    #   into the same square as the wolf.

    world.perform_entity_action(ostrich_id, action, current_turn)

    world.perform_entity_action(wolf_id, 4, current_turn) #for wolves, 0 is move up by default


    assert world._entities.iloc[0]["Visible"] == False
    assert world._entities.iloc[0]["Entity_Object"].get_status() == 2
    assert world._current_turn == 1



