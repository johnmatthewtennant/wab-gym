import World
import pytest


def test_get_visible_objects_no_wrap():
    world = World.World(20, 20)

    world.create_wolf(5, 5)
    world.create_bush(10, 5)
    ostrich_id = world.create_ostrich(10, 10)
    world.create_bush(10, 10)
    world.create_bush(15, 10)
    world.create_wolf(15, 15)

    nearby_objects_df = world._get_visible_objects(ostrich_id, 8)

    assert len(nearby_objects_df) == 6
    assert len(nearby_objects_df[nearby_objects_df["Type"] == "Wolf"]) == 2
    assert len(nearby_objects_df[nearby_objects_df["Type"] == "Bush"]) == 3
    assert len(nearby_objects_df[nearby_objects_df["Type"] == "Ostrich"]) == 1

    assert nearby_objects_df.loc[0, "Type"] == "Wolf"
    assert nearby_objects_df.loc[0, "Delta_X"] == -5
    assert nearby_objects_df.loc[0, "Delta_Y"] == -5
    assert nearby_objects_df.loc[0, "Additional_Data"] == []
    assert nearby_objects_df.loc[1, "Type"] ==  "Bush"
    assert nearby_objects_df.loc[1, "Delta_X"] == 0
    assert nearby_objects_df.loc[1, "Delta_Y"] == -5
    assert nearby_objects_df.loc[1, "Additional_Data"] == [20]
    assert nearby_objects_df.loc[2, "Type"] == "Ostrich"
    assert nearby_objects_df.loc[2, "Delta_X"] == 0
    assert nearby_objects_df.loc[2, "Delta_Y"] == 0
    assert nearby_objects_df.loc[2, "Additional_Data"] == []
    assert nearby_objects_df.loc[3, "Type"] == "Bush"
    assert nearby_objects_df.loc[3, "Delta_X"] == 0
    assert nearby_objects_df.loc[3, "Delta_Y"] == 0
    assert nearby_objects_df.loc[3, "Additional_Data"] == [20]
    assert nearby_objects_df.loc[4, "Type"] == "Bush"
    assert nearby_objects_df.loc[4, "Delta_X"] == 5
    assert nearby_objects_df.loc[4, "Delta_Y"] == 0
    assert nearby_objects_df.loc[4, "Additional_Data"] == [20]
    assert nearby_objects_df.loc[5, "Type"] == "Wolf"
    assert nearby_objects_df.loc[5, "Delta_X"] == 5
    assert nearby_objects_df.loc[5, "Delta_Y"] == 5
    assert nearby_objects_df.loc[5, "Additional_Data"] == []



def test_get_visible_objects_wrap_horizontal():
    world = World.World(20, 20)

    world.create_wolf(5, 5)
    #world.create_bush(10, 5)
    ostrich_id = world.create_ostrich(19, 10)
    world.create_bush(10, 10)
    world.create_bush(15, 10)
    world.create_wolf(15, 15)
    other_ostrich_id = world.create_ostrich(15, 15)

    world.perform_entity_action(other_ostrich_id, 0, 0)
    world.update_game_state()

    nearby_objects_df = world._get_visible_objects(ostrich_id, 10)

    assert len(nearby_objects_df) == 5
    assert len(nearby_objects_df[nearby_objects_df["Type"] == "Wolf"]) == 2
    assert len(nearby_objects_df[nearby_objects_df["Type"] == "Bush"]) == 2
    assert len(nearby_objects_df[nearby_objects_df["Type"] == "Ostrich"]) == 1

    assert nearby_objects_df.loc[0, "Type"] == "Wolf"
    assert nearby_objects_df.loc[0, "Delta_X"] == 6
    assert nearby_objects_df.loc[0, "Delta_Y"] == -5
    assert nearby_objects_df.loc[0, "Additional_Data"] == []
    assert nearby_objects_df.loc[1, "Type"] == "Ostrich"
    assert nearby_objects_df.loc[1, "Delta_X"] == 0
    assert nearby_objects_df.loc[1, "Delta_Y"] == 0
    assert nearby_objects_df.loc[1, "Additional_Data"] == []
    assert nearby_objects_df.loc[2, "Type"] == "Bush"
    assert nearby_objects_df.loc[2, "Delta_X"] == -9
    assert nearby_objects_df.loc[2, "Delta_Y"] == 0
    assert nearby_objects_df.loc[2, "Additional_Data"] == [20]
    assert nearby_objects_df.loc[3, "Type"] == "Bush"
    assert nearby_objects_df.loc[3, "Delta_X"] == -4
    assert nearby_objects_df.loc[3, "Delta_Y"] == 0
    assert nearby_objects_df.loc[3, "Additional_Data"] == [20]
    assert nearby_objects_df.loc[4, "Type"] == "Wolf"
    assert nearby_objects_df.loc[4, "Delta_X"] == -4
    assert nearby_objects_df.loc[4, "Delta_Y"] == 5
    assert nearby_objects_df.loc[4, "Additional_Data"] == []


if __name__ == "__main__":
    test_get_visible_objects_no_wrap()
    test_get_visible_objects_wrap_horizontal()
