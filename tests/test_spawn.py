import math

from redrocks.spawn import CAPSULE_RADIUS_M, player_spawn


def test_spawn_is_behind_the_edge_and_faces_up_the_bowl():
    s = player_spawn(10.0, 20.0, (0.0, 2.0), 1800.0, lambda e, n: 1700.0)
    assert (round(s.east_m, 6), round(s.north_m, 6)) == (10.0, 15.0)
    assert math.isclose(s.yaw_deg, 90.0)
    assert s.ground_m == 1800.0  # deck above terrain: stand on the deck


def test_spawn_rises_above_terrain_that_buries_the_deck():
    s = player_spawn(0.0, 0.0, (1.0, 0.0), 1800.0, lambda e, n: 1801.0 + 0.1 * e)
    # Highest sample is on the capsule's edge furthest along +east.
    assert math.isclose(s.ground_m, 1801.0 + 0.1 * (-5.0 + CAPSULE_RADIUS_M))
    assert s.terrain_m == s.ground_m and s.deck_top_m == 1800.0
