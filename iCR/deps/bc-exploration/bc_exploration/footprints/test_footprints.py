from __future__ import print_function, absolute_import, division

import numpy as np

from bc_exploration.footprints.footprint_points import get_tricky_oval_footprint, get_tricky_circular_footprint
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import rc_to_xy, load_occupancy_map_data


def test_custom_footprint(debug=False):
    map_data = load_occupancy_map_data('test', 'three_rooms.png')
    occupancy_map = Costmap(data=map_data, resolution=0.03, origin=[0, 0])

    free_state = rc_to_xy(np.array([110, 220, 0]), occupancy_map)
    colliding_state = rc_to_xy(np.array([90, 215, 0]), occupancy_map)

    footprint_points = get_tricky_oval_footprint()
    footprint = CustomFootprint(footprint_points, 2 * np.pi / 180.)
    assert not footprint.check_for_collision(free_state, occupancy_map, debug=debug)
    assert footprint.check_for_collision(colliding_state, occupancy_map, debug=debug)
    assert not footprint.check_for_collision(free_state, occupancy_map, debug=debug, use_python=True)
    assert footprint.check_for_collision(colliding_state, occupancy_map, debug=debug, use_python=True)

    rotated_colliding_state = rc_to_xy(np.array([110, 220, -np.pi / 2]), occupancy_map)
    assert footprint.check_for_collision(rotated_colliding_state, occupancy_map, debug=debug)
    assert footprint.check_for_collision(rotated_colliding_state, occupancy_map, debug=debug, use_python=True)

    footprint_points = get_tricky_circular_footprint()
    footprint = CustomFootprint(footprint_points, 2 * np.pi / 180.)
    assert not footprint.check_for_collision(free_state, occupancy_map, debug=debug)
    assert footprint.check_for_collision(colliding_state, occupancy_map, debug=debug)
    assert not footprint.check_for_collision(free_state, occupancy_map, debug=debug, use_python=True)
    assert footprint.check_for_collision(colliding_state, occupancy_map, debug=debug, use_python=True)

    rotated_colliding_state = rc_to_xy(np.array([115, 155, np.pi / 4]), occupancy_map)
    assert footprint.check_for_collision(rotated_colliding_state, occupancy_map, debug=debug)
    assert footprint.check_for_collision(rotated_colliding_state, occupancy_map, debug=debug, use_python=True)

    rotated_colliding_state = rc_to_xy(np.array([0, 0, np.pi / 2]), occupancy_map)
    assert footprint.check_for_collision(rotated_colliding_state, occupancy_map, debug=debug)
    assert footprint.check_for_collision(rotated_colliding_state, occupancy_map, debug=debug, use_python=True)


def test_footprint_orientation(debug=False):
    state = np.array([0.71, 0.35, np.pi / 4])
    map_data = load_occupancy_map_data('test', 'footprint_orientation_map.png')
    occupancy_map = Costmap(data=map_data, resolution=0.03, origin=[-19.72, -1.54])

    footprint_points = get_tricky_circular_footprint()
    footprint = CustomFootprint(footprint_points, 2 * np.pi / 180., inflation_scale=1.3)

    assert not footprint.check_for_collision(state, occupancy_map, debug=debug)
    assert not footprint.check_for_collision(state, occupancy_map, debug=debug, use_python=True)


def main():
    debug = False
    test_custom_footprint(debug=debug)
    test_footprint_orientation(debug=debug)


if __name__ == '__main__':
    main()
