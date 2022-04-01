from __future__ import print_function, absolute_import, division

from functools import partial

import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint
from bc_exploration.utilities.util import load_occupancy_map_data, rc_to_xy, xy_to_rc
from bc_exploration.planners.astar_cpp import astar, oriented_astar, get_astar_angles
from bc_exploration.utilities.visualization import draw_footprint_path
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.mapping.costmap import Costmap


def test_one_astar(debug=False):
    occupancy_map_data = load_occupancy_map_data('test', 'maze_small.png')
    occupancy_map = Costmap(occupancy_map_data, 0.03, origin=[0., 0.])

    obstacle_values = np.array([Costmap.OCCUPIED, Costmap.UNEXPLORED], dtype=np.uint8)

    start = rc_to_xy(np.argwhere(occupancy_map.data == Costmap.FREE)[0, :], occupancy_map)
    goal = rc_to_xy(np.argwhere(occupancy_map.data == Costmap.FREE)[-1, :], occupancy_map)

    start_time = time.time()
    success, path = astar(goal=goal,
                          start=start,
                          occupancy_map=occupancy_map,
                          obstacle_values=obstacle_values,
                          allow_diagonal=False)
    time_elapsed = time.time() - start_time

    assert success

    if debug:
        map_solution = np.dstack((occupancy_map.data, occupancy_map.data, occupancy_map.data))
        path_px = xy_to_rc(path, occupancy_map).astype(np.int)
        map_solution[path_px[:, 0], path_px[:, 1]] = [0, 0, 255]

        if success:
            print("found solution in", time_elapsed, "seconds, with length:", path.shape[0])
        else:
            print("failed to find solution")

        plt.imshow(map_solution, interpolation='nearest')
        plt.show()


def test_multithread_astar(debug=False):
    num_instances = 10
    pool = multiprocessing.Pool()

    occupancy_map_data = load_occupancy_map_data('test', 'maze_small.png')
    occupancy_map = Costmap(occupancy_map_data, 0.03, origin=[0., 0.])

    obstacle_values = np.array([Costmap.OCCUPIED, Costmap.UNEXPLORED], dtype=np.uint8)

    start = rc_to_xy(np.argwhere(occupancy_map.data == Costmap.FREE)[0, :], occupancy_map)
    goals = rc_to_xy(np.argwhere(occupancy_map.data == Costmap.FREE)[-num_instances:, :], occupancy_map)

    astar_fn = partial(astar, start=start, occupancy_map=occupancy_map, obstacle_values=obstacle_values)

    start_time = time.time()
    results = pool.map(astar_fn, goals)
    time_elapsed = time.time() - start_time

    if debug:
        print("ran", num_instances, "astar instances, finished in", time_elapsed)

    for i, [successful, path] in enumerate(results):
        assert successful
        if debug:
            if successful:
                print("  instance", i, "\b: found solution with length:", path.shape[0])
            else:
                print("  instance", i, "\b: failed to find solution")


def test_oriented_astar(debug=False):
    occupancy_map_data = load_occupancy_map_data('brain', 'big_retail.png')
    occupancy_map = Costmap(occupancy_map_data, 0.03, origin=[0., 0.])

    footprint_points = get_tricky_circular_footprint()
    footprint = CustomFootprint(footprint_points, 2. * np.pi / 180., inflation_scale=2.0)

    angles = get_astar_angles()

    footprint_masks = footprint.get_footprint_masks(occupancy_map.resolution, angles=angles)
    outline_coords = footprint.get_outline_coords(occupancy_map.resolution, angles=angles)

    start = rc_to_xy([1956, 137, 0], occupancy_map)
    goal = rc_to_xy([841, 3403, 0.], occupancy_map)

    start_time = time.time()
    success, path = oriented_astar(goal=goal,
                                   start=start,
                                   occupancy_map=occupancy_map,
                                   footprint_masks=footprint_masks,
                                   outline_coords=outline_coords,
                                   obstacle_values=[0, 127],
                                   planning_scale=10)
    time_elapsed = time.time() - start_time

    assert success

    if debug:
        visualization_map = occupancy_map.copy()
        visualization_map.data = np.dstack((occupancy_map.data, occupancy_map.data, occupancy_map.data))
        draw_footprint_path(footprint, path, visualization_map, [255, 0, 0], [0, 0, 255])

        if success:
            print("found solution in", time_elapsed, "seconds, with length:", path.shape[0])
        else:
            print("failed to find solution")

        plt.imshow(visualization_map.data, interpolation='nearest')
        plt.show()


def test_impossible_path(debug=False):
    occupancy_map_data = load_occupancy_map_data('test', 'three_rooms.png')

    occupancy_map = Costmap(occupancy_map_data, 0.03, origin=[0., 0.])
    footprint_points = get_tricky_circular_footprint()
    footprint = CustomFootprint(footprint_points, 2. * np.pi / 180., inflation_scale=2.0)

    angles = get_astar_angles()

    footprint_masks = footprint.get_footprint_masks(occupancy_map.resolution, angles=angles)
    outline_coords = footprint.get_outline_coords(occupancy_map.resolution, angles=angles)

    start = rc_to_xy([425, 50, 0.], occupancy_map)
    goal = rc_to_xy([232, 339, 0.], occupancy_map)

    start_time = time.time()
    success, path = oriented_astar(goal=goal,
                                   start=start,
                                   occupancy_map=occupancy_map,
                                   footprint_masks=footprint_masks,
                                   outline_coords=outline_coords,
                                   obstacle_values=[0, 127],
                                   planning_scale=10)
    time_elapsed = time.time() - start_time

    if debug:
        visualization_map = occupancy_map.copy()
        visualization_map.data = np.dstack((occupancy_map.data, occupancy_map.data, occupancy_map.data))
        draw_footprint_path(footprint, path, visualization_map, [255, 0, 0], [0, 0, 255])

        if success:
            print("found solution in", time_elapsed, "seconds, with length:", path.shape[0])
        else:
            print("failed to find solution")

        plt.imshow(visualization_map.data, interpolation='nearest')
        plt.show()


def debug_real_case():
    c_angles = get_astar_angles()
    footprint_points = get_tricky_circular_footprint()

    data = np.load('debug2.npy')
    start = data[0]
    goal = data[1]
    occupancy_map = data[2]
    footprint_masks = data[3]
    outline_coords = data[5]
    obstacle_values = data[6]
    planning_scale = data[7]
    delta = data[8]
    epsilon = data[9]

    start_time = time.time()
    success, path = oriented_astar(goal=goal,
                                   start=start,
                                   occupancy_map=occupancy_map,
                                   footprint_masks=footprint_masks,
                                   outline_coords=outline_coords,
                                   obstacle_values=obstacle_values,
                                   planning_scale=planning_scale,
                                   delta=delta,
                                   epsilon=epsilon)
    time_elapsed = time.time() - start_time

    if success:
        print("found solution in", time_elapsed, "seconds, with length:", path.shape[0])
    else:
        print("failed to find solution")

    map_solution = np.dstack((occupancy_map.data, occupancy_map.data, occupancy_map.data))
    map_solution[path[:, 0].astype(np.int), path[:, 1].astype(np.int)] = [0, 0, 255]

    footprint = CustomFootprint(footprint_points, np.pi / 4., inflation_scale=1.0)
    outline_coords = np.array(footprint.get_outline_coords(occupancy_map.resolution))
    angle_inds = np.argmin(np.abs(path[:, 2:] - np.expand_dims(c_angles, axis=0)), axis=1)
    draw_footprints = outline_coords[angle_inds] + np.expand_dims(path[:, :2], axis=1)
    draw_footprints = draw_footprints.reshape(draw_footprints.shape[0] * draw_footprints.shape[1],
                                              draw_footprints.shape[2]).astype(np.int)
    map_solution[draw_footprints[:, 0], draw_footprints[:, 1]] = [255, 0, 0]

    plt.imshow(map_solution.data, interpolation='nearest')
    plt.show()


def main():
    debug = True
    test_one_astar(debug=debug)
    test_oriented_astar(debug=debug)
    test_multithread_astar(debug=debug)
    test_impossible_path(debug=debug)

    # debug_real_case()


if __name__ == '__main__':
    main()
