from __future__ import print_function, absolute_import, division

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from bc_exploration.agents.frontier_agent import extract_frontiers
from bc_exploration.algorithms.frontier_based_exploration import create_frontier_agent_from_params, visualize
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.paths import get_exploration_dir
from bc_exploration.utilities.util import xy_to_rc, which_coords_in_bounds, load_occupancy_map_data
from bc_exploration.utilities.visualization import draw_footprint_path


def test_extract_frontiers(debug=False):
    occupancy_map_data = load_occupancy_map_data('test', 'vw_partly_explored_test.png')
    occupancy_map = Costmap(data=occupancy_map_data, resolution=0.03, origin=np.array([0, 0]))

    frontiers = extract_frontiers(occupancy_map=occupancy_map, approx=False,
                                  kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    frontier_sizes = np.array([frontier.shape[0] if len(frontier.shape) > 1 else 1 for frontier in frontiers])
    significant_frontiers = [frontier for i, frontier in enumerate(frontiers) if frontier_sizes[i] > 70]
    assert len(significant_frontiers) == 4

    if debug:
        frontier_map = np.repeat([occupancy_map.data], repeats=3, axis=0).transpose((1, 2, 0))
        for frontier in significant_frontiers:
            frontier_px = xy_to_rc(frontier, occupancy_map).astype(np.int)
            frontier_px = frontier_px[which_coords_in_bounds(frontier_px, occupancy_map.get_shape())]
            frontier_map[frontier_px[:, 0], frontier_px[:, 1]] = [255, 0, 0]

        plt.imshow(frontier_map, cmap='gray', interpolation='nearest')
        plt.show()


def test_plan(debug=False):
    occupancy_map_data = load_occupancy_map_data('test', 'frontier_plan_map.png')
    occupancy_map = Costmap(data=occupancy_map_data, resolution=0.03, origin=np.array([-6.305, -6.305]))
    pose = np.array([.8, 0, -0.51759265])

    frontier_agent = create_frontier_agent_from_params(os.path.join(get_exploration_dir(), "params/params.yaml"))
    frontier_agent.is_first_plan = False

    plan = frontier_agent.plan(pose, occupancy_map)
    if debug:
        visualize(occupancy_map, pose, np.array([]), np.array([]), frontier_agent.get_footprint(), [], (1000, 1000), pose,
                  frontiers=frontier_agent.get_frontiers(compute=True, occupancy_map=occupancy_map))
    assert plan.shape[0] > 2


def debug_frontier_agent():
    data = np.load('debug3.npy')
    state = data[0]
    occupancy_map = data[1]
    frontier_agent = create_frontier_agent_from_params(get_exploration_dir() + '/params/params.yaml')

    path = frontier_agent.plan(state, occupancy_map)

    visualization_map = occupancy_map.copy()
    visualization_map.data = np.dstack((occupancy_map.data, occupancy_map.data, occupancy_map.data))
    draw_footprint_path(frontier_agent.get_footprint(), path, visualization_map, [0, 255, 0])

    plt.imshow(visualization_map.data, interpolation='nearest')
    plt.show()


def main():
    # debug_frontier_agent()
    test_extract_frontiers()
    test_plan()


if __name__ == '__main__':
    main()
