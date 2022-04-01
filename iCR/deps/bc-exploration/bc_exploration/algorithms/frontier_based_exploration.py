"""frontier_based_exploration.py
Main interface for the FrontierAgent.py
Contains construction method for frontier agent from params, as well as interface to run with grid world
"""
from __future__ import print_function, absolute_import, division

import os

import cv2
import numpy as np
import yaml

from bc_exploration.agents.frontier_agent import FrontierAgent
from bc_exploration.envs.grid_world import GridWorld
from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint, get_jackal_footprint
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.mapping.log_odds_mapper import LogOddsMapper
from bc_exploration.sensors.sensors import Lidar
from bc_exploration.utilities.paths import get_maps_dir, get_exploration_dir
from bc_exploration.utilities.util import xy_to_rc, which_coords_in_bounds, scan_to_points
from bc_exploration.utilities.visualization import draw_footprint_path, draw_frontiers, draw_scan_ranges


def create_frontier_agent_from_params(params_filename):
    """
    Creates a frontier agent from params given in the params file, the params file must be located in the params
    folder
    :param params_filename str: path to the params file
    :return FrontierAgent: object created from the params
    """
    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    assert params['algorithm'] == 'frontier'
    if params['footprint']['type'] == 'tricky_circle':
        footprint_points = get_tricky_circular_footprint()
    elif params['footprint']['type'] == 'tricky_oval':
        footprint_points = get_tricky_oval_footprint()
    elif params['footprint']['type'] == 'jackal':
        footprint_points = get_jackal_footprint()
    elif params['footprint']['type'] == 'circle':
        rotation_angles = np.arange(0, 2 * np.pi, 4 * np.pi / 180)
        footprint_points = \
            params['footprint']['radius'] * np.array([np.cos(rotation_angles), np.sin(rotation_angles)]).T
    elif params['footprint']['type'] == 'pixel':
        footprint_points = np.array([[0., 0.]])
    else:
        footprint_points = None
        assert False and "footprint type specified not supported."

    footprint = CustomFootprint(footprint_points=footprint_points,
                                angular_resolution=params['footprint']['angular_resolution'],
                                inflation_scale=params['footprint']['inflation_scale'])

    frontier_agent = FrontierAgent(footprint=footprint,
                                   mode=params['mode'],
                                   min_frontier_size=params['min_frontier_size'],
                                   max_replans=params['max_replans'],
                                   planning_resolution=params['planning_resolution'],
                                   planning_epsilon=params['planning_epsilon'],
                                   planning_delta_scale=params['planning_delta_scale'])

    return frontier_agent


def visualize(occupancy_map, state, scan_angles, scan_ranges, footprint, path, render_size, start_state, frontiers,
              wait_key=0, flipud=False, path_idx=None):
    """
    Visualization function for exploration.
    :param occupancy_map Costmap: object, map to visualize
    :param state array(3)[float]: corresponds to the pose of the robot [x, y, theta]
    :param scan_angles array(N)[float]: lidar scan angles for visualization
    :param scan_ranges array(N)[float]: lidar scan ranges for visualization
    :param footprint CustomFootprint: for visualization
    :param path array(N, 3)[float]: containing an array of poses for the robot to follow
    :param render_size Tuple(int, int): the size of the opencv window to render
    :param start_state array(3)[float]: starting state of the robot [x, y, theta]
    :param frontiers List[array(N,2)[float]): frontiers for plotting
    :param wait_key int: the opencv.waitKey on the visualization window
    :param flipud bool: whether to flip the map ud
    :param path_idx int: the index on the path we are currently visualizing, used to only plot path after the index
    """
    map_vis = occupancy_map.copy()
    map_vis.data = cv2.cvtColor(map_vis.data, cv2.COLOR_GRAY2BGR)

    if np.array(path).shape[0]:
        # todo draw arrow with heading
        draw_footprint_path(footprint=footprint, path=path[:-1] if path_idx is None else path[path_idx + 1:-1],
                            visualization_map=map_vis, footprint_color=[200, 255, 200], path_color=None,
                            footprint_thickness=-1)

    if len(frontiers):
        draw_frontiers(visualization_map=map_vis, frontiers=frontiers, color=[255, 150, 80])

    if start_state is not None:
        footprint.no_inflation().draw([0, 0, start_state[2]], map_vis, [10, 122, 127])
    footprint.no_inflation().draw(state, map_vis, [127, 122, 10])

    if scan_ranges.shape[0]:
        draw_scan_ranges(visualization_map=map_vis, state=state,
                         scan_angles=scan_angles, scan_ranges=scan_ranges, color=[0, 255, 0])

    # visualize map
    cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
    if flipud:
        cv2.imshow('map', np.flipud(map_vis.data))
    else:
        cv2.imshow('map', map_vis.data)
    cv2.resizeWindow('map', *render_size)
    cv2.waitKey(wait_key)


def run_frontier_exploration(map_filename, params_filename, start_state, sensor_range, map_resolution,
                             completion_percentage, render=True, render_interval=1, render_size_scale=1.7,
                             completion_check_interval=1, render_wait_for_key=True, max_exploration_iterations=None):
    """
    Interface for running frontier exploration on the grid world environment that is initialized via map_filename.. etc
    :param map_filename str: path of the map to load into the grid world environment, needs to be a uint8 png with
                         values 127 for unexplored, 255 for free, 0 for occupied.
    :param params_filename str: path of the params file for setting up the frontier agent etc.
                            See exploration/params/ for examples
    :param start_state array(3)[float]: starting state of the robot in the map (in meters) [x, y, theta], if None the starting
                        state is random
    :param sensor_range float: range of the sensor (lidar) in meters
    :param map_resolution float: resolution of the map desired
    :param completion_percentage float: decimal of the completion percentage desired i.e (.97), how much of the ground
                                  truth environment to explore, note that 1.0 is not always reachable because of
                                  footprint.
    :param render bool: whether or not to visualize
    :param render_interval int: visualize every render_interval iterations
    :param render_size_scale Tuple(int): (h, w), the size of the render window in pixels
    :param completion_check_interval int: check for exploration completion every completion_check_interval iterations
    :param render_wait_for_key bool: if render is enabled, if render_wait_for_key is True then the exploration algorithm
                                will wait for key press to start exploration. Timing is not affected.
    :param max_exploration_iterations int: number of exploration cycles to run before exiting
    :return Costmap: occupancy_map, final map from exploration), percentage explored, time taken to explore
    """

    # some parameters
    frontier_agent = create_frontier_agent_from_params(params_filename)
    footprint = frontier_agent.get_footprint()

    # pick a sensor
    sensor = Lidar(sensor_range=sensor_range,
                   angular_range=250 * np.pi / 180,
                   angular_resolution=1.0 * np.pi / 180,
                   map_resolution=map_resolution)

    # setup grid world environment
    env = GridWorld(map_filename=map_filename,
                    map_resolution=map_resolution,
                    sensor=sensor,
                    footprint=footprint,
                    start_state=start_state)

    render_size = (np.array(env.get_map_shape()[::-1]) * render_size_scale).astype(np.int)

    # setup log-odds mapper, we assume the measurements are very accurate,
    # thus one scan should be enough to fill the map
    padding = 1.
    map_shape = np.array(env.get_map_shape()) + int(2. * padding // map_resolution)
    initial_map = Costmap(data=Costmap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8),
                          resolution=env.get_map_resolution(),
                          origin=[-padding - env.start_state[0], -padding - env.start_state[1]])

    clearing_footprint_points = footprint.get_clearing_points(map_resolution)
    clearing_footprint_coords = xy_to_rc(clearing_footprint_points, initial_map).astype(np.int)
    initial_map.data[clearing_footprint_coords[:, 0], clearing_footprint_coords[:, 1]] = Costmap.FREE

    mapper = LogOddsMapper(initial_map=initial_map,
                           sensor_range=sensor.get_sensor_range(),
                           measurement_certainty=0.8,
                           max_log_odd=8,
                           min_log_odd=-8,
                           threshold_occupied=.5,
                           threshold_free=.5)

    # reset the environment to the start state, map the first observations
    pose, [scan_angles, scan_ranges] = env.reset()

    occupancy_map = mapper.update(state=pose, scan_angles=scan_angles, scan_ranges=scan_ranges)

    if render:
        visualize(occupancy_map=occupancy_map, state=pose, footprint=footprint,
                  start_state=start_state, scan_angles=scan_angles, scan_ranges=scan_ranges, path=[],
                  render_size=render_size, frontiers=[], wait_key=0 if render_wait_for_key else 1)

    iteration = 0
    is_last_plan = False
    was_successful = True
    percentage_explored = 0.0
    while True:
        if iteration % completion_check_interval == 0:
            percentage_explored = env.compare_maps(occupancy_map)
            if percentage_explored >= completion_percentage:
                is_last_plan = True

        if max_exploration_iterations is not None and iteration > max_exploration_iterations:
            was_successful = False
            is_last_plan = True

        # using the current map, make an action plan
        path = frontier_agent.plan(state=pose, occupancy_map=occupancy_map, is_last_plan=is_last_plan)

        # if we get empty lists for policy/path, that means that the agent was
        # not able to find a path/frontier to plan to.
        if not path.shape[0]:
            print("No more frontiers! Either the map is 100% explored, or bad start state, or there is a bug!")
            break

        # if we have a policy, follow it until the end. update the map sparsely (speeds it up)
        for j, desired_state in enumerate(path):
            if footprint.check_for_collision(desired_state, occupancy_map, unexplored_is_occupied=True):
                footprint_coords = footprint.get_ego_points(desired_state[2], map_resolution) + desired_state[:2]
                footprint_coords = xy_to_rc(footprint_coords, occupancy_map).astype(np.int)
                footprint_coords = footprint_coords[which_coords_in_bounds(footprint_coords, occupancy_map.get_shape())]
                occupancy_map.data[footprint_coords[:, 0], footprint_coords[:, 1]] = Costmap.FREE

            pose, [scan_angles, scan_ranges] = env.step(desired_state)
            occupancy_map = mapper.update(state=pose, scan_angles=scan_angles, scan_ranges=scan_ranges)

            # put the current laserscan on the map before planning
            occupied_coords = scan_to_points(scan_angles + pose[2], scan_ranges) + pose[:2]
            occupied_coords = xy_to_rc(occupied_coords, occupancy_map).astype(np.int)
            occupied_coords = occupied_coords[which_coords_in_bounds(occupied_coords, occupancy_map.get_shape())]
            occupancy_map.data[occupied_coords[:, 0], occupied_coords[:, 1]] = Costmap.OCCUPIED

            # shows a live visualization of the exploration process if render is set to true
            if render and j % render_interval == 0:
                visualize(occupancy_map=occupancy_map, state=pose, footprint=footprint,
                          start_state=start_state, scan_angles=scan_angles, scan_ranges=scan_ranges,
                          path=path, render_size=render_size,
                          frontiers=frontier_agent.get_frontiers(compute=True, occupancy_map=occupancy_map), wait_key=1,
                          path_idx=j)

        if is_last_plan:
            break

        iteration += 1

    if render:
        cv2.waitKey(0)

    return occupancy_map, percentage_explored, iteration, was_successful


def main():
    """
    Main Function
    """
    # big target 270 plans crash
    np.random.seed(3)
    _, percent_explored, iterations_taken, _ = \
        run_frontier_exploration(map_filename=os.path.join(get_maps_dir(), "brain/vw_ground_truth_full_edited.png"),
                                 params_filename=os.path.join(get_exploration_dir(), "params/params.yaml"),
                                 map_resolution=0.03,
                                 start_state=None,
                                 sensor_range=10.0,
                                 completion_percentage=0.98,
                                 max_exploration_iterations=None,
                                 render_size_scale=2.0,
                                 render_interval=5)

    print("Map", "{:.2f}".format(percent_explored * 100), "\b% explored!",
          "This is " + str(iterations_taken) + " iterations!")


if __name__ == '__main__':
    main()
