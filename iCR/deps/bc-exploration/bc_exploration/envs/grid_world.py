"""grid_world.py
A simple grid world environment. Edges of the map are treated like obstacles
Map must be a image file whose values represent free (255, white), occupied (0, black).
"""

from __future__ import print_function, absolute_import, division

import cv2
import numpy as np
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import wrap_angles, compute_connected_pixels
from bc_exploration.utilities.util import xy_to_rc


class GridWorld:
    """
    A simple grid world environment. Edges of the map are treated like obstacles
    Map must be a image file whose values represent free (255, white), occupied (0, black).
    """

    def __init__(self,
                 map_filename,
                 map_resolution,
                 sensor,
                 footprint,
                 start_state=None,
                 render_size=(500, 500),
                 thicken_obstacles=True):
        """
        Creates an interactive grid world environment structured similar to open ai gym.
        Allows for moving, sensing, and visualizing within the space. Map loaded is based off the map_filename.
        :param map_filename str: path of image file whose values represent free (255, white), occupied (0, black).
        :param map_resolution float: resolution of which to represent the map
        :param sensor Sensor: sensor to use to sense the environment
        :param footprint Footprint: footprint of the robot
        :param start_state bool: if None, will randomly sample a free space point as the starting point
                                  else it must be [row, column] of the position you want the robot to start.
                                  it must be a valid location (free space and footprint must fit)
        :param render_size Tuple(int): size of which to render the map (for visualization env.render() )
        :param thicken_obstacles bool: thicken the obstacles in the map by 1 pixel, to avoid 1 pixel thick walls
        """

        self.footprint = footprint
        self.footprint_no_inflation = footprint.no_inflation()
        self.map_resolution = map_resolution

        self.render_size = render_size
        assert render_size.shape[0] == 2 if isinstance(render_size, np.ndarray) else len(render_size) == 2

        self.state = None
        self.map = None
        self.truth_free_coords = None

        self._load_map(map_filename, map_resolution, thicken_obstacles=thicken_obstacles)

        self.start_state = np.array(start_state).astype(np.float) \
            if start_state is not None else self._get_random_start_state()
        assert self.start_state.shape[0] == 3

        self.sensor = sensor
        assert self.map is not None
        self.sensor.set_map(occupancy_map=self.map)

        self.reset()
        assert self._is_state_valid(self.start_state)

    def _get_random_start_state(self):
        """
        Samples a random valid state from the map
        :return array(3)[float]: state [x, y, theta] of the robot
        """
        valid_points = self.map_resolution * np.argwhere(self.map.data == Costmap.FREE)[:, ::-1]
        choice = np.random.randint(valid_points.shape[0])

        valid = False
        while not valid:
            choice = np.random.randint(valid_points.shape[0])
            valid = self._is_state_valid(np.concatenate((valid_points[choice], [0])))

        # todo add random angle
        return np.concatenate((valid_points[choice], [0])).astype(np.float)

    def _is_state_valid(self, state, use_inflation=True):
        """
        Make sure state is not out of bounds or on an obstacle (footprint check)
        :param state array(3)[float]: [x, y, theta], the position/orientation of the robot
        :param use_inflation bool: whether to use the inflated footprint to collision check, or the normal footprint.
                              usually the environment will need use the actual footprint of the robot (because thats the
                              physical limitation we want to simulate)
        :return bool: whether it is valid or not
        """
        return 0 <= state[0] < self.map.get_size()[0] and 0 <= state[1] < self.map.get_size()[1] \
            and not (self.footprint.check_for_collision(state=state, occupancy_map=self.map)
                     if use_inflation else self.footprint_no_inflation.check_for_collision(state=state, occupancy_map=self.map))

    def _load_map(self, filename, map_resolution, thicken_obstacles=True):
        """
        Loads map from file into costmap object
        :param filename str: location of the map file
        :param map_resolution float: desired resolution of the loaded map
        :param thicken_obstacles bool: thicken the obstacles in the map by 1 pixel, to avoid 1 pixel thick walls
        """
        map_data = cv2.imread(filename)
        assert map_data is not None and "map file not able to be loaded. Does the file exist?"
        map_data = cv2.cvtColor(map_data, cv2.COLOR_RGB2GRAY)
        map_data = map_data.astype(np.uint8)
        assert np.max(map_data) == 255

        if thicken_obstacles:
            occupied_coords = np.argwhere(map_data == Costmap.OCCUPIED)
            occupied_mask = np.zeros_like(map_data)
            occupied_mask[occupied_coords[:, 0], occupied_coords[:, 1]] = 1
            occupied_mask = cv2.dilate(occupied_mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            occupied_coords = np.argwhere(occupied_mask == 1)
            map_data[occupied_coords[:, 0], occupied_coords[:, 1]] = Costmap.OCCUPIED

        self.map = Costmap(data=map_data, resolution=map_resolution, origin=[0., 0.])

    def compare_maps(self, occupancy_map):
        """
        Does a comparison of the ground truth map with the input map,
        and will return a percentage completed
        :param occupancy_map Costmap: input map to be compared with ground truth
        :return float: percentage covered of the input map to the ground truth map
        """
        if self.truth_free_coords is None:
            start_state_px = xy_to_rc(self.start_state, self.map)
            self.truth_free_coords = compute_connected_pixels(start_state_px, self.map.data)

        free_coords = np.argwhere(occupancy_map.data == Costmap.FREE)
        return free_coords.shape[0] / float(self.truth_free_coords.shape[0])

    def step(self, desired_state):
        """
        Execute the given action with the robot in the environment, return the next robot position,
        and the output of sensor.measure() (sensor data)
        :param desired_state array(3)[float]: desired next state of the robot
        :return array(3)[float]: new_state (new robot position), sensor_data (output from sensor)
        """
        new_state = np.array(desired_state, dtype=np.float)
        new_state[2] = wrap_angles(desired_state[2])

        # # todo maybe keep angle of desired state
        if not self._is_state_valid(new_state + self.start_state, use_inflation=False):
            new_state = self.state.copy()

        # compute sensor_data
        measure_state = new_state.copy()
        measure_state[:2] += self.start_state[:2]
        sensor_data = self.sensor.measure(measure_state)

        # set state
        self.state = new_state

        return new_state.copy(), sensor_data

    def reset(self):
        """
        Resets the robot to the starting state in the environment
        :return array(3)[float]: robot position, sensor data from start state
        """
        self.state = np.array((0, 0, self.start_state[2]))
        sensor_data = self.sensor.measure(self.start_state)
        return self.state.copy(), sensor_data

    def render(self, wait_key=0):
        """
        Renders the environment and the robots position
        :param wait_key int: the opencv waitKey arg
        """
        # convert to colored image
        map_vis = cv2.cvtColor(self.map.data.copy(), cv2.COLOR_GRAY2BGR)

        state_px = xy_to_rc(self.state + self.start_state, self.map)[:2].astype(np.int)
        map_vis[state_px[0], state_px[1]] = [127, 122, 10]
        # # todo figure out programmatic way to pick circle size (that works well)
        # cv2.circle(map_vis, tuple(self.state[:2][::-1].astype(int)), 1, [127, 122, 10], thickness=-1)

        # resize map
        map_vis = cv2.resize(map_vis, tuple(self.render_size), interpolation=cv2.INTER_NEAREST)

        # visualize map
        cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('map', map_vis)
        cv2.resizeWindow('map', *self.render_size)
        cv2.waitKey(wait_key)

    def get_sensor(self):
        """
        Gets the sensor object
        :return Sensor: the sensor used by the grid world
        """
        return self.sensor

    def get_map_shape(self):
        """
        Returns the shape of the map
        :return Tuple(int): map shape
        """
        return self.map.get_shape()

    def get_map_size(self):
        """
        Returns the size of the map (x, y) in meters
        :return Tuple(float): map size (meters)
        """
        return self.map.get_size()

    def get_map_resolution(self):
        """
        Returns the resolution of the map
        :return float: resolution
        """
        return self.map_resolution

    def __del__(self):
        """
        Destructor, delete opencv windows
        """
        cv2.destroyAllWindows()
