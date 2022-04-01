"""sensors.py
Variety of sensors to be used with grid world, not all are working yet.
"""
from __future__ import print_function, absolute_import, division

import numpy as np
from matplotlib import pyplot as plt

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.sensors.sensor_util import bresenham2d
from bc_exploration.utilities.util import wrap_angles, which_coords_in_bounds, xy_to_rc, scan_to_points, get_rotation_matrix_2d


class Sensor:
    """
    Sensor base class
    """
    def __init__(self, sensor_range):
        """
        Base class for a variety of sensors.
        Assumes edges of map are obstacles.
        :param sensor_range float: sensor range in meters
        """
        self._range = sensor_range
        self._map = None

    def set_map(self, occupancy_map):
        """
        initializes the map
        :param occupancy_map Costmap: the map for the sensor to use for measurements
        """
        self._map = occupancy_map

    def get_sensor_range(self):
        """
        returns the range of the sensor
        :return float: range of sensor in meters
        """
        return self._range

    def measure(self, state, debug=False):
        """
        Returns data using state and map
        :param state array(3)[float]: state for measuring in meters [x, y, theta]
        :param debug bool: show plots?
        """
        raise NotImplementedError


class Neighborhood(Sensor):
    """
    Gives a NxN square of information around the robot.
    """
    def __init__(self, sensor_range, values=(0, 255)):
        """
        Neighboorhood sensor, ignores obstacles
        :param sensor_range float: range of sensor in meters
        :param values Tuple[uint8]: values of which to return locations in map, in order
        """
        Sensor.__init__(self, sensor_range)
        self.values = values
        self._range_px = None

    def set_map(self, occupancy_map):
        """
        Set the map for the sensor, must be setup for the object before calling measure
        :param occupancy_map Costmap: the ground truth of which to sample from
        """
        self._map = occupancy_map
        self._range_px = np.rint(self._range / occupancy_map.resolution).astype(np.int)

    def measure(self, state, debug=False):
        """
        Using the state, and referencing the map, isolate the window around the state in the map, and return
        occupied and free pixels
        :param state array(3)[float]: robot position in coordinate space [row, col, yaw]
        :param debug bool: show plots?
        :return array(N,2)[int]: ego coordinates corresponding to self.values in the costmap
        """
        if self._map is None or self._range_px is None:
            assert False and "Sensor's map is not currently set, please initialize with set_map()," \
                             " the set_map() function should initialize map and range_px"

        state = xy_to_rc(state, self._map)
        pad_map = np.pad(self._map.data.copy(), pad_width=self._range_px, mode='constant', constant_values=0)

        min_range = np.array([state[0], state[1]]).astype(int)
        max_range = np.array([state[0] + 2 * self._range_px, state[1] + 2 * self._range_px]).astype(int)

        ego_map = pad_map[min_range[0]:max_range[0] + 1,
                          min_range[1]:max_range[1] + 1]

        ego_state = np.array([self._range_px, self._range_px])

        if debug:
            vis_map = ego_map.copy()
            vis_map[ego_state[0], ego_state[1]] = 75
            plt.imshow(vis_map, cmap='gray')
            plt.show()

        coords = []
        for value in self.values:
            value_coords = np.argwhere(ego_map == value) - ego_state + min_range
            is_valid = np.logical_and(np.all(value_coords >= 0, axis=1),
                                      np.logical_and(value_coords[:, 0] < self._map.data.shape[0],
                                                     value_coords[:, 1] < self._map.data.shape[1]))

            coords.append(value_coords[is_valid] - state[:2].astype(np.int))

        return coords


class Laser(Sensor):
    """
    Python lidar implementation
    """
    def __init__(self, sensor_range, map_resolution, angle_range=(-45, 45), angle_increment=1):
        """
        Python implementation of a lidar
        :param sensor_range float: range of sensor in meters
        :param map_resolution float: resolution of the map
        :param angle_range Tuple[int]: (min angle, max angle) around 0, in degrees ie (-45, 45)
        :param angle_increment float: angle increment of which to precompute rays in degrees
        """
        Sensor.__init__(self, sensor_range)
        assert angle_increment >= 1
        self.angle_range = wrap_angles(np.array(angle_range))
        self.map_resolution = map_resolution
        self.range_px = np.rint(sensor_range / map_resolution).astype(np.int)
        self.angle_increment = angle_increment
        self.ray_angles, self.circle_rays = self._generate_circle_rays()

    def _generate_circle_rays(self, debug=False):
        """
        Precomputes the circle rays at the specified angles
        :param debug bool: show plots?
        :return Tuple[array(N)[int], array(N,2)[int]]: angles which were generated, and the rays at those angles
        """
        angles = np.arange(-180, 180, 1)

        # we define - to be left of y axis and + to be right of y axis, and y axis to be 0 degrees
        # todo change this to brains format 0 degrees right neg down pos up
        range_points = self.range_px * np.vstack(([-np.cos(angles * np.pi / 180)], [np.sin(angles * np.pi / 180)])).T
        circle_coords = np.round(range_points).astype(np.int)
        if debug:
            plt.scatter(circle_coords[:, 0], circle_coords[:, 1])
            plt.show()

        circle_rays = [bresenham2d([0, 0], circle_coord).astype(np.int) for circle_coord in circle_coords.tolist()]

        return angles, circle_rays

    def _compute_state_rays(self, state):
        """
        Uses the precomputed rays to compute the rays at the current state
        :param state array(3)[float]: state of the robot
        :return List[array(N,2)[int]]: precomputed rays shifted by state
        """
        shifted_angle_range = self.angle_range + np.round(state[2] * 180 / np.pi)

        if shifted_angle_range[0] == 180:
            shifted_angle_range[0] *= -1

        if shifted_angle_range[1] == -180:
            shifted_angle_range[1] *= -1

        desired_angles = wrap_angles(np.arange(shifted_angle_range[0],
                                               shifted_angle_range[1] + self.angle_increment,
                                               self.angle_increment), is_radians=False)

        state_rays = [(circle_ray + state[:2]).astype(np.int) for i, circle_ray in enumerate(self.circle_rays)
                      if self.ray_angles[i] in desired_angles]

        return state_rays

    def measure(self, state, debug=False):
        """
        Returns the occupied free ego coordinates at the current state
        :param state array(3)[float]: state of the robot
        :param debug bool: show plots?
        :return Union[array(N,2)[int], array(N,2)[int]]: ego occupied, ego free coordinates
        """
        if self._map is None:
            assert False and "Sensor's map is not currently set, please initialize with set_map()"

        state_rays = self._compute_state_rays(state)

        # remove ray points that are out of bounds
        rays = []
        for state_ray in state_rays:
            in_bound_point_inds = which_coords_in_bounds(state_ray, map_shape=self._map.data.shape)
            rays.append(state_ray[in_bound_point_inds])

        if debug:
            vis_map = self._map.copy()
            for ray in rays:
                vis_map[ray[:, 0], ray[:, 1]] = 75
            plt.imshow(vis_map)
            plt.show()

        occupied = []
        free = []
        for ray in rays:
            ind = -1
            for ind, point in enumerate(ray):
                if self._map[int(point[0]), int(point[1])] == 0:
                    occupied.append(point)
                    break
            if ind != -1:
                free.extend(ray[:ind, :])
            else:
                free.extend(ray)

        return [np.array(occupied) - state[:2].astype(np.int) if len(occupied) else np.empty((0,)),
                np.array(free) - state[:2].astype(np.int) if len(free) else np.empty((0,))]


class Lidar(Sensor):
    """
    Lidar sensor that conforms to the Sensor class
    """
    def __init__(self, sensor_range, angular_range, angular_resolution, map_resolution):
        """
        Realistic lidar.
        :param sensor_range float: range of the lidar in meters
        :param angular_range array(N)[float]: the total amount of angle to be covered, all around is 2 * pi, centered at 0 radians in
                              the robots frame
        :param angular_resolution float: angular increment for each ray
        :param map_resolution float: resolution of the map
        """
        Sensor.__init__(self, sensor_range)
        self._angular_range = angular_range
        self._angular_resolution = angular_resolution
        self._map_resolution = map_resolution
        self._ray_angles, self._ego_rays = self.precompute_ego_rays()

    def precompute_ego_rays(self):
        """
        Precomputes the angles and rays in ego coordinates of the lidar.
        :return Tuple[array(N)[float], array(N, 2)[float]]: the lidar angles, ray ego coords
        """
        angles = np.arange(-self._angular_range / 2, self._angular_range / 2 + self._angular_resolution, self._angular_resolution)

        ray_points = self._range * np.array([np.cos(angles), np.sin(angles)]).T
        ray_points_px = np.rint(ray_points / self._map_resolution).astype(np.int)[:, ::-1] * [-1, 1]
        ego_rays = [bresenham2d([0, 0], ray_point_px).astype(np.int) for ray_point_px in ray_points_px]

        return angles, ego_rays

    def set_map(self, occupancy_map):
        """
        Initialize the map for the sensor to read off of. It will compute self.costmap to be used with the pixel lidar
        :param occupancy_map Costmap: object corresponding to the map
        """
        self._map = occupancy_map.copy()

    def measure(self, state, debug=False):
        """
        given the current robot pose, return the lidar data
        :param state array(3)[float]: [x, y, theta (radians)] pose of the robot
        :param debug bool: show debug plot?
        :return Tuple[array(N)[float], array(N)[float]]: [lidar angles radians, lidar ranges meters] if ray did not hit obstacle, value is np.nan
        """
        assert self._map is not None \
            and "Please set the map using set_map() before calling measure, this will initialize lidar as well."

        pose = state.copy()
        pose_px = xy_to_rc(pose, self._map)

        ranges = np.zeros_like(self._ray_angles)
        for i, ego_ray in enumerate(self._ego_rays):
            rotation_matrix = get_rotation_matrix_2d(-state[2])
            rotated_ray = np.rint(ego_ray.dot(rotation_matrix))
            ray = (pose_px[:2] + rotated_ray).astype(np.int)
            ray = ray[which_coords_in_bounds(ray, self._map.get_shape())]
            occupied_ind = np.argwhere(self._map.data[ray[:, 0], ray[:, 1]] == Costmap.OCCUPIED)
            if occupied_ind.shape[0]:
                ranges[i] = np.linalg.norm(pose_px[:2] - ray[int(occupied_ind[0])]) * self._map.resolution
            else:
                ranges[i] = np.nan

        if debug:
            points = scan_to_points(self._ray_angles + state[2],
                                    ranges) + pose_px[:2]
            plt.plot(points[:, 0], points[:, 1])
            plt.show()

        return self._ray_angles, ranges
