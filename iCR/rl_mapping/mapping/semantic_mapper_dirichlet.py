import numpy as np

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import xy_to_rc, which_coords_in_bounds, wrap_angles
from bc_exploration.sensors.sensor_util import bresenham2d

from sem_slam.utilities.utils import entropy


def compute_map_entropy(parameter_map):
    s1, s2, s3 = parameter_map.get_shape()
    alphas = np.reshape(parameter_map.data, (s1 * s2, s3))
    return np.sum(entropy(alphas))

class SemanticMapper:
    def __init__(self, initial_map, sensor_range, num_class, max_alpha=None, update_inc=2):
        self._sensor_range = sensor_range
        self._num_class = num_class
        if max_alpha is None:
            self._max_alpha = 150 / num_class
        else:
            self._max_alpha = max_alpha

        self._update_inc = update_inc

        self._map = initial_map
        self._origin = initial_map.origin

        map_shape = initial_map.get_shape()
        self._parameter_map = Costmap(data=(1 + 1e-10) * np.ones((map_shape[0], map_shape[1], self._num_class + 1)),
                                      resolution=initial_map.resolution,
                                      origin=initial_map.origin)

    def _scan_to_class_coords(self, state, scan_angles, scan_ranges, scan_categories):
        world_angles = wrap_angles(state[2] + scan_angles)

        free_inds = np.logical_or(np.isnan(scan_ranges), scan_ranges > self._sensor_range)
        scan_ranges[free_inds] = self._sensor_range

        robot_xy = state[:2]
        robot_rc = xy_to_rc(robot_xy, self._map)
        class_coords = np.array([robot_rc])
        classes = np.array([0])
        for i, range_measure in enumerate(scan_ranges):
            end_xy = robot_xy + range_measure * np.array([np.cos(world_angles[i]), np.sin(world_angles[i])])
            end_rc = xy_to_rc(end_xy, self._map)
            bresenham_line = bresenham2d(robot_rc, end_rc)
            if bresenham_line.shape[0] >= 2:
                class_coords = np.vstack((class_coords, bresenham_line[1:]))
                classes = np.concatenate((classes, np.zeros(bresenham_line.shape[0] - 2)))
                classes = np.concatenate((classes, scan_categories[i, None]))

        class_coords = class_coords.astype(np.int)
        classes = classes.astype(np.int)
        good_inds = which_coords_in_bounds(class_coords, self._map.get_shape())
        class_coords = class_coords[good_inds]
        classes = classes[good_inds]

        return class_coords, classes

    def update(self, state, scan_angles, scan_ranges, scan_categories):
        map_state = state.copy()
        class_coords, categories = self._scan_to_class_coords(map_state, scan_angles, scan_ranges, scan_categories)

        self._parameter_map.data[class_coords[:, 0], class_coords[:, 1], categories] += self._update_inc
        self._parameter_map.data = np.clip(self._parameter_map.data, 0, self._max_alpha)

        mode_map = self._get_mode()

        self._map.data[class_coords[:, 0], class_coords[:, 1]] = np.argmax(mode_map[class_coords[:, 0],
                                                                           class_coords[:, 1], :], axis=1)

        np.place(self._map.data, self._map.data == 0, Costmap.FREE)

        return self._map

    def get_parameter_map(self):
        return self._parameter_map

    def _get_mode(self):
        mode = (self._parameter_map.data - 1) / (np.sum(self._parameter_map.data, axis=2) -
                                                 (self._num_class + 1))[:, :, None]
        return mode