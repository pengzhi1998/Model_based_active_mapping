import numpy as np

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import xy_to_rc, which_coords_in_bounds, wrap_angles
from bc_exploration.sensors.sensor_util import bresenham2d


def compute_map_entropy(probability_map):
    return -1 * np.sum(probability_map.data * np.log(probability_map.data))


class SemanticMapper:
    def __init__(self, initial_map, sensor_range, num_class, l_0, phi_f, phi_o, psi_o, min_l, max_l):
        self._sensor_range = sensor_range
        self._num_class = num_class

        self._map = initial_map.copy()

        self._l_0 = np.clip(l_0 - l_0[0], min_l, max_l)

        self._phi_f = phi_f - phi_f[0]
        self._phi_o = phi_o - phi_o[0]
        self._psi_o = psi_o

        self._min_l = min_l
        self._max_l = max_l

        map_shape = initial_map.get_shape()
        self._log_odds_map = Costmap(data=self._l_0[None, None, :] * np.ones((map_shape[0], map_shape[1],
                                                                              self._num_class + 1)),
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

        free_coords = class_coords[classes == 0, :]
        occ_coords = class_coords[classes != 0, :]
        categories = classes[classes != 0]

        return free_coords, occ_coords, categories

    def _update_map_coords(self, free_coords, occ_coords, categories):
        self._log_odds_map.data[free_coords[:, 0], free_coords[:, 1], :] = \
            np.clip(self._log_odds_map.data[free_coords[:, 0], free_coords[:, 1], :] + self._phi_f - self._l_0,
                    self._min_l, self._max_l)

        inv_obs_log_odds = np.eye(self._num_class + 1)[:, categories] * self._psi_o[:, None] + self._phi_o[:, None]

        self._log_odds_map.data[occ_coords[:, 0], occ_coords[:, 1], :] = \
            np.clip(self._log_odds_map.data[occ_coords[:, 0], occ_coords[:, 1], :] + inv_obs_log_odds.T - self._l_0,
                    self._min_l, self._max_l)

        self._map.data[free_coords[:, 0], free_coords[:, 1]] = np.argmax(self._log_odds_map.data[free_coords[:, 0],
                                                                         free_coords[:, 1], :], axis=1)
        self._map.data[occ_coords[:, 0], occ_coords[:, 1]] = np.argmax(self._log_odds_map.data[occ_coords[:, 0],
                                                                       occ_coords[:, 1], :], axis=1)

        np.place(self._map.data, self._map.data == 0, Costmap.FREE)

        return self._map

    def update_obstructed(self, state, scan_angles, scan_ranges, scan_categories):
        map_state = state.copy()
        free_coords, occ_coords, categories = self._scan_to_class_coords(map_state, scan_angles, scan_ranges,
                                                                         scan_categories)

        return self._update_map_coords(free_coords, occ_coords, categories)

    def update_aerial(self, state, obs):
        pose_px = xy_to_rc(state, self._map).astype(np.int)
        obs[:, :2] += pose_px[:2]
        free_coords = obs[np.nonzero(obs[:, 2] == Costmap.FREE)[0], :2]
        occ_rows = np.nonzero(obs[:, 2] != Costmap.FREE)[0]
        occ_coords = obs[occ_rows, :2]
        categories = obs[occ_rows, 2]

        return self._update_map_coords(free_coords, occ_coords, categories)

    def get_log_odds_map(self):
        return self._log_odds_map

    def get_probability_map(self):
        return np.exp(self._log_odds_map.data) / np.sum(np.exp(self._log_odds_map.data), axis=2)[:, :, None]

    def get_occupancy_map(self):
        occupancy_map = self._map.copy()
        np.place(occupancy_map.data, np.logical_and(occupancy_map.data != Costmap.FREE,
                                                    occupancy_map.data != Costmap.UNEXPLORED), Costmap.OCCUPIED)
        return  occupancy_map

    def get_binary_prob_map(self):
        return Costmap(data=1 - self.get_probability_map()[:,:,0],
                       resolution=self._map.resolution,
                       origin=self._map.origin)
