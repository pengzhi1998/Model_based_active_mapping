"""log_odds_mapper.py
Traditional log odds mapper, works with lidar data
"""
from __future__ import print_function, absolute_import, division

import numpy as np
from matplotlib import pyplot as plt

from bc_exploration.sensors.sensor_util import bresenham2d_with_intensities
from bc_exploration.utilities.util import xy_to_rc, which_coords_in_bounds, wrap_angles, scan_to_points


class LogOddsMapper:
    """
    Traditional log-odds based occupancy mapper.
    Call the update function with your occupied and free coords to start mapping!
    """
    def __init__(self,
                 initial_map,
                 sensor_range,
                 measurement_certainty=.8,
                 max_log_odd=50,
                 min_log_odd=-8,
                 threshold_occupied=.7,
                 threshold_free=.3):
        """
        Traditional log-odds based occupancy mapper. Call the update function with your occupied and free coords
        to start mapping!

        :param initial_map Costmap: initial map to start mapping on. ex. Costmap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8)
        :param sensor_range float: max range of the sensor, used to plot freespace for empty scans / filter invalid scans
        :param measurement_certainty float: 0 < measurement_certainty < 1 (cant be 1)
                                      how much you trust your sensor's measurements
        :param max_log_odd float: maximum log odd, log-odds will not increase past this (clipped)
        :param min_log_odd float: minimum log odd, log-odds will not decrease past this (clipped)
        :param threshold_occupied float: the threshold of which to call a pixel occupied (probability (0,1))
        :param threshold_free float: the threshold of which to call a pixel free (probability (0,1))
        """

        assert measurement_certainty != 1
        assert sensor_range is not None
        self.sensor_range = sensor_range
        self.measurement_certainty = measurement_certainty
        self.max_log_odd = max_log_odd
        self.min_log_odd = min_log_odd
        self.threshold_filled = threshold_occupied
        self.threshold_empty = threshold_free

        self._map = initial_map
        self._origin = initial_map.origin
        self._log_odds_map = initial_map.copy()
        self._log_odds_map = np.zeros(self._map.get_shape())
        self._probability_map = initial_map.copy()
        self._probability_map = 0.5 * np.ones(self._map.get_shape())

    def _scan_to_occupied_free_coords(self, state, angles, ranges, debug=False):
        """
        converts laser scan to occupied and free coordinates for mapping
        :param state array(3)[float]: current state of the robot
        :param angles array(N)[float]: lidar angles of the robot
        :param ranges array(N)[float]: lidar ranges corresponding to the angles
        :param debug bool: show debug?
        :return Tuple[array(N,2)[int], array(N,2)[int]]: occupied coordinates, and free coordinates (in row column)
        """

        new_angles = wrap_angles(angles.copy() + state[2])
        state_px = xy_to_rc(state, self._map)
        position_px = state_px[:2].astype(np.int)

        with np.errstate(invalid='ignore'):
            free_inds = np.logical_or(np.isnan(ranges), ranges >= self.sensor_range)
            occ_inds = np.logical_not(free_inds)

        ranges = ranges.copy()
        ranges[free_inds] = self.sensor_range

        occupied_coords = scan_to_points(new_angles[occ_inds], ranges[occ_inds]) + state[:2]
        occupied_coords = xy_to_rc(occupied_coords, self._map).astype(np.int)

        free_endcoords = scan_to_points(new_angles[free_inds], ranges[free_inds]) + state[:2]
        free_endcoords = xy_to_rc(free_endcoords, self._map).astype(np.int)

        if debug:
            occupied_coords_vis = occupied_coords[which_coords_in_bounds(occupied_coords, self._map.get_shape())]
            free_coords_vis = free_endcoords[which_coords_in_bounds(free_endcoords, self._map.get_shape())]
            map_vis = np.repeat([self._map.data], repeats=3, axis=0).transpose((1, 2, 0)).copy()
            map_vis[occupied_coords_vis[:, 0], occupied_coords_vis[:, 1]] = [255, 0, 0]
            map_vis[free_coords_vis[:, 0], free_coords_vis[:, 1]] = [0, 255, 0]
            plt.imshow(map_vis)
            plt.show()

        free_coords = np.array([position_px])
        for i in range(occupied_coords.shape[0]):
            bresenham_line = bresenham2d_with_intensities(position_px, occupied_coords[i, :], self._map.data.T)[:-1]
            free_coords = np.vstack((free_coords, bresenham_line[:, :2]))

        for i in range(free_endcoords.shape[0]):
            bresenham_line = bresenham2d_with_intensities(position_px, free_endcoords[i, :], self._map.data.T)
            free_coords = np.vstack((free_coords, bresenham_line[:, :2]))

        free_coords = free_coords.astype(np.int)

        occupied_coords = occupied_coords[which_coords_in_bounds(occupied_coords, self._map.get_shape())]
        free_coords = free_coords[which_coords_in_bounds(free_coords, self._map.get_shape())]

        if debug:
            map_vis = np.repeat([self._map.copy()], repeats=3, axis=0).transpose((1, 2, 0))

            map_vis[occupied_coords[:, 0], occupied_coords[:, 1]] = [10, 10, 127]
            map_vis[free_coords[:, 0], free_coords[:, 1]] = [127, 10, 127]
            map_vis[int(state[0]), int(state[1])] = [127, 122, 10]

            plt.imshow(map_vis, interpolation='nearest')
            plt.show()

        return occupied_coords.astype(np.int), free_coords.astype(np.int)

    def update(self, state, scan_angles, scan_ranges):
        """
        update the current map with the current laser scan using a log odds approach
        :param state array(3)[float]: current state of the robot
        :param scan_angles array(N)[float]: lidar angles of the robot
        :param scan_ranges array(N)[float]: lidar ranges corresponding to the angles
        :return Costmap: an occupancy map showing free, occupied and unexplored space
        """
        map_state = state.copy()
        measured_occupied, measured_free = self._scan_to_occupied_free_coords(map_state, scan_angles, scan_ranges)

        # calculate log likelihood factors for adjusting the log odds
        g_occupied = self.measurement_certainty / (1. - self.measurement_certainty)
        g_free = 1. / g_occupied

        try:
            # update the log odds for the coordinates measured
            if measured_occupied.shape[0]:
                self._log_odds_map[measured_occupied[:, 0], measured_occupied[:, 1]] += np.log(g_occupied)
                over_inds = np.argwhere(self._log_odds_map > self.max_log_odd)
                self._log_odds_map[over_inds[:, 0], over_inds[:, 1]] = self.max_log_odd

            if measured_free.shape[0]:
                self._log_odds_map[measured_free[:, 0], measured_free[:, 1]] += np.log(g_free)
                under_inds = np.argwhere(self._log_odds_map < self.min_log_odd)
                self._log_odds_map[under_inds[:, 0], under_inds[:, 1]] = self.min_log_odd

            # compute the probability map from the log odds map
            self._probability_map = 1 - (1 / (1 + np.exp(self._log_odds_map)))

            # threshold the map to get the occupancy grid
            occupied_coords = np.argwhere(self._probability_map > self.threshold_filled).astype(np.int)
            free_coords = np.argwhere(self._probability_map < self.threshold_empty).astype(np.int)

            self._map.data[occupied_coords[:, 0], occupied_coords[:, 1]] = np.zeros((occupied_coords.shape[0],),
                                                                                    dtype=np.uint8)
            self._map.data[free_coords[:, 0], free_coords[:, 1]] = 255 * np.ones((free_coords.shape[0],),
                                                                                 dtype=np.uint8)

            return self._map

        except IndexError:
            raise IndexError("Map size is too small to fit your measurements! Increase allocated map size!!")

    def get_log_odds_map(self):
        """
        Returns the log odds map
        :return Costmap: the log odds aggregate map which is converted to the probability map
        """
        return self._log_odds_map

    def get_probability_map(self):
        """
        Returns the probability map
        :return Costmap: probability map which is occupancy map before thresholding
        """
        return self._probability_map

    def get_map(self):
        """
        Returns the thresholded map (occupany map)
        :return Costmap: occupancy map
        """
        return self._map

    def get_origin(self):
        """
        Returns the origin of the map
        :return array(2)[float]: origin in meters of the map
        """
        return self._origin
