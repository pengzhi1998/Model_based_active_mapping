"""visualization.py
Visualization functions for exploration
"""
from __future__ import print_function, absolute_import, division

import cv2
import numpy as np

from bc_exploration.utilities.util import which_coords_in_bounds, xy_to_rc, scan_to_points


def draw_footprint_path(footprint, path, visualization_map, footprint_color=None, path_color=None, footprint_thickness=1):
    """
    Draws the path specified on the map, overlaying footprints on each pose.
    :param footprint CustomFootprint: footprint to draw
    :param path array(N, 3)[float]: path to draw
    :param visualization_map Costmap: costmap to draw on
    :param footprint_color Union[int, array(3)[uint8]]: depending on the shape of visualization_map.data,
                            a valid value for the color to draw the footprints
    :param path_color Union[int, array(3)[uint8]]: depending on the shape of visualization_map.data,
                       a valid value for the color to draw the path
    :param footprint_thickness int: thickness of which to draw the footprint, a negative value will fill the footprint
    """
    path_px = xy_to_rc(path, visualization_map)
    if path_color is not None:
        visualization_map.data[path_px[:, 0].astype(np.int), path_px[:, 1].astype(np.int)] = path_color

    if footprint_color is not None:
        actual_footprint = footprint.no_inflation()
        outline_coords = np.array(actual_footprint.get_outline_coords(visualization_map.resolution))
        angles = footprint.get_mask_angles()
        angle_inds = np.argmin(np.abs(-path_px[:, 2:] - np.expand_dims(angles, axis=0)), axis=1)
        footprints_coords = outline_coords[angle_inds] + np.expand_dims(path_px[:, :2], axis=1)

        for footprint_coords in footprints_coords:
            cv2.drawContours(visualization_map.data, [footprint_coords[:, ::-1].astype(np.int)], 0, footprint_color, footprint_thickness)


def draw_frontiers(visualization_map, frontiers, color):
    """
    Draw the frontiers onto the map.
    :param visualization_map Costmap: costmap to draw on
    :param frontiers List[np.ndarray]: of frontiers, each frontier is a set of coordinates
    :param color Union[int, array(3)[uint8]]: depending on the shape of visualization_map.data,
                  a valid value for the color to draw the path
    """
    for frontier in frontiers:
        frontier_px = xy_to_rc(frontier, visualization_map).astype(np.int)
        frontier_px = frontier_px[which_coords_in_bounds(frontier_px, visualization_map.get_shape())]
        cv2.drawContours(visualization_map.data, [frontier_px[:, ::-1]], 0, color, 2)


def draw_scan_ranges(visualization_map, state, scan_angles, scan_ranges, color):
    """
    Draw the occupied points of a lidar scan onto the map
    :param visualization_map Costmap: costmap to draw on
    :param state array(3)[float]: pose of the robot [x, y, theta]
    :param scan_angles array(N)[float]: angles of the lidar scan
    :param scan_ranges array(N)[float]: ranges of the lidar scan
    :param color Union[int, array(3)[uint8]]: depending on the shape of visualization_map.data,
              a valid value for the color to draw the path
    """
    occupied_coords = scan_to_points(scan_angles + state[2], scan_ranges) + state[:2]
    occupied_coords = xy_to_rc(occupied_coords, visualization_map).astype(np.int)
    occupied_coords = occupied_coords[which_coords_in_bounds(occupied_coords, visualization_map.get_shape())]
    visualization_map.data[occupied_coords[:, 0], occupied_coords[:, 1]] = color


def make_visualization_map(occupancy_map):
    """
    Convert occupancy map to rgb image for visualization
    :param occupancy_map Costmap: costmap to convert
    :return Costmap: same as occupancy map but now .data is a rgb image instead of grayscale
    """
    visualization_map = occupancy_map.copy()
    visualization_map.data = np.dstack((occupancy_map.data, occupancy_map.data, occupancy_map.data))
    return visualization_map
