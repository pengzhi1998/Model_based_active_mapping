"""util.py
Utility functions for exploration
"""
from __future__ import print_function, absolute_import, division

import os
import cv2
import numpy as np

from bc_exploration.utilities.paths import get_maps_dir


def load_occupancy_map_data(group, filename):
    """
    Reads in an occupancy map image from the maps folder, given the group name, and the image name.
    :param group str: the subfolder within maps to browse
    :param filename str: the image filename within the subfolder group.
    :return array(N,M)[uint8]: occupancy map data corresponded to the loaded image file
    """
    return cv2.cvtColor(cv2.imread(os.path.join(get_maps_dir(), group, filename)), cv2.COLOR_BGR2GRAY).astype(np.uint8)


def get_rotation_matrix_2d(angle):
    """
    Returns a 2D rotation matrix numpy array corresponding to angle
    :param angle float: angle in radians
    :return array(2, 2)[float]: 2D rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def which_coords_in_bounds(coords, map_shape):
    """
    Checks the coordinates given to see if they are in bounds
    :param coords Union[array(2)[int], array(N,2)[int]]: [int, int] or [[int, int], ...], Nx2 ndarray
    :param map_shape Tuple[int]: shape of the map to check bounds
    :return Union[bool array(N)[bool]]: corresponding to whether the coord is in bounds (if array is given, then it will be
             array of bool)
    """
    assert isinstance(coords, np.ndarray) and coords.dtype == np.int
    assert np.array(map_shape).dtype == np.int
    if len(coords.shape) == 1:
        return coords[0] >= 0 and coords[0] < map_shape[0] and coords[1] >= 0 and coords[1] < map_shape[1]
    else:
        return np.logical_and(np.logical_and(coords[:, 0] >= 0, coords[:, 0] < map_shape[0]),
                              np.logical_and(coords[:, 1] >= 0, coords[:, 1] < map_shape[1]))


def wrap_angles(angles, is_radians=True):
    """
    Wraps angles between -180 and 180 or -pi and pi based off is_radians
    :param angles array(N)[float]: ndarray containing the angles
    :param is_radians bool: whether to wrap in degrees or radians
    :return array(N)[float]: same shape ndarray where angles are wrapped
    """
    if is_radians:
        wrapped_angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi
    else:
        wrapped_angles = np.mod(angles + 180, 2 * 180) - 180
    return wrapped_angles


def clip_range(min_range, max_range, map_shape):
    """
    clips range to stay in map_shape
    :param min_range array(2)[int]: [int, int] min row col range
    :param max_range array(2)[int]: [int, int] max row col range
    :param map_shape array(2)[int]: (int, int) map shape
    :return Tuple[array(2), array(2)]: the min range and max range, clipped.
    """
    clipped_min_range = [max(min_range[0], 0), max(min_range[1], 0)]
    clipped_max_range = [min(max_range[0], map_shape[0]), min(max_range[1], map_shape[1])]
    return clipped_min_range, clipped_max_range


def numpy_diff2d(set1, set2):
    """
    take the set difference between to coordinate arrays
    :param set1 array(N, 2)[int]: ndarray of coordinates (2d)
    :param set2 array(N, 2)[int]: ndarray of coordinates (2d)
    :return array(N, 2)[int]: coordinates that appear in both sets
    """
    set2 = set2.astype(set1.dtype)
    dtype = {'names': ['f{}'.format(i) for i in range(set1.shape[1])],
             'formats': set1.shape[1] * [set1.dtype]}

    common_coords = np.intersect1d(set1.view(dtype), set2.view(dtype))
    common_coords = common_coords.view(set1.dtype).reshape(-1, set1.shape[1])
    return common_coords


def compute_circumscribed_radius(points):
    """
    Compute the circumscribed radius for the given points
    :param points array(N, 2)[float]: Nx2 points to compute the radius
    :return float: circumscribed radius
    """
    distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    return np.max(distances)


def compute_connected_pixels(start_state, image, flood_value=77, debug=False):
    """
    Floodfill on map from starting location, find connected pixels
    :param start_state array(2)[int]: starting state for flood fill IN PIXELS (for now)
    :param image array(N, M)[uint8]: 2d ndarray, image of which to compute on
    :param flood_value uint8: value to set the flood values to for np.argwhere (should be an unused uint8 value)
    :param debug bool: show plots?
    :return array(N, 2)[int]: ndarray, indicies of the flood
    """
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)

    truth_floodfill = image.copy()

    assert flood_value not in np.unique(image).tolist()
    cv2.floodFill(truth_floodfill, mask, tuple(start_state[:2][::-1].astype(np.int)), flood_value)

    if debug:
        cv2.imshow("", truth_floodfill)
        cv2.waitKey()

    flood_inds = np.argwhere(truth_floodfill == flood_value)

    return flood_inds


def scan_to_points(angles, ranges):
    """
    Convert laserscan to points
    :param angles array(N)[float]: angles corresponding to the ranges
    :param ranges array(N)[float]: ranges corresponding to the angles
    :return array(N,2)[float]: ndarray, points
    """
    points = np.nan * np.ones((angles.shape[0], 2))
    good_inds = np.logical_not(np.logical_or(np.isinf(ranges), np.isnan(ranges)))
    points[good_inds, :] = np.expand_dims(ranges[good_inds], axis=1) * \
        np.array([np.cos(angles[good_inds]), np.sin(angles[good_inds])]).T
    return points


def xy_to_rc(pose, occupancy_map):
    """
    Convert x, y points to row, column coordinates
    :param pose Union[array(2)[float], array(3)[float], array(N,2)[float], array(N,3)[float]]: [x, y, theta pose]
    :param occupancy_map Costmap: current map
    :return Union[array(2)[float], array(3)[float], array(N,2)[float], array(N,3)[float]]: [r, c, theta pose]
    """
    new_pose = np.array(pose, dtype=np.float)
    if len(new_pose.shape) == 1:
        new_pose[:2] -= occupancy_map.origin
        new_pose[1] = (occupancy_map.get_size()[1] - occupancy_map.resolution) - new_pose[1]
        new_pose[[0, 1]] = new_pose[[1, 0]]
        new_pose[:2] = np.rint(new_pose[:2] / occupancy_map.resolution)
    else:
        new_pose[:, :2] -= occupancy_map.origin
        new_pose[:, 1] = (occupancy_map.get_size()[1] - occupancy_map.resolution) - new_pose[:, 1]
        new_pose[:, [0, 1]] = new_pose[:, [1, 0]]
        new_pose[:, :2] = np.rint(new_pose[:, :2] / occupancy_map.resolution)
    return new_pose


def rc_to_xy(pose, occupancy_map):
    """
    Convert row, column coordinates to x, y points
    :param pose Union[array(2)[float], array(3)[float], array(N,2)[float], array(N,3)[float]]: x,y pose
    :param occupancy_map Costmap: current map
    :return Union[array(2)[float], array(3)[float], array(N,2)[float], array(N,3)[float]]: x, y pose
    """
    new_pose = np.array(pose, dtype=np.float)
    if len(new_pose.shape) == 1:
        new_pose[0] = (occupancy_map.get_shape()[0] - 1) - new_pose[0]
        new_pose[[0, 1]] = new_pose[[1, 0]]
        new_pose[:2] *= occupancy_map.resolution
        new_pose[:2] += occupancy_map.origin
    else:
        new_pose[:, 0] = (occupancy_map.get_shape()[0] - 1) - new_pose[:, 0]
        new_pose[:, [0, 1]] = new_pose[:, [1, 0]]
        new_pose[:, :2] *= occupancy_map.resolution
        new_pose[:, :2] += occupancy_map.origin

    return new_pose


def round_to_increment(x, increment):
    """
    rounds number to increment
    :param x float: number to round
    :param increment float: increment to round to
    :return float: rounded number
    """
    precision = len(str(increment).split('.')[1]) if '.' in str(increment) else 0
    answer = np.round(increment * np.round(float(x) / increment), precision)
    return answer


def round_state_to_increment(state, increment):
    """
    Extension of round to increment to the state vector
    :param state array(3)[float]: [x, y, theta]
    :param increment float: increment to round state to (just the position)
    :return array(3)[float]: rounded state
    """
    new_state = np.array(state)
    new_state[0] = round_to_increment(state[0], increment)
    new_state[1] = round_to_increment(state[1], increment)
    if isinstance(state, np.ndarray):
        return new_state
    else:
        return new_state.tolist()
