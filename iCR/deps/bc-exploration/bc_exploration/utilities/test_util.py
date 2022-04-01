from __future__ import print_function, absolute_import, division
import cv2
import numpy as np

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.paths import get_maps_dir
from bc_exploration.utilities.util import round_to_increment, compute_connected_pixels


def test_round_to_increment():
    assert round_to_increment(1.14, 0.05) == 1.15
    assert round_to_increment(22., 1.00) == 22
    assert round_to_increment(1.11, 0.1) == 1.1


def test_compare_maps_with_floodfill():
    ground_truth = cv2.imread(get_maps_dir() + "/test/vw_ground_truth_test.png", cv2.COLOR_BGR2GRAY)
    current_map = cv2.imread(get_maps_dir() + "/test/vw_partly_explored_test.png", cv2.COLOR_BGR2GRAY)
    current_map = Costmap(current_map, 1, np.array([0., 0.]))

    start_state = np.array([200, 230, 0])
    num_free = compute_connected_pixels(start_state, ground_truth).shape[0]

    measured_num_free = np.argwhere(current_map.data == Costmap.FREE).shape[0]
    x = measured_num_free / float(num_free)
    assert np.round(x, 4) == 0.2790


def test_scan_to_points():
    pass


def main():
    test_round_to_increment()
    test_compare_maps_with_floodfill()


if __name__ == '__main__':
    main()
