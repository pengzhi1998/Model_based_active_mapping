from __future__ import print_function, absolute_import, division

import numpy as np
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.mapping.log_odds_mapper import LogOddsMapper


def test_update():
    initial_map = Costmap(127 * np.ones((11, 11), dtype=np.uint8), 1, np.array([-5., -5.]))
    sensor_range = 7.0
    mapper = LogOddsMapper(initial_map=initial_map,
                           sensor_range=sensor_range,
                           measurement_certainty=.8,
                           max_log_odd=50,
                           min_log_odd=-8,
                           threshold_occupied=.7,
                           threshold_free=.3)

    pose = np.array([0., 0., 0.])
    scan_angles = np.linspace(-np.pi, np.pi, 360)
    scan_ranges = sensor_range * np.ones_like(scan_angles)
    scan_ranges[scan_ranges.shape[0] // 2:] = sensor_range / 2.
    occupancy_map = mapper.update(state=pose, scan_angles=scan_angles, scan_ranges=scan_ranges)

    ground_truth = np.array([[127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127],
                             [127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127],
                             [127, 127, 127, 0, 0, 0, 0, 0, 127, 127, 127],
                             [127, 127, 0, 0, 255, 255, 255, 0, 0, 127, 127],
                             [127, 127, 0, 255, 255, 255, 255, 255, 0, 127, 127],
                             [255, 255, 127, 255, 255, 255, 255, 255, 127, 255, 255],
                             [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
                             [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]], dtype=np.uint8)

    assert np.all(occupancy_map.data == ground_truth)


def main():
    test_update()


if __name__ == '__main__':
    main()
