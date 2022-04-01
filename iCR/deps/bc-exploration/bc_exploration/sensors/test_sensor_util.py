from __future__ import print_function, absolute_import, division

import numpy as np

from bc_exploration.sensors.sensor_util import bresenham2d


def test_bresenham2d():
    sx = 0
    sy = 1
    r1 = bresenham2d([sx, sy], [10, 5])
    r1_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]]).T
    r2 = bresenham2d([sx, sy], [9, 6])
    r2_ex = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 2, 3, 3, 4, 4, 5, 5, 6]]).T
    assert np.logical_and(np.sum(r1 == r1_ex) == np.size(r1_ex), np.sum(r2 == r2_ex) == np.size(r2_ex))


def main():
    test_bresenham2d()


if __name__ == '__main__':
    main()
