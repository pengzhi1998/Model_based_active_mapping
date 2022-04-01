from __future__ import print_function, absolute_import, division

import numpy as np
from bc_exploration.mapping.costmap import Costmap


def test_creation():
    costmap = Costmap(data=np.zeros((100, 100), dtype=np.uint8), resolution=1.0, origin=np.array([-50., -50.]))
    assert np.all(costmap.data == np.zeros((100, 100), dtype=np.uint8))
    assert costmap.resolution == 1.0
    assert np.all(costmap.origin == np.array([-50., -50.]))


def test_get_shape_size():
    costmap = Costmap(data=np.zeros((100, 200), dtype=np.uint8), resolution=0.05, origin=np.array([-50., -50.]))
    assert costmap.get_shape() == (100, 200)
    assert costmap.get_size() == (200 * 0.05, 100 * 0.05)


def test_get_downscaled():
    # todo grab a real case to test obstacle preservation
    costmap = Costmap(data=np.zeros((100, 200), dtype=np.uint8), resolution=0.05, origin=np.array([-50., -50.]))
    downscaled_costmap = costmap.get_downscaled(0.1)
    assert downscaled_costmap.get_shape() == (50, 100)
    assert downscaled_costmap.resolution == 0.1
    assert np.all(downscaled_costmap.origin == [-50., -50.])


def main():
    test_creation()
    test_get_shape_size()
    test_get_downscaled()
    # test_to_brain_costmap()
    # test_from_brain_costmap()


if __name__ == '__main__':
    main()
