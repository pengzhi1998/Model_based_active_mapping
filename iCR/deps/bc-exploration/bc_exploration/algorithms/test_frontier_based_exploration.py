from __future__ import print_function, absolute_import, division

import os
import numpy as np
from bc_exploration.algorithms.frontier_based_exploration import run_frontier_exploration
from bc_exploration.utilities.paths import get_maps_dir, get_exploration_dir


def test_frontier_based_exploration():
    np.random.seed(3)
    _, _, _, _ = \
        run_frontier_exploration(map_filename=os.path.join(get_maps_dir(), "test/vw_ground_truth_test.png"),
                                 params_filename=os.path.join(get_exploration_dir(), "params/params.yaml"),
                                 map_resolution=0.03,
                                 start_state=None,
                                 sensor_range=10.0,
                                 completion_percentage=0.97,
                                 max_exploration_iterations=2,
                                 render=False)


def main():
    test_frontier_based_exploration()


if __name__ == '__main__':
    main()
