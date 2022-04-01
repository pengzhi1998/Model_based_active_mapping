from __future__ import print_function, absolute_import, division

import os

import numpy as np
from bc_exploration.envs.grid_world import GridWorld
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.sensors.sensors import Neighborhood
from bc_exploration.utilities.paths import get_maps_dir


def test_creation():
    test_map_filename = os.path.join(get_maps_dir(), "test/circle.png")
    footprint = CustomFootprint(np.array([[0., 0.]]), np.pi / 2)
    env = GridWorld(map_filename=test_map_filename, footprint=footprint,
                    sensor=Neighborhood(sensor_range=1, values=[0, 255]),
                    start_state=[1, 3, 0], map_resolution=1, thicken_obstacles=False)
    assert np.all(env.state == [0, 0, 0])
    assert np.all(env.map.data == [[0, 0, 0, 0, 0],
                                   [0, 255, 255, 255, 0],
                                   [0, 255, 0, 255, 0],
                                   [0, 255, 255, 255, 0],
                                   [0, 0, 0, 0, 0]])


def test_reset():
    """
    Tests the reset function on an environment
    ---maps/test/circle.png
    0  0   0   0  0
    0 255 255 255 0
    0 255  0  255 0
    0 255 255 255 0
    0  0   0   0  0
    ---end file
    """
    test_map_filename = os.path.join(get_maps_dir(), "test/circle.png")
    footprint = CustomFootprint(np.array([[0., 0.]]), np.pi / 2)
    env = GridWorld(map_filename=test_map_filename, footprint=footprint,
                    sensor=Neighborhood(sensor_range=1, values=[0, 255]),
                    start_state=[1, 3, 0], map_resolution=1, thicken_obstacles=False)
    env.state = np.array([70, 70, 0])
    env.reset()
    assert np.all(env.state == [0, 0, 0])


def test_step_movement():
    """
    Tests the step function on an environment, checking out of bounds, and collision
    ---maps/test/circle.png
    0  0   0   0  0
    0 255 255 255 0
    0 255  0  255 0
    0 255 255 255 0
    0  0   0   0  0
    ---end file
    """
    test_map_filename = os.path.join(get_maps_dir(), "test/circle.png")
    footprint = CustomFootprint(np.array([[0., 0.]]), np.pi / 2)
    env = GridWorld(map_filename=test_map_filename, footprint=footprint,
                    sensor=Neighborhood(sensor_range=1, values=[0, 255]),
                    start_state=[1, 3, 0], map_resolution=1, thicken_obstacles=False)

    # todo test is broken (not out of bounds, all obstacles) we arent testing out of bounds
    # test up out of bounds
    new_state, _ = env.step([0, 1, 0])
    assert np.all(new_state == [0, 0, 0])

    # test left out of bounds
    new_state, _ = env.step([-1, 0, 0])
    assert np.all(new_state == [0, 0, 0])

    # test down movement
    new_state, _ = env.step([0, -1, 0])
    assert np.all(new_state == [0, -1, 0])

    # test right obstacle
    new_state, _ = env.step([1, -1, 0])
    assert np.all(new_state == [0, -1, 0])

    new_state, _ = env.step([0, -2, 0])
    assert np.all(new_state == [0, -2, 0])

    # test down out of bounds
    new_state, _ = env.step([0, -3, 0])
    assert np.all(new_state == [0, -2, 0])

    # test right movement
    new_state, _ = env.step([1, -2, 0])
    assert np.all(new_state == [1, -2, 0])

    # test up obstacle
    new_state, _ = env.step([1, -1, 0])
    assert np.all(new_state == [1, -2, 0])

    new_state, _ = env.step([2, -2, 0])
    assert np.all(new_state == [2, -2, 0])

    # test right out of bounds
    new_state, _ = env.step([2, -2, 0])
    assert np.all(new_state == [2, -2, 0])

    # test up movement
    new_state, _ = env.step([2, -1, 0])
    assert np.all(new_state == [2, -1, 0])

    # test left obstacle
    new_state, _ = env.step([1, -1, 0])
    assert np.all(new_state == [2, -1, 0])

    new_state, _ = env.step([2, 0, 0])
    assert np.all(new_state == [2, 0, 0])

    # test left movement
    new_state, _ = env.step([1, 0, 0])
    assert np.all(new_state == [1, 0, 0])

    # test down obstacle
    new_state, _ = env.step([1, -1, 0])
    assert np.all(new_state == [1, 0, 0])

    new_state, _ = env.step([0, 0, 0])
    assert np.all(new_state == [0, 0, 0])


def test_step_data():
    """
    Tests the step function on an environment, checking out of bounds, and collision
    ---maps/test/circle.png
    0  0   0   0  0
    0 255 255 255 0
    0 255  0  255 0
    0 255 255 255 0
    0  0   0   0  0
    ---end file
    """
    test_map_filename = os.path.join(get_maps_dir(), "test/circle.png")
    footprint = CustomFootprint(np.array([[0., 0.]]), np.pi / 2)
    env = GridWorld(map_filename=test_map_filename, footprint=footprint,
                    sensor=Neighborhood(sensor_range=1, values=[0, 255]),
                    start_state=[1, 3, 0], map_resolution=1, thicken_obstacles=False)

    _, data = env.reset()

    # check occupied coords
    assert np.all(data[0] == [[-1, -1],
                              [-1, 0],
                              [-1, 1],
                              [0, -1],
                              [1, -1],
                              [1, 1]])

    # check free coords
    assert np.all(data[1] == [[0, 0],
                              [0, 1],
                              [1, 0]])

    # todo in reality this sensor_range change test may belong in sensors.py
    footprint = CustomFootprint(np.array([[0., 0.]]), np.pi / 2)
    env = GridWorld(map_filename=test_map_filename, footprint=footprint,
                    sensor=Neighborhood(sensor_range=2, values=[0, 255]),
                    start_state=[1, 3, 0], map_resolution=1, thicken_obstacles=False)

    _, data = env.step([0, 1, 0])

    # check occupied coords
    assert np.all(data[0] == [[-1, -1],
                              [-1, 0],
                              [-1, 1],
                              [-1, 2],
                              [0, -1],
                              [1, -1],
                              [1, 1],
                              [2, -1]])

    # check free coords
    assert np.all(data[1] == [[0, 0],
                              [0, 1],
                              [0, 2],
                              [1, 0],
                              [1, 2],
                              [2, 0],
                              [2, 1],
                              [2, 2]])


def test_compare_maps():
    """
    Tests the compare maps function
    ---maps/test/test.png
    0  0   0   0  0
    0 255 255 255 0
    0 255  0  255 0
    0 255 255 255 0
    0  0   0   0  0
    ---end file
    """
    test_map_filename = os.path.join(get_maps_dir(), "test/circle.png")
    footprint = CustomFootprint(np.array([[0., 0.]]), np.pi / 2)
    env = GridWorld(map_filename=test_map_filename, footprint=footprint,
                    sensor=Neighborhood(sensor_range=1, values=[0, 255]),
                    start_state=[1, 3, 0], map_resolution=1, thicken_obstacles=False)

    map_data = np.array([[0, 0, 0, 0, 0],
                         [0, 255, 255, 0, 0],
                         [0, 255, 0, 255, 0],
                         [0, 0, 255, 0, 0],
                         [0, 0, 0, 0, 0]]).astype(np.uint8)
    occupancy_map = Costmap(data=map_data, resolution=1, origin=[0, 0])

    percentage_completed = env.compare_maps(occupancy_map)
    assert percentage_completed == 0.625


def main():
    test_creation()
    test_reset()
    test_step_movement()
    test_step_data()
    test_compare_maps()

    run_example = False
    if run_example:
        test_map_filename = os.path.join(get_maps_dir(), "test/circle.png")
        footprint = CustomFootprint(np.array([[0., 0.]]), np.pi / 2)
        env = GridWorld(map_filename=test_map_filename, footprint=footprint,
                        sensor=Neighborhood(sensor_range=1, values=[0, 255]),
                        start_state=[1, 3, 0], map_resolution=1, thicken_obstacles=False)
        _, _ = env.reset()

        path = np.array([[0, -1, 0],
                         [0, -2, 0],
                         [1, -2, 0],
                         [2, -2, 0],
                         [2, -1, 0],
                         [2, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])

        for point in path:
            _, _ = env.step(point)
            env.render(wait_key=0)


if __name__ == '__main__':
    main()
