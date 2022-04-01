import numpy as np
from matplotlib import pyplot as plt

from bc_exploration.sensors.sensors import Lidar
from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import which_coords_in_bounds, xy_to_rc, rc_to_xy, scan_to_points, \
    get_rotation_matrix_2d


class SemanticLidar(Lidar):
    def __init__(self, sensor_range, angular_range, angular_resolution, map_resolution, num_classes=None,
                 range_std=None, true_class_prob=None, aerial_view=True):
        Lidar.__init__(self, sensor_range, angular_range, angular_resolution, map_resolution)
        self.aerial_view = aerial_view
        self._noisy_obs = False
        if range_std is not None and true_class_prob is not None:
            self._noisy_obs = True
            self._range_std = range_std
            self._true_class_prob = true_class_prob
            self._num_classes = num_classes

    def _measure_obstructed(self, state, debug=False):
        assert self._map is not None \
            and "Please set the map using set_map() before calling measure, this will initialize lidar as well."

        pose = state.copy()
        pose_px = xy_to_rc(pose, self._map)

        ranges = np.zeros_like(self._ray_angles)
        categories = np.zeros_like(self._ray_angles, dtype=np.uint8)
        for i, ego_ray in enumerate(self._ego_rays):
            rotation_matrix = get_rotation_matrix_2d(-state[2])
            rotated_ray = np.rint(ego_ray.dot(rotation_matrix))
            ray = (pose_px[:2] + rotated_ray).astype(np.int)
            ray = ray[which_coords_in_bounds(ray, self._map.get_shape())]
            occupied_ind = np.argwhere(self._map.data[ray[:, 0], ray[:, 1]] != Costmap.FREE)
            if occupied_ind.shape[0]:
                ranges[i] = np.linalg.norm(pose_px[:2] - ray[int(occupied_ind[0])]) * self._map.resolution
                categories[i] = self._map.data[ray[int(occupied_ind[0]), 0], ray[int(occupied_ind[0]), 1]]
            else:
                ranges[i] = np.nan
                categories[i] = 0

        if self._noisy_obs is True:
            ranges, categories = self._add_noise(ranges, categories)

        if debug:
            points = scan_to_points(self._ray_angles + state[2],
                                    ranges) + pose_px[:2]
            plt.plot(points[:, 0], points[:, 1])
            plt.show()

        return self._ray_angles, ranges, categories

    def _measure_aerial(self, state, debug=False):
        assert self._map is not None \
            and "Please set the map using set_map() before calling measure, this will initialize lidar as well."

        pose = state.copy()
        pose_px = xy_to_rc(pose, self._map).astype(np.int)

        obs = np.empty((1,3), dtype=np.int)
        for i, ego_ray in enumerate(self._ego_rays):
            rotation_matrix = get_rotation_matrix_2d(-state[2])
            rotated_ray = np.rint(ego_ray.dot(rotation_matrix))
            ray = (pose_px[:2] + rotated_ray).astype(np.int)
            ray = ray[which_coords_in_bounds(ray, self._map.get_shape())]

            ray_obs = np.hstack((ray, (self._map.data[ray[:, 0], ray[:, 1]])[:,None]))
            obs = np.vstack((obs, ray_obs))

        if debug:
            points = scan_to_points(self._ray_angles + state[2],
                                    self._range) + pose_px[:2]
            plt.plot(points[:, 0], points[:, 1])
            plt.show()

        obs[:, :2] -= pose_px[:2]

        return obs[1:, :]

    def _add_noise(self, ranges, categories):
        noisy_range = np.random.normal(0, self._range_std, ranges.shape) + ranges
        class_error = np.random.sample(size=categories.shape) > self._true_class_prob
        noisy_categories = (categories + np.random.randint(1, self._num_classes + 1, categories.shape) *
                            class_error) % (self._num_classes + 1)
        np.place(noisy_categories, categories == 0, 0)

        return noisy_range, noisy_categories

    def measure(self, state, debug=False):
        if not self.aerial_view:
            return self._measure_obstructed(state=state, debug=debug)
        else:
            return self._measure_aerial(state=state, debug=debug)
