import numpy as np

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import xy_to_rc


class KFMapper:
    def __init__(self, initial_map, sigma2):

        self._map = initial_map.copy()

        self._sigma2 = sigma2

        mu_0 = 0
        s_0 = 0.0001

        map_shape = initial_map.get_shape()
        self._distrib_map = Costmap(data=np.array([mu_0, s_0]) * np.ones((map_shape[0], map_shape[1], 2)),
                                    resolution=initial_map.resolution,
                                    origin=initial_map.origin)

    def update(self, state, obs):
        print("obs:", obs, np.shape(obs), np.unique(obs[:, 2]))
        pose_px = xy_to_rc(state, self._map).astype(np.int)
        obs[:, :2] += pose_px[:2]
        free_coords = obs[np.nonzero(obs[:, 2] == Costmap.FREE)[0], :2]
        occ_rows = np.nonzero(obs[:, 2] != Costmap.FREE)[0]
        occ_coords = obs[occ_rows, :2]

        self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 0] = \
            self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 0] - \
            (1 + self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 0]) /\
            (1 + self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 1] * self._sigma2)

        self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 1] = \
            self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 1] + 1 / self._sigma2

        self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 0] = \
            self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 0] + \
            (1 - self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 0]) /\
            (1 + self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 1] * self._sigma2)

        self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 1] = \
            self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 1] + 1 / self._sigma2

        np.place(self._map.data,
                 self._distrib_map.data[:, :, 0] < 0, Costmap.FREE)
        np.place(self._map.data,
                 self._distrib_map.data[:, :, 0] > 0, Costmap.OCCUPIED)

        # np.place(self._map.data[free_coords[:, 0], free_coords[:, 1]],
        #          self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 0] < 0, Costmap.FREE)
        #
        # np.place(self._map.data[free_coords[:, 0], free_coords[:, 1]],
        #          self._distrib_map.data[free_coords[:, 0], free_coords[:, 1], 0] > 0, Costmap.OCCUPIED)
        #
        # np.place(self._map.data[occ_coords[:, 0], occ_coords[:, 1]],
        #          self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 0] < 0, Costmap.FREE)
        #
        # np.place(self._map.data[occ_coords[:, 0], occ_coords[:, 1]],
        #          self._distrib_map.data[occ_coords[:, 0], occ_coords[:, 1], 0] > 0, Costmap.OCCUPIED)

        return self._map

    def get_distrib_map(self):
        return self._distrib_map

    def get_occupancy_map(self):
        return  self._map

    def get_info_reward(self):
        return np.sum(np.log(self._distrib_map.data[:, :, 1]))
