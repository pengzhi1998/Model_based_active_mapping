import sys
import os
import math
import gym
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import slogdet
from gym import spaces
from scipy import stats
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
from rl_mapping.utilities.utils import state_to_T, T_to_state, SE2_motion, triangle_SDF, Gaussian_CDF, exp_hat,\
                                       Grad_exp_hat
from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.utilities.util import rc_to_xy

with open(os.path.join(os.path.join(os.path.abspath(os.path.join("", os.pardir)), "params/params.yaml"))) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

class VolumetricQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    def __init__(self):
        super(VolumetricQuadrotor, self).__init__()

        self.observation_space = None
        self.action_space = None

        self._dt = int(np.floor(params['horizon']/params['dt']))
        self._kappa = params['kappa']

        """ values which could be set from the arguments """
        self._sigma2 = 0.01**2
        self._max_range = 1.5
        self._downsample_coeff = 25
        self._angular_range = 2 * np.pi

    def step(self, action):
        self.step_count += 1
        self.T_new[:, :] = SE2_motion(self.T_old[:, :], action, self._dt)
        for i in range(0, self.S.shape[0], self._downsample_coeff):
            for j in range(0, self.S.shape[1], self._downsample_coeff):
                p_ij = rc_to_xy(np.array([i, j]), self.distrib_map)
                q = self.T_old[:2, :2].transpose() @ (p_ij - self.T_old[:2, 2])
                d, _ = triangle_SDF(q, self._angular_range / 2, self._max_range)
                Phi, _ = Gaussian_CDF(d, self._kappa)
                self.S[i, j] += 1 / (self._sigma2) * (1 - Phi)

        self.T_old = self.T_new

        self.reward_sum = np.sum(np.log(self.S[::self._downsample_coeff, ::self._downsample_coeff]))
        if self.step_count == 0:
            reward = self.reward_sum
        else:
            reward = self.reward_sum - self.reward_sum_old
        self.reward_sum_old = self.reward_sum

        if self.step_count >= 50:
            done = True
        else:
            done = False

        return obs, reward, done, {}

    def reset(self):
        occ_map = (self.distrib_map.data[:, :, 0] == 0) * 255
        img = cv2.threshold(occ_map.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1]
        num_labels, labels = cv2.connectedComponents(img)
        biggest_region_label = stats.mode(labels[labels > 0], axis=None)[0]
        center_px = np.floor(np.mean(np.nonzero(labels == biggest_region_label), axis=1))

        center = rc_to_xy(center_px, self.distrib_map)
        self.init_state[2] = np.arctan2(center[1] - self.init_state[1], center[0] - state[0])

        self.T_old = state_to_T(self.init_state)
        self.T_new = np.zeros((3, 3))

        self.S = self.distrib_map.data[:, :, 1].copy()

        self.step_count = 0
        return obs

    def render(self, mode='human'):

        raise NotImplementedError

    def close (self):
        pass

    def initial_inputs(self, distrib_map, initial_state):
        self.distrib_map = distrib_map
        self.init_state = initial_state

if __name__ == '__main__':
    env = VolumetricQuadrotor()
    check_env(env)