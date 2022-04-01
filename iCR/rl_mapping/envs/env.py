import math
import numpy as np
from numpy.linalg import slogdet

import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt

class VolumetricQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    def __init__(self):
        super(VolumetricQuadrotor, self).__init__()

        self.observation_space = None
        self.action_space = None

    def step(self, action):

        return obs, r, done, info

    def reset(self):

        return obs

    def render(self, mode='human'):

        raise NotImplementedError

    def close (self):
        pass

if __name__ == '__main__':
    env = VolumetricQuadrotor()
    check_env(env)