import os
import sys
import yaml
import math
import numpy as np
from numpy.linalg import slogdet

import gym
from gym import spaces

import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import rc_to_xy
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint

from rl_mapping.utilities.utils import state_to_T, T_to_state, SE2_motion, triangle_SDF, Gaussian_CDF, circle_SDF
from rl_mapping.sensors.semantic_sensors import SemanticLidar
from rl_mapping.envs.semantic_grid_world import SemanticGridWorld
from rl_mapping.mapping.mapper_kf import KFMapper

CONTROL = 2.
DOWNSAMPLE = 25
SENSOR_RANGE = 1.5
STD = 0.01

class VolumetricQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    def __init__(self, distrib_map: Costmap, params_filename: str):
        super(VolumetricQuadrotor, self).__init__()

        # agent position + diagonal entries of the information matrix (i.e. the information vector)
        # [x, y, info_vec_0, info_vec_1, ...]
        h, w, _ = distrib_map.get_shape()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + h * w, ), dtype=np.float32)

        # linearly controlling the x, y velocities, yaw angle is fixed as 0
        # (x, y): {-CONTROL, CONTROL}^2
        # scaled in the step function
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32) 

        # parameters
        params = self.__load_params(params_filename)
        self.distrib_map = distrib_map.copy()
        self.kappa = params['kappa']
        self.total_time = params['horizon']
        self.dt = params['dt']
        self.total_step = np.floor(self.total_time / self.dt)

        # init
        self.info_vec = self.distrib_map.data[:, :, 1].copy() # (h, w)
        self.agent_pos = np.zeros(3, dtype=np.float32) # (3, )
        self.last_r = np.sum(np.log(self.info_vec[::DOWNSAMPLE, ::DOWNSAMPLE]))
        self.current_step = -1

    def __load_params(self, params_filename: str):
        with open(os.path.join(params_filename)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return params

    def step(self, action):
        self.current_step += 1

        # rescale actions
        action *= CONTROL
        control = np.hstack([
            action, 
            0
        ]).astype(np.float32)

        # apply dynamics and update agent's pos
        T_old = state_to_T(self.agent_pos)
        T_new = SE2_motion(T_old, control, self.dt)
        self.agent_pos = T_to_state(T_new)

        # update information
        for i in range(0, self.info_vec.shape[0], DOWNSAMPLE):
            for j in range(0, self.info_vec.shape[1], DOWNSAMPLE):
                p_ij = rc_to_xy(np.array([i, j]), self.distrib_map)
                q = T_old[:2, :2].transpose() @ (p_ij - T_old[:2, 2])
                d, _ = circle_SDF(q, SENSOR_RANGE)
                Phi, _ = Gaussian_CDF(d, self.kappa)
                self.info_vec[i, j] += 1 / (STD**2) * (1 - Phi)

        # calculate reward
        cur_r = np.sum(np.log(self.info_vec[::DOWNSAMPLE, ::DOWNSAMPLE]))
        r = cur_r - self.last_r
        self.last_r = cur_r

        # obs
        obs = np.hstack([
            self.agent_pos[:2],
            self.info_vec.flatten()
        ]).astype(np.float32)

        done = False
        if self.current_step >= self.total_step-1:
            done = True

        # info
        info = {}

        return obs, r, done, info

    def reset(self):

        # init
        self.info_vec = self.distrib_map.data[:, :, 1].copy() # (h, w)
        self.agent_pos = np.zeros(3, dtype=np.float32) # (3, )
        self.last_r = np.sum(np.log(self.info_vec[::DOWNSAMPLE, ::DOWNSAMPLE]))
        self.current_step = -1

        # construct observation
        obs = np.hstack([
            self.agent_pos[:2],
            self.info_vec.flatten()
        ]).astype(np.float32)

        return obs

    def render(self, mode='human'):

        raise NotImplementedError

    def close (self):
        pass

if __name__ == '__main__':
    params_filename = '../params/params.yaml'
    map_filename = '../maps/map10_converted.png'
    

    ### create map for env ###
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if params['footprint']['type'] == 'tricky_circle':
        footprint_points = get_tricky_circular_footprint()
    elif params['footprint']['type'] == 'tricky_oval':
        footprint_points = get_tricky_oval_footprint()
    elif params['footprint']['type'] == 'circle':
        rotation_angles = np.arange(0, 2 * np.pi, 4 * np.pi / 180)
        footprint_points = \
            params['footprint']['radius'] * np.array([np.cos(rotation_angles), np.sin(rotation_angles)]).T
    elif params['footprint']['type'] == 'pixel':
        footprint_points = np.array([[0., 0.]])
    else:
        footprint_points = None
        assert False and "footprint type specified not supported."

    footprint = CustomFootprint(footprint_points=footprint_points,
                            angular_resolution=params['footprint']['angular_resolution'],
                            inflation_scale=params['footprint']['inflation_scale'])
    sensor = SemanticLidar(sensor_range=SENSOR_RANGE,
                           angular_range=np.pi*2,
                           angular_resolution=0.5 * np.pi / 180,
                           map_resolution=0.03,
                           num_classes=1,
                           aerial_view=True)
    env = SemanticGridWorld(map_filename=map_filename,
                            map_resolution=0.03,
                            sensor=sensor,
                            num_class=1,
                            footprint=footprint,
                            start_state=None,
                            no_collision=True)
    padding = 0.
    map_shape = np.array(env.get_map_shape()) + int(2. * padding // 0.03)
    initial_map = Costmap(data=Costmap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8),
                          resolution=env.get_map_resolution(),
                          origin=[-padding - env.start_state[0], -padding - env.start_state[1]])

    mapper = KFMapper(initial_map=initial_map, sigma2=STD**2)


    ### create env & test random actions ###
    env = VolumetricQuadrotor(mapper.get_distrib_map(), params_filename)
    check_env(env)

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()

        obs, r, done, info = env.step(action)

        total_reward += r

        print("reward: ", r)

    print("return:", total_reward)

    env.close()

