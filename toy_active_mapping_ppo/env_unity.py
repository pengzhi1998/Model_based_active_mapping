import torch
import uuid
import os
import gym
import math
import numpy as np
import random
import matplotlib.pyplot as plt

from typing import List
from numpy.linalg import slogdet
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from utils import unicycle_dyn, diff_FoV_land
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# env setting
STATE_DIM = 3
RADIUS = 2
STD = 0.5
KAPPA = 1.  # TODO: increase

# time & step
STEP_SIZE = 1



class PosChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass

    def assign_landmark_pos(self, data: List[float]) -> None:
        msg = OutgoingMessage()
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)

class landmark_based_mapping(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}
    def __init__(self, num_landmarks, horizon, landmarks, test=False):
        super(landmark_based_mapping, self).__init__()

        # variables
        self.num_landmarks = num_landmarks
        self.test = test
        self.total_time = horizon
        self.step_size = STEP_SIZE
        self.total_step = math.floor(self.total_time / STEP_SIZE)

        # action space
        # defined as {-1, 1} as suggested by stable_baslines3, rescaled to {-2, 2} later in step()
        self.action_space = spaces.Box(low=-1, high=1, shape=(STATE_DIM,),
                                       dtype=np.float32)  # (x, y, \theta): {-2, 2}^3

        # state space
        # agent state + diag of info mat
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM - 1 + self.num_landmarks * 4,),
                                            dtype=np.float32)  # (x, y, \theta, info_mat_0, info_mat_1, info_mat_2, info_mat_3): {-inf, inf}^7

        # landmark and info_mat init
        if self.test == False:
            self.landmarks = landmarks
        else:
            if num_landmarks == 5:
                self.landmarks = np.array(
                    [[0.43404942], [-2.21630615], [-0.75482409], [3.44776132], [-4.95281144], [-3.78430879],
                     [1.70749085], [3.25852755], [-3.6329341], [0.75093329]])
            elif num_landmarks == 15:
                self.landmarks = np.array(
                    [[0.86809884], [-4.4326123], [-1.50964819], [6.89552265], [-9.90562288], [-7.56861758],
                     [3.41498169], [6.5170551], [-7.26586821], [1.50186659], [7.82643909], [-5.81595756], [-6.29343561],
                     [-7.83246219],
                     [-5.60605015], [9.57247569], [6.23366298], [-6.56117975], [6.32449497], [-4.51852506],
                     [-1.36591633], [8.80059639],
                     [6.35298758], [-3.277761], [-6.49179093], [-2.54335907], [-9.88622985], [-4.95147293],
                     [5.91325017], [-9.69490058]])
            elif num_landmarks == 25:
                self.landmarks = np.array(
                    [[-9.82638023], [-0.88652246], [-10.30192964], [1.37910453], [-11.98112458], [-1.51372352],
                     [-9.31700366], [1.30341102], [-11.45317364], [0.30037332], [1.56528782], [8.83680849],
                     [-1.25868712], [8.43350756],
                     [-1.12121003], [11.91449514], [1.2467326], [8.68776405], [1.26489899], [9.09629499], [9.72681673],
                     [1.76011928],
                     [11.27059752], [-0.6555522], [8.70164181], [-0.50867181], [8.02275403], [-0.99029459],
                     [11.18265003], [-1.93898012],
                     [0.39537351], [-9.58478184], [-1.57940926], [-10.47222622], [-1.85409577], [-8.43835375],
                     [1.92368343], [-11.76023204],
                     [1.56218378], [-9.692394], [0.96991876], [0.52073575], [0.32736877], [-1.91824347], [-1.15989369],
                     [0.17873951],
                     [1.07646068], [-0.99721908], [-0.85641724], [1.40958035]])
        self.info_mat_init = np.diag([.5] * self.num_landmarks * 2).astype(np.float32)
        print("landmarks' positions:", [self.landmarks])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_info = PosChannel()
        config_channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(os.path.abspath("./") + "/Unity_envs/Landmark",
                                     side_channels=[config_channel, self.pos_info])


        config_channel.set_configuration_parameters(time_scale=10, capture_frame_rate=100)
        self.env_unity = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

    def step(self, action):
        self.current_step += 1

        # rescale actions
        action *= 3

        # record history action
        self.history_actions.append(action.copy())

        # enforece the 3rd dim to zero
        action[-1] = 0.

        # dynamics
        next_agent_pos = unicycle_dyn(self.agent_pos, action, self.step_size).astype(np.float32)

        obs_unity, _, _, _ = self.env_unity.step([next_agent_pos[0], next_agent_pos[1]])  # reward, termination, and other info aren't needed

        # update the estimated landmarks' positions
        for i in range(self.num_landmarks):
            if np.linalg.norm(next_agent_pos[0:2] - self.landmarks[i*2: i*2+2].flatten()) < RADIUS:
                sensor_value = self.landmarks[i*2: i*2+2].flatten() + np.random.normal(0, STD, [2,])
                info_sensor = np.array([[self.info_mat[i*2,i*2], 0], [0, self.info_mat[i*2+1,i*2+1]]])
                kalman_gain = np.linalg.inv(np.identity(2) + STD ** 2 * info_sensor)
                landmarks_estimate = self.landmarks_estimate[i*2: i*2+2].flatten() + \
                                              kalman_gain @ (sensor_value - self.landmarks_estimate[i*2: i*2+2].flatten())  # no dynamics for the landmarks
                self.landmarks_estimate[i*2] = landmarks_estimate[0]
                self.landmarks_estimate[i*2+1] = landmarks_estimate[1]

        # reward
        V_jj_inv = diff_FoV_land(next_agent_pos, self.landmarks_estimate, self.num_landmarks, RADIUS, KAPPA, STD).astype(np.float32) # TODO replace self.landmarks with an estimated one
        next_info_mat = self.info_mat + V_jj_inv # update info
        reward = float(slogdet(next_info_mat)[1] - slogdet(self.info_mat)[1])

        # terminate at time
        done = False
        if self.current_step >= self.total_step-1:
            done = True

        # info
        info = {'info_mat': next_info_mat}

        # update variables
        self.agent_pos = next_agent_pos
        self.info_mat = next_info_mat

        # update state
        self.state = np.hstack([
            self.agent_pos[:2],
            self.info_mat.diagonal(),
            self.landmarks_estimate.flatten()
        ]).astype(np.float32)

        # record history poses
        self.history_poses.append(self.agent_pos)
        # print(np.sum(np.abs(self.landmarks - self.landmarks_estimate)))

        return self.state, reward, done, info

    def reset(self):
        # landmark and info_mat init
        self.env_unity.reset()

        self.info_mat = self.info_mat_init
        lx = np.random.uniform(low=-10, high=10, size=(self.num_landmarks, 1))
        ly = np.random.uniform(low=-10, high=10, size=(self.num_landmarks, 1))
        self.landmarks = np.concatenate((lx, ly), 1).reshape(self.num_landmarks*2, 1)
        self.pos_info.assign_landmark_pos(self.landmarks)
        self.landmarks_estimate = self.landmarks + np.random.normal(0, STD, np.shape(self.landmarks))

        # agent pose init
        # self.agent_pos = np.zeros(STATE_DIM, dtype=np.float32)
        self.agent_pos = np.array([random.uniform(-2, 2), random.uniform(-2, 2), 0])
        print("agent_pos:", self.agent_pos)
        obs_unity, _, _, _ = self.env_unity.step([self.agent_pos[0], self.agent_pos[1]])
        # self.agent_pos = np.array([0, 0, 0])

        # state init
        self.state = np.hstack([
            self.agent_pos[:2],
            self.info_mat.diagonal(),
            self.landmarks_estimate.flatten()
        ]).astype(np.float32)

        # step counter init
        self.current_step = -1

        # plot
        self.history_poses = [self.agent_pos]
        self.history_actions = []
        if self.test == True:
            self.fig = plt.figure(1)
            self.ax = self.fig.gca()

        return self.state

    def _plot(self, legend, title='trajectory'):

        # plot agent trajectory
        plt.tick_params(labelsize=11)
        history_poses = np.array(self.history_poses)
        self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='black', linewidth=2, label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=50, c='red', label="start")
        self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='s', s=50, c='red', label="end")

        # plot landmarks
        self.ax.scatter(self.landmarks[list(range(0, self.num_landmarks*2, 2)), :],
                        self.landmarks[list(range(1, self.num_landmarks*2+1, 2)), :], s=50, c='blue', label="landmark")

        # annotate theta value to each position point
        # for i in range(0, len(self.history_poses)-1):
        #     self.ax.annotate(round(self.history_actions[i][2], 4), history_poses[i, :2])

        # axes
        self.ax.set_xlabel("x", fontdict={'size': 16})
        self.ax.set_ylabel("y", fontdict={'size': 16})

        # title
        self.ax.set_title(title, fontdict={'size': 16})

        # legend
        if legend == True:
            self.ax.legend()

    def render(self, mode='human'):
        if mode == 'terminal':
            print(f">>>> step {self.current_step} <<<<")
            print(f"agent pos = {self.agent_pos}")
            print("information matrix:")
            print(self.info_mat)
        elif mode == 'human':
            # clear axes
            self.ax.cla()

            # plot
            self._plot(True)

            # display
            plt.draw()
            plt.pause(0.5)

        else:
            raise NotImplementedError

    def save_plot(self, name='default.png', title='trajectory', legend=False):
        self.ax.cla()
        self._plot(legend, title=title)
        self.fig.savefig(name)

    def close (self):
        plt.close('all')
