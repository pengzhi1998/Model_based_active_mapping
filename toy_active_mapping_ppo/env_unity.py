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
from utils import unicycle_dyn, diff_FoV_land, diff_FoV_land_square
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
KAPPA = .5

# time & step
STEP_SIZE = 1



class InfoChannel(SideChannel):
    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.infos = msg.read_float32_list()

    def prints(self, agent_pos):
        print(self.infos, agent_pos)

    def assign_landmark_pos(self, data: List[float]) -> None:
        msg = OutgoingMessage()
        msg.write_float32_list(data)
        super().queue_message_to_send(msg)

class landmark_based_mapping(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}
    def __init__(self, num_landmarks, horizon, landmarks, boundary, test=False):
        super(landmark_based_mapping, self).__init__()

        # variables
        self.num_landmarks = num_landmarks
        self.boundary = boundary
        self.test = test
        self.total_time = horizon
        self.step_size = STEP_SIZE
        self.total_step = math.floor(self.total_time / STEP_SIZE)
        self.current_step = 0
        self.radius = np.array([RADIUS, RADIUS])

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
        self.info_mat = self.info_mat_init.copy()
        # print("landmarks' positions:", [self.landmarks])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.info_channel = InfoChannel()
        config_channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(os.path.abspath("./") + "/Unity_envs/Landmark",
                                     side_channels=[config_channel, self.info_channel], worker_id=1, base_port=5001)

        config_channel.set_configuration_parameters(time_scale=10, capture_frame_rate=100)
        self.env_unity = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
        self.width, self.height = 100, 100
        # homogenous intrinsic matrix with f_x and f_y both 50 millimeters, and the central point is (50, 50) in the image
        self.intrinsic = np.matrix([[0.05, 0, 50], [0, 0.05, 50], [0, 0, 1]], dtype=np.float32)
        self.intrinsic_homogeneous = np.matrix([[0.05, 0, 50, 0], [0, 0.05, 50, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.rotation = np.matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        x_index = np.array([list(range(100)) * 100])
        y_index = np.array([[i] * 100 for i in list(range(100))]).ravel()
        self.xy_index = np.vstack((x_index, y_index)).T  # x,y
        self.xyd_vect = np.zeros([100 * 100, 3])  # x,y,depth
        self.XYZ_vect = np.zeros([100 * 100, 3])  # real world coord
        self.pxToMetre = 0.1/self.width  # sensor_size/image_size
        self.landmarks_sem_color = np.array([[0, 0, 1],
                                             [0, 1, 0.7],
                                             [0, 1, 0],
                                             [1, 0, 0],
                                             [1, 1, 0]], dtype=np.float32)
        self.landmarks_sem_color_mapping = np.ones((5, 100, 100, 3))
        for i in range(len(self.landmarks_sem_color)):
            mask_r = np.full((self.height, self.width), self.landmarks_sem_color[i, 0])
            mask_g = np.full((self.height, self.width), self.landmarks_sem_color[i, 1])
            mask_b = np.full((self.height, self.width), self.landmarks_sem_color[i, 2])
            self.landmarks_sem_color_mapping[i] = np.stack((mask_r, mask_g, mask_b), 2)

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
        # print("\n\ncurrent agent position:", next_agent_pos[:2])
        # self.info_channel.prints(next_agent_pos)

        # update the estimated landmarks' positions
        for i in range(self.num_landmarks):
            sensor_result = np.abs(next_agent_pos[0:2] - self.landmarks[i * 2: i * 2 + 2].flatten()) < self.radius
            if sensor_result[0] and sensor_result[1]:  # sensor shape in square
                # restore the real depth values
                normalized_depth = obs_unity[0] ** 2.2
                normalized_depth = normalized_depth ** (1 / 0.25)
                depth_img = (normalized_depth * 10 + (1 - normalized_depth) * 0.3)[:, :, 0] - 0.05  # remove the focal length in depth values

                semantic_img = np.around(obs_unity[1], 2)
                mask_ = semantic_img == self.landmarks_sem_color_mapping[i]
                mask = mask_[:, :, 0] & mask_[:, :, 1] & mask_[:, :, 2]
                mask_index = np.stack(np.where(mask)).T
                mask_depth_img = depth_img[mask_index.T.tolist()].reshape(-1, 1)

                xyd_vect_uv = mask_index * mask_depth_img * self.pxToMetre
                xyd_vect_z = mask_depth_img * self.pxToMetre  # pxToMetre for unit conversion

                xyd_vect_homogeneous = np.hstack((xyd_vect_uv, xyd_vect_z, np.ones(np.shape(xyd_vect_z))))
                XYZ_vect_homogeneous = self.intrinsic_homogeneous.I.dot(xyd_vect_homogeneous.T)
                XYZ_vect_homogeneous[[0, 1], :] = XYZ_vect_homogeneous[[1, 0], :]
                XYZ_vect_homogeneous[1, :] *= -1
                XYZ_vect_homogeneous[2, :] /= self.pxToMetre
                extrinsic = np.vstack(
                    (np.hstack((self.rotation, np.array([[-next_agent_pos[1]], [next_agent_pos[0]], [-1]]))), np.array([[0, 0, 0, 1]])))
                XYZ_world = extrinsic.I.dot(XYZ_vect_homogeneous)
                sensor_value = np.array([np.mean(XYZ_world[0, :]), np.mean(XYZ_world[2, :])])
                # print("estimated landmark position:", sensor_value)
                if np.isnan(sensor_value)[0] == True:
                    r_test = obs_unity[1][:, :, 0]
                    g_test = obs_unity[1][:, :, 1]
                    b_test = obs_unity[1][:, :, 2]
                    depth_test = obs_unity[0][:, :, 0]
                    plt.imshow(obs_unity[1])
                    plt.show()
                    plt.imshow(obs_unity[0])
                    plt.show()
                info_sensor = np.array([[self.info_mat[i * 2, i * 2], 0], [0, self.info_mat[i * 2 + 1, i * 2 + 1]]])
                kalman_gain = np.linalg.inv(np.identity(2) + STD ** 2 * info_sensor)
                landmarks_estimate = self.landmarks_estimate[i * 2: i * 2 + 2].flatten() + \
                                     kalman_gain @ (sensor_value - self.landmarks_estimate[
                                                                   i * 2: i * 2 + 2].flatten())  # no dynamics for the landmarks
                self.landmarks_estimate[i * 2] = landmarks_estimate[0]
                self.landmarks_estimate[i * 2 + 1] = landmarks_estimate[1]
                # print("landmark_id:", i, "estimated_position:", np.mean(XYZ_world[0, :]), np.mean(XYZ_world[2, :]))

        # reward
        V_jj_inv = diff_FoV_land_square(next_agent_pos, self.landmarks_estimate, self.num_landmarks, RADIUS, KAPPA, STD).astype(np.float32) # TODO replace self.landmarks with an estimated one
        next_info_mat = self.info_mat + V_jj_inv # update info
        reward = float(slogdet(next_info_mat)[1] - slogdet(self.info_mat)[1])
        # print(slogdet(next_info_mat)[1], slogdet(self.info_mat)[1], next_info_mat)

        # terminate at time
        done = False
        if self.current_step >= self.total_step:
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
        # print("state:", self.state)

        # record history poses
        self.history_poses.append(self.agent_pos)
        # print(np.sum(np.abs(self.landmarks - self.landmarks_estimate)))

        return self.state, reward, done, info

    def reset(self):
        # landmark and info_mat init
        # print(self.info_mat)
        self.env_unity.reset()

        self.info_mat = self.info_mat_init.copy()
        # an extremely large value which guarantee this landmark's position has much lower uncertainty
        self.random_serial = np.random.randint(0, self.num_landmarks)
        self.info_mat[self.random_serial * 2, self.random_serial * 2], \
        self.info_mat[self.random_serial * 2 + 1, self.random_serial * 2 + 1] = 100, 100
        lx = np.random.uniform(low=-self.boundary, high=self.boundary, size=(self.num_landmarks, 1))
        ly = np.random.uniform(low=-self.boundary, high=self.boundary, size=(self.num_landmarks, 1))
        dis_mat = np.sqrt((lx - lx.T) ** 2 + (ly - ly.T) ** 2) + np.eye(self.num_landmarks)
        self.check_mat = dis_mat < 1
        while self.check_mat.__contains__(True):
            lx = np.random.uniform(low=-self.boundary, high=self.boundary, size=(self.num_landmarks, 1))
            ly = np.random.uniform(low=-self.boundary, high=self.boundary, size=(self.num_landmarks, 1))
            dis_mat = np.sqrt((lx - lx.T) ** 2 + (ly - ly.T) ** 2) + np.eye(self.num_landmarks)
            self.check_mat = dis_mat < 1

        # print("landmarks positions:", lx, ly, "\n\n")

        self.landmarks = np.concatenate((lx, ly), 1).reshape(self.num_landmarks*2, 1)
        self.info_channel.assign_landmark_pos(self.landmarks)
        self.landmarks_estimate = self.landmarks + np.random.normal(0, STD, np.shape(self.landmarks))

        # agent pose init
        # self.agent_pos = np.zeros(STATE_DIM, dtype=np.float32)
        self.agent_pos = np.array([random.uniform(-self.boundary, self.boundary), random.uniform(-self.boundary, self.boundary), 0])
        # print("agent_pos:", self.agent_pos, self.current_step)
        obs_unity, _, _, _ = self.env_unity.step([0, 1])
        # self.agent_pos = np.array([0, 0, 0])

        # print("after reset:")
        # self.info_channel.prints(self.agent_pos)

        # state init
        self.state = np.hstack([
            self.agent_pos[:2],
            self.info_mat.diagonal(),
            self.landmarks_estimate.flatten()
        ]).astype(np.float32)

        # step counter init
        self.current_step = 0

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
                        self.landmarks[list(range(1, self.num_landmarks*2+1, 2)), :], s=50, c='blue', label="landmark_0")
        self.ax.scatter(self.landmarks[2 * self.random_serial, :],
                        self.landmarks[2 * self.random_serial + 1, :], s=50, c='green',
                        label="landmark_100")
        print(self.landmarks, self.random_serial)

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
