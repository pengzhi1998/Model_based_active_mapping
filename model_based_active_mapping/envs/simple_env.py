import torch
import cv2
import numpy as np

from typing import Tuple
from torch import tensor
from model_based_active_mapping.utilities.utils import SE2_kinematics, landmark_motion


class SimpleEnv:

    def __init__(self, num_landmarks, horizon, width, height, tau, A, B, landmark_motion_scale, psi, radius):
        self._num_landmarks = num_landmarks
        self._horizon = horizon
        self._env_size = tensor([width, height])
        self._tau = tau
        self._A = A
        self._B = B
        self._landmark_motion_scale = landmark_motion_scale

        self._mu = None
        self._v = None
        self._x = None
        self._step_num = None

        self._psi = psi
        self._radius = radius

    def reset(self):
        mu = torch.rand((self._num_landmarks, 2)) * self._env_size

        v = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._landmark_motion_scale

        x = torch.empty(3)
        x[:2] = torch.rand(2) * self._env_size
        x[2] = (torch.rand(1) * 2 - 1) * torch.pi

        self._mu = mu
        self._v = v
        self._x = x
        self._step_num = 0

        return mu, v, x, False

    def step(self, action: tensor) -> Tuple[tensor, tensor, tensor, bool]:
        self._x = SE2_kinematics(self._x, action, self._tau)
        # self._x[:2] = torch.clip(self._x[:2], min=torch.zeros(2), max=self._env_size)

        self._mu = torch.clip(landmark_motion(self._mu, self._v, self._A, self._B),
                              min=torch.zeros(2), max=self._env_size)

        self._v = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._landmark_motion_scale

        done = False
        self._step_num += 1
        if self._step_num >= self._horizon:
            done = True

        return self._mu, self._v, self._x, done

    def render(self):
        render_size = 50
        arrow_length = 20
        canvas = 255 * np.ones((self._env_size[1] * render_size, self._env_size[0] * render_size), dtype=np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

        # cv2.circle(canvas, (0, int(self._env_size[1] / 4 * render_size)), 10, (255, 0, 0), -1)

        for landmark_pos in self._mu:
            cv2.circle(canvas, (int(landmark_pos[0] * render_size), int(landmark_pos[1] * render_size)), 10,
                       (255, 0, 0), -1)

        robot_pose = self._x.detach().numpy()
        # robot_pose = np.array([0, 5, np.pi * 0.0])

        cv2.circle(canvas, (int(robot_pose[0] * render_size), int(robot_pose[1] * render_size)), 10, (0, 0, 255), -1)
        canvas = cv2.arrowedLine(canvas, (int(robot_pose[0] * render_size), int(robot_pose[1] * render_size)),
                                 (int(robot_pose[0] * render_size + arrow_length * np.cos(robot_pose[2])),
                                  int(robot_pose[1] * render_size + arrow_length * np.sin(robot_pose[2]))),
                                 (0, 0, 255), 2, tipLength=0.5)

        FoV_corners = np.array([(int(robot_pose[0] * render_size), int(robot_pose[1] * render_size)),
                                (int((robot_pose[0] + self._radius *
                                      np.cos(robot_pose[2] + self._psi) / np.cos(self._psi)) * render_size),
                                 int((robot_pose[1] + self._radius *
                                      np.sin(robot_pose[2] + self._psi) / np.cos(self._psi)) * render_size)),
                                (int((robot_pose[0] + self._radius *
                                      np.cos(robot_pose[2] - self._psi) / np.cos(self._psi)) * render_size),
                                 int((robot_pose[1] + self._radius *
                                      np.sin(robot_pose[2] - self._psi) / np.cos(self._psi)) * render_size))])

        cv2.polylines(canvas, [FoV_corners], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('map', canvas)
        # cv2.resizeWindow('map', *render_size)
        cv2.waitKey(100)
