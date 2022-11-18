import torch

from typing import Tuple
from torch import tensor
from model_based_active_mapping.utilities.utils import SE2_kinematics, landmark_motion


class SimpleEnv:

    def __init__(self, num_landmarks, horizon, width, height, tau, A, B, landmark_motion_scale):
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

        return mu, v, x

    def step(self, action: tensor) -> Tuple[tensor, tensor, tensor, bool]:
        self._x = torch.clip(SE2_kinematics(self._x, action, self._tau), min=torch.zeros(2), max=self._env_size)

        self._mu = torch.clip(landmark_motion(self._mu, self._v, self._A, self._B),
                              min=torch.zeros(2), max=self._env_size)

        self._v = (torch.rand((self._num_landmarks, 2)) - 0.5) * self._landmark_motion_scale

        done = False
        self._step_num += 1
        if self._step_num >= self._horizon:
            done = True

        return self._mu, self._v, self._x, done
