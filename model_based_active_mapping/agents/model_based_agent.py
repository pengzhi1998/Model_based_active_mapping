import torch

from torch.optim import Adam
from torch import tensor
from model_based_active_mapping.models.policy_net import PolicyNet
from model_based_active_mapping.utilities.utils import landmark_motion, triangle_SDF, get_transformation, phi


class ModelBasedAgent:

    def __init__(self, num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr):
        self._info = init_info * torch.ones((num_landmarks, 2))

        self._num_landmarks = num_landmarks
        self._A = A
        self._B = B
        self._W = W
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._inv_V = V ** (-1)

        input_dim = num_landmarks * 4 + 3
        self._policy = PolicyNet(input_dim=input_dim)
        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def plan(self, mu, v, x):
        next_mu = landmark_motion(mu, v, self._A, self._B)
        net_input = torch.hstack((x, self._info.flatten(), next_mu.flatten()))
        action = self._policy.forward(net_input)
        return action

    def update_info(self, mu, x):
        transformation = get_transformation(x)
        mu_h = torch.hstack((mu, torch.ones((mu.shape[0], 1))))
        q = (mu_h @ transformation.T)[:, :2]
        SDF = triangle_SDF(q, self._psi, self._radius)
        M = (1 - phi(SDF, self._kappa)) * self._inv_V.repeat(self._num_landmarks, 1)
        # Assuming A = I:
        self._info = (self._info**(-1) + self._W)**(-1) + M

    def update_policy(self):
        self._policy_optimizer.zero_grad()
        reward = - torch.sum(torch.log(self._info))
        reward.backward()
        self._policy_optimizer.step()
