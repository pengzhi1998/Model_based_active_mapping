import torch

from torch.optim import Adam
from model_based_active_mapping.models.policy_net import PolicyNet
from model_based_active_mapping.utilities.utils import landmark_motion, triangle_SDF, get_transformation, phi


class ModelBasedAgent:

    def __init__(self, num_landmarks, init_info, A, B, W, radius, psi, kappa, V, lr):
        self._init_info = init_info
        self._info = None

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
        self._policy.train()

        # for p in self._policy.parameters():
        #     p.retain_grad()

        self._policy_optimizer = Adam(self._policy.parameters(), lr=lr)

    def reset_agent_info(self):
        self._info = self._init_info * torch.ones((self._num_landmarks, 2))

    def eval_policy(self):
        self._policy.eval()

    def plan(self, mu, v, x):
        next_mu = landmark_motion(mu, v, self._A, self._B)
        net_input = torch.hstack((x, self._info.flatten(), next_mu.flatten()))
        action = self._policy.forward(net_input)
        return action

    def update_info(self, mu, x):
        # transformation = get_transformation(x)
        # mu_h = torch.hstack((mu, torch.ones((mu.shape[0], 1))))
        # q = (mu_h @ transformation.T)[:, :2]

        q = torch.vstack((mu[:, 0] * torch.cos(x[2]) - mu[:, 1] * torch.sin(x[2]) + x[0],
                          mu[:, 0] * torch.sin(x[2]) + mu[:, 1] * torch.cos(x[2]) + x[1])).T

        SDF = triangle_SDF(q, self._psi, self._radius)
        M = (1 - phi(SDF, self._kappa))[:, None] * self._inv_V.repeat(self._num_landmarks, 1)
        # Assuming A = I:
        self._info = (self._info**(-1) + self._W)**(-1) + M

    def update_policy(self, debug=False):
        self._policy_optimizer.zero_grad()

        if debug:
            param_list = []
            grad_power = 0
            for i, p in enumerate(self._policy.parameters()):
                param_list.append(p.data.detach().clone())
                if p.grad is not None:
                    grad_power += (p.grad**2).sum()
                else:
                    grad_power += 0

            print("Gradient power before backward: {}".format(grad_power))

        reward = - torch.sum(torch.log(self._info))
        reward.backward()
        self._policy_optimizer.step()

        if debug:
            grad_power = 0
            total_param_ssd = 0
            for i, p in enumerate(self._policy.parameters()):
                if p.grad is not None:
                    grad_power += (p.grad ** 2).sum()
                else:
                    grad_power += 0
                total_param_ssd += ((param_list[i] - p.data) ** 2).sum()

            print("Gradient power after backward: {}".format(grad_power))
            print("SSD of weights after applying the gradient: {}".format(total_param_ssd))

        return -reward.item()
