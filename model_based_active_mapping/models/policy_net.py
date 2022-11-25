import torch

from torch import nn


class PolicyNet(nn.Module):

    def __init__(self,
                 input_dim: int,
                 policy_dim: int = 2):

        super(PolicyNet, self).__init__()

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # self.num_landmark = int((input_dim - 3) / 4)
        #
        # self.agent_pos_fc = nn.Sequential(*[nn.Linear(3, 32), self.relu])
        # self.landmark_info_fc = nn.Sequential(*[nn.Linear(self.num_landmark * 2, 32), self.relu])
        # self.landmark_pose_fc = nn.Sequential(*[nn.Linear(self.num_landmark * 2, 32), self.relu])
        # self.mixer_fc = nn.Sequential(*[nn.Linear(3 * 32, 32), self.relu])
        # self.pi_fc = nn.Sequential(*[nn.Linear(32, policy_dim), self.tanh])

        self.fc_1 = nn.Sequential(*[nn.Linear(input_dim, 64), self.relu])
        self.fc_2 = nn.Sequential(*[nn.Linear(64, 64), self.relu])
        self.pi_fc = nn.Sequential(*[nn.Linear(64, policy_dim), self.tanh])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if len(observation.size()) == 1:
            observation = observation[None, :]

        # mixer_input = torch.cat((self.agent_pos_fc(observation[:, :3]).squeeze(),
        #                          self.landmark_info_fc(observation[:, 3:3 + 2 * self.num_landmark]).squeeze(),
        #                          self.landmark_pose_fc(observation[:, 3 + 2 * self.num_landmark:]).squeeze()))

        # action = self.pi_fc(self.mixer_fc(mixer_input))

        action = self.pi_fc(self.fc_2(self.fc_1(observation)))

        if action.size()[0] == 1:
            action = action.flatten()

        scaled_action = torch.hstack((action[0] * 0.2, action[1] * 0.05))

        return scaled_action
