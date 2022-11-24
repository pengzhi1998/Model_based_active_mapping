import torch

from torch import nn


class SimplePolicyNet(nn.Module):

    def __init__(self,
                 input_dim: int,
                 policy_dim: int = 2):

        super(SimplePolicyNet, self).__init__()

        self.agent_pos_fc1_pi = nn.Linear(3, 32)
        self.agent_pos_fc2_pi = nn.Linear(32, 32)
        self.landmark_fc1_pi = nn.Linear(2, 64)
        self.landmark_fc2_pi = nn.Linear(64, 32)
        self.info_fc1_pi = nn.Linear(64, 64)
        self.action_fc1_pi = nn.Linear(64, 64)
        self.action_fc2_pi = nn.Linear(64, policy_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if len(observation.size()) == 1:
            observation = observation[None, :]

        # compute the policy
        # embeddings of agent's position
        agent_pos_embedding = self.relu(self.agent_pos_fc1_pi(observation[:, :3]))
        agent_pos_embedding = self.relu(self.agent_pos_fc2_pi(agent_pos_embedding))

        # embeddings of landmarks
        estimated_landmark_pos = observation[:, 3:]
        landmark_embedding = self.relu(self.landmark_fc1_pi(estimated_landmark_pos))
        landmark_embedding = self.relu(self.landmark_fc2_pi(landmark_embedding))

        # attention
        attention = torch.dot(landmark_embedding.squeeze(), agent_pos_embedding.squeeze()) / 4
        att = self.softmax(attention)
        landmark_embedding_att = self.relu(att * landmark_embedding.squeeze())

        info_embedding = self.relu(self.info_fc1_pi(torch.cat((agent_pos_embedding.squeeze(),
                                                               landmark_embedding_att))))
        action = self.tanh(self.action_fc1_pi(info_embedding))
        action = self.action_fc2_pi(action)

        if action.size()[0] == 1:
            action = action.flatten()

        return action
