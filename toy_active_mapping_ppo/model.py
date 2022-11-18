from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.num_landmark = int((feature_dim - 2) / 5)
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # policy net
        self.policy_net = torch.nn.Sequential(

        )

        self.agent_pos_fc1_pi = nn.Linear(2, 32)
        self.agent_pos_fc2_pi = nn.Linear(32, 32)
        self.landmark_fc1_pi = nn.Linear(4, 64)
        self.landmark_fc2_pi = nn.Linear(64, 32)
        self.info_fc1_pi = nn.Linear(64, 64)
        self.action_fc1_pi = nn.Linear(64, self.latent_dim_pi)

        # value net
        self.agent_pos_fc1_vf = nn.Linear(2, 32)
        self.agent_pos_fc2_vf = nn.Linear(32, 32)
        self.landmark_fc1_vf = nn.Linear(4, 64)
        self.landmark_fc2_vf = nn.Linear(64, 32)
        self.info_fc1_vf = nn.Linear(64, 64)
        self.value_fc1_vf = nn.Linear(64, self.latent_dim_vf)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)
        # self.Norm = nn.LayerNorm()

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # print(landmark_info, action, self.latent_dim_pi, self.latent_dim_vf, "\n\n")
        return self.forward_actor(observation), self.forward_critic(observation)

    def forward_actor(self, observation: torch.Tensor) -> torch.Tensor:
        # compute the policy
        # embeddings of agent's position
        agent_pos_embedding = self.relu(self.agent_pos_fc1_pi(observation[:, :2]))
        agent_pos_embedding = self.relu(self.agent_pos_fc2_pi(agent_pos_embedding))

        # embeddings of landmarkss
        info_vector = observation[:, 2: 2 + 2 * self.num_landmark]
        estimated_landmark_pos = observation[:, 2 + 2 * self.num_landmark: - self.num_landmark]
        landmark_info = torch.cat((estimated_landmark_pos.reshape(observation.size()[0], self.num_landmark, 2),
                                   info_vector.reshape(observation.size()[0], self.num_landmark, 2)), 2)
        landmark_embedding = self.relu(self.landmark_fc1_pi(landmark_info))
        landmark_embedding = self.relu(self.landmark_fc2_pi(landmark_embedding))

        # attention
        landmark_embedding_tr = torch.transpose(landmark_embedding, 1, 2)

        # mask
        mask = observation[:, - self.num_landmark:].unsqueeze(1)
        attention = torch.matmul(agent_pos_embedding.unsqueeze(1), landmark_embedding_tr) / 4
        attention = attention.masked_fill(mask == 0, -1e10)

        att = self.softmax(attention)
        landmark_embedding_att = self.relu((torch.matmul(att, torch.transpose(landmark_embedding_tr, 1, 2)).squeeze(1)))

        info_embedding = self.relu(self.info_fc1_pi(torch.cat((agent_pos_embedding, landmark_embedding_att), 1)))
        action = self.tanh(self.action_fc1_pi(info_embedding))
        return action

    def forward_critic(self, observation: torch.Tensor) -> torch.Tensor:
        # compute the value
        # embeddings of agent's position
        agent_pos_embedding = self.relu(self.agent_pos_fc1_vf(observation[:, :2]))
        agent_pos_embedding = self.relu(self.agent_pos_fc2_vf(agent_pos_embedding))

        # embeddings of landmarkss
        info_vector = observation[:, 2: 2 + 2 * self.num_landmark]
        estimated_landmark_pos = observation[:, 2 + 2 * self.num_landmark: - self.num_landmark]
        landmark_info = torch.cat((estimated_landmark_pos.reshape(observation.size()[0], self.num_landmark, 2),
                                   info_vector.reshape(observation.size()[0], self.num_landmark, 2)), 2)
        landmark_embedding = self.relu(self.landmark_fc1_vf(landmark_info))
        landmark_embedding = self.relu(self.landmark_fc2_vf(landmark_embedding))

        # attention
        landmark_embedding_tr = torch.transpose(landmark_embedding, 1, 2)

        # mask
        mask = observation[:, - self.num_landmark:].unsqueeze(1)
        attention = torch.matmul(agent_pos_embedding.unsqueeze(1), landmark_embedding_tr) / 4
        attention = attention.masked_fill(mask == 0, -1e10)

        att = self.softmax(attention)
        landmark_embedding_att = self.relu((torch.matmul(att, torch.transpose(landmark_embedding_tr, 1, 2)).squeeze(1)))

        info_embedding = self.relu(self.info_fc1_vf(torch.cat((agent_pos_embedding, landmark_embedding_att), 1)))
        value = self.tanh(self.value_fc1_vf(info_embedding))
        return value


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)