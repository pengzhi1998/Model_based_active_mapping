import torch
import gym

from torch import nn
from typing import Callable, Dict, List, Optional, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from toy_active_mapping_ppo.model import CustomNetwork


class CustomModelNetwork(CustomNetwork):

    def __init__(self,
                 feature_dim: int,
                 last_layer_dim_pi: int = 64,
                 last_layer_dim_vf: int = 64):

        super(CustomModelNetwork, self).__init__(feature_dim,
                                                 last_layer_dim_pi,
                                                 last_layer_dim_vf)

    def forward_critic(self, observation: torch.Tensor) -> torch.Tensor:
        return


class CustomActorModelPolicy(ActorCriticPolicy):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 *args,
                 **kwargs):

        super(CustomActorModelPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs)

        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomModelNetwork(self.features_dim)
