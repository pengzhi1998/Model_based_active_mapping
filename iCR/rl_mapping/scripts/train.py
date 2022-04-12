import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from rl_mapping.envs.env import VolumetricQuadrotor
from rl_mapping.networks.policy import CustomCombinedExtractor

NUM_STEPS = 1e5
LOG_INTERVAL=50

def make_ddpg_agent(env):
    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # policy network
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )

    # model
    # model = DDPG('MlpPolicy', env, action_noise=action_noise, buffer_size=1024, learning_starts=10000, gamma=0.98, policy_kwargs=dict(net_arch=[400, 300]), verbose=1, tensorboard_log="./tensorboard/ddpg/") # zoo
    # model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./tensorboard/ddpg-cnn/", buffer_size=1024) # default
    model = DDPG('MultiInputPolicy', env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboard/ddpg-cnn/", buffer_size=1024) # default

    return model

if __name__ == '__main__':
    exp_name = 'default'

    params_filename = '../params/env_params.yaml'
    map_filename = '../maps/map6_converted.png'
    
    ### create env & test random actions ###
    env = VolumetricQuadrotor(map_filename, params_filename)

    # wrap with vector env
    env = make_vec_env(lambda: env, n_envs=1)

    # train agent
    model = make_ddpg_agent(env)
    model.learn(total_timesteps=NUM_STEPS, log_interval=LOG_INTERVAL, tb_log_name=exp_name)
    model.save("checkpoints/ddpg-cnn/" + exp_name)