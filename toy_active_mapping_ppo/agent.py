import numpy as np
import sys
import os
from env import SimpleQuadrotor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

NUM_STEPS = 1e5
LOG_INTERVAL=1

def make_ppo_agent(env):
    model = PPO('MlpPolicy', env, verbose=1, n_steps=2048,
                tensorboard_log=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                             "tensorboard/ddpg_toy_active_mapping/")) # default

    return model

if __name__ == '__main__':
    # init env
    env = SimpleQuadrotor()

    # wrap with vector env
    env = make_vec_env(lambda: env, n_envs=1)

    # train agent
    model = make_ppo_agent(env)
    model.learn(total_timesteps=NUM_STEPS, log_interval=LOG_INTERVAL, tb_log_name="default")
    model.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoints/ddpg_toy_active_mapping/default"))