import numpy as np
from toy_env import ToyEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

NUM_STEPS = 1e5
LOG_INTERVAL=50

class OptAgent:
    def __init__(self):
        pass

    def predict(self, obs: np.float32):
        action = np.array(- 2 / (1 + np.sqrt(5)) ).astype(np.float32) * obs
        new_obs = obs + action
        return action, new_obs

def make_ddpg_agent(env):
    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model
    model = DDPG('MlpPolicy', env, action_noise=action_noise, buffer_size=200000, learning_starts=10000, gamma=0.98, policy_kwargs=dict(net_arch=[400, 300]), verbose=1, tensorboard_log="./tensorboard/ddpg_toyenv_bounded/")
    # model = DDPG('MlpPolicy', env, action_noise=action_noise, gamma=1, verbose=1, tensorboard_log="./tensorboard/ddpg_toyenv_bounded/")

    return model

if __name__ == '__main__':
    # init env
    env = ToyEnv()

    # wrap with vector env
    env = make_vec_env(lambda: env, n_envs=1)

    # train agent
    model = make_ddpg_agent(env)
    model.learn(total_timesteps=NUM_STEPS, log_interval=LOG_INTERVAL, tb_log_name="zoo0-tb")
    model.save("checkpoints/ddpg_toyenv_bounded/zoo0-tb")