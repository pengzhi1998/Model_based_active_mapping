import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

TERMINAL_STATE = 0.
MAX_STEPS = 50
class ToyEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ToyEnv, self).__init__()

        # spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) # (-inf, inf)
        # self.observation_space = spaces.Box(low=-200, high=200, shape=(1,), dtype=np.float32) # (-200, 200)
        # self.action_space      = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # (-1, 1), recommended
        self.action_space      = spaces.Box(low=-25, high=25, shape=(1,), dtype=np.float32) # (-25, 25)

        self.init_observation_space = spaces.Box(low=-25, high=25, shape=(1,), dtype=np.float32) # (-25, 25)

        # random state init
        self.agent_pos = self.init_observation_space.sample()

        # init step counter
        self.step_counter = 0

    def reset(self):
        # reset agent pos
        self.agent_pos = self.init_observation_space.sample()

        # reset step counter
        self.step_counter = 0

        return self.agent_pos # obs/state

    def step(self, action: np.float32):
        # update step counter
        self.step_counter += 1

        # convert action to float32 
        action = np.array(action).astype(np.float32)

        # reward
        reward = - float(self.agent_pos**2 + action**2)

        # new obs/state
        self.agent_pos += action

        # termination condition (-0.1, 0.1)
        done = False
        if ((self.agent_pos < 0.1) & (self.agent_pos > -0.1)) or self.step_counter >= MAX_STEPS:
            done = True

        # additional info
        info = {}

        return self.agent_pos.copy(), reward, done, info # obs/state, reward, done, info

    def render(self, mode='human'):
        print("current obs:", float(np.round(self.agent_pos, 3)), end=" ")

    def close(self):
        pass

if __name__ == '__main__':
    env = ToyEnv()
    # check if env follows Gym interface and is compatible with Stable Baselines
    check_env(env, warn=True)
