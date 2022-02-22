from stable_baselines3 import DDPG
from toy_env import ToyEnv
from agent import OptAgent
import numpy as np

NUM_TEST = 10

def test_agent(agent):
    # get env
    env = ToyEnv()

    # test
    for i in range(NUM_TEST):
        obs = env.reset()
        done = False

        # episode total reward
        ep_rew = 0

        # action / state
        ratio = list()

        print(f"=== episode {i} ====")
        print(f"init state  = {obs}")
        while not done:
            action, _state = agent.predict(obs)
            ratio.append(action/obs)
            obs, reward, done, info = env.step(action)
            ep_rew += reward
            # env.render()
        print(f"final state = {obs}")
        print(f"num steps   = {env.step_counter}")
        print(f"eps reward  = {ep_rew}")
        m = np.mean(ratio)
        std = np.std(ratio)
        print(f"optimality  = [{m-std}, {m+std}]")

if __name__ == '__main__':
    # load model
    model = DDPG.load('checkpoints/ddpg_toyenv_bounded/zoo2')
    # model = OptAgent()

    # test
    test_agent(model)

        
