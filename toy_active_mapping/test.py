from stable_baselines3 import DDPG
from env import SimpleQuadrotor
import numpy as np

NUM_TEST = 2

def test_agent(agent):
    # get env
    env = SimpleQuadrotor()

    # test
    for eps in range(NUM_TEST):
        obs = env.reset()
        done = False
        print(f"\n------ Eps {eps} ------")
        print(f"init state = {obs}")
        while not done:
            action, _state = agent.predict(obs)
            obs, r, done, info = env.step(action)
            env.render(mode='human')
            print(f"reward = {r}")
        env.save_plot(name=f'plots/test_eps{eps}.png')
    env.close()

if __name__ == '__main__':
    # load model
    model = DDPG.load('checkpoints/ddpg_toy_active_mapping/default')

    # test
    test_agent(model)

        
