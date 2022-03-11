from stable_baselines3 import DDPG
from env import SimpleQuadrotor
import numpy as np

NUM_TEST = 1

def test_agent(agent):
    # get env
    env = SimpleQuadrotor()

    # get parameters
    try:
        gamma = float(agent.gamma)
    except:
        gamma = 1.
    print(f"gamma = {gamma}")

    # test
    for eps in range(NUM_TEST):

        obs = env.reset()
        done = False
        total_reward = 0

        print(f"\n------ Eps {eps} ------")
        print(f"init state = {obs}")

        while not done:
            # get action
            action, _state = agent.predict(obs)

            # step env
            obs, r, done, info = env.step(action)

            # calc return
            total_reward += r * (gamma ** env.current_step)

            # render
            env.render(mode='human')
            print(f"reward = {r}")

        # summary
        print("---")
        print(f"return = {total_reward}")
        env.save_plot(name=f'plots/test_eps{eps}.png', title=f'return = {total_reward}')
    env.close()

if __name__ == '__main__':
    # load model
    model = DDPG.load('checkpoints/ddpg_toy_active_mapping/default')

    # test
    test_agent(model)

        
