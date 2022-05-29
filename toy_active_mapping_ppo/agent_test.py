import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from env import SimpleQuadrotor

NUM_TEST = 10

ACTION = np.array([[4, 3, 0], [0, -2, 0],
    [-4, -5, 0], [-2, -1, 0], [-1, 4, 0], [0, 4, 0]])/5

parser = argparse.ArgumentParser(description='landmark-based mapping')
parser.add_argument('--num-landmarks', type=int, default=15)
parser.add_argument('--learning-curve-path', default="tensorboard/ppo_toy_active_mapping/")
parser.add_argument('--model-path', default="checkpoints/ppo_toy_active_mapping/default")
args = parser.parse_args()

def test_agent(agent):
    # get env
    landmarks = np.random.uniform(low=-10, high=10.0, size=(args.num_landmarks * 2, 1))
    env = SimpleQuadrotor(args.num_landmarks, landmarks, True)

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

        # print(f"\n------ Eps {eps} ------")
        # print(f"init state = {obs}")

        t = 0
        while not done:
            # get action
            action, _state = agent.predict(obs, deterministic=True)
            # action = ACTION[t]
            t += 1

            # step env
            obs, r, done, info = env.step(action)

            # calc return
            total_reward += r

            # render
            env.render(mode='human')
            print(f"reward = {r}")

        # summary
        print("---")
        print(f"return = {total_reward}")
        env.save_plot(name=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                             "plots/test_eps{}.png".format(eps)), title=f'return = {total_reward}')
    env.close()

if __name__ == '__main__':
    # load model
    model = PPO.load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "checkpoints/ppo_toy_active_mapping/default/best_model.zip"))

    # test
    test_agent(model)

        
