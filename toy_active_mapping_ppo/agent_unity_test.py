import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from env_unity import landmark_based_mapping

NUM_TEST = 20

ACTION = np.array([[4, 3, 0], [0, -2, 0],
                   [-4, -5, 0], [-2, -1, 0], [-1, 4, 0], [0, 4, 0]]) / 5

parser = argparse.ArgumentParser(description='landmark-based mapping')
parser.add_argument('--num-landmarks', type=int, default=5)
parser.add_argument('--horizon', type=int, default=15)
parser.add_argument('--bound', type=int, default=10)
parser.add_argument('--learning-curve-path', default="tensorboard/ppo_toy_active_mapping/")
parser.add_argument('--model-path', default="checkpoints/ppo_toy_active_mapping/default")
args = parser.parse_args()


def test_agent(agent):
    # get env
    landmarks = np.random.uniform(low=-args.bound, high=args.bound, size=(args.num_landmarks * 2, 1))
    env = landmark_based_mapping(args.num_landmarks, args.horizon, landmarks, args.bound, True)

    # get parameters
    try:
        gamma = float(agent.gamma)
    except:
        gamma = 1.
    print(f"gamma = {gamma}")

    # test
    reward_list = []
    for eps in range(NUM_TEST):

        obs = env.reset()
        done = False
        total_reward = 0

        # print(f"\n------ Eps {eps} ------")
        # print(f"init state = {obs}")

        t = 0
        while not done:
            if t == 0 or t == args.horizon / 3 - 1 or t == args.horizon * 2 / 3 - 1:
                total_reward_ = format(total_reward, '.2f')
                if t == 0:
                    env.save_plot(name=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                    "plots/test_landmarknum{}_eps{}_step{}.png".format(
                                                        args.num_landmarks, eps, t)), title=f'return = {total_reward}',
                                  legend=True)
                else:
                    env.save_plot(name=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                    "plots/test_landmarknum{}_eps{}_step{}.png".format(
                                                        args.num_landmarks, eps, t)), title=f'return = {total_reward_}')

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

        # summary
        print("---")
        print(f"return = {total_reward}")
        reward_list.append(total_reward)
        total_reward_ = format(total_reward, '.2f')
        env.save_plot(name=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        "plots/test_landmarknum{}_eps{}_step{}.png".format(args.num_landmarks, eps, t)),
                      title=f'return = {total_reward_}')
    env.close()
    print(f"mean and std of total return = {np.mean(reward_list), np.std(reward_list)}")


if __name__ == '__main__':
    # load model
    if args.num_landmarks == 5:
        model = PPO.load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                      "checkpoints/ppo_toy_active_mapping/default/best_model.zip"))
    elif args.num_landmarks == 15:
        model = PPO.load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                      "checkpoints/ppo_toy_active_mapping/best_model.zip"))
    elif args.num_landmarks == 25:
        model = PPO.load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                      "checkpoints/ppo_toy_active_mapping/25landmarks/best_model.zip"))

    # test
    test_agent(model)


