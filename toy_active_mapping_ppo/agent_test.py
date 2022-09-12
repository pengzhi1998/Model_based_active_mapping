import argparse
import os
import numpy as np

from stable_baselines3 import PPO
from env import SimpleQuadrotor

NUM_TEST = 10

ACTION = np.array([[4, 3, 0], [0, -2, 0],
    [-4, -5, 0], [-2, -1, 0], [-1, 4, 0], [0, 4, 0]])/5

parser = argparse.ArgumentParser(description='landmark-based mapping')
parser.add_argument('--num-landmarks', type=int, default=5)
parser.add_argument('--horizon', type=int, default=15)
parser.add_argument('--bound', type=int, default=10)
parser.add_argument('--learning-curve-path', default="tensorboard/ppo_toy_active_mapping/")
parser.add_argument('--model-path', default="checkpoints/ppo_toy_active_mapping/default")
parser.add_argument('--model', default="attention")
parser.add_argument('--seed', type=int, default=0, help="use multiple seeds to train, values should be 0, 10, and 100")
parser.add_argument('--for-comparison', type=int, default=0, help="0 means this code is running for training or simple tests,"
                                                                  "otherwise this is for large-scale tests, while we need to read the"
                                                                  "randomized landmarks' and agent's initial positions from a generated"
                                                                  "txt file for fair comparison with landmark-based iCR")
parser.add_argument('--special-case', type=int, default=0, help="0 means all the initial information values are the same, 0.5,"
                                                                "otherwise there would be one with a special value, like 100")
parser.add_argument('--dynamics-noise', type=int, default=0, help="0 means there would be dynamics noise during testing, "
                                                                  "otherwise there is no noise for both testing and training")
args = parser.parse_args()

def test_agent(agent):
    # get env
    landmarks = np.random.uniform(low=-args.bound, high=args.bound, size=(args.num_landmarks * 2, 1))
    env = SimpleQuadrotor(args.bound, args.num_landmarks, args.horizon, landmarks, args.for_comparison, args.special_case, True)

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
            if t == 0 or t == args.horizon/3 - 1 or t == args.horizon*2/3 - 1:
                total_reward_ = format(total_reward, '.2f')
                if t == 0:
                    env.save_plot(name=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                "plots/test_landmarknum{}_eps{}_step{}.png".format(args.num_landmarks, eps, t)), title=f'return = {total_reward}', legend=True)
                else:
                    env.save_plot(name=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                    "plots/test_landmarknum{}_eps{}_step{}.png".format(args.num_landmarks, eps, t)), title=f'return = {total_reward_}')

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
                                             "plots/test_landmarknum{}_eps{}_step{}.png".format(args.num_landmarks, eps, t)), title=f'return = {total_reward_}')
    env.close()
    print(f"mean and std of total return = {np.mean(reward_list), np.std(reward_list)}")

if __name__ == '__main__':
    # load model
    model = PPO.load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "checkpoints/ppo_toy_active_mapping/default/landmark_{}-seed_{}-model_{}/best_model.zip".format(args.num_landmarks, args.seed, args.model)))

    # test
    test_agent(model)

        
