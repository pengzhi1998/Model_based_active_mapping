from stable_baselines3 import DDPG

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from rl_mapping.envs.env import VolumetricQuadrotor

NUM_TEST = 1

def test_agent(env: VolumetricQuadrotor, agent):

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
        print(f"init pose = {obs[:2]}")

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
    env.close()

if __name__ == '__main__':
    # create env
    params_filename = '../params/env_params.yaml'
    map_filename = '../maps/map6_converted.png'
    env = VolumetricQuadrotor(map_filename, params_filename)

    # load model
    model = DDPG.load('checkpoints/ddpg/zoo', device='cpu')

    # test
    test_agent(env, model)

        
