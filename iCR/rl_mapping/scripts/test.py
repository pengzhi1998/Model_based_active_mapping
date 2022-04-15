from stable_baselines3 import DDPG

import os, sys
import yaml

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from rl_mapping.envs.env import VolumetricQuadrotor

NUM_TEST = 1

def test(params_filepath: str):
    ### read parameters ###
    with open(os.path.join(params_filepath)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    ### create env ###
    env = VolumetricQuadrotor(params['map_filepath'], params['env_params_filepath'])

    ### load model ###
    agent = DDPG.load(os.path.join(params['checkpoints_folder'], params['exp_name'], params['exp_name']), device='cpu')

    ### get parameters ###
    try:
        gamma = float(agent.gamma)
    except:
        gamma = 1.
    print(f"gamma = {gamma}")

    ### test ###
    for eps in range(NUM_TEST):

        obs = env.reset()
        done = False
        total_reward = 0

        print(f"\n------ Eps {eps} ------")
        print(f"init pose = {obs['pose']}")

        while not done:
            # get action
            action, _state = agent.predict(obs)

            # step env
            obs, r, done, info = env.step(action)
            print("obs =", obs['pose'])

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
    test("../params/training_params.yaml")

        
