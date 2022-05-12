from stable_baselines3 import PPO
from env import SimpleQuadrotor
import os
import numpy as np
NUM_TEST = 1

ACTION = np.array([[4, 3, 0], [0, -2, 0],
    [-4, -5, 0], [-2, -1, 0], [-1, 4, 0], [0, 4, 0]])/5

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

        t = 0
        while not done:
            # get action
            action, _state = agent.predict(obs, deterministic=True)
            # action = ACTION[t]
            t += 1

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
        env.save_plot(name=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                             "plots/test_eps{}.png".format(eps)), title=f'return = {total_reward}')
    env.close()

if __name__ == '__main__':
    # load model
    model = PPO.load(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "checkpoints/ppo_toy_active_mapping/default"))

    # test
    test_agent(model)

        
