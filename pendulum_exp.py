import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

######### training
# # create env
# env = gym.make('Pendulum-v0')

# # wrap with vector env
# env = make_vec_env(lambda: env, n_envs=1)

# # action noise
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# # model
# model = DDPG('MlpPolicy', env, action_noise=action_noise, buffer_size=200000, learning_starts=10000, gamma=0.98, gradient_steps=-1, train_freq=(1, "episode"), learning_rate=1e-3, policy_kwargs=dict(net_arch=[400, 300]), verbose=1)

# # train agent
# NUM_STEPS = 1e5
# LOG_INTERVAL=20
# model.learn(total_timesteps=NUM_STEPS, log_interval=LOG_INTERVAL)
# model.save("checkpoints/Pendulum-v0-2")

######### testing

# create env
env = gym.make('Pendulum-v0')

# load model
model = DDPG.load("checkpoints/Pendulum-v0-2")

# test
NUM_TEST = 10
for i in range(NUM_TEST):
    obs = env.reset()
    done = False

    # episode total reward
    ep_rew = 0

    print(f"=== episode {i} ====")
    print(f"init state  = {obs}")
    while not done:
        env.render()
        action, _state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        ep_rew += reward
        # env.render()
    print(f"final state = {obs}")
    print(f"eps reward  = {ep_rew}")
