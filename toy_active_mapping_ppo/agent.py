import argparse
import sys
import os
import numpy as np
from model import CustomActorCriticPolicy
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

NUM_STEPS = 1e6
LOG_INTERVAL = 1
parser = argparse.ArgumentParser(description='landmark-based mapping')
parser.add_argument('--num-landmarks', type=int, default=15)
parser.add_argument('--horizon', type=int, default=45)
parser.add_argument('--bound', type=int, default=10)
parser.add_argument('--learning-curve-path', default="tensorboard/ppo_toy_active_mapping/")
parser.add_argument('--model-path', default="checkpoints/ppo_toy_active_mapping/default")
args = parser.parse_args()

register(
    # unique identifier for the env `name-version`
    id="SimpleQuadrotor-v0",
    entry_point="env:SimpleQuadrotor",
)

def make_ppo_agent(env):
    # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
    #                      net_arch=[dict(pi=[512, 256], vf=[512, 256])])
    # model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, seed=0, policy_kwargs=policy_kwargs,
    #             tensorboard_log=os.path.join(os.path.abspath(os.path.dirname(__file__)),
    #                                          args.learning_curve_path))  # default
    model = PPO(CustomActorCriticPolicy, env, verbose=1, n_steps=2048, seed=0,
                tensorboard_log=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                             args.learning_curve_path))  # default

    return model

if __name__ == '__main__':
    # init env
    if args.num_landmarks < 20:
        landmarks = np.random.uniform(low=-args.bound, high=args.bound, size=(args.num_landmarks * 2, 1))
    else:
        if args.num_landmarks == 25:
            landmarks = np.concatenate(
                (np.reshape(np.random.uniform(low=[-12., -2.], high=[-8., 2.], size=(args.num_landmarks//5, 2)), [10, 1]),
                 np.reshape(np.random.uniform(low=[-2., 8.], high=[2., 12.], size=(args.num_landmarks//5, 2)), [10, 1]),
                 np.reshape(np.random.uniform(low=[8., -2.], high=[12., 2.], size=(args.num_landmarks//5, 2)), [10, 1]),
                 np.reshape(np.random.uniform(low=[-2., -12.], high=[2., -8.], size=(args.num_landmarks//5, 2)), [10, 1]),
                 np.reshape(np.random.uniform(low=[-2., -2.], high=[2., 2.], size=(args.num_landmarks//5, 2)), [10, 1])), 0)

    # wrap with vector env
    env = make_vec_env("SimpleQuadrotor-v0", n_envs=1,
                       env_kwargs={"num_landmarks" : args.num_landmarks, "horizon" : args.horizon,
                                       "landmarks" : landmarks, "test" : False})

    # train agent
    model = make_ppo_agent(env)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path))
    evaluation_callback = EvalCallback(env, n_eval_episodes=20, eval_freq=10240,
        best_model_save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path))

    model.learn(total_timesteps=NUM_STEPS,log_interval=LOG_INTERVAL, tb_log_name="default", callback=evaluation_callback)
    model.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path))
    # print([landmarks])