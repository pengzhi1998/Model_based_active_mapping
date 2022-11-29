import argparse
import sys
import os
import torch
import numpy as np
from toy_active_mapping_ppo.model import CustomActorCriticPolicy
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

NUM_STEPS = 1e6
LOG_INTERVAL = 1
parser = argparse.ArgumentParser(description='landmark-based mapping')
parser.add_argument('--num-landmarks', type=int, default=5)
parser.add_argument('--horizon', type=int, default=15)
parser.add_argument('--bound', type=int, default=10)
parser.add_argument('--model', default="attention")
parser.add_argument('--seed', type=int, default=0, help="use multiple seeds to train, values should be 0, 10, and 100")
parser.add_argument('--SE3-control', type=int, default=1, help="if true: use SE3 control to directly control v_x and v_yaw while v_y is always 0,"
                                                           "if false: directly control x and y while yaw is always kept 0")
parser.add_argument('--motion-model', type=int, default=1, help="if 1: noisy static model (landmarks moving in smaller area),"
                                                           "if 2: noisy motion model (landmarks moving in one direction with noise)")
parser.add_argument('--for-comparison', type=int, default=0, help="0 means this code is running for training or simple tests,"
                                                                  "otherwise this is for large-scale tests, while we need to read the"
                                                                  "randomized landmarks' and agent's initial positions from a generated"
                                                                  "txt file for fair comparison with landmark-based iCR")
parser.add_argument('--special-case', type=int, default=0, help="0 means all the initial information values are the same, 0.5,"
                                                                "otherwise there would be one with a special value, like 100")
parser.add_argument('--dynamics-noise', type=int, default=0, help="0 means there would be dynamics noise during testing, "
                                                                  "otherwise there is no noise for both testing and training")
args = parser.parse_args()

register(
    # unique identifier for the env `name-version`
    id="SimpleQuadrotor-v0",
    entry_point="env:SimpleQuadrotor",
)

def make_ppo_agent(env):
    if args.model != 'attention':
        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                             net_arch=[dict(pi=[512, 256], vf=[512, 256])])
        model = PPO('MlpPolicy', env, verbose=1, n_steps=2048, seed=args.seed, policy_kwargs=policy_kwargs,
                    tensorboard_log=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                 "tensorboard/ppo_toy_active_mapping/landmark_{}-seed_{}-model_{}".format(args.num_landmarks, args.seed, args.model)))  # default
    else:
        model = PPO(CustomActorCriticPolicy, env, verbose=1, n_steps=2048, seed=args.seed,
                tensorboard_log=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                             "tensorboard/ppo_toy_active_mapping/landmark_{}-seed_{}-model_{}".format(args.num_landmarks, args.seed, args.model)))  # default

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
                       env_kwargs={"bound": args.bound, "num_landmarks" : args.num_landmarks, "horizon" : args.horizon,
                                   "SE3_control" : args.SE3_control, "motion_model" : args.motion_model, "for_comparison" : args.for_comparison,
                                   "special_case" : args.special_case, "test" : False})

    # train agent
    model = make_ppo_agent(env)
    # checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path))
    evaluation_callback = EvalCallback(env, n_eval_episodes=20, eval_freq=10240,
        best_model_save_path=os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                          "checkpoints/ppo_toy_active_mapping/default/landmark_{}-seed_{}-model_{}".format(args.num_landmarks, args.seed, args.model)))

    model.learn(total_timesteps=NUM_STEPS,log_interval=LOG_INTERVAL, tb_log_name="default", callback=evaluation_callback)
    model.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), args.model_path))
    # print([landmarks])
