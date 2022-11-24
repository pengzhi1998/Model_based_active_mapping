import os, yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import tensor
from model_based_active_mapping.envs.simple_env import SimpleEnv
from model_based_active_mapping.agents.model_based_agent import ModelBasedAgent


def run_model_based_training(params_filename):
    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    num_landmarks = params['num_landmarks']
    horizon = params['horizon']
    env_width = params['env_width']
    env_height = params['env_height']
    tau = params['tau']

    A = torch.zeros((2, 2))
    A[0, 0] = params['motion']['A']['_1']
    A[1, 1] = params['motion']['A']['_2']

    B = torch.zeros((2, 2))
    B[0, 0] = params['motion']['B']['_1']
    B[1, 1] = params['motion']['B']['_2']

    W = torch.zeros(2)
    W[0] = params['motion']['W']['_1']
    W[1] = params['motion']['W']['_2']

    landmark_motion_scale = params['motion']['landmark_motion_scale']

    init_info = params['init_info']

    radius = params['FoV']['radius']
    psi = tensor([params['FoV']['psi']])
    kappa = params['FoV']['kappa']

    V = torch.zeros(2)
    V[0] = params['FoV']['V']['_1']
    V[1] = params['FoV']['V']['_2']

    lr = params['lr']
    max_epoch = params['max_epoch']
    num_test_trials = params['num_test_trials']

    env = SimpleEnv(num_landmarks=num_landmarks, horizon=horizon, width=env_width, height=env_height, tau=tau,
                    A=A, B=B, landmark_motion_scale=landmark_motion_scale)
    agent = ModelBasedAgent(num_landmarks=num_landmarks, init_info=init_info, A=A, B=B, W=W,
                            radius=radius, psi=psi, kappa=kappa, V=V, lr=lr)

    reward_list = np.empty(max_epoch)
    action_list = np.empty((max_epoch, horizon, 2))
    for i in range(max_epoch):
        mu, v, x, done = env.reset()
        agent.reset_agent_info()
        step = 0
        while not done:
            action = agent.plan(mu, v, x)
            action_list[i, step, :] = action.detach().numpy()
            mu, v, x, done = env.step(action)
            agent.update_info(mu, x)
            step += 1

        reward_list[i] = agent.update_policy(debug=False) / num_landmarks
        print('Normalized reward at iteration {}: {}'.format(i, reward_list[i]))

    plt.figure()
    plt.plot(reward_list)
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Reward")
    plt.show()

    plt.figure()
    plt.plot(np.mean(action_list[:, :, 0], axis=0), 'b-', label='Linear Velocity')
    plt.plot(np.mean(action_list[:, :, 0], axis=0) + 5 * np.std(action_list[:, :, 0], axis=0), 'b--')
    plt.plot(np.mean(action_list[:, :, 0], axis=0) - 5 * np.std(action_list[:, :, 0], axis=0), 'b--')

    plt.plot(np.mean(action_list[:, :, 1], axis=0), 'r-', label='Angular Velocity')
    plt.plot(np.mean(action_list[:, :, 1], axis=0) + 5 * np.std(action_list[:, :, 1], axis=0), 'r--')
    plt.plot(np.mean(action_list[:, :, 1], axis=0) - 5 * np.std(action_list[:, :, 1], axis=0), 'r--')

    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.show()

    agent.eval_policy()
    for i in range(num_test_trials):
        mu, v, x, done = env.reset()
        agent.reset_agent_info()
        env.render()
        while not done:
            action = agent.plan(mu, v, x)
            mu, v, x, done = env.step(action)
            agent.update_info(mu, x)
            env.render()


if __name__ == '__main__':
    # torch.manual_seed(0)
    # torch.autograd.set_detect_anomaly(True)
    run_model_based_training(params_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)),
                                                          "params/params.yaml"))