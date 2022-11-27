import math
import numpy as np
import gym
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from numpy.linalg import slogdet
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from utils import unicycle_dyn, diff_FoV_land, diff_FoV_land_triangle, triangle_SDF, state_to_T

# env setting
STATE_DIM = 3
RADIUS = 2
STD_sensor = 0.2
STD_motion = 0.05
KAPPA = 0.4

# another option
# RADIUS = 2
# KAPPA = 0.25

# time & step
STEP_SIZE = 1
random.seed(100)
np.random.seed(100)

class SimpleQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    def __init__(self, bound, num_landmarks, horizon, SE3_control=True, motion_model=1,
                 for_comparison=False, special_case=False, test=False):
        super(SimpleQuadrotor, self).__init__()

        # variables
        self.num_landmarks = num_landmarks
        self.test = test
        self.total_time = horizon
        self.step_size = STEP_SIZE
        self.total_step = math.floor(self.total_time / STEP_SIZE)
        self.bound = bound

        # action space
        # defined as {-1, 1} as suggested by stable_baslines3, rescaled to {-2, 2} later in step()
        self.action_space = spaces.Box(low=-1, high=1, shape=(STATE_DIM, ), dtype=np.float32) # (x, y, \theta): {-2, 2}^3

        # state space
        # agent state + diag of info mat
        self.max_num_landmarks = 7
        self.x_pos = np.array([1, 0])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM + self.max_num_landmarks * 5, ), dtype=np.float32) # (x, y, \theta, info_mat_0, info_mat_1, info_mat_2, info_mat_3): {-inf, inf}^7

        # info_mat init
        self.info_mat_init = np.diag([.5] * self.num_landmarks * 2).astype(np.float32)
        self.info_mat = self.info_mat_init.copy()

        self.SE3_control = SE3_control
        self.motion_model = motion_model
        self.for_comparison = for_comparison
        self.special_case = special_case

    def step(self, action):
        self.current_step += 1

        if self.SE3_control:
            # rescale actions
            action[0] = (action[0] + 1) * 2
            action[1] = 0
            action[2] *= np.pi/3

            # record history action
            self.history_actions.append(action.copy())

            # robot dynamics (precise x, y positions, we don't incorporate noise for robot's state or motion)
            next_agent_pos = unicycle_dyn(self.agent_pos, action, self.step_size).astype(np.float32)

        else:
            # rescale actions
            action[0] *= 3
            action[1] *= 3
            action[2] = 0

            # record history action
            self.history_actions.append(action.copy())

            # robot dynamics (precise x, y positions, we don't incorporate noise for robot's state or motion)
            # next_agent_pos = unicycle_dyn(self.agent_pos, action, self.step_size).astype(np.float32)
            next_agent_pos_xy = (self.agent_pos[:2] + action[:2]).tolist()

            _Norm = np.linalg.norm(action[:2]) * np.linalg.norm(self.x_pos)
            rho = np.rad2deg(np.arcsin(np.cross(action[:2], self.x_pos) / _Norm))
            alpha = np.arccos(np.dot(action[:2], self.x_pos) / _Norm)
            if rho > 0:
                alpha = - alpha
            next_agent_pos = np.array(next_agent_pos_xy + [alpha])


        # landmark dynamics, we store the ground truth landmark positions into "self.landmarks",
        # using the same parameters (self.B_mat, self.u_land) for predicting estimated landmark
        # positions (self.landmarks_estimate_pred) in the previous loop.
        self.landmarks = self.A_mat @ self.landmarks + self.B_mat @ self.u_land + np.random.normal(0, STD_motion, np.shape(self.landmarks))

        sensor_value = np.zeros([self.num_landmarks * 2])
        # landmarks estimation with sensor
        T_pose = state_to_T(next_agent_pos)
        for i in range(self.num_landmarks):
            q = T_pose[:2, :2].transpose() @ (self.landmarks.flatten()[i * 2: i * 2 + 2] - T_pose[:2, 2])
            # print(q, triangle_SDF(q, np.pi/3, RADIUS)[0])

            if triangle_SDF(q, np.pi/3, RADIUS)[0] <= 0:
                sensor_value[i * 2: i * 2 + 2] = self.landmarks[i * 2: i * 2 + 2].flatten()\
                                                 + np.random.normal(0, STD_sensor, [2, ])
            else:
                sensor_value[i * 2: i * 2 + 2] = self.landmarks_estimate_pred[i * 2: i * 2 + 2].flatten()

        H_mat = np.eye(self.num_landmarks * 2)  # sensor matrix, I set it to be an identity matrix
        R_mat = np.eye(self.num_landmarks * 2) * STD_sensor ** 2  # sensor uncertainty covariance matrix
        S_mat = H_mat @ np.linalg.inv(self.info_mat) @ H_mat.transpose() + R_mat
        kalman_gain = np.linalg.inv(self.info_mat) @ H_mat.transpose() @ np.linalg.inv(S_mat)

        self.landmarks_estimate = self.landmarks_estimate_pred.flatten() + (kalman_gain @ (sensor_value - (H_mat @ self.landmarks_estimate_pred).flatten())).flatten()

        # reward
        V_jj_inv = diff_FoV_land_triangle(next_agent_pos, self.landmarks_estimate, self.num_landmarks, RADIUS, KAPPA,
                                 STD_sensor).astype(np.float32)
        next_info_mat = self.info_mat + V_jj_inv  # update info
        reward = float(slogdet(next_info_mat)[1] - slogdet(self.info_mat_update)[1])  # slogdet(Y_{k+1}) - slogdet(Y_k) for model-free stage-wise reward
        self.info_mat_update = next_info_mat

        '''
        Next landmark pos trick:
        To make the landmarks better controllable, I picked time-variant u_land values from a uniform distribution.
        At the same time, this motion model is known to the agent so it wouldn't affect the Kalman filter
        Real landmark motion model: x_{k+1} = A @ x_k + B @ u_k + w
        Prediction: x^_{k+1} = A @ x^_k + B @ u_k
        while w is the gaussian noise with STD_motion which is unknown by the agent.
        '''
        self.u_land = np.random.uniform(-1, 1, size=(self.num_landmarks * 2, 1)) + self.U

        # Estimated landmark positions (x^_{k+1}) after moving for the next time step, which is used as an observation
        # to get rid of incorporating the motion model into our RL policy model.
        # We also update the information matrix here.
        self.landmarks_estimate_pred = self.A_mat @ self.landmarks_estimate.flatten() + (self.B_mat @ self.u_land).flatten()
        self.Q_mat = np.eye(self.num_landmarks * 2) * STD_motion ** 2
        next_info_mat = np.linalg.inv(self.A_mat @ np.linalg.inv(next_info_mat) @ self.A_mat.T + self.Q_mat)

        # terminate at time
        done = False
        if self.current_step >= self.total_step-1:
            done = True

        # info
        if self.for_comparison == False:
            info = {'info_mat': next_info_mat}
        else:
            info = np.mean(np.abs(self.landmarks.flatten() - self.landmarks_estimate))

        # update variables
        self.agent_pos = next_agent_pos
        self.info_mat = next_info_mat

        # update state
        self.state = np.hstack([
            self.agent_pos,
            self.info_mat.diagonal(),
            self.padding,
            self.landmarks_estimate_pred,
            self.padding,
            self.mask
        ]).astype(np.float32)
        # print("state:", self.state)

        # record history poses
        self.history_poses.append(self.agent_pos)
        # print(np.sum(np.abs(self.landmarks.flatten() - self.landmarks_estimate)))

        return self.state, reward, done, info

    def reset(self, init_agent_landmarks=None):
        # landmark and info_mat init
        self.num_landmarks = np.random.randint(3, 8)  # randomized number for landmarks
        self.total_step = self.num_landmarks * 3
        self.padding = np.array([0.] * 2 * (self.max_num_landmarks - self.num_landmarks))
        self.mask = np.array([True] * self.num_landmarks + [False] * (self.max_num_landmarks - self.num_landmarks))
        self.info_mat_init = np.diag([.5] * self.num_landmarks * 2).astype(np.float32)
        self.info_mat = self.info_mat_init.copy()
        self.info_mat_update = self.info_mat
        # an extremely large value which guarantee this landmark's position has much lower uncertainty
        # self.random_serial = np.random.randint(0, self.num_landmarks)

        # if self.special_case == True:
        #     self.info_mat[self.random_serial * 2, self.random_serial * 2], \
        #     self.info_mat[self.random_serial * 2 + 1, self.random_serial * 2 + 1] = 25, 25

        if self.for_comparison == False:
            self.bound = 10 * self.num_landmarks / 5
            lx = np.random.uniform(low=-self.bound, high=self.bound, size=(self.num_landmarks, 1))
            ly = np.random.uniform(low=-self.bound, high=self.bound, size=(self.num_landmarks, 1))
            self.landmarks = np.concatenate((lx, ly), 1).reshape(self.num_landmarks*2, 1)
            self.agent_pos = np.array([random.uniform(-self.bound-2, self.bound+2), random.uniform(-self.bound-2, self.bound+2), random.uniform(-np.pi, np.pi)])
            if self.motion_model == 1:
                self.U = np.zeros((self.num_landmarks * 2, 1))
            elif self.motion_model == 2:
                # in this case, all landmarks move in one similar direction with Gaussian motion noise
                self.U = np.array(np.random.uniform(-.8, .8, size=(2, )).tolist()
                                  * self.num_landmarks).reshape(self.num_landmarks * 2, 1)

        else:
            self.landmarks = np.array(init_agent_landmarks[0])
            self.agent_pos = np.array(init_agent_landmarks[1])

        # if self.special_case == True:
        #     self.random_serial = 2
        #     self.info_mat[self.random_serial * 2, self.random_serial * 2], \
        #     self.info_mat[self.random_serial * 2 + 1, self.random_serial * 2 + 1] = 5, 5
        #     self.landmarks = np.array([[-3.735633361080981], [-5.389339529873403], [9.30832441039854], [-2.102614130995417], [1.7693017013749888],
        #      [2.376171291827143], [3.1933682413882565], [-0.5026496918211603], [0.6641250783329653], [-0.5973562148846963]])
        #     self.agent_pos = np.array([1.1351943561390905, -0.7867490956842902, 0.0])

        # self.landmarks = np.array([[7.277112118464629], [-2.0788059438541246], [-7.649362880759338], [1.3084262371701794], [0.34758214308228297],
        #  [-6.334403275718428], [-7.358637873096933], [-7.103044813132455], [4.337193623851874], [-0.23887438702090869]])
        # self.agent_pos = np.array([-0.5775490486001775, 1.7617277810112522, 0.0])

        self.landmarks_estimate = self.landmarks + np.random.normal(0, STD_sensor, np.shape(self.landmarks))

        # landmarks control vector
        # landmarks' control motion for the initial time step, note it's varying over time
        self.A_mat = np.eye(self.num_landmarks * 2)
        self.u_land = np.random.uniform(-1, 1, size=(self.num_landmarks * 2, 1)) + self.U
        self.B_mat = np.eye(self.num_landmarks * 2)

        self.landmarks_estimate_pred = self.A_mat @ self.landmarks_estimate + self.B_mat @ self.u_land
        self.Q_mat = np.eye(self.num_landmarks * 2) * STD_motion ** 2
        self.info_mat = np.linalg.inv(self.A_mat @ np.linalg.inv(self.info_mat) @ self.A_mat.T + self.Q_mat)

        # state init
        self.state = np.hstack([
            self.agent_pos,
            self.info_mat.diagonal(),
            self.padding,
            self.landmarks_estimate_pred.flatten(),
            self.padding,
            self.mask
        ]).astype(np.float32)

        # step counter init
        self.current_step = -1

        # plot
        self.history_poses = [self.agent_pos]
        self.history_actions = []
        if self.test == True:
            self.fig = plt.figure(1)
            self.ax = self.fig.gca()

        return self.state

    def _plot(self, legend, title='trajectory'):

        # plot agent trajectory
        plt.tick_params(labelsize=15)
        history_poses = np.array(self.history_poses)
        self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='black', linewidth=3, label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=70, c='red', label="start")
        self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='s', s=70, c='red', label="end")

        self.ax.scatter(history_poses[-1, 0] + np.cos(history_poses[-1, 2])*0.5,
                     history_poses[-1, 1] + np.sin(history_poses[-1, 2])*0.5, marker='o', c='black')

        # plot landmarks
        if self.special_case == False:
            self.ax.scatter(self.landmarks[list(range(0, self.num_landmarks*2, 2)), :],
                        self.landmarks[list(range(1, self.num_landmarks*2+1, 2)), :], s=50, c='blue', label="landmark")
        if self.special_case == True:
            self.ax.scatter(self.landmarks[list(range(0, self.num_landmarks * 2, 2)), :],
                            self.landmarks[list(range(1, self.num_landmarks * 2 + 1, 2)), :], s=50, c='blue',
                            label="landmark_0.5")
            self.ax.scatter(self.landmarks[2 * self.random_serial, :],
                        self.landmarks[2 * self.random_serial + 1, :], s=50, c='green',
                        label="landmark_25")

        # annotate theta value to each position point
        # for i in range(0, len(self.history_poses)-1):
        #     self.ax.annotate(round(self.history_actions[i][2], 4), history_poses[i, :2])

        # axes
        self.ax.set_xlabel("x", fontdict={'size': 20})
        self.ax.set_ylabel("y", fontdict={'size': 20})

        # title
        # self.ax.set_title(title, fontdict={'size': 16})

        self.ax.set_facecolor('whitesmoke')
        plt.grid(alpha=0.4)
        # legend
        if legend == True:
            self.ax.legend()
            plt.legend(prop={'size': 14})


    def render(self, mode='human'):
        if mode == 'terminal':
            print(f">>>> step {self.current_step} <<<<")
            print(f"agent pos = {self.agent_pos}")
            print("information matrix:")
            print(self.info_mat)
        elif mode == 'human':
            # clear axes
            self.ax.cla()

            # plot
            self._plot(True)

            # display
            plt.draw()
            plt.pause(0.5)

        else:
            raise NotImplementedError

    def save_plot(self, name='default.png', title='trajectory', legend=False):
        self.ax.cla()
        self._plot(legend, title=title)
        self.fig.savefig(name, bbox_inches = 'tight')

    def close (self):
        plt.close('all')

if __name__ == '__main__':
    num_eps = 4
    gamma = 0.98
    ## create env
    env = SimpleQuadrotor()
    check_env(env)

    ## testing actions
    # eps0: diagonal /
    actions0 = np.array([
        [-0.5, -0.5, 0.],
        [-0.5, -0.5, 0.],
        [1.0, 1.0, 0.],
        [0.5, 0.5, 0.],
        [0.5, 0.5, 0.]
    ], dtype=np.float32).T/2
    # eps1: no move
    actions1 = np.zeros((5, 3), dtype=np.float32).T/2
    # eps2: diagonal \
    actions2 = np.array([
        [-0.5, 0.5, 0.],
        [-0.5, 0.5, 0.],
        [1.0, -1.0, 0.],
        [0.5, -0.5, 0.],
        [0.5, -0.5, 0.]
    ], dtype=np.float32).T/2
    action_spaces = [actions0, actions1, actions2]

    ## run examples
    for eps in range(num_eps):
        obs = env.reset()
        done = False
        total_reward = 0

        print(f"\n------ Eps {eps} ------")
        print(f"init state = {obs}")

        while not done:
            # get action
            if eps < len(action_spaces):
                action = action_spaces[eps][:, env.current_step+1]
            else:
                action = env.action_space.sample()

            # step env
            obs, r, done, info = env.step(action)

            # calc return
            total_reward += r * (gamma ** env.current_step)

            # render
            env.render(mode='human')
            print(f"reward = {r}")

        # summary
        print("---")
        print(env.history_actions)
        print(f"return = {total_reward}")
        env.save_plot(name=f'plots/eps{eps}.png', title=f'return = {total_reward}')
    env.close()
