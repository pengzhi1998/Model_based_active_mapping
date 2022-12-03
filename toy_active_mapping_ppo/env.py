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
                 for_comparison=False, special_case=False, test=False, visualcomp=False):
        super(SimpleQuadrotor, self).__init__()

        # variables
        self.num_landmarks = num_landmarks
        self.test = test
        self.total_time = horizon
        self.step_size = STEP_SIZE
        self.total_step = math.floor(self.total_time / STEP_SIZE)
        self.bound = bound
        self.episode = 0
        self.history_poses_modelbased = [[[-1.149999976158142, -1.7000000476837158, -1.9900000095367432], [-1.69974946975708, -2.208278179168701, -2.8007428646087646], [-4.27127742767334, -2.3559494018554688, -3.3677175045013428], [-4.688115119934082, -2.027696132659912, -4.249593257904053], [-5.194333553314209, 1.4362142086029053, -4.884957313537598], [-5.009426593780518, 1.6938984394073486, -5.784665584564209], [-4.6100945472717285, 1.710342288017273, -6.699394226074219], [-3.864501953125, 0.8598446846008301, -7.569036483764648], [-3.9347124099731445, -2.4482622146606445, -8.181367874145508], [-4.590812683105469, -3.1260178089141846, -9.064925193786621]],
                                         [[-0.23000000417232513, 1.5399999618530273, -2.569999933242798],
                                          [-2.4652209281921387, -1.6528544425964355, -1.7931382656097412],
                                          [-4.086528301239014, -5.245113372802734, -2.1963791847229004],
                                          [-7.035381317138672, -7.385009765625, -2.831310987472534],
                                          [-8.743014335632324, -7.167046070098877, -3.705782890319824],
                                          [-11.005727767944336, -4.05103063583374, -4.462863922119141],
                                          [-10.916420936584473, -0.10777831077575684, -5.007201671600342],
                                          [-9.164227485656738, 3.464153289794922, -5.329686641693115],
                                          [-6.534713268280029, 6.459982872009277, -5.535834312438965],
                                          [-3.1865108013153076, 8.563847541809082, -5.908517837524414],
                                          [0.6061131954193115, 9.507128715515137, -6.170315742492676],
                                          [0.7562276721000671, 9.449602127075195, -7.127962589263916],
                                          [0.8617058396339417, 9.027420997619629, -8.090343475341797],
                                          [-1.2440288066864014, 5.725836753845215, -8.753113746643066],
                                          [-4.455248832702637, 3.3488640785217285, -8.82203197479248],
                                          [-7.91359281539917, 1.3728870153427124, -8.989327430725098]],
                                         [[1.5399999618530273, -1.899999976158142, 1.8300000429153442],
                                          [-1.0522727966308594, 0.9385453462600708, 2.7917566299438477],
                                          [-4.972511291503906, 1.0030066967010498, 3.458545207977295],
                                          [-7.725115776062012, -1.694014549255371, 4.375038146972656],
                                          [-8.396942138671875, -5.596652030944824, 4.7087883949279785],
                                          [-8.158711433410645, -9.293306350708008, 4.844701290130615],
                                          [-8.268288612365723, -11.899846076965332, 4.496047019958496],
                                          [-8.295351028442383, -11.930764198303223, 3.490757465362549],
                                          [-11.432254791259766, -11.814326286315918, 2.7182247638702393],
                                          [-13.950688362121582, -9.049277305603027, 1.900878667831421],
                                          [-14.539946556091309, -5.120270729064941, 1.5384478569030762],
                                          [-13.657492637634277, -1.264183759689331, 1.1531985998153687],
                                          [-10.824315071105957, 1.32832670211792, 0.3289419412612915],
                                          [-6.890037536621094, 1.4344112873077393, -0.2750266194343567],
                                          [-3.6323249340057373, -0.7655861377716064, -0.9129118323326111],
                                          [-0.44481992721557617, -3.0911102294921875, -0.34768927097320557],
                                          [2.679649829864502, -5.477355003356934, -0.9567691087722778],
                                          [5.099102973937988, -8.654967308044434, -0.8833014965057373],
                                          [6.829303741455078, -12.163270950317383, -1.3419486284255981],
                                          [6.302855491638184, -15.433151245117188, -2.1189029216766357],
                                          [3.0545527935028076, -17.5157470703125, -3.0240478515625],
                                          [-0.9331700801849365, -17.79505157470703, -3.119283676147461]]]

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
        self.visual_comp = visualcomp

    def step(self, action, init_agent_landmarks_moving_visual=False):
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
        if self.visual_comp == True:
            self.u_land = np.array(init_agent_landmarks_moving_visual[3][self.current_step]) + self.U
        else:
            self.u_land = np.random.uniform(-.5, .5, size=(self.num_landmarks * 2, 1)) + self.U

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
        if self.visual_comp == True:
            info = slogdet(self.info_mat_update)[1] / self.num_landmarks
        else:
            if self.for_comparison == False:
                info = {'info_mat': next_info_mat}
            else:
                info = slogdet(self.info_mat_update)[1]/self.num_landmarks

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
        print(self.landmarks_estimate, self.landmarks, "\n")

        return self.state, reward, done, info

    def reset(self, init_agent_landmarks=None):
        # landmark and info_mat init
        self.history_poses_current_model_based = np.array(self.history_poses_modelbased[self.episode])
        self.episode += 1
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

        if self.visual_comp == False:
            if self.for_comparison == False:
                self.bound = 10 * self.num_landmarks / 5
                lx = np.random.uniform(low=-self.bound, high=self.bound, size=(self.num_landmarks, 1))
                ly = np.random.uniform(low=-self.bound, high=self.bound, size=(self.num_landmarks, 1))
                self.landmarks = np.concatenate((lx, ly), 1).reshape(self.num_landmarks*2, 1)
                self.agent_pos = np.array([random.uniform(-self.bound*1.25, self.bound*1.25), random.uniform(-self.bound*1.25, self.bound+1.25), random.uniform(-np.pi, np.pi)])
                if self.motion_model == 1:
                    self.U = np.zeros((self.num_landmarks * 2, 1))
                elif self.motion_model == 2:
                    # in this case, all landmarks move in one similar direction with Gaussian motion noise
                    self.U = np.array(np.random.uniform(-.8, .8, size=(2, )).tolist()
                                      * self.num_landmarks).reshape(self.num_landmarks * 2, 1)

            else:
                self.landmarks = np.array(init_agent_landmarks[0])
                self.agent_pos = np.array(init_agent_landmarks[1])
                self.num_landmarks = int(np.shape(self.landmarks)[0] / 2)
                self.U = np.array(init_agent_landmarks[2])
                self.total_step = self.num_landmarks * 3
                self.padding = np.array([0.] * 2 * (self.max_num_landmarks - self.num_landmarks))
                self.mask = np.array([True] * self.num_landmarks + [False] * (self.max_num_landmarks - self.num_landmarks))
                self.info_mat_init = np.diag([.5] * self.num_landmarks * 2).astype(np.float32)
                self.info_mat = self.info_mat_init.copy()
                self.info_mat_update = self.info_mat

        else:
            self.landmarks = np.array(init_agent_landmarks[0])
            self.agent_pos = np.array(init_agent_landmarks[1])
            self.num_landmarks = int(np.shape(self.landmarks)[0] / 2)
            self.U = np.array(init_agent_landmarks[2])
            self.total_step = self.num_landmarks * 3
            self.padding = np.array([0.] * 2 * (self.max_num_landmarks - self.num_landmarks))
            self.mask = np.array([True] * self.num_landmarks + [False] * (self.max_num_landmarks - self.num_landmarks))
            self.info_mat_init = np.diag([.5] * self.num_landmarks * 2).astype(np.float32)
            self.info_mat = self.info_mat_init.copy()
            self.info_mat_update = self.info_mat
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
        if self.visual_comp == True:
            self.u_land = np.array(init_agent_landmarks[3][0]) + self.U
        else:
            self.u_land = np.random.uniform(-.5, .5, size=(self.num_landmarks * 2, 1)) + self.U
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
        if self.visual_comp == False:
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
        else:
            history_poses_modelbased = self.history_poses_current_model_based[: self.current_step + 2,:]


            # plot agent trajectory
            plt.tick_params(labelsize=15)
            history_poses = np.array(self.history_poses)
            self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='green', linewidth=2, label='model-free traj', alpha=0.7)

            # plot agent trajectory start & end
            self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=70, c='green', label="model-free start")
            self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='o', s=70, c='green', label="model-free end")

            # self.ax.scatter(history_poses[-1, 0] + np.cos(history_poses[-1, 2]) * 0.5,
            #                 history_poses[-1, 1] + np.sin(history_poses[-1, 2]) * 0.5, marker='o', c='black')

            sensor_x_1 = np.array([history_poses[-1, 0], history_poses[-1, 0] + np.cos(history_poses[-1, 2] + np.pi/4) * 3])
            sensor_y_1 = np.array([history_poses[-1, 1], history_poses[-1, 1] + np.sin(history_poses[-1, 2] + np.pi/4) * 3])
            sensor_x_2 = np.array(
                [history_poses[-1, 0], history_poses[-1, 0] + np.cos(history_poses[-1, 2] - np.pi / 4) * 3])
            sensor_y_2 = np.array(
                [history_poses[-1, 1], history_poses[-1, 1] + np.sin(history_poses[-1, 2] - np.pi / 4) * 3])
            sensor_x_3 = np.array(
                [history_poses[-1, 0] + np.cos(history_poses[-1, 2] + np.pi/4) * 3, history_poses[-1, 0] + np.cos(history_poses[-1, 2] - np.pi / 4) * 3])
            sensor_y_3 = np.array(
                [history_poses[-1, 1] + np.sin(history_poses[-1, 2] + np.pi/4) * 3, history_poses[-1, 1] + np.sin(history_poses[-1, 2] - np.pi / 4) * 3])
            self.ax.plot(sensor_x_1, sensor_y_1, c='black', linewidth=3, alpha=0.2)
            self.ax.plot(sensor_x_2, sensor_y_2, c='black', linewidth=3, alpha=0.2)
            self.ax.plot(sensor_x_3, sensor_y_3, c='black', linewidth=3, alpha=0.2)








            self.ax.plot(history_poses_modelbased[:, 0], history_poses_modelbased[:, 1], c='red', linewidth=2, label='model-based traj', alpha=0.7)

            # plot agent trajectory start & end
            self.ax.scatter(history_poses_modelbased[0, 0], history_poses_modelbased[0, 1], marker='>', s=70, c='red', label="model-based start")
            self.ax.scatter(history_poses_modelbased[-1, 0], history_poses_modelbased[-1, 1], marker='o', s=70, c='red', label="model-based end")

            # self.ax.scatter(history_poses_modelbased[-1, 0] + np.cos(history_poses_modelbased[-1, 2]) * 0.5,
            #                 history_poses_modelbased[-1, 1] + np.sin(history_poses_modelbased[-1, 2]) * 0.5, marker='o', c='black')
            sensor_x_1_modelbased = np.array(
                [history_poses_modelbased[-1, 0], history_poses_modelbased[-1, 0] + np.cos(history_poses_modelbased[-1, 2] + np.pi / 4) * 3])
            sensor_y_1_modelbased = np.array(
                [history_poses_modelbased[-1, 1], history_poses_modelbased[-1, 1] + np.sin(history_poses_modelbased[-1, 2] + np.pi / 4) * 3])
            sensor_x_2_modelbased = np.array(
                [history_poses_modelbased[-1, 0], history_poses_modelbased[-1, 0] + np.cos(history_poses_modelbased[-1, 2] - np.pi / 4) * 3])
            sensor_y_2_modelbased = np.array(
                [history_poses_modelbased[-1, 1], history_poses_modelbased[-1, 1] + np.sin(history_poses_modelbased[-1, 2] - np.pi / 4) * 3])
            sensor_x_3_modelbased = np.array(
                [history_poses_modelbased[-1, 0] + np.cos(history_poses_modelbased[-1, 2] + np.pi / 4) * 3,
                 history_poses_modelbased[-1, 0] + np.cos(history_poses_modelbased[-1, 2] - np.pi / 4) * 3])
            sensor_y_3_modelbased = np.array(
                [history_poses_modelbased[-1, 1] + np.sin(history_poses_modelbased[-1, 2] + np.pi / 4) * 3,
                 history_poses_modelbased[-1, 1] + np.sin(history_poses_modelbased[-1, 2] - np.pi / 4) * 3])
            self.ax.plot(sensor_x_1_modelbased, sensor_y_1_modelbased, c='black', linewidth=3, alpha=0.2)
            self.ax.plot(sensor_x_2_modelbased, sensor_y_2_modelbased, c='black', linewidth=3, alpha=0.2)
            self.ax.plot(sensor_x_3_modelbased, sensor_y_3_modelbased, c='black', linewidth=3, alpha=0.2)



            # plot landmarks
            if self.special_case == False:
                self.ax.scatter(self.landmarks[list(range(0, self.num_landmarks * 2, 2)), :],
                                self.landmarks[list(range(1, self.num_landmarks * 2 + 1, 2)), :], marker='s', s=50, c='blue',
                                label="landmark")
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
            if self.episode == 1:
                plt.xlim((-15, 5))
                plt.ylim((-15, 5))
            elif self.episode == 2:
                plt.xlim((-20, 10))
                plt.ylim((-10, 20))
            else:
                plt.xlim((-20, 20))
                plt.ylim((-20, 20))
            plt.gca().set_aspect(1)
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
            plt.pause(0.3)

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
