import numpy as np
import os
import yaml
import cv2
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
from scipy import stats

from rl_mapping.utilities.utils import state_to_T, T_to_state, SE2_motion, triangle_SDF, Gaussian_CDF, exp_hat,\
                                       Grad_exp_hat

from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.utilities.util import rc_to_xy


def create_gradient_agent_from_params(params_filename, sensor_range, angular_range, sigma2):

    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if params['footprint']['type'] == 'tricky_circle':
        footprint_points = get_tricky_circular_footprint()
    elif params['footprint']['type'] == 'tricky_oval':
        footprint_points = get_tricky_oval_footprint()
    elif params['footprint']['type'] == 'circle':
        rotation_angles = np.arange(0, 2 * np.pi, 4 * np.pi / 180)
        footprint_points = \
            params['footprint']['radius'] * np.array([np.cos(rotation_angles), np.sin(rotation_angles)]).T
    elif params['footprint']['type'] == 'pixel':
        footprint_points = np.array([[0., 0.]])
    else:
        footprint_points = None
        assert False and "footprint type specified not supported."

    footprint = CustomFootprint(footprint_points=footprint_points,
                                angular_resolution=params['footprint']['angular_resolution'],
                                inflation_scale=params['footprint']['inflation_scale'])

    horizon = params['horizon']
    dt = params['dt']
    num_iter = params['num_iter']
    kappa = params['kappa']
    v_0 = params['v_0']
    alpha = np.array([params['alpha']['_1'], params['alpha']['_2'], params['alpha']['_3']])

    gradient_agent = GradientAgent(footprint=footprint,
                                   max_range=sensor_range,
                                   angular_range=angular_range,
                                   horizon=horizon,
                                   dt=dt,
                                   num_iter=num_iter,
                                   kappa=kappa,
                                   sigma2=sigma2,
                                   v_0=v_0,
                                   alpha=alpha)

    return gradient_agent


class GradientAgent:
    def __init__(self, footprint, max_range, angular_range, horizon, dt, num_iter,
                 kappa, sigma2, v_0, alpha):

        self._max_range = max_range
        self._angular_range = angular_range

        self._footprint = footprint
        self._footprint_masks = None
        self._footprint_outline_coords = None
        self._footprint_mask_radius = None

        self._dt = dt
        self._tf = int(np.floor(horizon / dt))

        self._num_iter = num_iter

        self._kappa = kappa
        self._sigma2= sigma2

        self._v_0 = v_0

        self._alpha = alpha

        self._downsample_coeff = 25

    def _initialize_motion(self):
        u = np.zeros((3, self._tf))
        u[0, :] = self._v_0
        u[2, :] = np.random.uniform(low=-np.pi/10, high=np.pi/10, size=self._tf)
        return u

    def get_footprint(self):
        return self._footprint

    def plan(self, state, distrib_map, face_unknown=False):
        print("Planning started...")

        T = np.zeros((3, 3, self._tf + 1))

        if face_unknown:
            occ_map = (distrib_map.data[:, :, 0] == 0) * 255
            img = cv2.threshold(occ_map.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1]
            num_labels, labels = cv2.connectedComponents(img)
            biggest_region_label = stats.mode(labels[labels > 0], axis=None)[0]
            center_px = np.floor(np.mean(np.nonzero(labels == biggest_region_label), axis=1))
        else:
            center_px = np.floor(np.array([distrib_map.get_shape()[0], distrib_map.get_shape()[1]]) / 2)

        center = rc_to_xy(center_px, distrib_map)
        state[2] = np.arctan2(center[1] - state[1], center[0] - state[0])
        T[:, :, 0] = state_to_T(state)

        u = self._initialize_motion()

        path_init_sampled = np.zeros((3, self._tf))
        T_init_sampled = np.zeros((3, 3, self._tf + 1))
        T_init_sampled[:, :, 0] = state_to_T(state)
        # Find the optimal trajectory
        for t in range(self._tf):
            T_init_sampled[:, :, t + 1] = SE2_motion(T_init_sampled[:, :, t], u[:, t], self._dt)
            path_init_sampled[:, t] = T_to_state(T_init_sampled[:, :, t + 1])

        reward = []
        for it in range(self._num_iter):
            print("Iteration: ", it)
            S = distrib_map.data[:, :, 1].copy()

            # Forward pass to evaluate reward
            for t in range(self._tf):
                T[:, :, t + 1] = SE2_motion(T[:, :, t], u[:, t], self._dt)
                for i in range(0, S.shape[0], self._downsample_coeff):
                    for j in range(0, S.shape[1], self._downsample_coeff):
                        p_ij = rc_to_xy(np.array([i, j]), distrib_map)
                        q = T[:2, :2, t].transpose() @ (p_ij - T[:2, 2, t])
                        d, _ = triangle_SDF(q, self._angular_range / 2, self._max_range)
                        Phi, _ = Gaussian_CDF(d, self._kappa)
                        S[i, j] += 1 / (self._sigma2) * (1 - Phi)

            reward.append(np.sum(np.log(S[::self._downsample_coeff, ::self._downsample_coeff])))
            print("Reward = ", reward[-1])

            # Gradient backpropagation
            for t in range(self._tf - 1, -1, -1):
                Grad_exp_1, Grad_exp_2, Grad_exp_3 = Grad_exp_hat(u[:, t], self._dt)
                dTk_dut_1, dTk_dut_2, dTk_dut_3 = T[:, :, t] @ Grad_exp_1, T[:, :, t] @ Grad_exp_2,\
                                                  T[:, :, t] @ Grad_exp_3
                dR_du_1, dR_du_2, dR_du_3 = 0, 0, 0

                for k in range(t + 1, self._tf):
                    for i in range(0, S.shape[0], self._downsample_coeff):
                        for j in range(0, S.shape[1], self._downsample_coeff):
                            p_ij = rc_to_xy(np.array([i, j]), distrib_map)
                            q = T[:2, :2, k].transpose() @ (p_ij - T[:2, 2, k])
                            d, d_der = triangle_SDF(q, self._angular_range / 2, self._max_range)
                            _, Phi_der = Gaussian_CDF(d, self._kappa)

                            dq_du_1 = dTk_dut_1[:2, :2].transpose() @ (p_ij - T[:2, 2, k]) - \
                                      T[:2, :2, k].transpose() @ dTk_dut_1[:2, 2]
                            dq_du_2 = dTk_dut_2[:2, :2].transpose() @ (p_ij - T[:2, 2, k]) - \
                                      T[:2, :2, k].transpose() @ dTk_dut_2[:2, 2]
                            dq_du_3 = dTk_dut_3[:2, :2].transpose() @ (p_ij - T[:2, 2, k]) - \
                                      T[:2, :2, k].transpose() @ dTk_dut_3[:2, 2]

                            dR_du_1 += (-1 * Phi_der * d_der @ dq_du_1) / (S[i, j] * self._sigma2)
                            dR_du_2 += (-1 * Phi_der * d_der @ dq_du_2) / (S[i, j] * self._sigma2)
                            dR_du_3 += (-1 * Phi_der * d_der @ dq_du_3) / (S[i, j] * self._sigma2)

                    exp_uhat = exp_hat(u[:, k], self._dt)
                    dTk_dut_1, dTk_dut_2, dTk_dut_3 = dTk_dut_1 @ exp_uhat, dTk_dut_2 @ exp_uhat, dTk_dut_3 @ exp_uhat

                u[0, t] += self._alpha[0] * dR_du_1
                u[1, t] += self._alpha[1] * dR_du_2
                u[2, t] += self._alpha[2] * dR_du_3

        path = np.zeros((3, self._tf * 5))
        T_final = np.zeros((3, 3, self._tf * 5 + 1))
        T_final[:, :, 0] = state_to_T(state)
        # Find the optimal trajectory
        for t in range(self._tf * 5):
            T_final[:, :, t + 1] = SE2_motion(T_final[:, :, t], u[:, np.floor(t / 5).astype(np.int)], self._dt / 5)
            path[:, t] = T_to_state(T_final[:, :, t + 1])

        path_final_sampled = np.zeros((3, self._tf))
        T_final_sampled = np.zeros((3, 3, self._tf + 1))
        T_final_sampled[:, :, 0] = state_to_T(state)
        # Find the optimal trajectory
        for t in range(self._tf):
            T_final_sampled[:, :, t + 1] = SE2_motion(T_final_sampled[:, :, t], u[:, t], self._dt)
            path_final_sampled[:, t] = T_to_state(T_final_sampled[:, :, t + 1])

        return path.transpose(), path_init_sampled.transpose(), path_final_sampled.transpose(), reward
