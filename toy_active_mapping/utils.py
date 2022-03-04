import numpy as np

from scipy.special import erf

def state_to_T(state):
    return np.array([[np.cos(state[2]), -np.sin(state[2]), state[0]], [np.sin(state[2]), np.cos(state[2]), state[1]],
                     [0, 0, 1]])

def T_to_state(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def SE2_motion(T, u, dt):
    exp_uhat = exp_hat(u,dt)
    T_next = T @ exp_uhat
    return T_next


def exp_hat(u, dt):
    u_hat = np.array([[0, -u[2], u[0]], [u[2], 0, u[1]], [0, 0, 0]])
    u_hat_sq = u_hat @ u_hat
    if u[2] == 0:
        exp_uhat = np.identity(3) + dt * u_hat + dt**2 * u_hat_sq / 2
    else:
        exp_uhat = np.identity(3) + np.sin(u[2] * dt) / u[2] * u_hat + (1 - np.cos(u[2] * dt)) / u[2]**2 * u_hat_sq
    return exp_uhat

def unicycle_dyn(state,u,dt):
    T = state_to_T(state)
    T_next = SE2_motion(T,u,dt)
    state_next = T_to_state(T_next)
    return state_next


def l_function(x, psi, r, p_x):
    if x < 0:
        l_1_low, l_2_up = - x / np.tan(psi), x / np.tan(psi)
    elif 0 <= x < p_x:
        l_1_low, l_2_up = 0, 0
    elif p_x <= x < r:
        l_1_low = np.tan(np.pi / 4 + psi / 2) * x - r / np.cos(psi)
        l_2_up = - np.tan(np.pi / 4 + psi / 2) * x + r / np.cos(psi)
    else:
        l_1_low, l_2_up = r * np.tan(psi), -r * np.tan(psi)

    if x < r:
        l_1_up, l_2_low = - (x - r) / np.tan(psi) + r * np.tan(psi), (x - r) / np.tan(psi) - r * np.tan(psi)
    else:
        l_1_up, l_2_low = r * np.tan(psi), -r * np.tan(psi)
    return l_1_low, l_1_up, l_2_low, l_2_up


def triangle_SDF(q, psi, r):
    x, y = q[0], q[1]
    p_x = r / (1 + np.sin(psi))

    a_1, a_2, a_3 = np.array([-1, 1 / np.tan(psi)]), np.array([-1, -1 / np.tan(psi)]), np.array([1, 0])
    b_1, b_2, b_3 = 0, 0, -r
    q_1, q_2, q_3 = np.array([r, r * np.tan(psi)]), np.array([r, -r * np.tan(psi)]), np.array([0, 0])
    l_1_low, l_1_up, l_2_low, l_2_up = l_function(x, psi, r, p_x)
    if y >= l_1_up:
        # P_1
        SDF, Grad = np.linalg.norm(q - q_1), (q - q_1) / np.linalg.norm(q - q_1)
    elif l_1_low <= y < l_1_up:
        # D_1
        SDF, Grad = (a_1 @ q + b_1) / np.linalg.norm(a_1), a_1 / np.linalg.norm(a_1)
    elif x < 0 and l_2_up <= y < l_1_low:
        # P_3
        SDF, Grad = np.linalg.norm(q - q_3), (q - q_3) / np.linalg.norm(q - q_3)
    elif x > p_x and l_2_up <= y < l_1_low:
        # D_3
        SDF, Grad = (a_3 @ q + b_3) / np.linalg.norm(a_3), a_3 / np.linalg.norm(a_3)
    elif y < l_2_up and y > l_2_low:
        # D_2
        SDF, Grad = (a_2 @ q + b_2) / np.linalg.norm(a_2), a_2 / np.linalg.norm(a_2)
    else:
        # P_2
        SDF, Grad = np.linalg.norm(q - q_2), (q - q_2) / np.linalg.norm(q - q_2)
    return SDF, Grad

def circle_SDF(q, r):
    SDF, Grad = np.linalg.norm(q) ** 2 - r ** 2, 2 * q
    return SDF, Grad


def Gaussian_CDF(x, kap):
    Psi = (1 + erf(x / (np.sqrt(2) * kap) - 2)) / 2
    Psi_der = 1 / (np.sqrt(2 * np.pi) * kap) * np.exp(- (x / (np.sqrt(2) * kap) - 2) ** 2)
    return Psi, Psi_der

def diff_FoV_land(x,y,n_y,r,kap,std):
    V_jj_inv = np.zeros((2 * n_y, 2 * n_y)) 
    for j in range(n_y):
        q = x[:2] - y[j * 2: j * 2 + 2]
        SDF, Grad = circle_SDF(q,r)
        Phi, Phi_der =  Gaussian_CDF(SDF, kap)
        V_jj_inv[2 * j, 2 * j] = 1 / (std ** 2) * (1 - Phi)
        V_jj_inv[2 * j + 1, 2 * j+1] = 1 / (std ** 2) * (1 - Phi)
    return V_jj_inv
