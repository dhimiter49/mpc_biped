import sys
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from qpsolvers import solve_qp


# Constants
MAX_CONSTRAINT = 0.17
MIN_CONSTRAINT = -MAX_CONSTRAINT
MAX_MIN_CONSTRAINT = -0.03
MIN_MAX_CONSTRAINT = -MAX_MIN_CONSTRAINT
START_TIME = 2.5  # in s
END_TIME = 7.5  # in s
G = 9.81  # gravity
H_COM = 0.8  # height of CoM
R = 1
Q = 1e6
DT = 0.005  # in s
N = 300  # lookahead
T = 9 # in s
PERIOD = 1 // DT  # in s
SHORT_PERIOD = 0.8 // DT  # in s
DIFF_PERIOD = (PERIOD - SHORT_PERIOD) // 2
factor = float(sys.argv[sys.argv.index("-f") + 1]) if "-f" in sys.argv else 1


# Iteration dynamics
x_0 = np.zeros(3)
DYN_MAT = np.array([[1, DT, DT ** 2 / 2], [0, 1, DT], [0, 0, 1]])
DYN_JERK = np.array([[DT ** 3 / 6, DT ** 2 / 2, DT]])
Z_COMP = np.array([1, 0, -H_COM / G])
next_x = lambda x, x_jerk : (DYN_MAT @ x + x_jerk * DYN_JERK).flatten()
compute_z = lambda x : np.sum(Z_COMP * x)
def iterative_x_k(x_k, jerks):
    q = np.zeros(N)
    for i, jerk in enumerate(jerks):
        next_x_k = next_x(x_k, jerk)
        q[i] = np.dot(DYN_MAT[0], x_k) * DYN_JERK[0][0]
        x_k = next_x_k
    return q


# Constraints/bounds
Z_MAX = MAX_CONSTRAINT * np.ones(int(T / DT) + N)
Z_MIN = MIN_CONSTRAINT * np.ones(int(T / DT) + N)
Z_MAX[int(START_TIME / DT) : int(START_TIME / DT + SHORT_PERIOD)] = MAX_MIN_CONSTRAINT
Z_MAX[
    int(START_TIME / DT + SHORT_PERIOD + PERIOD) :
    int(START_TIME / DT + 2 * SHORT_PERIOD + PERIOD)
] = MAX_MIN_CONSTRAINT
Z_MAX[
    int(START_TIME / DT + 2 * SHORT_PERIOD + 2 * PERIOD) :
    int(START_TIME / DT + 3 * SHORT_PERIOD + 2 * PERIOD)
] = MAX_MIN_CONSTRAINT

Z_MIN[
    int(START_TIME / DT + SHORT_PERIOD + DIFF_PERIOD) :
    int(START_TIME / DT + 2 * SHORT_PERIOD + DIFF_PERIOD)
] = MIN_MAX_CONSTRAINT
Z_MIN[
    int(START_TIME / DT + 2 * SHORT_PERIOD + PERIOD + DIFF_PERIOD) :
    int(START_TIME / DT + 3 * SHORT_PERIOD + PERIOD + DIFF_PERIOD)
] = MIN_MAX_CONSTRAINT
Z_MIN[
    int(START_TIME / DT + 3 * SHORT_PERIOD + 2 * PERIOD + DIFF_PERIOD) :
    int(START_TIME / DT + 4 * SHORT_PERIOD + 2 * PERIOD + DIFF_PERIOD)
] = MIN_MAX_CONSTRAINT

P_x = np.ones((N, 3))
P_x[:, 1] *= np.array([DT * i for i in range(1, N + 1)])
P_x[:, 2] *= np.array([(DT ** 2) * (i ** 2) / 2 - H_COM / G for i in range(1, N + 1)])
P_xx = np.ones((N, 3))
P_xx[:, 1] *= np.array([DT * i for i in range(1, N + 1)])
P_xx[:, 2] *= np.array([(DT ** 2) * (i ** 2) / 2 for i in range(1, N + 1)])

P_u = scipy.linalg.toeplitz(
    np.array(
        [(1 + 3 * i + 3 * i ** 2) * DT ** 3 / 6 - DT * H_COM / G for i in range(N)],
        dtype=np.float128,
    ),
    np.zeros(N)
)
P_u_inv = scipy.linalg.inv(P_u)

P_ux = scipy.linalg.toeplitz(
    np.array(
        [(1 + 3 * i + 3 * i ** 2) * DT ** 3 / 6 for i in range(N)],
        dtype=np.float128,
    ),
    np.zeros(N)
)
def get_x_ks(x_k, jerks):
    return P_xx @ x_k + P_ux @ jerks


def calc_z_ref():
    z_ref = (Z_MAX + Z_MIN) / 2
    if "-sw" in sys.argv:
        z_ref = np.convolve(z_ref, np.ones(20)/20, mode="valid")
        z_ref = np.concatenate((
            (MIN_CONSTRAINT + MAX_CONSTRAINT) / 2 * np.ones(10),
            z_ref,
            (MIN_CONSTRAINT + MAX_CONSTRAINT) / 2 * np.ones(9),
        ))
    elif "-gf" in sys.argv:
        z_ref = gaussian_filter1d(z_ref, sigma=7)
    elif "-pr" in sys.argv:  # perfect reference, smoothed step func between min max
        for i in range(len(z_ref)):
            if i < 10 or i > len(z_ref) - 10:
                continue
            if np.mean(np.abs(Z_MAX[i : i + 10])) > np.mean(np.abs(Z_MIN[i - 10 : i])):
                z_ref[i] = (MAX_CONSTRAINT + MIN_MAX_CONSTRAINT) / 2
            elif np.mean(np.abs(Z_MAX[i : i + 10])) < np.mean(np.abs(Z_MIN[i - 10 : i])):
                z_ref[i] = (MIN_CONSTRAINT + MAX_MIN_CONSTRAINT) / 2
            elif np.mean(np.abs(Z_MAX[i -10 : i])) > np.mean(np.abs(Z_MIN[i : i + 10])):
                z_ref[i] = (MAX_CONSTRAINT + MIN_MAX_CONSTRAINT) / 2
            elif np.mean(np.abs(Z_MAX[i - 10 : i])) < np.mean(np.abs(Z_MIN[i : i + 10])):
                z_ref[i] = (MIN_CONSTRAINT + MAX_MIN_CONSTRAINT) / 2
        z_ref = gaussian_filter1d(z_ref, sigma=5)

    if "-tref" in sys.argv:
        plt.plot(
            np.arange(0, T, DT), Z_MAX[:int(T / DT)], color='red', linestyle="dashed"
        )
        plt.plot(
            np.arange(0, T, DT), Z_MIN[:int(T / DT)], color='blue', linestyle="dashed"
        )
        plt.plot(
            np.arange(0, T, DT), z_ref[:int(T / DT)], color='green', linestyle="dashed"
        )
        plt.show()
        exit()
    z_ref = np.concatenate((z_ref, (MIN_CONSTRAINT + MAX_CONSTRAINT) / 2 * np.ones(N)))
    return z_ref


def analytical_solution():
    z_ref = calc_z_ref()
    x_k = x_0
    z_cop = []
    x_com = []
    for k in tqdm(range(int(T / DT))):
        x_jerk = -np.matmul(
            np.linalg.inv(P_u.transpose() @ P_u + R / Q * np.ones((N, N))),
            P_u.transpose() @ (P_x @ x_k - z_ref[k : k + N]),
        )
        x_k = next_x(x_k, x_jerk[0])
        # x_k += np.random.normal([0.0, 0.0, 0.0], [0.001, 0.0005, 0.0001])
        x_com.append(x_k[0])
        z_cop.append(compute_z(x_k))

    return z_ref, x_com, z_cop


def qp_solution():
    z_ref = calc_z_ref()
    x_k = x_0
    z_cop = []
    x_com = []
    opt_M = np.eye(N)
    opt_V = np.zeros(N)
    con_M = P_u
    x_jerk = np.zeros(N)
    for k in tqdm(range(int(T / DT))):
        horizon = min(int(T / DT) - k, N) if "-dh" in sys.argv else N

        if "-mr" in sys.argv:
            opt_V = factor * P_u @ (P_x @ x_k - z_ref[k : k + N])
        if "-mx" in sys.argv:
            opt_V = factor * iterative_x_k(x_k, x_jerk)
        if "-tc" in sys.argv and int(T / DT) - k <= N:
            last_M = np.zeros(horizon)
            last_M[-1] = 1
            last_M = np.diag(last_M)
            b = -last_M @ P_xx @ x_k
            eqcon_M = last_M @ P_ux

        z_min_jerk = Z_MIN[k : k + horizon] - P_x[: horizon] @ x_k
        z_max_jerk = Z_MAX[k : k + horizon] - P_x[: horizon] @ x_k
        con_M = con_M[:horizon, :horizon]
        opt_M = opt_M[:horizon, :horizon]
        opt_V = opt_V[:horizon]

        if "-tc" in sys.argv and int(T / DT) - k <= N:
            x_jerk = solve_qp(
                opt_M, opt_V,
                G=np.vstack([con_M, -con_M]), h=np.hstack([z_max_jerk, -z_min_jerk]),
                A=eqcon_M,
                b=b,
                solver="clarabel"
                #clarabel, cvxopt, daqp, ecos, highs, osqp, piqp, proxqp, scs
            )
        else:
            x_jerk = solve_qp(
                opt_M, opt_V,
                G=np.vstack([con_M, -con_M]), h=np.hstack([z_max_jerk, -z_min_jerk]),
                solver="clarabel",
                #clarabel, cvxopt, daqp, ecos, highs, osqp, piqp, proxqp, scs
            )
        x_k = next_x(x_k, x_jerk[0])

        # x_k += np.random.normal([0.0, 0.0, 0.0], [0.001, 0.0005, 0.0001])
        x_com.append(x_k[0])
        z_cop.append(compute_z(x_k))
    return z_ref, x_com, z_cop


if "-qp" in sys.argv:
    z_ref, x_com, z_cop = qp_solution()
else:
    z_ref, x_com, z_cop = analytical_solution()

# plotting
plt.plot(np.arange(0, T, DT), Z_MAX[:int(T / DT)], color='red', linestyle="dashed")
plt.plot(np.arange(0, T, DT), Z_MIN[:int(T / DT)], color='blue', linestyle="dashed")
plt.plot(np.arange(0, T, DT), z_ref[:int(T / DT)], color='green', linestyle="dashed")
plt.plot(np.arange(0, T, DT), np.array(x_com), color='black', linewidth=3)
plt.plot(np.arange(0, T, DT), np.array(z_cop), color='black')
# plt.ylim(-0.2, 0.2)
plt.show()
