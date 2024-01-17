import sys
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
# time when to stop can be different then time T when episode ends
stop_at = float(sys.argv[sys.argv.index("-sa") + 1]) if "-sa" in sys.argv else T


# Iteration dynamics
x_0 = np.zeros(3)
DYN_MAT = np.array([[1, DT, DT ** 2 / 2], [0, 1, DT], [0, 0, 1]])
DYN_JERK = np.array([[DT ** 3 / 6, DT ** 2 / 2, DT]])
Z_COMP = np.array([1, 0, -H_COM / G])
next_x = lambda x, x_jerk : (DYN_MAT @ x + x_jerk * DYN_JERK).flatten()
compute_z = lambda x : np.sum(Z_COMP * x)


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

P_x = np.ones((N, 3), dtype=np.float128)
P_x[:, 1] *= np.array([DT * i for i in range(1, N + 1)])
P_x[:, 2] *= np.array([(DT ** 2) * (i ** 2) / 2 - H_COM / G for i in range(1, N + 1)])
P_xx = np.ones((N, 3), dtype=np.float128)
P_xx[:, 1] *= np.array([DT * i for i in range(1, N + 1)])
P_xx[:, 2] *= np.array([(DT ** 2) * (i ** 2) / 2 for i in range(1, N + 1)])
P_xv = np.zeros((N, 3), dtype=np.float128)
P_xv[:, 1] = 1
P_xv[:, 2] = np.array([(DT * i) for i in range(1, N + 1)])
P_xa = np.zeros((N, 3), dtype=np.float128)
P_xa[:, 2] = 1

P_u = scipy.linalg.toeplitz(
    np.array(
        [(1 + 3 * i + 3 * i ** 2) * DT ** 3 / 6 - DT * H_COM / G for i in range(N)],
        dtype=np.float128
    ),
    np.zeros(N)
)
P_u_inv = scipy.linalg.inv(P_u)

P_ux = scipy.linalg.toeplitz(
    np.array(
        [(1 + 3 * i + 3 * i ** 2) * DT ** 3 / 6 for i in range(N)], dtype=np.float128,
    ),
    np.zeros(N)
)
P_uv = scipy.linalg.toeplitz(
    np.array([(1 + 2 * i) * DT ** 2 / 2 for i in range(N)], dtype=np.float128),
    np.zeros(N)
)
P_ua = scipy.linalg.toeplitz(
    np.array([DT] * N, dtype=np.float128), np.zeros(N)
)


def calc_z_ref() -> np.ndarray:
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
    jerks= []
    P_u_ = P_u.astype(np.float64)
    P_x_ = P_x.astype(np.float64)
    for k in tqdm(range(int(T / DT))):
        x_jerk = -np.matmul(
            np.linalg.inv(P_u_.transpose() @ P_u_ + R / Q * np.ones((N, N))),
            P_u_.transpose() @ (P_x_ @ x_k - z_ref[k : k + N]),
        )
        x_k = next_x(x_k, x_jerk[0])
        # x_k += np.random.normal([0.0, 0.0, 0.0], [0.001, 0.0005, 0.0001])
        x_com.append(x_k[0])
        z_cop.append(compute_z(x_k))
        jerks.append(x_jerk[0])
    return z_ref, x_com, z_cop, jerks


def qp_solution():
    z_ref = calc_z_ref()
    x_k = x_0
    z_cop = []
    x_com = []
    jerks = []
    opt_M = np.eye(N)
    # if "-mx" in sys.argv:
    #     opt_M += factor * P_ux ** 2
    opt_V = np.zeros(N)
    con_M = P_u
    x_jerk = np.zeros(N)
    for k in tqdm(range(int(T / DT))):
        horizon = min(int(T / DT) - k, N) if "-dh" in sys.argv else N
        z_min_jerk = Z_MIN[k : k + horizon] - P_x[: horizon] @ x_k
        z_max_jerk = Z_MAX[k : k + horizon] - P_x[: horizon] @ x_k
        con_M = con_M[:horizon, :horizon]
        opt_M = opt_M[:horizon, :horizon]

        if "-mr" in sys.argv:
            opt_V = factor * P_u.T @ (P_x @ x_k - z_ref[k : k + N])
        if "-mx" in sys.argv:
            opt_V = factor * P_xx @ x_k @ P_ux
        opt_V = opt_V[:horizon]
        if "-tc" in sys.argv and int(stop_at / DT) - k <= N:
            start_max = horizon - 1 if int(T / DT) - k <= N else 0
            start_min = int(stop_at / DT) - k - 1
            start = max(start_min, start_max)
            eqcon_x_M = P_ux[start : horizon, :horizon]
            eqcon_v_M = P_uv[start : horizon, :horizon]
            eqcon_a_M = P_ua[start : horizon, :horizon]
            b_x, b_v, b_a = P_xx @ x_k, P_xv @ x_k, P_xa @ x_k
            b_x, b_v, b_a = (
                -b_x[start : horizon],
                -b_v[start : horizon],
                -b_a[start : horizon]
            )
            x_jerk = solve_qp(
                opt_M, opt_V,
                G=np.vstack([con_M, -con_M]), h=np.hstack([z_max_jerk, -z_min_jerk]),
                A=np.vstack([eqcon_x_M, eqcon_v_M, eqcon_a_M]),
                b=np.hstack([b_x, b_v, b_a]),
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

        assert x_jerk is not None
        x_k = next_x(x_k, x_jerk[0])

        # x_k += np.random.normal([0.0, 0.0, 0.0], [0.001, 0.0005, 0.0001])
        x_com.append(x_k[0])
        z_cop.append(compute_z(x_k))
        jerks.append(x_jerk[0])
    print("Final position, velocity and acceleration: ", x_k)
    return z_ref, x_com, z_cop, jerks


z_ref, x_com, z_cop, jerks = qp_solution() if "-qp" in sys.argv else analytical_solution()
jerks = np.array(jerks)
pulse = np.arange(0, T, DT)
# pulse = np.arange(0, len(jerks))

# plotting
norm = plt.Normalize(-1, 1)
jerk_points = np.array([pulse, jerks * 0 + 0.16]).T.reshape(-1, 1, 2)
segments = np.concatenate([jerk_points[:-1], jerk_points[1:]], axis=1)
lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=10)
lc.set_array(jerks)
ax = plt.gca()
line = ax.add_collection(lc)

plt.plot(pulse, Z_MAX[:len(pulse)], color='red', linestyle="dashed")
plt.plot(pulse, Z_MIN[:len(pulse)], color='blue', linestyle="dashed")
plt.plot(pulse, z_ref[:len(pulse)], color='green', linestyle="dashed")
plt.plot(pulse, np.array(x_com), color='black', linewidth=3)
plt.plot(pulse, np.array(z_cop), color='black')
# plt.ylim(-0.2, 0.2)
plt.show()
