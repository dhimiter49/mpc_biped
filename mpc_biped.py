import sys
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from qpsolvers import solve_qp

# Constants
max_constraint = 0.17
min_constraint = -max_constraint
max_min_constraint = -0.03
min_max_constraint = -max_min_constraint
start_time = 2.5  # in s
end_time = 7.5  # in s
g = 9.81  # gravity
h_com = 0.8  # height of CoM
R = 1
Q = 1e6
dt = 0.005  # in s
N = 300  # lookahead
T = 9  # in s
period = 1 // dt  # in s
short_period = 0.8 // dt  # in s
diff_period = (period - short_period) // 2


# Iteration dynamics
x_0 = np.zeros(3)
dyn_mat = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
dyn_jerk = np.array([[dt ** 3 /6, dt ** 2 / 2, dt]])
z_comp = np.array([1, 0, -h_com / g])
next_x = lambda x, x_jerk : (dyn_mat @ x + x_jerk * dyn_jerk).flatten()
compute_z = lambda x : np.sum(z_comp * x)

# Constraints/bounds
z_max = max_constraint * np.ones(int(T / dt) + N)
z_min = min_constraint * np.ones(int(T / dt) + N)
z_max[int(start_time / dt) : int(start_time / dt + short_period)] = max_min_constraint
z_max[
    int(start_time / dt + short_period + period) :
    int(start_time / dt + 2 * short_period + period)
] = max_min_constraint
z_max[
    int(start_time / dt + 2 * short_period + 2 * period) :
    int(start_time / dt + 3 * short_period + 2 * period)
] = max_min_constraint

z_min[
    int(start_time / dt + short_period + diff_period) :
    int(start_time / dt + 2 * short_period + diff_period)
] = min_max_constraint
z_min[
    int(start_time / dt + 2 * short_period + period + diff_period) :
    int(start_time / dt + 3 * short_period + period + diff_period)
] = min_max_constraint
z_min[
    int(start_time / dt + 3 * short_period + 2 * period + diff_period) :
    int(start_time / dt + 4 * short_period + 2 * period + diff_period)
] = min_max_constraint

P_x = np.ones((N, 3))
P_x[:, 1] *= np.array([dt * i for i in range(1, N + 1)])
P_x[:, 2] *= np.array([(dt ** 2) * (i ** 2) / 2 - h_com / g for i in range(1, N + 1)])

row = np.zeros(N, dtype=np.float128)
col = np.array(
    [
        (1 + 3 * i + 3 * i ** 2) * dt ** 3 / 6 - dt * h_com / g
        for i in range(N)
    ],
    dtype=np.float128
)
P_u = scipy.linalg.toeplitz(col, row)
# print(np.linalg.cond(P_u))  # equal to 1259
P_u_inv = scipy.linalg.inv(P_u)


def calc_z_ref():
    z_ref = (z_max + z_min) / 2
    if "-sw" in sys.argv:
        z_ref = np.convolve(z_ref, np.ones(20)/20, mode="valid")
        z_ref = np.concatenate((
            (min_constraint + max_constraint) / 2 * np.ones(10),
            z_ref,
            (min_constraint + max_constraint) / 2 * np.ones(9),
        ))
    elif "-gf" in sys.argv:
        z_ref = gaussian_filter1d(z_ref, sigma=7)
    elif "-pr" in sys.argv:  # perfect reference, smoothed step func between min max
        for i in range(len(z_ref)):
            if i < 10 or i > len(z_ref) - 10:
                continue
            if np.mean(np.abs(z_max[i : i + 10])) > np.mean(np.abs(z_min[i - 10 : i])):
                z_ref[i] = (max_constraint + min_max_constraint) / 2
            elif np.mean(np.abs(z_max[i : i + 10])) < np.mean(np.abs(z_min[i - 10 : i])):
                z_ref[i] = (min_constraint + max_min_constraint) / 2
            elif np.mean(np.abs(z_max[i -10 : i])) > np.mean(np.abs(z_min[i : i + 10])):
                z_ref[i] = (max_constraint + min_max_constraint) / 2
            elif np.mean(np.abs(z_max[i - 10 : i])) < np.mean(np.abs(z_min[i : i + 10])):
                z_ref[i] = (min_constraint + max_min_constraint) / 2
        z_ref = gaussian_filter1d(z_ref, sigma=5)

    if "-tref" in sys.argv:
        plt.plot(
            np.arange(0, T, dt), z_max[:int(T / dt)], color='red', linestyle="dashed"
        )
        plt.plot(
            np.arange(0, T, dt), z_min[:int(T / dt)], color='blue', linestyle="dashed"
        )
        plt.plot(
            np.arange(0, T, dt), z_ref[:int(T / dt)], color='green', linestyle="dashed"
        )
        plt.show()
        exit()
    z_ref = np.concatenate((z_ref, (min_constraint + max_constraint) / 2 * np.ones(N)))
    return z_ref


def analytical_solution():
    z_ref = calc_z_ref()
    x_k = x_0
    z_cop = []
    x_com = []
    for k in tqdm(range(int(T / dt))):
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
    z_ref = calc_z_ref()[:int(T / dt)]
    x_k = x_0
    z_cop = []
    x_com = []
    P = np.eye(N)
    q = np.zeros(N)
    G = P_u  # np.eye(N)
    for k in tqdm(range(int(T / dt))):
        z_min_jerk = z_min[k : k + N] - P_x @ x_k
        z_max_jerk = z_max[k : k + N] - P_x @ x_k

        # z_min_jerk = P_u_inv @ z_min_jerk
        # z_max_jerk = P_u_inv @ z_max_jerk
        # lb = np.minimum(z_max_jerk, z_min_jerk)
        # ub = np.maximum(z_max_jerk, z_min_jerk)

        # # if k >= 350:
        # #     plt.plot(np.arange(0, N), 1000 * z_max[k : k + N], color='red', linestyle="dashed")
        # #     plt.plot(np.arange(0, N), 1000 * z_min[k : k + N], color='blue', linestyle="dashed")
        # #     plt.plot(np.arange(0, N), z_max_jerk, color='red', linestyle="dashed")
        # #     plt.plot(np.arange(0, N), z_min_jerk, color='blue', linestyle="dashed")
        # #     plt.plot(np.arange(0, N), ub, color='red')
        # #     plt.plot(np.arange(0, N), lb, color='blue')
        # #     # plt.show()

        # x_jerk = solve_qp(P, q, lb=lb, ub=ub, solver="osqp")
        # x_k = next_x(x_k, x_jerk[0])
        # x_jerk = solve_qp(P, q, G, 1000 * np.ones(lb.shape), lb=, ub=ub, solver="highs")

        try:
            x_jerk = solve_qp(
                P, q, G=np.vstack([G, G]), h=np.hstack([z_max_jerk, -z_min_jerk]),
                solver="osqp"
                # verbose=True,
                #'clarabel', 'cvxopt', 'daqp', 'ecos', 'highs', 'osqp', 'piqp', 'proxqp', 'scs'
            )
            x_k = next_x(x_k, x_jerk[0])

            # if k > 350:
            #     plt.subplot(211)
            #     plt.plot(np.arange(0, N), z_max_jerk, color='red', linestyle="dashed")
            #     plt.plot(np.arange(0, N), z_min_jerk, color='blue', linestyle="dashed")
            #     plt.plot(np.arange(0, N), G @ x_jerk, color='black', linestyle="dashed")
            #     plt.plot(np.arange(0, N), x_jerk, color='black')
            #     plt.subplot(212)
            #     plt.plot(np.arange(0, len(x_com)), z_max[:len(x_com)], color='red', linestyle="dashed")
            #     plt.plot(np.arange(0, len(x_com)), z_min[:len(x_com)], color='blue', linestyle="dashed")
            #     plt.plot(np.arange(0, len(x_com)), np.array(z_cop), color='black')
            #     plt.show()
        except (ValueError, TypeError):
            plt.plot(np.arange(0, N), z_max_jerk, color='red', linestyle="dashed")
            plt.plot(np.arange(0, N), z_min_jerk, color='blue', linestyle="dashed")
            plt.show()
            return z_ref, x_com, z_cop


        # x_k += np.random.normal([0.0, 0.0, 0.0], [0.001, 0.0005, 0.0001])
        x_com.append(x_k[0])
        z_cop.append(compute_z(x_k))
    return z_ref, x_com, z_cop


if "-qp" in sys.argv:
    z_ref, x_com, z_cop = qp_solution()
else:
    z_ref, x_com, z_cop = analytical_solution()

# plotting
plt.plot(np.arange(0, T, dt), z_max[:int(T / dt)], color='red', linestyle="dashed")
plt.plot(np.arange(0, T, dt), z_min[:int(T / dt)], color='blue', linestyle="dashed")
plt.plot(np.arange(0, T, dt), z_ref[:int(T / dt)], color='green', linestyle="dashed")
plt.plot(np.arange(0, T, dt), np.array(x_com), color='black', linewidth=3)
plt.plot(np.arange(0, T, dt), np.array(z_cop), color='black')
# plt.ylim(-0.2, 0.2)
plt.show()
