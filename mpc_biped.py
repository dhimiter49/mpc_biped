import sys
import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Constants
max_constraint = 0.17
min_constraint = -max_constraint
max_min_constraint = -0.03
min_max_constraint = -max_min_constraint
start_time = 2.5  # in ms
end_time = 7.5  # in ms
g = 9.81  # gravity
h_com = 0.8  # height of CoM
R = 1
Q = 1e6
dt = 0.005  # in ms
N = 300  # lookahead
T = 9  # in ms
period = 1 // dt  # in ms
short_period = 0.8 // dt  # in ms
diff_period = (period - short_period) // 2


# iteration dynamics
x_0 = np.zeros(3)
dyn_mat = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
dyn_jerk = np.array([[dt ** 3 /6, dt ** 2 / 2, dt]])
z_comp = np.array([1, 0, -h_com / g])
next_x = lambda x, x_jerk : (dyn_mat @ x + x_jerk * dyn_jerk).flatten()
compute_z = lambda x : np.sum(z_comp * x)

# constraints/bounds
z_max = max_constraint * np.ones(int(T / dt))
z_min = min_constraint * np.ones(int(T / dt))
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
z_ref = np.concatenate((z_ref, (min_constraint + max_constraint) / 2 * np.ones(N)))


# plt.plot(np.arange(0, T, dt), z_max, color='red', linestyle="dashed")
# plt.plot(np.arange(0, T, dt), z_min, color='blue', linestyle="dashed")
# plt.plot(np.arange(0, T, dt), z_ref[:int(T / dt)], color='green', linestyle="dashed")
# plt.show()

# analytical solution
P_x = np.ones((N, 3))
P_x[:, 1] *= np.array([dt * i for i in range(1, N + 1)])
P_x[:, 2] *= np.array([dt ** 2 * i ** 2 / 2 - h_com / g for i in range(1, N + 1)])

P_u = scipy.linalg.toeplitz(
    np.array(
        [(1 + 3 * i + 3 * i ** 2) * dt ** 3 / 6 - dt * h_com / g for i in range(N)]
    ),
    np.zeros(N),
)

x_k = x_0
z_cop = []
x_com = []
for k in range(int(T / dt)):
    x_jerk = -np.matmul(
        np.linalg.inv(P_u.transpose() @ P_u + R / Q * np.ones((N, N))),
        P_u.transpose() @ (P_x @ x_k - z_ref[k : k + N]),
    )
    x_k = next_x(x_k, x_jerk[0])
    x_com.append(x_k[0])
    z_cop.append(compute_z(x_k))

# plotting
plt.plot(np.arange(0, T, dt), z_max, color='red', linestyle="dashed")
plt.plot(np.arange(0, T, dt), z_min, color='blue', linestyle="dashed")
plt.plot(np.arange(0, T, dt), z_ref[:int(T / dt)], color='green', linestyle="dashed")
plt.plot(np.arange(0, T, dt), np.array(x_com), color='black', linewidth=4)
plt.plot(np.arange(0, T, dt), np.array(z_cop), color='black')
# plt.ylim(-0.2, 0.2)
plt.show()
