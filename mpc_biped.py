import numpy as np
import scipy
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
N = 5  # lookahead
T = 9  # in ms
period = 1 // dt  # in ms
short_period = 0.8 // dt  # in ms
diff_period = (period - short_period) // 2


# iteration dynamics
x_0 = np.zeros(3)
dyn_mat = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
dyn_jerk = np.array([[1, dt, dt ** 2 / 2], [0, 1, dt], [0, 0, 1]])
z_comp = np.array([1, 0, h_com / g])
next_x = lambda x, x_jerk : dyn_mat @ x + x_jerk * dyn_jerk
compute_z = lambda x : np.sum(z_comp * x)
