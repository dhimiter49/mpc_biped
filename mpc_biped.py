import numpy as np
import matplotlib

# Constants
max_constraint = 0.17
min_constraint = -max_constraint
max_min_constraint = -0.3
min_max_constraint = -max_min_constraint
g = 9.81  # gravity
h_com = 0.8  # height of CoM
R = 1
Q = 1e6
dt = 5
N = 300  # lookahead
