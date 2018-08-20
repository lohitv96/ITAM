from SRM import *
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# Number of Dimensions
n = 3

# Number of Variables
m = 3

# Input Data
# Time
T = 10  # Simulation Time
dt = 0.1
nt = int(T / dt) + 1
t = np.linspace(0, T, m)

# Frequency
nw = 100
W = np.array([1.5, 2.5, 2.0])
dw = W / (nw - 1)
x_list = [np.linspace(dw[i], W[i], nw) for i in range(n)]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))

S_11 = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-5 * np.linalg.norm(xy_list, axis=0))
S_22 = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-3 * np.linalg.norm(xy_list, axis=0))
S_33 = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-7 * np.linalg.norm(xy_list, axis=0))

g_12 = np.exp(-0.1757 * np.linalg.norm(xy_list, axis=0))
g_13 = np.exp(-3.478 * np.linalg.norm(xy_list, axis=0))
g_23 = np.exp(-3.392 * np.linalg.norm(xy_list, axis=0))

S_list = np.array([S_11, S_22, S_33])
g_list = np.array([g_12, g_13, g_23])

SRM_object = SRM(10, S_list, dw, nt, nw, case='multi', g=g_list)
samples = SRM_object.samples
