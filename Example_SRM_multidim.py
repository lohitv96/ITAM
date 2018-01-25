from SRM import *
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# Input Data
# Time
dim = 3
T = 10
dt = 0.1
m = int(T / dt) + 1
t = np.linspace(0, T, m)

# Frequency
n = 100 + 1
W = np.array([1.5, 2.5, 2.0])
dw = W / (n - 1)
x_list = [np.linspace(0, W[i], n) for i in range(dim)]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
S = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-5 * np.linalg.norm(xy_list, axis=0))

SRM_object = SRM(1, S, dw, m, n)
samples = SRM_object.samples
