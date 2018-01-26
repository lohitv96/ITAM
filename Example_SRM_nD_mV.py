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
S_list = np.sqrt(S_list)
S_jk = np.einsum('i...,j...->ij...', S_list, S_list)

# Use transpose technique to deal with symmetry
g_jk = np.zeros_like(S_jk)
g_jk[0, 0, :] = np.ones(shape=[nw, nw, nw])
g_jk[0, 1, :] = g_12
g_jk[0, 2, :] = g_13
g_jk[1, 0, :] = g_12
g_jk[1, 1, :] = np.ones(shape=[nw, nw, nw])
g_jk[1, 2, :] = g_23
g_jk[2, 0, :] = g_13
g_jk[2, 1, :] = g_23
g_jk[2, 2, :] = np.ones(shape=[nw, nw, nw])

S = S_jk * g_jk
S = np.einsum('ij...->...ij', S)
H_jk = np.linalg.cholesky(S)

for i in range(nw):
    for j in range(nw):
        for k in range(nw):
            H_jk[:, :, i, j, k] = np.linalg.cholesky(S[:, :, i, j, k])
    # except:
    #     H_jk[:, :, i] = np.linalg.cholesky(nearestPD(S[:, :, i]))

