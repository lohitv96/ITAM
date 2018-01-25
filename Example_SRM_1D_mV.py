from SRM import *
from correlation_matrix1 import *
import matplotlib.pyplot as plt
import scipy

plt.style.use('seaborn')

# Kaimal et al. (1972)
# Example from Deodatis Book Chapter4

m = 501

W = 10
n = 400 + 1
dw = W / (n - 1)
w = np.linspace(0, W, n)

S_11 = 38.3 / (1 + 6.19 * w) ** (5 / 3)
S_22 = 43.4 / (1 + 6.98 * w) ** (5 / 3)
S_33 = 135 / (1 + 21.8 * w) ** (5 / 3)

g_12 = np.exp(-0.1757 * w)
g_13 = np.exp(-3.478 * w)
g_23 = np.exp(-3.392 * w)

var = 3
S_list = np.array([S_11, S_22, S_33])
S_jk = np.einsum('i...,j...->ij...', S_list, S_list)

g_jk = np.array([[np.ones(n), g_12, g_13], [g_12, np.ones(n), g_23], [g_13, g_23, np.ones(n)]])

S = S_jk * g_jk
S = np.einsum('ijk->kij', S)
H_jk = np.zeros_like(S)

for i in range(n):
    try:
        H_jk[i] = scipy.linalg.cholesky(S[i])
    except:
        H_jk[i] = scipy.linalg.cholesky(nearestPD(S[i]))

phi = np.random.uniform(size=n) * 2 * np.pi
B = 2 * np.abs(H_jk[:, 0, 0]) * np.sqrt(dw) * np.exp(phi*1.0j)
f_t = np.real(np.fft.fftn(B, s=[m]))

# SRM_object = SRM(1, H_jk[:, 0, 0], dw, m, n)
