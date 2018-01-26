from SRM import *
from correlation_matrix1 import *
import matplotlib.pyplot as plt
import scipy
from statsmodels.distributions.empirical_distribution import ECDF

plt.style.use('seaborn')

# Kaimal et al. (1972)
# Example from Deodatis Book Chapter 4

nt = 501

W = 0.5
nw = 400 + 1
dw = W / (nw - 1)
w = np.linspace(0, W, nw)

S_11 = 38.3 / (1 + 6.19 * w) ** (5 / 3)
S_22 = 43.4 / (1 + 6.98 * w) ** (5 / 3)
S_33 = 135 / (1 + 21.8 * w) ** (5 / 3)

g_12 = np.exp(-0.1757 * w)
g_13 = np.exp(-3.478 * w)
g_23 = np.exp(-3.392 * w)

m = 3
S_list = np.array([S_11, S_22, S_33])
S_list = np.sqrt(S_list)
S_jk = np.einsum('i...,j...->ij...', S_list, S_list)

# Use transpose technique to deal with symmetry
g_jk = np.zeros_like(S_jk)
g_jk[0, 0, :] = np.ones(nw)
g_jk[0, 1, :] = g_12
g_jk[0, 2, :] = g_13
g_jk[1, 0, :] = g_12
g_jk[1, 1, :] = np.ones(nw)
g_jk[1, 2, :] = g_23
g_jk[2, 0, :] = g_13
g_jk[2, 1, :] = g_23
g_jk[2, 2, :] = np.ones(nw)

S = S_jk * g_jk
# S = np.einsum('ijk->kij', S)
H_jk = np.zeros_like(S)

for i in range(nw):
    try:
        H_jk[:, :, i] = scipy.linalg.cholesky(S[:, :, i])
    except:
        H_jk[:, :, i] = scipy.linalg.cholesky(nearestPD(S[:, :, i]))

phi = np.random.uniform(size=nw) * 2 * np.pi
B = 2 * H_jk[0, 0] * np.sqrt(dw) * np.exp(phi * 1.0j)
f_t = np.real(np.fft.fftn(B, s=[nt]))

plt.figure()
plt.plot(f_t)
plt.show()

plt.figure()
ecdf = ECDF(f_t)
plt.plot(np.sort(f_t), ecdf(np.sort(f_t)), ':')
plt.show()

# SRM_object = SRM(1, H_jk[:, 0, 0], dw, m, n)
