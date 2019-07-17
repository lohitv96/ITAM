import numpy as np
import copy
import math
from scipy.stats import skew, kurtosis, moment

########################################################################################################################
# Input Part

nsamples = 10000
n = 1  # Number of dimensions
m = 2  # Number of variables

W = 2.0  # Cutoff Frequency
nw = 400  # Number of frequency steps
dw = W / nw  # Length of frequency step
w = np.linspace(dw, W, nw)  # frequency vector
wx, wy = np.meshgrid(w, w)  # Frequency mesh

nt = 800  # Number of time steps
T = 1 / W * nt / 2  # Total Simulation time
dt = T / nt  # Duration of time step
t = np.linspace(dt, T, nt)  # Vector of time

t_u = 2 * np.pi / (2 * W)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

# Diagonal elements of the Multi-variate Power Spectrum
S_11 = 38.3 / (1 + 6.19 * w) ** (5 / 3)
S_22 = 43.4 / (1 + 6.98 * w) ** (5 / 3)

# Gamma values of the Multi-variate Power Spectrum
g_s_12 = np.exp(-2 * w)

# Diagonal elements of the Multi-variate Bispectrum
B_111 = 50 / (1 + 6.19 * (wx + wy)) ** (5 / 3)
B_222 = 50 / (1 + 21.8 * (wx + wy)) ** (5 / 3)

# Gamma values of the Multi-variate Bispectrum
g_b_111 = np.ones_like(B_111)
g_b_121 = np.exp(-3.478 * (wx + wy))
g_b_211 = np.exp(-3.478 * (wx + wy))
g_b_221 = np.exp(-3.392 * (wx + wy))
g_b_112 = np.exp(-3.478 * (wx + wy))
g_b_122 = np.exp(-3.392 * (wx + wy))
g_b_212 = np.exp(-3.392 * (wx + wy))
g_b_222 = np.ones_like(B_222)

########################################################################################################################
# Assembly of the cross 2nd-order Spectrum

S_list = np.array([S_11, S_22])
g_s_list = np.array([g_s_12])

S_sqrt = np.sqrt(S_list)
S_ij = np.einsum('i...,j...->ij...', S_sqrt, S_sqrt)

# Assembly of G_ij
G_ij = np.zeros_like(S_ij)
counter = 0
for i in range(m):
    for j in range(i + 1, m):
        G_ij[i, j] = g_s_list[counter]
        counter = counter + 1
G_ij = np.einsum('ij...->ji...', G_ij) + G_ij

for i in range(m):
    G_ij[i, i] = np.ones_like(S_ij[0, 0])
S = S_ij * G_ij

########################################################################################################################
# Assembly of the cross 3rd-order Spectrum

B_list = np.array([B_111, B_222])
g_b_list = np.array([g_b_112, g_b_122])

B_cube_root = np.power(B_list, 1 / 3)
B_ijk = np.einsum('i...,j..., k...->ijk...', B_cube_root, B_cube_root, B_cube_root)

# Assembly of G_ijk
G_ijk = np.zeros_like(B_ijk)
G_ijk[0, 0, 0] = B_ijk[0, 0, 0] * g_b_111
G_ijk[0, 1, 0] = B_ijk[0, 1, 0] * g_b_121
G_ijk[1, 0, 0] = B_ijk[1, 0, 0] * g_b_211
G_ijk[1, 1, 0] = B_ijk[1, 1, 0] * g_b_221
G_ijk[0, 0, 1] = B_ijk[0, 0, 1] * g_b_112
G_ijk[0, 1, 1] = B_ijk[0, 1, 1] * g_b_122
G_ijk[1, 0, 1] = B_ijk[1, 0, 1] * g_b_212
G_ijk[1, 1, 1] = B_ijk[1, 1, 1] * g_b_222
B = copy.deepcopy(G_ijk)

########################################################################################################################
# Decomposiong the 2nd-order spectrum into pure and interactive parts based on the 3rd-order spectrum

S = np.einsum('ij...->...ij', S)
B = np.einsum('ijk...->...ijk', B)

B[0, :] = 0
B[:, 0] = 0

B_Ampl = np.absolute(B)
Biphase = np.zeros_like(B)
Bc2 = np.zeros(shape=[nw, nw, m, m])
SP = np.zeros_like(S)
sum_Bc2 = np.zeros_like(S)

# random phase angles for the simulation
phi = np.random.uniform(size=[nsamples, nw, m]) * 2 * np.pi
Phi_e = np.exp(phi * 1.0j)

# Make this for loop computationally efficient
Fi = np.zeros(shape=[nsamples, nw, m])
Fi = Fi + Fi * 1.0j
for i in range(nw):
    wk = i
    print(wk)
    for j in range(int(math.ceil((wk + 1) / 2))):
        wj = j
        wi = wk - wj
        if np.all(B_Ampl[wi, wj]) > 0 and np.all(SP[wi] * SP[wj]) != 0:
            # Bc2[wi, wj] = B_Ampl[wi, wj] ** 2 / (SP[wi] * SP[wj] * S[wk]) * dw ** n
            Ui, si, Vi = np.linalg.svd(SP[wi])
            Uj, sj, Vj = np.linalg.svd(SP[wj])
            Ri = np.einsum('ij, jk->ik', Ui, np.diag(np.sqrt(si)))
            Rj = np.einsum('ij, jk->ik', Uj, np.diag(np.sqrt(sj)))
            Rii = np.linalg.inv(Ri)
            Rji = np.linalg.inv(Rj)
            # Bc2[wi, wj] = np.einsum('cba, gfe, pa, re, qb, sf, pr, qs->cg', B[wi, wj], B[wi, wj], Rii, Rii, Rji, Rji,
            #                         np.eye(m), np.eye(m)) * dw
            # Bc2[wi, wj] = np.einsum('cba, gfe, pa, pe, qb, qf->cg', B[wi, wj], B[wi, wj], Rii, Rii, Rji, Rji) * dw
            Bc2[wi, wj] = np.einsum('cba, gfe, pa, pe, qb, qf->cg', B[wi, wj], B[wi, wj], Rii, Rii, Rji, Rji) * dw
            sum_Bc2[wk] = sum_Bc2[wk] + Bc2[wi, wj]
            Fi[:, wk, :] = Fi[:, wk, :] + np.einsum('np, nq, gfe, pe, qf -> ng', Phi_e[:, wi], Phi_e[:, wj], B[wi, wj],
                                                    Rii, Rji) * 2 * dw
        else:
            Bc2[wi, wj] = 0
    SP[wk] = S[wk] - sum_Bc2[wk]
print(np.min(SP))

########################################################################################################################
# Simulation Part
# Biphase_e = np.exp(Biphase * 1.0j)  # Only when the Imaginary part of the cross Bispectrum exists

# Original Spectral Representation Theorem Method
Coeff = 2 * np.sqrt(dw)
U, s, V = np.linalg.svd(S)
R = np.einsum('wij,wj->wij', U, np.sqrt(s))
F = Coeff * np.einsum('wij,nwj -> nwi', R, Phi_e)
samples_SRM = np.real(np.fft.fft(F, nt, axis=1))

# Making sure that the Pure part of the 2nd order power spectrum is symmetric, asymmetry might arise from computations
# involved
SP = (SP + np.einsum('wij->wji', SP)) / 2
# Simulating the pure component of the samples_SRM
Coeff = 2 * np.sqrt(dw)
U, s, V = np.linalg.svd(SP)
R = np.einsum('wij,wj->wij', U, np.sqrt(s))
Fp = Coeff * np.einsum('wij,nwj -> nwi', R, Phi_e)

Fp[np.isnan(Fp)] = 0
samplesp = np.real(np.fft.fft(Fp, nt, axis=1))

Fi[np.isnan(Fi)] = 0
samplesi = np.real(np.fft.fft(Fi, nt, axis=1))

samples_BSRM = samplesp + samplesi

B1 = np.zeros_like(B)
for i in range(nw):
    wk = i
    for j in range(int(math.ceil((wk + 1) / 2))):
        wj = j
        wi = wk - wj
        B1[wi, wj] = B[wi, wj]

for i in range(2):
    for j in range(2):
        for k in range(2):
            B1[:, :, i, j, k] = B1[:, :, i, j, k] + np.transpose(B1[:, :, i, j, k])
            B1[:, :, i, j, k] = B1[:, :, i, j, k] - np.diag(np.diag(B1[:, :, i, j, k])) / 2

np.save('data_multi_variate/samples_SRM.npy', samples_SRM)
np.save('data_multi_variate/samples_BSRM.npy', samples_BSRM)
np.save('data_multi_variate/S.npy', S)
np.save('data_multi_variate/SP.npy', SP)
np.save('data_multi_variate/B.npy', B)
np.save('data_multi_variate/B1.npy', B1)
