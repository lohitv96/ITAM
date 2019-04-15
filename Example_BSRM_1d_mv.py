import numpy as np
import copy
import math
from scipy.stats import skew, kurtosis, moment


def estimate_cross_power_spectrum(samples):
    nsamples, m, nt = samples.shape
    nw = int(nt / 2)
    Xw = np.fft.ifft(samples, axis=2)
    Xw = Xw[:, :, :nw]
    # Initializing the array before hand
    s_P = np.zeros([nsamples, m, m, nw])
    s_P = s_P + 1.0j * s_P
    for i1 in range(nw):
        s_P[..., i1] = s_P[..., i1] + np.einsum('ij, ik-> ijk', Xw[..., i1], np.conj(Xw[..., i1]))
    m_P = np.mean(s_P, axis=0)
    return m_P


def estimate_cross_bispectrum(samples):
    nsamples, m, nt = samples.shape
    nw = int(nt / 2)
    Xw = np.fft.ifft(samples, axis=2)
    Xw = Xw[:, :, :nw]
    # Initializing the array before hand
    s_B = np.zeros([nsamples, m, m, m, nw, nw])
    s_B = s_B + 1.0j * s_B
    for i1 in range(nw):
        for i2 in range(nw - i1):
            s_B[..., i1, i2] = s_B[..., i1, i2] + np.einsum('ij, ik, il-> ijkl', Xw[..., i1], Xw[..., i2],
                                                            np.conj(Xw[..., i1 + i2]))
    m_B = np.mean(s_B, axis=0)
    return m_B


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
B_ijk = np.einsum('i...,j..., k...->ijk...', B_cube_root, B_cube_root, B_cube_root)\

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
Fi = Fi + Fi *1.0j
for i in range(nw):
    wk = i
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
            Bc2[wi, wj] = np.einsum('cba, gfe, pa, re, qb, sf, pr, qs->cg', B[wi, wj], B[wi, wj], Rii, Rii, Rji, Rji,
                                    np.eye(m), np.eye(m))*dw
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

# Making sure that the Pure part of the 2nd order power spectrum is symmetric, asymmetry might arise from computations
# involved
SP = (SP + np.einsum('wij->wji', SP)) / 2
# Simulating the pure component of the samples
Coeff = 2*np.sqrt(dw)
U, s, V = np.linalg.svd(SP)
R = np.einsum('wij,wj->wij', U, np.sqrt(s))
Fp = Coeff*np.einsum('wij,nwj -> nwi', R, Phi_e)

Fp[np.isnan(Fp)] = 0
samplesp = np.real(np.fft.fft(Fp, nt, axis=1))

Fi[np.isnan(Fi)] = 0
samplesi = np.real(np.fft.fft(Fi, nt, axis=1))

samples = samplesp + samplesi

print(np.var(samples[:, :, 0]))
print(np.sum(2*S[:, 0, 0]*dw))

print(skew(samples[:, :, 0].flatten()))
print(6*np.sum(B[:, :, 0, 0, 0])*dw**2/(2*np.sum(S[:, 0, 0])*dw)**(3/2))

########################################################################################################################
# Simulation Checks
# Checking if the 2nd-order spectrum can be recovered
# temp = estimate_cross_power_spectrum(samplesp)

# Checking if the 2nd-order spectrum can be recovered
# temp1 = estimate_cross_bispectrum(samplesp)

########################################################################################################################
# Statistics Checks
# Checking if the 2nd-order statistcs are satisfied

# Checking if the 3rd-order statistcs are satisfied
