import numpy as np
import copy
import math


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

nsamples = 1000
n = 1  # Number of dimensions
m = 2  # Number of variables

T = 10.0  # Total Simulation time
nt = 800  # Number of time steps
dt = T / nt  # Duration of time step
t = np.linspace(0, T - dt, nt)  # Vector of time

W = 0.5  # Cutoff Frequency
nw = 400  # Number of frequency steps
dw = W / nw  # Length of frequency step
w = np.linspace(dw, W, nw)  # frequency vector
wx, wy = np.meshgrid(w, w)  # Frequency mesh

# Diagonal elements of the Multi-variate Power Spectrum
S_11 = 38.3 / (1 + 6.19 * w) ** (5 / 3)
S_22 = 43.4 / (1 + 6.98 * w) ** (5 / 3)

# Gamma values of the Multi-variate Power Spectrum
g_s_12 = np.exp(-0.1757 * w)

# Diagonal elements of the Multi-variate Bispectrum
# B_111 = 38.3 / (1 + 6.19 * (wx + wy)) ** (5 / 3)
# B_222 = 135 / (1 + 21.8 * (wx + wy)) ** (5 / 3)
B_111 = 2 / (1 + 6.19 * (wx + wy)) ** (5 / 3)
B_222 = 2 / (1 + 21.8 * (wx + wy)) ** (5 / 3)

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
PP = np.zeros_like(S)
sum_Bc2 = np.zeros_like(S)

# Make this for loop computationally efficient
for i in range(nw):
    wk = i
    for j in range(int(math.ceil((wk + 1) / 2))):
        wj = j
        wi = wk - wj
        if np.all(B_Ampl[wi, wj]) > 0 and np.all(PP[wi] * PP[wj]) != 0:
            # Bc2[wi, wj] = B_Ampl[wi, wj] ** 2 / (PP[wi] * PP[wj] * S[wk]) * dw ** n
            Ui, si, Vi = np.linalg.svd(PP[wi])
            Uj, sj, Vj = np.linalg.svd(PP[wj])
            Uk, sk, Vk = np.linalg.svd(S[wk])
            Ri = np.einsum('ij, jk->ik', Ui, np.diag(np.sqrt(si)))
            Rj = np.einsum('ij, jk->ik', Uj, np.diag(np.sqrt(sj)))
            Rk = np.einsum('ij, jk->ik', Uk, np.diag(np.sqrt(sk)))
            Rii = np.linalg.inv(Ri)
            Rji = np.linalg.inv(Rj)
            Rki = np.linalg.inv(Rk)
            Bc2[wi, wj] = np.einsum('cba, gfe, pa, re, qb, sf, pr, qs->cg', B[wi, wj], B[wi, wj], Rii, Rii, Rji, Rji,
                                    np.eye(m), np.eye(m))
            sum_Bc2[wk] = sum_Bc2[wk] + Bc2[wi, wj]
        else:
            Bc2[wi, wj] = 0
    # if np.any(sum_Bc2[wk]) > 1:
    #     print('Results may not be as expected as sum of partial bicoherences is greater than 1')
    #     for j in range(int(math.ceil((wk + 1) / 2))):
    #         wj = j
    #         wi = wk - wj
    #         Bc2[wi, wj] = Bc2[wi, wj] / sum_Bc2[wk]
    #     sum_Bc2[wk] = 1
    PP[wk] = S[wk] - sum_Bc2[wk]

########################################################################################################################
# Simulation Part

Biphase_e = np.exp(Biphase * 1.0j)  # Only when the Imaginary part of the cross Bispectrum exists

Coeff1 = np.sqrt(dw)
# random phase angles for the simulation
phi = np.random.uniform(size=[nsamples, nw, m]) * 2 * np.pi
Phi_e = np.exp(phi * 1.0j)
U, s, V = np.linalg.svd(PP)
R = np.einsum('wij,wjk->wik', U, np.sqrt(S))
Fp = Coeff1*np.einsum('wij,nwj -> nwi', R, Phi_e)
Fp[np.isnan(Fp)] = 0
samplesp = np.real(np.fft.fftn(Fp, s=[nt], axes=(1,)))
samplesp = np.einsum('ijk->ikj', samplesp)


# Bc = np.sqrt(Bc2)
#
# Phi_e = np.einsum('i...->...i', Phi_e)
# F = np.einsum('i...->...i', F)
#
# for i in range(nw):
#     wk = i
#     for j in range(int(math.ceil((wk + 1) / 2))):
#         wj = j
#         wi = wk - wj
#         F[wk] = F[wk] + Bc[wi, wj] * Biphase_e[wi, wj] * Phi_e[wi] * Phi_e[wj]
#
# F = np.einsum('...i->i...', F)
# Phi_e = np.einsum('...i->i...', Phi_e)
# F = F * Coeff
# F[np.isnan(F)] = 0
# samples = np.fft.fftn(F, [nt for _ in range(n)])

# print(
#   np.einsum('efg, cba, ep, ra, fq, sb, pr, qs -> gc', B[wi, wj], B[wi, wj], Rii, Rii, Rji, Rji, np.eye(m), np.eye(m)))
# checking if the Bispectrum is recovered
# B_est = np.einsum('ab, cd, pb, qd, gfe, pe, qf->gca', Ri, Rj, np.eye(m), np.eye(m), B[wi, wj], Rii, Rji)
#
# # Understanding the tensor product
# wi = 1
# wj = 3
# wk = wi + wj
#
# Ui, si, Vi = np.linalg.svd(PP[wi])
# Uj, sj, Vj = np.linalg.svd(PP[wj])
# Uk, sk, Vk = np.linalg.svd(S[wk])
# Ri = np.einsum('ij, jk->ik', Ui, np.diag(np.sqrt(si)))
# Rj = np.einsum('ij, jk->ik', Uj, np.diag(np.sqrt(sj)))
# Rk = np.einsum('ij, jk->ik', Uk, np.diag(np.sqrt(sk)))
# Rii = np.linalg.inv(Ri)
# Rji = np.linalg.inv(Rj)
# Rki = np.linalg.inv(Rk)
#
# prod1 = np.zeros(shape=[m, m])
# prod2 = np.zeros(shape=[m, m, m])
# for i in range(nsamples):
#     temp1 = np.real(np.einsum('ab, b -> a', Ri, phi_e[i, wi]))
#     temp2 = np.real(np.einsum('cd, d -> c', Rj, phi_e[i, wj]))
#     temp3 = np.real(np.einsum('p, q, gfe, pe, qf -> g', phi_e[i, wi], phi_e[i, wj], B[wi, wj], Rii, Rji))
#     prod1 = prod1 + np.einsum('i, j->ij', temp3, temp3)
#     prod2 = prod2 + np.einsum('i, j, k->ijk', temp1, temp2, temp3)
# print(prod1/nsamples*2)
# print(prod2/nsamples*4)
# print(prod2/nsamples*4 - B[wi, wj])

########################################################################################################################
# Simulation Checks
# Checking if the 2nd-order spectrum can be recovered
temp = estimate_cross_power_spectrum(samplesp)

# Checking if the 2nd-order spectrum can be recovered
temp1 = estimate_cross_bispectrum(samplesp)

########################################################################################################################
# Statistics Checks
# Checking if the 2nd-order statistcs are satisfied

# Checking if the 3rd-order statistcs are satisfied
