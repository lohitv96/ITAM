from scipy.stats import moment
import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy


def estimate_second_order_interactions(samples):
    nsamples = samples.shape[0]
    Xw1 = np.fft.fft(samples, axis=1) / nt
    Xw = np.zeros_like(Xw1[:, :nf])
    Xw[:, 0] = Xw1[:, 0] + Xw1[:, nf]
    Xw[:, 1:] = Xw1[:, 1:nf] + np.flip(Xw1[:, nf + 1:], axis=1)
    temp = np.zeros(shape=[nf, nf])
    for iter1 in range(len(samples)):
        temp = temp + np.einsum('i, j->ij', np.conj(Xw[iter1]), Xw[iter1]) / nsamples
    return temp / df


def estimate_third_order_interactions(samples):
    nsamples = samples.shape[0]
    Xw1 = np.fft.fft(samples, axis=1) / nt
    Xw = np.zeros_like(Xw1[:, :nf])
    Xw[:, 0] = Xw1[:, 0] + Xw1[:, nf]
    Xw[:, 1:] = Xw1[:, 1:nf] + np.flip(Xw1[:, nf + 1:], axis=1)
    temp = np.zeros(shape=[nf, nf, nf])
    for iter2 in range(len(samples)):
        temp = temp + np.einsum('i, j, k->ijk', np.conj(Xw[iter2]), np.conj(Xw[iter2]), Xw[iter2]) / nsamples
    return temp / df ** 2


def estimate_fourth_order_interactions(samples):
    nsamples = samples.shape[0]
    Xw1 = np.fft.fft(samples, axis=1) / nt
    Xw = np.zeros_like(Xw1[:, :nf])
    Xw[:, 0] = Xw1[:, 0] + Xw1[:, nf]
    Xw[:, 1:] = Xw1[:, 1:nf] + np.flip(Xw1[:, nf + 1:], axis=1)
    temp = np.zeros(shape=[nf, nf, nf, nf])
    for iter3 in range(len(samples)):
        temp = temp + np.einsum('i, j, k, l->ijkl', np.conj(Xw[iter3]), np.conj(Xw[iter3]), np.conj(Xw[iter3]),
                                Xw[iter3]) / nsamples
    return temp / df ** 3


def estimate_fifth_order_interactions(samples):
    nsamples = samples.shape[0]
    Xw1 = np.fft.fft(samples, axis=1) / nt
    Xw = np.zeros_like(Xw1[:, :nf])
    Xw[:, 0] = Xw1[:, 0] + Xw1[:, nf]
    Xw[:, 1:] = Xw1[:, 1:nf] + np.flip(Xw1[:, nf + 1:], axis=1)
    temp = np.zeros(shape=[nf, nf, nf, nf, nf])
    for iter4 in range(len(samples)):
        temp = temp + np.einsum('i, j, k, l, m->ijklm', np.conj(Xw[iter4]), np.conj(Xw[iter4]), np.conj(Xw[iter4]),
                                np.conj(Xw[iter4]), Xw[iter4]) / nsamples
    return temp / df ** 4


########################################################################################################################

# Input parameters
T = 20  # Time(1 / T = dw)
nt = 128  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 64  # Num of Discretized Freq.
nsamples = 100000  # Num.of samples

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

t_u = 2 * np.pi / (2 * F)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

########################################################################################################################
# Simulating with 2nd order interactions

# Target PSDF(stationary)
P = 20 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)
P[0] = 0.00001

phi = np.random.uniform(size=[nsamples, nf]) * 2 * np.pi
phi_e = np.exp(-1 * phi * 1.0j)
Coeff = np.sqrt(P * df)
B1 = phi_e * Coeff
B1[np.isnan(B1)] = 0
samples_SRM = np.fft.ifft(B1, nt) * nt

########################################################################################################################

# second_order_data_srm = estimate_second_order_interactions(samples_SRM)
# third_order_data_srm = estimate_third_order_interactions(samples_SRM)
# fourth_order_data_srm = estimate_fourth_order_interactions(samples_SRM)
# fifth_order_data_srm = estimate_fifth_order_interactions(samples_SRM)
#
# np.save('samples_srm_data.npy', samples_SRM)
# np.save('second_order_coupling_srm_data.npy', second_order_data_srm)
# np.save('third_order_coupling_srm_data.npy', third_order_data_srm)
# np.save('fourth_order_coupling_srm_data.npy', fourth_order_data_srm)

########################################################################################################################
# Simulating with 2nd and 3rd order interactions

# Generating the 2 dimensional mesh grid
fx = f
fy = f
Fx, Fy = np.meshgrid(fx, fy, indexing='ij')

b = 40 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
B_Real = b
B_Imag = b

B_Real[0, :] = 0
B_Real[:, 0] = 0
B_Imag[0, :] = 0
B_Imag[:, 0] = 0

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

Biphase = np.ones_like(B_Real) * np.pi / 4
Biphase_e = np.exp(Biphase * 1.0j)

PP = np.zeros_like(P)
sum_Bc2 = np.zeros_like(P)
Bc2 = np.zeros_like(B_Real)

PP[0] = P[0]
PP[1] = P[1]

for i in range(nf):
    wk = i
    for j in range(i):
        wj = np.array(j)
        wi = wk - wj
        if B_Ampl[wi, wj] > 0 and PP[wi] * PP[wj] != 0:
            Bc2[wi, wj] = B_Ampl[wi, wj] ** 2 / (PP[wi] * PP[wj] * P[wk]) * df
            sum_Bc2[wk] = sum_Bc2[wk] + Bc2[wi, wj]
        else:
            Bc2[wi, wj] = 0
    if sum_Bc2[wk] > 1:
        print('Results may not be as expected as sum of partial bicoherences is greater than 1')
        for j in range(np.int32(np.ceil((wk + 1) / 2))):
            wj = np.array(j)
            wi = wk - wj
            Bc2[wi, wj] = Bc2[wi, wj] / sum_Bc2[wk]
        sum_Bc2[wk] = 1
    PP[wk] = P[wk] * (1 - sum_Bc2[wk])

B2 = phi_e * np.sqrt(1 - sum_Bc2)
Bc = np.sqrt(Bc2)

for i in range(nf):
    for j in range(i):
        k = i - j
        B2[:, i] = B2[:, i] + Bc[k, j] * Biphase_e[k, j] * phi_e[:, k] * phi_e[:, j] / np.sqrt(2)

B2 = B2 * Coeff
B2[np.isnan(B2)] = 0
samples_BSRM = np.fft.ifft(B2, nt) * nt

########################################################################################################################

second_order_data_bsrm = estimate_second_order_interactions(samples_BSRM)
third_order_data_bsrm = estimate_third_order_interactions(np.real(samples_BSRM))
fourth_order_data_bsrm = estimate_fourth_order_interactions(samples_BSRM)

np.save('samples_bsrm_data.npy', samples_BSRM)
np.save('second_order_coupling_bsrm_data.npy', second_order_data_bsrm)
np.save('third_order_coupling_bsrm_data.npy', third_order_data_bsrm)
np.save('fourth_order_coupling_bsrm_data.npy', fourth_order_data_bsrm)

########################################################################################################################
# Simulating with 2nd ,3rd and 4th order interactions

# Generating the 3 dimensional mesh grid
fx = f
fy = f
fz = f
Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing='ij')

trispectrum = 40 / (2 * np.pi) * np.exp(-2 * (Fx ** 2 + Fy ** 2 + Fz ** 2))
T_Real = trispectrum
T_Imag = trispectrum

T_Real[0, :, :] = 0
T_Real[:, 0, :] = 0
T_Real[:, :, 0] = 0
T_Imag[0, :, :] = 0
T_Imag[:, 0, :] = 0
T_Imag[:, :, 0] = 0

T_Complex = T_Real + 1j * T_Imag
T_Ampl = np.absolute(T_Complex)

Triphase = np.ones_like(T_Real) * np.pi / 4
Triphase_e = np.exp(Triphase * 1.0j)

PP = np.zeros_like(P)
sum_Bc2_Tc2 = np.zeros_like(P)
Bc2 = np.zeros_like(B_Real)
Tc2 = np.zeros_like(T_Real)

PP[0] = P[0]
PP[1] = P[1]
PP[2] = P[2]

for i in range(nf):
    wl = i
    for j in range(i):
        wi = j
        wj = i - j
        if B_Ampl[wi, wj] > 0 and PP[wi] * PP[wj] != 0:
            Bc2[wi, wj] = B_Ampl[wi, wj] ** 2 / (PP[wi] * PP[wj] * P[wl]) * df
            sum_Bc2_Tc2[wl] = sum_Bc2_Tc2[wl] + Bc2[wi, wj]
        else:
            Bc2[wi, wj] = 0
        for k in range(wj):
            wj1 = k
            wk1 = wj - wj1
            if T_Ampl[wi, wj1, wk1] > 0 and PP[wi] * PP[wj1] * PP[wk1] != 0:
                Tc2[wi, wj1, wk1] = T_Ampl[wi, wj1, wk1] ** 2 / (PP[wi] * PP[wj1] * PP[wk1] * P[wl]) * df
                sum_Bc2_Tc2[wl] = sum_Bc2_Tc2[wl] + Tc2[wi, wj1, wk1]
            else:
                Tc2[wi, wj1, wk1] = 0
    if sum_Bc2_Tc2[wl] > 1:
        print('Results may not be as expected as sum of partial bicoherences is greater than 1')
        print(wl)
        break
    PP[wl] = P[wl] * (1 - sum_Bc2_Tc2[wl])

B3 = phi_e * np.sqrt(1 - sum_Bc2_Tc2)
Bc = np.sqrt(Bc2)
Tc = np.sqrt(Tc2)

for i in range(nf):
    for j in range(i):
        B3[:, i] = B3[:, i] + Bc[j, i - j] * Biphase_e[j, i - j] * phi_e[:, i - j] * phi_e[:, j] / np.sqrt(2)
        for k in range(j):
            B3[:, i] = B3[:, i] + Tc[k, j - k, i - j] * Triphase_e[k, j - k, i - j] * phi_e[:, k] * phi_e[:, j - k] * phi_e[:, i - j] / np.sqrt(6)

B3 = B3 * Coeff
B3[np.isnan(B3)] = 0
samples_TSRM = np.fft.ifft(B3, nt) * nt

samples = samples_TSRM
print(np.mean(samples.flatten() * np.conj(samples.flatten())))
print(np.mean(samples.flatten() * samples.flatten() * np.conj(samples.flatten())))
print(np.mean(samples.flatten() * samples.flatten() * samples.flatten() * np.conj(samples.flatten())))
print(np.mean(samples.flatten() * samples.flatten() * samples.flatten() * samples.flatten() * np.conj(samples.flatten())))

########################################################################################################################

second_order_data_tsrm = estimate_second_order_interactions(samples_TSRM)
third_order_data_tsrm = estimate_third_order_interactions(np.real(samples_TSRM))
fourth_order_data_tsrm = estimate_fourth_order_interactions(samples_TSRM)

np.save('samples_tsrm_data.npy', samples_BSRM)
np.save('second_order_coupling_tsrm_data.npy', second_order_data_tsrm)
np.save('third_order_coupling_tsrm_data.npy', third_order_data_tsrm)
np.save('fourth_order_coupling_tsrm_data.npy', fourth_order_data_tsrm)

########################################################################################################################

# print('The moment data is:')
# print('0th order moment', moment(samples_SRM.flatten(), moment=0))
# print('1st order moment', moment(samples_SRM.flatten(), moment=1))
# print('2nd order moment', moment(samples_SRM.flatten(), moment=2))
# print('3rd order moment', moment(samples_SRM.flatten(), moment=3))
# print('4th order moment', moment(samples_SRM.flatten(), moment=4))
# print('5th order moment', moment(samples_SRM.flatten(), moment=5))
#
# print('The cumulant data is:')
# print('0th order cumulant', moment(samples_SRM.flatten(), moment=0))
# print('1st order cumulant', moment(samples_SRM.flatten(), moment=1))
# print('2nd order cumulant', moment(samples_SRM.flatten(), moment=2))
# print('3rd order cumulant', moment(samples_SRM.flatten(), moment=3))
# print('4th order cumulant', moment(samples_SRM.flatten(), moment=4) - 3 * moment(samples_SRM.flatten(), moment=2) ** 2)
# print('5th order cumulant',
#       moment(samples_SRM.flatten(), moment=5) - 10 * moment(samples_SRM.flatten(), moment=3) * moment(samples_SRM.flatten(),
#                                                                                                       moment=2))
#

# soc_srm = np.load('second_order_coupling_srm_data.npy')
# soc_bsrm = np.load('second_order_coupling_bsrm_data.npy')
# toc_srm = np.load('third_order_coupling_srm_data.npy')
# toc_bsrm = np.load('third_order_coupling_bsrm_data.npy')
# foc_srm = np.load('fourth_order_coupling_srm_data.npy')
# foc_bsrm = np.load('fourth_order_coupling_bsrm_data.npy')
#
# soc_diff = soc_bsrm - soc_srm
# toc_diff = toc_bsrm - toc_srm
# foc_diff = foc_bsrm - foc_srm

# print('The moment data is:')
# print('0th order moment', moment(samples_BSRM.flatten(), moment=0))
# print('1st order moment', moment(samples_BSRM.flatten(), moment=1))
# print('2nd order moment', moment(samples_BSRM.flatten(), moment=2))
# print('3rd order moment', moment(samples_BSRM.flatten(), moment=3))
# print('4th order moment', moment(samples_BSRM.flatten(), moment=4))
# print('5th order moment', moment(samples_BSRM.flatten(), moment=5))
#
# print('The cumulant data is:')
# print('0th order cumulant', moment(samples_BSRM.flatten(), moment=0))
# print('1st order cumulant', moment(samples_BSRM.flatten(), moment=1))
# print('2nd order cumulant', moment(samples_BSRM.flatten(), moment=2))
# print('3rd order cumulant', moment(samples_BSRM.flatten(), moment=3))
# print('4th order cumulant', moment(samples_BSRM.flatten(), moment=4) - 3 * moment(samples_BSRM.flatten(), moment=2) ** 2)
# print('5th order cumulant',
#       moment(samples_BSRM.flatten(), moment=5) - 10 * moment(samples_BSRM.flatten(), moment=3) * moment(samples_BSRM.flatten(),
#                                                                                                         moment=2))



# # Checking simulations for the SRM Method
# Xw = np.fft.fft(np.real(samples_SRM), axis=1) / nt
# Xw1 = np.zeros_like(Xw[:, :nf])
# Xw1[:, 0] = Xw[:, 0] + Xw[:, nf]
# Xw1[:, 1:] = Xw[:, 1:nf] + np.flip(Xw[:, nf + 1:], axis=1)
#
# theoretical_srm = B1[0, :]
# practical_srm = Xw1[0, :]
#
# # Checking simulations for the BSRM Method
# Xw = np.fft.fft(samples_BSRM, axis=1) / nt
# Xw2 = np.zeros_like(Xw[:, :nf])
# Xw2[:, 0] = Xw[:, 0] + Xw[:, nf]
# Xw2[:, 1:] = Xw[:, 1:nf] + np.flip(Xw[:, nf + 1:], axis=1)
#
# theoretical_bsrm = B2[0, :]
# practical_bsrm = Xw2[0, :]

# # Checking simulations for the TSRM Method
# Xw = np.fft.fft(samples3, axis=1) / nt
# Xw3 = np.zeros_like(Xw[:, :nf])
# Xw3[:, 0] = Xw[:, 0] + Xw[:, nf]
# Xw3[:, 1:] = Xw[:, 1:nf] + np.flip(Xw[:, nf + 1:], axis=1)
#
# for i in range(nf):
#     for j in range(nf):
#         print(i, j)

# theoretical_tsrm = B3[0, :]
# practical_tsrm = Xw3[0, :]

# def estimate_spectra(samples):
#     Xw = np.fft.fft(samples, axis=1)
#     Xw = Xw[:, :nf]
#
#     s_P = np.zeros([nsamples, nf])
#     s_P = s_P + 1.0j * s_P
#     s_B = np.zeros([nsamples, nf, nf])
#     s_B = s_B + 1.0j * s_B
#     s_T = np.zeros([nsamples, nf, nf, nf])
#     s_T = s_T + 1.0j * s_T
#     # s_Q = np.zeros([nf, nf, nf, nf])
#     # s_Q = s_Q + 1.0j * s_Q
#
#     for i1 in range(nf):
#         s_P[:, i1] = s_P[:, i1] + Xw[:, i1] * np.conj(Xw[:, i1])
#         for i2 in range(nf - i1):
#             s_B[:, i1, i2] = s_B[:, i1, i2] + Xw[:, i1] * Xw[:, i2] * np.conj(Xw[:, i1 + i2])
#             for i3 in range(nf - i1 - i2):
#                 s_T[:, i1, i2, i3] = s_T[:, i1, i2, i3] + Xw[:, i1] * Xw[:, i2] * Xw[:, i3] * np.conj(
#                     Xw[:, i1 + i2 + i3])
#                 # for i4 in range(nf - i1 - i2 - i3):
#                 #     s_Q[i1, i2, i3, i4] = s_Q[i1, i2, i3, i4] + Xw[i, i1] * Xw[i, i2] * Xw[i, i3] * Xw[i, i4] * np.conj(
#                 #         Xw[i, i1 + i2 + i3 + i4])
#
#     m_P = np.mean(s_P, axis=0) * (T ** 1) / nt ** 2
#     m_B = np.mean(s_B, axis=0) * (T ** 2) / nt ** 3
#     m_T = np.mean(s_T, axis=0) * (T ** 3) / nt ** 4
#     # m_Q = s_Q * (T ** 4) / nt ** 5 / nsamples
#     return m_P, m_B, m_T
#
#
# spectra_list = Parallel(n_jobs=nbatches)(
#     delayed(estimate_spectra)(samples[i * nsamples:(i + 1) * nsamples]) for i in range(nbatches))
# P_spectra = np.zeros(shape=[nf])
# B_spectra = np.zeros(shape=[nf, nf])
# T_spectra = np.zeros(shape=[nf, nf, nf])
# for i in range(nbatches):
#     P_spectra = P_spectra + spectra_list[i][0] / nbatches
#     B_spectra = B_spectra + spectra_list[i][1] / nbatches
#     T_spectra = T_spectra + spectra_list[i][2] / nbatches

# m_P_Ampl = np.absolute(P_spectra)
# m_P_Real = np.real(P_spectra)
# m_P_Imag = np.imag(P_spectra)
#
# m_B[0, :] = 0
# m_B[:, 0] = 0
# m_B_Ampl = np.absolute(B_spectra)
# m_B_Real = np.real(B_spectra)
# m_B_Imag = np.imag(B_spectra)
#
# m_T[0, :, :] = 0
# m_T[:, 0, :] = 0
# m_T[:, :, 0] = 0
# m_T_Ampl = np.absolute(T_spectra)
# m_T_Real = np.real(T_spectra)
# m_T_Imag = np.imag(T_spectra)
#
# print('Order 2 comparision Samples:', moment(samples.flatten(), moment=2), ' Estimation:',
#       2 * np.sum(m_P_Real) * df ** 1)
# print('Order 3 comparision Samples:', moment(samples.flatten(), moment=3), ' Estimation:',
#       6 * np.sum(m_B_Real) * df ** 2)
# print('Order 4 comparision Samples:', moment(samples.flatten(), moment=4), ' Estimation:',
#       24 * np.sum(m_T_Real) * df ** 3)
#
# # Plotting the Estimated Real Bispectrum function
# fig = plt.figure(10)
# ax = fig.gca(projection='3d')
# h = ax.plot_surface(Fx, Fy, m_B_Real)
# plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()
#
# # Plotting the Estimated Imaginary Bispectrum function
# fig = plt.figure(11)
# ax = fig.gca(projection='3d')
# h = ax.plot_surface(Fx, Fy, m_B_Imag)
# plt.title('Estimated $\Im{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()

# # Simulation statistics checks
# print('The estimate of mean is', np.mean(samples), 'whereas the expected mean is 0.000')
# print('The estimate of variance is', np.var(samples.flatten(), axis=0), 'whereas the expected variance is',
#       np.sum(P) * 2 * df)
# print('The estimate of third moment is', moment(samples.flatten(), moment=3, axis=0), 'whereas the expected value is',
#       np.sum(b) * 6 * df ** 2)
# print('The estimate of skewness is', skew(samples.flatten(), axis=0), 'whereas the expected skewness is',
#       (np.sum(b) * 6 * df ** 2) / (np.sum(P) * 2 * df) ** (3 / 2))
# print('The estimate of skewness is', skew(samples.flatten()))
# print('The estimate of kurtosis is', kurtosis(samples.flatten()))
#
# plt.figure()
# plt.hist(samples.flatten(), bins=1000, normed=True)
# plt.show()

# import numpy as np
#
# # samples = np.load('samples.npy')
# # P_spectra = np.load('P_spectra.npy')
# # B_spectra = np.load('B_spectra.npy')
# # T_spectra = np.load('T_spectra.npy')
#
# # samples = np.load('samples.npy')
# P_spectra = np.load('P_spectra.npy')
# B_spectra = np.load('B_spectra.npy')
# T_spectra = np.load('T_spectra.npy')
# for i in range(1, nsim + 1):
#     # samples = np.concatenate((samples, np.load('data' + str(i) + '/samples.npy')), axis=0)
#     P_spectra = P_spectra + np.load('data' + str(i) + '/P_spectra.npy')
#     B_spectra = B_spectra + np.load('data' + str(i) + '/B_spectra.npy')
#     T_spectra = T_spectra + np.load('data' + str(i) + '/T_spectra.npy')
# P_spectra = P_spectra/(nsim+1)
# B_spectra = B_spectra/(nsim+1)
# T_spectra = T_spectra/(nsim+1)

# print(np.real(T_spectra[26])[1, 1], np.real(T_spectra[26])[2, 1])
# print(np.real(B_spectra)[1, 1], np.real(B_spectra)[2, 1])

# print('Order 2 comparision Samples:', moment(samples.flatten(), moment=2), ' Estimation:',
#       2 * np.sum(np.real(P_spectra)) * df ** 1)
# print('Order 3 comparision Samples:', moment(samples.flatten(), moment=3), ' Estimation:',
#       6 * np.sum(np.real(B_spectra)) * df ** 2)
# print('Order 4 comparision Samples:', moment(samples.flatten(), moment=4), ' Estimation:',
#       24 * np.sum(np.real(T_spectra)) * df ** 3)
#
# T_spectra[0, :, :] = 0
# T_spectra[:, 0, :] = 0
# T_spectra[:, :, 0] = 0

# # Plotting the Estimated Real Bispectrum function
# fig = plt.figure(10)
# ax = fig.gca(projection='3d')
# h = ax.plot_surface(Fx, Fy, np.absolute(T_spectra[26]))
# plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()

# # Plotting the Estimated Real Bispectrum function
# fig = plt.figure(10)
# ax = fig.gca(projection='3d')
# h = ax.plot_surface(Fx, Fy, np.imag(T_spectra[63]))
# plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()
