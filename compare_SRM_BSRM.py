import matplotlib.pyplot as plt
from scipy.stats import skew, moment, kurtosis
from mpl_toolkits.mplot3d import Axes3D
from BSRM import *
from SRM import *
from joblib import Parallel, delayed

plt.style.use('seaborn')

# Input parameters
T = 20  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128  # Num of Discretized Freq.
nsamples = 100  # Num.of samples
nbatches = 12
ncores = 4

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

# Target PSDF(stationary)
P = 20 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)

t_u = 2 * np.pi / (2 * 2 * np.pi * F)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

# Generating the 2 dimensional mesh grid
fx = f
fy = f
Fx, Fy = np.meshgrid(fx, fy)

b = 20 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
B_Real = b
B_Imag = b

B_Real[0, :] = 0
B_Real[:, 0] = 0
B_Imag[0, :] = 0
B_Imag[:, 0] = 0

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)


def simulate():
    obj = BSRM(nsamples, P, B_Complex, dt, df, nt, nf)
    samples_BSRM = obj.samples
    return samples_BSRM
    # obj = SRM(nsamples, P, df, nt, nf)
    # samples_SRM = obj.samples
    # return samples_SRM


def estimate_spectra(samples):
    Xw = np.fft.fft(samples, axis=1)
    Xw = Xw[:, :nf]

    s_P = np.zeros([nsamples, nf])
    s_P = s_P + 1.0j * s_P
    s_B = np.zeros([nsamples, nf, nf])
    s_B = s_B + 1.0j * s_B
    s_T = np.zeros([nsamples, nf, nf, nf])
    s_T = s_T + 1.0j * s_T
    # s_Q = np.zeros([nf, nf, nf, nf])
    # s_Q = s_Q + 1.0j * s_Q

    for i1 in range(nf):
        s_P[:, i1] = s_P[:, i1] + Xw[:, i1] * np.conj(Xw[:, i1])
        for i2 in range(nf - i1):
            s_B[:, i1, i2] = s_B[:, i1, i2] + Xw[:, i1] * Xw[:, i2] * np.conj(Xw[:, i1 + i2])
            for i3 in range(nf - i1 - i2):
                s_T[:, i1, i2, i3] = s_T[:, i1, i2, i3] + Xw[:, i1] * Xw[:, i2] * Xw[:, i3] * np.conj(
                    Xw[:, i1 + i2 + i3])
                # for i4 in range(nf - i1 - i2 - i3):
                #     s_Q[i1, i2, i3, i4] = s_Q[i1, i2, i3, i4] + Xw[i, i1] * Xw[i, i2] * Xw[i, i3] * Xw[i, i4] * np.conj(
                #         Xw[i, i1 + i2 + i3 + i4])

    m_P = np.mean(s_P, axis=0) * (T ** 1) / nt ** 2
    m_B = np.mean(s_B, axis=0) * (T ** 2) / nt ** 3
    m_T = np.mean(s_T, axis=0) * (T ** 3) / nt ** 4
    # m_Q = s_Q * (T ** 4) / nt ** 5 / nsamples
    return m_P, m_B, m_T


samples_list = Parallel(n_jobs=ncores)(delayed(simulate)() for _ in range(nbatches))
samples = np.concatenate(samples_list, axis=0)

spectra_list = Parallel(n_jobs=ncores)(
    delayed(estimate_spectra)(samples[i * nsamples:(i + 1) * nsamples]) for i in range(nbatches))

P_spectra = np.zeros(shape=[nf])
B_spectra = np.zeros(shape=[nf, nf])
T_spectra = np.zeros(shape=[nf, nf, nf])

for i in range(nbatches):
    P_spectra = P_spectra + spectra_list[i][0] / nbatches
    B_spectra = B_spectra + spectra_list[i][1] / nbatches
    T_spectra = T_spectra + spectra_list[i][2] / nbatches

print('Order 2 comparision Samples:', moment(samples.flatten(), moment=2), ' Estimation:',
      2 * np.sum(np.real(P_spectra)) * df ** 1,  ' Actual:', 2 * np.sum(P) * df ** 1)
print('Order 3 comparision Samples:', moment(samples.flatten(), moment=3), ' Estimation:',
      6 * np.sum(np.real(B_spectra)) * df ** 2, ' Actual:', 6 * np.sum(B_Real) * df ** 2)
print('Trying to understand:')
print('Order 4 comparision Samples:', moment(samples.flatten(), moment=4), ' Estimation:',
      3*(2 * np.sum(np.real(P_spectra)) * df ** 1)**2 + 24 * np.sum(np.real(T_spectra)) * df ** 3)

# print('Order 4 comparision Samples:', moment(samples.flatten(), moment=4), ' Estimation:',
#       64 * np.sum(np.real(T_spectra)) * df ** 3)


# MARCC data post processing
# import numpy as np

# samples = np.load('samples.npy')
# P_spectra = np.load('P_spectra.npy')
# B_spectra = np.load('B_spectra.npy')
# T_spectra = np.load('T_spectra.npy')

# nsim = 79
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
# h = ax.plot_surface(Fx, Fy, np.real(T_spectra[26]))
# plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()

# # Plotting the Estimated Real Bispectrum function
# fig = plt.figure(10)
# ax = fig.gca(projection='3d')
# h = ax.plot_surface(Fx, Fy, np.imag(T_spectra[26]))
# plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()
