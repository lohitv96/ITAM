import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from BSRM import *

plt.style.use('seaborn')

# Input parameters
T = 20  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128 # Num of Discretized Freq.
nsamples = 512  # Num.of samples

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
Fx, Fy = np.meshgrid(f, f)

b = 40 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
B_Real = b
B_Imag = b

B_Real[0, :] = 0
B_Real[:, 0] = 0
B_Imag[0, :] = 0
B_Imag[:, 0] = 0

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

import time

time1 = time.time()
obj = BSRM(nsamples, P, B_Complex, dt, df, nt, nf, test='old')
samples = obj.samples
print('Time taken is', time.time() - time1)

# np.savetxt('Phi.txt', obj.phi)
# np.savetxt('samples_python.txt', obj.samples)

# Xw = np.fft.fft(samples, axis=1)
# Xw = Xw[:, :nf]
#
# # Bispectrum
# s_B = np.zeros([nsamples, nf, nf])
# s_B = s_B + 1.0j*s_B
# for i1 in range(nf):
#     for i2 in range(nf - i1):
#         s_B[:, i1, i2] = s_B[:, i1, i2] + (Xw[:, i1] * Xw[:, i2] * np.conj(Xw[:, i1 + i2]) / nt ** 2 / (nt / T)) * T
# m_B = np.mean(s_B, axis=0)
# # # # Set zero on X & Y axis
# m_B[0, :] = 0
# m_B[:, 0] = 0
#
# m_B_Ampl = np.absolute(m_B)
# m_B_Real = np.real(m_B)
# m_B_Imag = np.imag(m_B)

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
#
# print('The estimate of mean is', np.mean(samples), 'whereas the expected mean is 0.000')
# print('The estimate of variance is', np.mean(np.var(samples, axis=0)), 'whereas the expected variance is',
#       np.sum(P) * 2 * df)
#
# from scipy.stats import skew, moment
#
# print('The estimate of third moment is', np.mean(moment(samples, moment=3, axis=0)), 'whereas the expected value is',
#       np.sum(b) * 6 * df ** 2)
#
# print('The estimate of skewness is', np.mean(skew(samples, axis=0)), 'whereas the expected variance is',
#       (np.sum(b) * 6 * df ** 2) / (np.sum(P) * 2 * df) ** (3 / 2))
