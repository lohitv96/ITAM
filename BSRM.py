from SRM import *
from tools import *
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from BSRM_class import *

plt.style.use('seaborn')

# Input parameters
T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128 # Num of Discretized Freq.
nsamples = 10000  # Num.of samples

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)
# dw = 2*np.pi*df
# w = 2*np.pi*f

# Target PSDF(stationary)
# case 1(var=20)
P = 20 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)
# case 2
# p = 20 * 1 / sqrt(2 * np.pi * 1) * exp(-1 / (2 * 1) * (f - 2). ^ 2)
# case 3(var=2 * 20 * 0.9772 = 39.0880)
# p = 40 * 1 / sqrt(2 * np.pi * 0.25) * exp(-1 / (2 * 0.25) * (f - 2). ^ 2)
# case 4
# p = (f <= 5) * 14
# case5
# p = (f == 10)* 0.25 / df + (f == 20) * 0.25 / df + (f == 30)* 0.25 / df + (f == 40) * 0.25 / df

# # Plotting the power spectrum
# plt.figure(1)
# plt.plot(f, P, label='Target')
# plt.title('$P(\omega)$')
# plt.xlabel('$\omega$')
# plt.show()

t_u = 2 * np.pi / (2 * 2 * np.pi * F)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

# Generating the 2 dimensional mesh grid
fx = f
fy = f
Fx, Fy = np.meshgrid(f, f)

# Generating the bispectrum
# case 1 (skew_unormal: 40 / 4 * 6 = 60)
b = 40 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
#
# # case 2 (skew_unnorml: 10 * 6 = 60)
# b = 10 * 1 / (2 * np.pi * 0.5) * np.exp((-1 / (2 * 0.5) * ((fx - 2.5)** 2 + (fy - 2.5)**2)))
#
# # case 3 (skew_unnorml: 10 * 6 = 120)
# b = 10 * 1 / (2 * np.pi * 0.0625) * np.exp((-1 / (2 * 0.0625) * ((fx - 1.0)** 2 + (fy - 1.0)** 2)))
#
# # case 4
# b = (fx <= 2.5) * (fy <= 2.5) * 20
#
# # case5(skew_unnorml: 20 * 6 = 120)
# b = (fx == 10)* (fy == 20) * 0.125 / df ^ 2 * 0.999999 + (fx == 20) * (fy == 10) * 0.125 / df ^ 2 * 0.999999
#
# b = 1 / np.sqrt(2) * ((fx == 10) * (fy == 20) * 0.125 / df ^ 2 * 0.999999 + (fx == 20) * (
#        fy == 10) * 0.125 / df ^ 2 * 0.999999 + (fx == 10) * (fy == 30) * 0.125 / df ^ 2 * 0.999999 + (fx == 30) * (
#                       fy == 10) * 0.125 / df ^ 2 * 0.999999)
# b = 1 / np.sqrt(2) * ((fx == 10) * (fy == 20) * 0.125 / df ^ 2 * 0.999999 + (fx == 20) * (
#        fy == 10) * 0.125 / df ^ 2 * 0.999999 + (fx == 10) * (fy == 30) * 0.125 / df ^ 2 * 0.999999 + (fx == 30) * (
#                       fy == 10) * 0.125 / df ^ 2 * 0.999999)

# real part
B_Real = b
# imaginary part
B_Imag = b

B_Real[0, :] = 0
B_Real[:, 0] = 0
B_Imag[0, :] = 0
B_Imag[:, 0] = 0

# # Plotting the Real part of the power spectrum
# fig = plt.figure(2)
# ax = fig.gca(projection='3d')
# ax.plot_surface(Fx, Fy, B_Real)
# plt.title('Target $\Re{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$(Hz)')
# ax.set_ylabel('$\omega_2$(Hz)')
# plt.show()
#
# # Plotting the Imaginary part of the power spectrum
# fig = plt.figure(3)
# ax = fig.gca(projection='3d')
# ax.plot_surface(Fx, Fy, B_Imag)
# plt.title('Target $\Im{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$(Hz)')
# ax.set_ylabel('$\omega_2$(Hz)')
# plt.show()

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

# Biphase
Biphase = np.arctan2(B_Imag, B_Real)
Biphase[np.isnan(Biphase)] = 0

# # Plotting the Biphase angles
# fig = plt.figure(4)
# ax = fig.gca(projection='3d')
# ax.plot_surface(Fx, Fy, Biphase * (180 / np.pi))
# plt.title('Target $Biphase(\omega_1, \omega_2)$')
# ax.set_xlabel('$\omega_1$(Hz)')
# ax.set_ylabel('$\omega_2$(Hz)')
# plt.show()

# Bicoherence(Bc2), Pure Power spectrum(PP)
Bc2 = np.zeros_like(B_Real)
PP = np.zeros_like(P)
sum_Bc2 = np.zeros(nf)
PP[0] = P[0]
PP[1] = P[1]

for i in range(nf):
    for j in range(int(np.ceil((i + 1) / 2))):
        w1 = i - j
        w2 = j
        if B_Ampl[w2, w1] > 0 and PP[w2] * PP[w1] != 0:
            Bc2[w2, w1] = B_Ampl[w2, w1] ** 2 / (PP[w2] * PP[w1] * P[i]) * df
            sum_Bc2[i] = sum_Bc2[i] + Bc2[w2, w1]
        else:
            Bc2[w2, w1] = 0
    if sum_Bc2[i] > 1:
        for j in range(int(np.ceil((i + 1) / 2))):
            w1 = i - j
            w2 = j
            Bc2[w2, w1] = Bc2[w2, w1] / sum_Bc2[i]
        sum_Bc2[i] = 1

    PP[i] = P[i] * (1 - sum_Bc2[i])

# # Plotting the Bicohorence function
# fig = plt.figure(5)
# ax = fig.gca(projection='3d')
# ax.plot_surface(Fx, Fy, (Bc2 + np.transpose(Bc2) - np.diag(np.diag(Bc2))))
# plt.title('$b^2(\omega_1, \omega_2)$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()

# # Plotting the pure power spectrum function
# plt.figure(6)
# plt.plot(f, sum_Bc2)
# plt.title('Sum of squared bicoherence function')
# plt.xlabel('$\omega$')
# plt.show()

# # Plotting the pure power spectrum function
# plt.figure(7)
# plt.plot(f, P, label='Total')
# plt.plot(f, PP, label='Pure')
# plt.title('Comparing the Pure part of the Power spectrum')
# plt.xlabel('$\omega$')
# plt.ylabel('$P(F)$')
# plt.legend()
# plt.show()

PP1 = deepcopy(PP)
PP1[0] = PP1[0]/2

Phi = np.random.uniform(size=[nsamples, nf]) * 2 * np.pi

P1 = deepcopy(P)
P1[0] = P1[0]/2

B_old = 2 * np.exp(Phi * 1.0j) * np.sqrt(P1 * df)
F_old = np.fft.fftn(B_old, [nt])
F_old = np.real(F_old)

B = 2 * np.exp(Phi * 1.0j) * np.sqrt(PP1 * df)
F1 = np.fft.fftn(B, [nt])
F1 = np.real(F1)

F2 = np.zeros(shape=[nsamples, nt])
Coeff = 2 * np.sqrt(df * P)

for k in range(nf):
    if sum_Bc2[k] > 0:
        for l in range(int(np.ceil((k+1)/2))):
            f1 = k - l
            f2 = l
            if Bc2[f2, f1] > 0:
                for j in range(nt):
                    F2[:, j] = F2[:, j] + Coeff[k] * np.sqrt(Bc2[f2, f1]) * np.cos(2 * np.pi * (f[f2] + f[f1]) * t[j] - Phi[:, f2] - Phi[:, f1] - Biphase[f2, f1])
F_new = F1 + F2

# Estimating from the BSRM simulation
plt.figure(9)
plt.plot(f, P, label='Target')
plt.plot(f, estimate_PSD(F_new, nt, T)[1], label='BSRM Simulation')
plt.legend()
plt.show()

Xw = np.fft.fft(F_new, axis=1)
Xw = Xw[:, :nf]

# Bispectrum
s_B = np.zeros([nsamples, nf, nf])
s_B = s_B + 1.0j*s_B
for i1 in range(nf):
    for i2 in range(nf - i1):
        s_B[:, i1, i2] = s_B[:, i1, i2] + (Xw[:, i1] * Xw[:, i2] * np.conj(Xw[:, i1 + i2]) / nt ** 2 / (nt / T)) * T
m_B = np.mean(s_B, axis=0)
# # # Set zero on X & Y axis
m_B[0, :] = 0
m_B[:, 0] = 0

m_B_Ampl = np.absolute(m_B)
m_B_Real = np.real(m_B)
m_B_Imag = np.imag(m_B)

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
# print('The total time taken is ', time.time() - time1)

