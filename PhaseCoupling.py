import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools import *
from BSRM import *

plt.style.use('seaborn')

# # Generate a series of theta
nsamples = 20000

# F = 50
# nf = 100
# df = F / nf
#
# T = 1 / df
# nt = 200
# dt = T / nt
#
# f = np.linspace(0, F - df, nf)
# t = np.linspace(0, T - dt, nt)
#
# P = (f == 10) * 0.25 / df + (f == 20) * 0.25 / df + (f == 30) * 0.25 / df + (f == 40) * 0.25 / df
# fx = f
# fy = f
# Fx, Fy = np.meshgrid(f, f)
# B = 1 / np.sqrt(2) * ((Fx == 20) * (Fy == 20) * 0.25 / df ** 2 * 0.999999 + (Fx == 10) * (
#         Fy == 30) * 0.125 / df ** 2 * 0.999999 + (Fx == 30) * (
#                               Fy == 10) * 0.125 / df ** 2 * 0.999999)
#
# B_Real = B
# B_Imag = B
# B_Complex = B_Real + 1j * B_Imag

# Input parameters
T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128 # Num of Discretized Freq.

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
Biphase = np.arctan2(B_Imag, B_Real)
Biphase[np.isnan(Biphase)] = 0

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
            f1 = i - j
            f2 = j
            Bc2[f2, f1] = Bc2[f2, f1] / sum_Bc2[i]
        sum_Bc2[i] = 1

    PP[i] = P[i] * (1 - sum_Bc2[i])

########################################################################################################################
# BSRM Case
import time

time1 = time.time()
P1 = np.copy(P)
P1[0] = P1[0]/2
obj = BSRM(nsamples, P1, B_Complex, dt, df, nt, nf)
samples_BSRM = obj.samples
print('Time to run BSRM is', time.time() - time1)

Phi = obj.phi

P_BSRM = estimate_PSD(samples_BSRM, nt, T)[1]

plt.figure()
plt.plot(P_BSRM)
plt.plot(P)
plt.show()

Xw = np.fft.fft(samples_BSRM, axis=1)
Xw = Xw[:, :nf]

# Bispectrum
s_B = np.zeros([nsamples, nf, nf])
s_B = s_B + 1.0j * s_B
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

########################################################################################################################
# Reconstructing the phase angles

B1 = np.fft.ifft(samples_BSRM)
B2 = B1[:, :nf]
B2[:, 0] = B2[:, 0] + B1[:, 128]
temp = np.conjugate(np.flip(B1[:, nf+1:], axis=1))
B2[:, 1:] = B2[:, 1:] + np.conjugate(np.flip(B1[:, nf+1:], axis=1))

########################################################################################################################
# BSRM with Fourier Transform

time2 = time.time()
Coeff = 2 * np.sqrt(df * P)
Coeff[0] = Coeff[0] / np.sqrt(2)

Phi_e = np.exp(Phi * 1.0j)
Biphase_e = np.exp(Biphase * 1.0j)
B = np.sqrt(1 - sum_Bc2) * Phi_e
Bc = np.sqrt(Bc2)

for i in range(nf):
    for j in range(1, int(np.ceil((i+1)/2))):
        f1 = j
        f2 = i - j
        B[:, i] = B[:, i] + Bc[f1, f2] * Biphase_e[f1, f2] * Phi_e[:, f1] * Phi_e[:, f2]

B = B * Coeff
B[np.isnan(B)] = 0
samples_PC = np.fft.fftn(B, [nt])
samples_PC = np.real(samples_PC)
print('Time taken for BSRM with FFt is', time.time() - time2)

P_PC = estimate_PSD(samples_PC, nt, T)[1]

plt.figure()
plt.plot(P_PC)
plt.plot(P)
plt.show()