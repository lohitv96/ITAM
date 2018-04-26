import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools import *
from BSRM import *

plt.style.use('seaborn')

# # Generate a series of theta
nsamples = 10000

F = 50
nf = 100
df = F / nf

T = 1 / df
nt = 200
dt = T / nt

f = np.linspace(0, F - df, nf)
t = np.linspace(0, T - dt, nt)

P = (f == 10) * 0.25 / df + (f == 20) * 0.25 / df + (f == 30) * 0.25 / df + (f == 40) * 0.25 / df
fx = f
fy = f
Fx, Fy = np.meshgrid(f, f)
B = 1 / np.sqrt(2) * ((Fx == 20) * (Fy == 20) * 0.25 / df ** 2 * 0.999999 + (Fx == 10) * (
        Fy == 30) * 0.125 / df ** 2 * 0.999999 + (Fx == 30) * (
                              Fy == 10) * 0.125 / df ** 2 * 0.999999)

B_Real = B
B_Imag = B
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

obj = BSRM(nsamples, P, B_Complex, dt, df, nt, nf)
samples = obj.samples
Phi = obj.phi

Xw = np.fft.fft(samples, axis=1)
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

# Plotting the Estimated Real Bispectrum function
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Fx, Fy, m_B_Real)
plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$ - BSRM Case')
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
plt.show()

print('Results for BSRM case')
print(B_Real[40, 40] / m_B_Real[40, 40])
print(B_Real[20, 60] / m_B_Real[20, 60])

B1 = np.fft.ifft(samples)
Phi_re = np.imag(np.log(B1[:, :nf] / 2 / np.sqrt(df * P)))

########################################################################################################################

theta_1 = Phi[:, 0]

theta_11 = np.exp((theta_1 + theta_1 + Biphase[0, 0]) * 1.0j)
theta_2 = np.imag(np.log(np.sqrt(1 - sum_Bc2[1]) * Phi[:, 1] + np.sqrt(Bc2[0, 0]) * theta_11))

theta_21 = np.exp((theta_2 + theta_1 + Biphase[1, 0]) * 1.0j)
theta_3 = np.imag(np.log(np.sqrt(1 - sum_Bc2[2]) * Phi[:, 2] + np.sqrt(Bc2[1, 0]) * theta_21))

# Mu_22 = Biphase[40, 40]
# rho_22 = np.sqrt(0.8)
# biphase_22 = stats.wrapcauchy.rvs(rho_22, size=nsamples) + Mu_22

# Mu_31 = Biphase[60, 20]
# rho_31 = np.sqrt(0.2)
# biphase_31 = stats.wrapcauchy.rvs(rho_31, size=nsamples) + Mu_31

# theta_22 = np.exp((theta_2 + theta_2 + Mu_22) * (1.0j))
# theta_31 = np.exp((theta_3 + theta_1 + Mu_31) * (1.0j))

# theta_4 = np.imag(np.log(np.sqrt(1 - sum_Bc2[80])*Phi[:,80] + rho_22*theta_22 + rho_31*theta_31))

Coeff = np.sqrt(df * P)
Coeff[0] = Coeff[0] / np.sqrt(2)
Phi = 2 * np.pi * np.random.uniform(size=[nsamples, nf])
Phi[:, 20] = theta_1
Phi[:, 40] = theta_2
Phi[:, 60] = theta_3
Phi[:, 80] = theta_4

B = 2 * np.exp(Phi * 1.0j) * Coeff
F = np.fft.fftn(B, [nt])
F = np.real(F)

Xw = np.fft.fft(F, axis=1)
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

# Plotting the Estimated Real Bispectrum function
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Fx, Fy, m_B_Real)
plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$ - Phase Coupling Case')
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
plt.show()

print('Results for the Phase Coupling case')
print(B_Real[40, 40] / m_B_Real[40, 40])
print(B_Real[20, 60] / m_B_Real[20, 60])
