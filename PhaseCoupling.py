import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tools import *

plt.style.use('seaborn')

# # Generate a series of theta
nsamples = 100000

F = 50
nf = 100
df = F/nf

T = 1/df
nt = 200
dt = T/nt

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

# Plotting the Real part of the Bispectrum
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Fx, Fy, B_Real)
plt.title('Target $\Re{B(\omega_1, \omega_2)}$')
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
plt.show()

# # Plotting the Imaginary part of the Bispectrum
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Fx, Fy, B_Imag)
# plt.title('Target $\Im{B(\omega_1, \omega_2)}$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()
#
# # Plotting the Amplitude of the Bispectrum
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Fx, Fy, B_Ampl)
# plt.title('Target $\B(\omega_1, \omega_2)$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()
#
# # Plotting the Biphase Angle of the Bispectrum
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Fx, Fy, Biphase)
# plt.title('Target $Biphase(\omega_1, \omega_2)$')
# ax.set_xlabel('$\omega_1$')
# ax.set_ylabel('$\omega_2$')
# plt.show()

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

plt.figure()
plt.plot(f, PP)
plt.show()

# Plotting the Bicohorence function
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Fx, Fy, (Bc2 + np.transpose(Bc2) - np.diag(np.diag(Bc2))))
plt.title('Target $b^2(\omega_1, \omega_2)$')
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
plt.show()

theta_1 = 2*np.pi*np.random.uniform(0, 1, nsamples)

# # Plotting the Theta_1 Angles
# plt.figure()
# ax = plt.subplot(111, polar=True)
# ax.hist(theta_1, bins=100, normed='True')
# ax.set_title('Estimated PDF of $theta_1$')
# plt.show()

Mu_11 = 0
rho_11 = np.sqrt(0)
if rho_11 == 0:
    biphase_11 = np.random.uniform(0, 2*np.pi, size=nsamples) + Mu_11
elif 0 < rho_11 < 1:
    biphase_11 = stats.wrapcauchy.rvs(rho_11, size=nsamples) + Mu_11
else:
    biphase_11 = 0

# # Plotting the Biphase_11 random variates
# plt.figure(2)
# ax = plt.subplot(111, polar=True)
# ax.hist(biphase_11, bins=100, normed='True')
# ax.set_title('Estimated PDF of $biphase_{11}$')
# plt.show()

theta_2 = theta_1 + theta_1 + biphase_11

# # Plotting the Theta_2 Angles
# plt.figure()
# ax = plt.subplot(111, polar=True)
# ax.hist(theta_2, bins=100, normed='True')
# ax.set_title('Estimated PDF of $theta_2$')
# plt.show()

Mu_21 = 0
rho_21 = 0
if rho_11 == 0:
    biphase_21 = np.random.uniform(0, 2*np.pi, size=nsamples) + Mu_11
elif 0 < rho_11 < 1:
    biphase_21 = stats.wrapcauchy.rvs(rho_11, size=nsamples) + Mu_11
else:
    biphase_21 = 0

# # Plotting the Biphase_21 random variates
# plt.figure(2)
# ax = plt.subplot(111, polar=True)
# ax.hist(biphase_21, bins=100, normed='True')
# ax.set_title('Estimated PDF of $biphase_{11}$')
# plt.show()

theta_3 = theta_2 + theta_1 + biphase_21

# # Plotting the Theta_3 Angles
# plt.figure()
# ax = plt.subplot(111, polar=True)
# ax.hist(theta_3, bins=100, normed='True')
# ax.set_title('Estimated PDF of $theta_{21}$')
# plt.show()

Mu_22 = 0
rho_22 = np.sqrt(0.8)
biphase_22 = stats.wrapcauchy.rvs(rho_22, size=nsamples)

Mu_31 = 0
rho_31 = np.sqrt(0.2)
biphase_31 = stats.wrapcauchy.rvs(rho_31, size=nsamples)

theta_22 = theta_2 + theta_2 + biphase_22
theta_31 = theta_3 + theta_1 + biphase_31

theta_4 = 0.5*(theta_22 + theta_31)

# Plotting the Theta_4 Angles
plt.figure()
ax = plt.subplot(111, polar=True)
ax.hist(theta_4 - 0.5*(theta_2 + theta_2 + theta_3 + theta_1), bins=100, normed='True')
ax.set_title('Estimated PDF of $theta_4$')
plt.show()

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

# Estimate PSDF
plt.figure()
plt.plot(f, P, label='Target')
plt.plot(f, estimate_PSD(F, nt, T)[1], label='Simulation')
plt.legend()
plt.show()

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
plt.title('Estimated $\Re{B(\omega_1, \omega_2)}$')
ax.set_xlabel('$\omega_1$')
ax.set_ylabel('$\omega_2$')
plt.show()
