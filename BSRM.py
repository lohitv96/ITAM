from SRM import *
from tools import *
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

plt.style.use('seaborn')

# Make note of initial time
# Input parameters
norm = 'SRM'
Beta = 1.0
T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128 # Num of Discretized Freq.
nsamples = 100000  # Num.of samples
# W = 2*np.pi*F
# nw = nf

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

# Plotting the power spectrum
plt.figure(1)
plt.plot(f, P, label='True')
plt.title('Power spectrum')
plt.xlabel('$\omega$')
plt.ylabel('$S(\omega)$')
plt.show()

t_u = 2 * np.pi / (2 * 2 * np.pi * F)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

# Simulation step
P1 = deepcopy(P)
P1[0] = P1[0]/2
P1 = P1*2*np.pi
object = SRM(nsamples, P1, df, nt, nf)
samples = object.samples

# Estimate PSDF
plt.figure(1)
plt.plot(f, estimate_PSD(samples, nt, T)[1], label='estimated')
plt.legend()
plt.show()

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

# Plotting the Real part of the power spectrum
fig = plt.figure(2)
ax = fig.gca(projection='3d')
ax.plot_surface(Fx, Fy, B_Real)
plt.title('Target Real Bispectrum')
ax.set_xlabel('$\omega_1$(Hz)')
ax.set_ylabel('$\omega_2$(Hz)')
ax.set_zlabel('Target $\Re{B(\omega_1, \omega_2)}$')
plt.show()

# Plotting the Imaginary part of the power spectrum
fig = plt.figure(3)
ax = fig.gca(projection='3d')
ax.plot_surface(Fx, Fy, B_Imag)
plt.title('Target Imaginary Bispectrum')
ax.set_xlabel('$\omega_1$(Hz)')
ax.set_ylabel('$\omega_2$(Hz)')
ax.set_zlabel('Target $\Im{B(\omega_1, \omega_2)}$')
plt.show()

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

# Biphase
Biphase = np.arctan2(B_Imag, B_Real)
Biphase[np.isnan(Biphase)] = 0

# Plotting the Biphase angles
fig = plt.figure(4)
ax = fig.gca(projection='3d')
h = ax.plot_surface(Fx, Fy, Biphase * (180 / np.pi))
plt.title('Target Biphase')
ax.set_xlabel('$\omega_1$(Hz)')
ax.set_ylabel('$\omega_2$(Hz)')
ax.set_zlabel('Target $\Re{B(\omega_1, \omega_2)}$')
plt.show()

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

# Plotting the Bicohorence function
fig = plt.figure(5)
ax = fig.gca(projection='3d')
h = ax.plot_surface(Fx, Fy, (Bc2 + np.transpose(Bc2) - np.diag(np.diag(Bc2))))
plt.title('Bicoherence')
ax.set_xlabel('$\omega_1$')
ax.set_xlabel('$\omega_2$')
ax.set_xlabel('$b^2(\omega_1, \omega_2)$')
plt.show()

# Plotting the pure power spectrum function
plt.figure(6)
plt.plot(f, sum_Bc2)
plt.title('Sum of squared bicoherence function')
plt.xlabel('$F$')
plt.show()

# Plotting the pure power spectrum function
plt.figure(7)
plt.plot(f, P)
plt.plot(f, PP)
plt.title('Pure Power spectrum')
plt.xlabel('$F$')
plt.ylabel('$S(F)$')
plt.show()

# Simulation step - check again
# samples = np.zeros(shape=[nsamples, nt])

PP1 = deepcopy(PP)
PP1[0] = PP1[0]/2
PP1 = PP1*2*np.pi

F_old = SRM(nsamples, P1, df, nt, nf).samples
F1 = SRM(nsamples, PP1, df, nt, nf).samples

# Estimate Pure PSDF
plt.figure(8)
plt.plot(f, estimate_PSD(F_old, nt, T)[1], label='Actual')
plt.plot(f, estimate_PSD(F1, nt, T)[1], label='Pure')
plt.legend()
plt.show()

samples2 = np.zeros(shape=[10000, nt])

for m in range(10000):
    if m % 1000 == 0:
        print(m)
    Phi = 2 * np.pi * np.random.uniform(size=nf)
    for k in range(nf):
        if sum_Bc2[k] > 0:
            Coeff3 = 2 * np.sqrt(df * P[k])
            for l in range(int(np.ceil((k+1)/2))):
                f1 = k - l
                f2 = l
                if Bc2[f2, f1] > 0:
                    for j in range(nt):
                        samples2[m, j] = samples2[m, j] + Coeff3 * math.sqrt(Bc2[f2, f1]) * math.cos(2 * math.pi * (f[f2] + f[f1]) * t[j] - Phi[f2] - Phi[f1] - Biphase[f2, f1])

F_new = F1 + samples2

# Estimating from the BSRM simulation
plt.figure(9)
plt.plot(f, estimate_PSD(F_new, nt, T)[1], label='estimated')
plt.legend()
plt.show()

# Xw = np.fft.fft(F_new, axis=1)
# Xw = Xw[:, :int(nt/2)]
#
# # Bispectrum
# s_B = np.zeros([int(nt/2), int(nt/2)])
# s_B = s_B + 1.0j*s_B
# for k in range(1000):
#     for i1 in range(int(nt / 2)):
#         for i2 in range(int(nt / 2) - i1):
#             s_B[i1, i2] = s_B[i1, i2] + (Xw[k, i1] * Xw[k, i2] * np.conj(Xw[k, i1 + i2]) / nt ** 2 / (nt / T)) * T
# m_B = s_B / nsamples
#
# # # # Set zero on X & Y axis
# m_B[0, :] = 0
# m_B[:, 0] = 0
#
# # amplitude
# m_B_Ampl = np.absolute(m_B)
# # real & imag
# m_B_Real = np.real(m_B)
# m_B_Imag = np.imag(m_B)

# # Plotting the Estimated Real Bispectrum function
# fig = plt.figure(7)
# ax = fig.gca(projection='3d')
# h = ax.plot_surface(w[1:], w[1:], m_B_Real)
# plt.title('Estimated Real Bispectrum function')
# ax.set_xlabel('$\omega_1$')
# ax.set_xlabel('$\omega_2$')
# ax.set_xlabel('$B(\omega_1, \omega_2)$')
# plt.show()

# plt.figure(7)
# h = mesh(0:(1 / T): (1 / (2 * dt) - 1 / T), 0: (1 / T):(1 / (2 * dt) - 1 / T), m_B_Real)
# plt.title('Estimated Real Bispectrum')
# plt.xlim([0 50])
# plt.ylim([0 50])
# plt.zlim([-0.1 0.1])
# plt.xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# plt.ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# plt.zlabel('Estimated $\Re{B}(f_1, f_2)$', 'Interpreter', 'latex')
# plt.show()

# plt.figure(8)
# h = mesh(0:(1 / T): (1 / (2 * dt) - 1 / T), 0: (1 / T):(1 / (2 * dt) - 1 / T), m_B_Imag)
# plt.title('Estimated Imaginary Bispectrum')
# plt.xlim([0 50])
# plt.ylim([0 50])
# plt.zlim([-0.1 0.1])
# plt.xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# plt.ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# plt.zlabel('Estimated $\Im{B}(f_1, f_2)$', 'Interpreter', 'latex')

# D_P11 = (diff(P11')/dt)'
# D_P33 = (diff(P33')/dt)'
"""
# variance
Target_var = 2 * np.dot(P, df)
Target_var_D = 2 * sum((2 * np.pi) ^ 2 * f. ^ 2. * P * df)

P11_var = mean(var(P11))
P33_var = mean(var(P33))
D_P11_var = mean(var(D_P11))
D_P33_var = mean(var(D_P33))

            # skewness
Target_skew_unnorm = sum(sum(B_Real * df * df)) * 6
Target_skew = Target_skew_unnorm / (sqrt(Target_var)) ^ 3
sum_B = 0
for i = 1:Nf / 2
for j=1:Nf / 2 \
        # sum_B = sum_B + (2 * np.pi) ^ 3 * B_Imag(i, j) * f(i) * f(j) * f(i + j) * df * df
sum_B = sum_B + (2 * np.pi) ^ 3 * B_Imag(i, j) * f(i) * f(j) * f(i + j - 1) * df * df
end
end
Target_skew_D_unnorm = -sum_B * 6 # # # # # # # # # (MINUS?!?!?!?!?!?!??!?!?!?!?!?!)
Target_skew_D = Target_skew_D_unnorm / (sqrt(Target_var_D)) ^ 3

P11_skew = mean(skewness(P11))
P33_skew = mean(skewness(P33))
D_P11_skew = mean(skewness(D_P11))
D_P33_skew = mean(skewness(D_P33))
P11_skew_unnorm = mean(skewness(P11). * ((var(P11)). ^ (3 / 2)))
P33_skew_unnorm = mean(skewness(P33). * ((var(P33)). ^ (3 / 2)))
D_P11_skew_unnorm = mean(skewness(D_P11). * ((var(D_P11)). ^ (3 / 2)))
D_P33_skew_unnorm = mean(skewness(D_P33). * ((var(D_P33)). ^ (3 / 2)))

                    # estimated
histogram
figure(41)
histogram(P33(:), 100, 'Normalization', 'pdf')
# estimated
histogram
figure(42)
histogram(D_P33(:), 100, 'Normalization', 'pdf')

# estimated
histogram
figure(51)
histogram(P11(:), 100, 'Normalization', 'pdf')
# estimated
histogram
figure(52)
histogram(D_P11(:), 100, 'Normalization', 'pdf')

# samples
for i=1:1
figure(10 + i)
h = plot(t, P11(i,:), 'k--')
set(h, 'Linewidth', 1.5)
hold
on
h = plot(t, P33(i,:), 'r')
set(h, 'Linewidth', 1.5)
# title('Sample process')
end

# Derivative
samples
for i=1:1
figure(20 + i)
h = plot(t(1:size(D_P11, 2)), D_P11(i,:), 'k--')
set(h, 'Linewidth', 1.5)
hold
on
h = plot(t(1:size(D_P33, 2)), D_P33(i,:), 'r')
set(h, 'Linewidth', 1.5)
# title('Derivative of Sample process')
end

# figure(14)
# xlim([0 T / 2])
# ylim([-4 4])
# # title('Sample process')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$t$(sec.)', 'Interpreter', 'latex')
# ylabel('$f(t)$', 'Interpreter', 'latex')
# h0 = legend('$f_0(t)$', '$f_1(t)$')
# set(h0, 'Interpreter', 'latex') \
  # set(gca, 'LineWidth', 1.5)
# saveas(gcf, 'Sample.fig')
  # saveas(gcf, 'Sample', 'epsc')
  #
  #
  # figure(24)
  # xlim([0 T / 2])
  # ylim([-800 800])
  # # title('Sample process of derivative')
  # set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
  # xlabel('$t$(sec.)', 'Interpreter', 'latex')
# ylabel('$\frac{\partial{f(t)}}{\partial{t}}$', 'Interpreter', 'latex')
# h0 = legend('$\frac{\partial{f_0(t)}}{\partial{t}}$', '$\frac{\partial{f_1(t)}}{\partial{t}}$')
# set(h0, 'Interpreter', 'latex') \
  # set(gca, 'LineWidth', 1.5)
# saveas(gcf, 'D_Sample.fig')
  # saveas(gcf, 'D_Sample', 'epsc')

# print('BSRM: CPU time = #d \n', cputime - start)
"""