import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')

# # parameter for von Mises distributions
# Mu = pi / 2
# Kappa = 1 / 2
# numsample = 10000
#
# samples = vmrand(Mu, Kappa, numsample, 1)
# figure
# histogram(samples, 100)
# xlim([-pi, pi])
# title('test')
# figure
# polarhistogram(samples, 100)
# title('test')
#
# population & sample variance
# PopVar = 1 - besseli(1, Kappa) / besseli(0, Kappa)
# SampleVar = 1 - abs(mean(exp(1i * samples)))
#
# # population $ sample mean
# PopMean = Mu
# SampleMean = atan2(imag(mean(exp(1i * samples)) / abs(mean(exp(1i * samples)))), ...
                 # real(mean(exp(1i * samples)) / abs(mean(exp(1i * samples)))))

# # Generate a series of theta
numsample = 10000
jumpfreq = 10

F = 50  # Frequency.(Hz)
nf = 100
df = F/nf
f = np.arange(0, nf)*df
P = (f == 10) * 0.25 / df + (f == 20) * 0.25 / df + (f == 30) * 0.25 / df + (f == 40) * 0.25 / df
fx = f
fy = f
Fx, Fy = np.meshgrid(f, f)
B = 1 / np.sqrt(2) * ((Fx == 20) * (Fy == 20) * 0.125 / df ** 2 * 0.999999 + (Fx == 10) * (
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

theta_1 = np.random.vonmises(0, 0, numsample)

# Plotting the Theta_1 Angles
plt.figure()
ax = plt.subplot(111, polar=True)
ax.hist(theta_1, bins=100, normed='True')
ax.set_title('Estimated PDF of $theta_1$')
plt.show()

Mu_11 = 0
rho_11 = np.sqrt(0)
if rho_11 == 0:
    biphase_11 = np.random.uniform(-np.pi, np.pi, size=numsample) + Mu_11
elif 0 < rho_11 < 1:
    biphase_11 = stats.wrapcauchy.rvs(rho_11, size=numsample) + Mu_11
else:
    biphase_11 = 0

# Plotting the Biphase_11 random variates
plt.figure(2)
ax = plt.subplot(111, polar=True)
ax.hist(biphase_11, bins=100, normed='True')
ax.set_title('Estimated PDF of $biphase_{11}$')
plt.show()

theta_2 = theta_1 + theta_1 + biphase_11

# Plotting the Theta_2 Angles
plt.figure()
ax = plt.subplot(111, polar=True)
ax.hist(theta_2, bins=100, normed='True')
ax.set_title('Estimated PDF of $theta_2$')
plt.show()

Mu_21 = 0
rho_21 = 0
if rho_11 == 0:
    biphase_21 = np.random.uniform(-np.pi, np.pi, size=numsample) + Mu_11
elif 0 < rho_11 < 1:
    biphase_21 = stats.wrapcauchy.rvs(rho_11, size=numsample) + Mu_11
else:
    biphase_21 = 0

# Plotting the Biphase_21 random variates
plt.figure(2)
ax = plt.subplot(111, polar=True)
ax.hist(biphase_21, bins=100, normed='True')
ax.set_title('Estimated PDF of $biphase_{11}$')
plt.show()

theta_3 = theta_2 + theta_1 + biphase_21

# Plotting the Theta_3 Angles
plt.figure()
ax = plt.subplot(111, polar=True)
ax.hist(theta_3, bins=100, normed='True')
ax.set_title('Estimated PDF of $theta_{21}$')
plt.show()

Mu_22_31 = 0
rho_22_31 = np.sqrt(0.5)
biphase_22_31 = stats.wrapcauchy.rvs(rho_22_31, size=numsample)
theta_4 = 0.5*(theta_2 + theta_2 + theta_3 + theta_1) + biphase_22_31

# Plotting the Theta_4 Angles
plt.figure()
ax = plt.subplot(111, polar=True)
ax.hist(theta_4, bins=100, normed='True')
ax.set_title('Estimated PDF of $theta_4$')
plt.show()

# # Mu_211 = 0
# # rho_211 = np.sqrt(0.5)
# # triphase_211 = wcrand(Mu_211, rho_211, numsample, 1)
# # theta_4 = theta_2 + theta_1 + theta_1 + triphase_211
#
# #
# #  \theta_2 + \theta_1 + \theta_1
# # Mu_211 = 0
# # Kappa_211 = 1
# # pdf_211 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_211(i,:)= circ_vmpdf(theta, Mu_211 + theta_2(i) + theta_1(i) + theta_1(i), Kappa_211)
# # end
# #
# # #  \theta_3 + \theta_2 - \theta_1
# # Mu_32m1 = 0
# # Kappa_32m1 = 100
# # pdf_32m1 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_32m1(i,:)= circ_vmpdf(theta, Mu_32m1 + theta_3(i) + theta_2(i) - theta_1(i), Kappa_32m1)
# # end
# #
# # #  \theta_3 + \theta_3 - \theta_2
# # Mu_33m2 = 0
# # Kappa_33m2 = 1
# # pdf_33m2 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_33m2(i,:)= circ_vmpdf(theta, Mu_33m2 + theta_3(i) + theta_3(i) - theta_2(i), Kappa_33m2)
# # end
# #
# #
# # theta_4 = np.zeros(numsample, 1)
# # for i=1:numsample
# # theta_4(i, 1) = UserCircRVGen(pdf_31(i,:).*pdf_22(i,:).*pdf_211(i,:).*pdf_32m1(i,:).*pdf_33m2(i,:), theta)
# # end
# #
# #
# #
# # figure
# # # figure
# histogram(theta_4 - (theta_3 + theta_1), 100)
# xlim([-pi, pi])
# # # title('\beta_{31}')
# # subplot(3, 2, 1)
# polarhistogram(theta_4 - (theta_3 + theta_1), 100)
# # title('\theta_4: \beta_{31}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # figure
# histogram(theta_4 - (theta_2 + theta_2), 100)
# xlim([-pi, pi])
# # title('\beta_{22}')
# # subplot(3, 2, 2)
# polarhistogram(theta_4 - (theta_2 + theta_2), 100)
# # title('\theta_4: \beta_{22}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# # #
# # # figure
# histogram(theta_4 - (theta_2 + theta_2 + theta_1), 100)
# xlim([-pi, pi])
# # # title('\gamma_{211}')
# # subplot(3, 2, 3)
# polarhistogram(theta_4 - (theta_2 + theta_1 + theta_1), 100)
# # title('\theta_4: \gamma_{211}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # # figure
# histogram(theta_4 - (theta_3 + theta_2 - theta_1), 100)
# xlim([-pi, pi])
# # # title('\gamma_{32-1}')
# # subplot(3, 2, 4)
# polarhistogram(theta_4 - (theta_3 + theta_2 - theta_1), 100)
# # title('\theta_4: \gamma_{32-1}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # # figure
# histogram(theta_4 - (theta_3 + theta_3 - theta_2), 100)
# xlim([-pi, pi])
# # # title('\gamma_{33-2}')
# # subplot(3, 2, 5)
# polarhistogram(theta_4 - (theta_3 + theta_3 - theta_2), 100)
# # title('\theta_4: \gamma_{33-2}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# 0
# p
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 2, 6)
# polarhistogram(theta_4, 100)
# # title('\theta_4')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
#
# # figure(6)
# h = polarhistogram(biphase_22, 100, 'Normalization', 'pdf')
# # set(gca, 'ThetaAxisUnits', 'radians')
# # set(gca, 'FontSize', 15, 'FontName', 'Times New Roman')
# # set(gca, 'GridAlpha', 0.5)
# # set(gca, 'Rlim', [0 1])
# # set(gca, 'RTickLabel', {'0.00', '0.20', '0.40', '0.60', '0.80', '1.00'})
# # set(gca, 'RMinorGrid', 'on')
# # set(gca, 'MinorGridLinestyle', ':')
# # set(gca, 'MinorGridAlpha', 0.5)
# # h.FaceColor = [0, 0, 1]
# # h.FaceAlpha = 0.5
# # h.LineStyle = 'none'
# # title('Estimated PDF of $\theta_4 - (\theta_2 \ + \theta_2)$', 'Interpreter', 'latex')
# # thetalim([-pi, pi])
#
# # figure(6)
# h = polarhistogram(triphase_211, 100, 'Normalization', 'pdf')
# # set(gca, 'ThetaAxisUnits', 'radians')
# # set(gca, 'FontSize', 15, 'FontName', 'Times New Roman')
# # set(gca, 'GridAlpha', 0.5)
# # set(gca, 'Rlim', [0 1])
# # set(gca, 'RTickLabel', {'0.00', '0.20', '0.40', '0.60', '0.80', '1.00'})
# # set(gca, 'RMinorGrid', 'on')
# # set(gca, 'MinorGridLinestyle', ':')
# # set(gca, 'MinorGridAlpha', 0.5)
# # h.FaceColor = [0, 0, 1]
# # h.FaceAlpha = 0.5
# # h.LineStyle = 'none'
# # title('Estimated PDF of $\theta_4 - (\theta_2 \ + \theta_2 + \theta_1)$', 'Interpreter', 'latex')
# # thetalim([-pi, pi])
# #
# #
# # figure(7)
# h = polarhistogram(theta_4, 100, 'Normalization', 'pdf')
# # # title(['cr: ', num2str(circ_r(a_Biphase(:, 11, 11)))])
# # set(gca, 'ThetaAxisUnits', 'radians')
# # set(gca, 'FontSize', 15, 'FontName', 'Times New Roman')
# # set(gca, 'GridAlpha', 0.5)
# # set(gca, 'Rlim', [0 0.2])
# # set(gca, 'RTickLabel', {'0.00', '0.05', '0.10', '0.15', '0.20'})
# # set(gca, 'RMinorGrid', 'on')
# # set(gca, 'MinorGridLinestyle', ':')
# # set(gca, 'MinorGridAlpha', 0.5)
# # h.FaceColor = [1, 0, 0]
# # h.FaceAlpha = 0.5
# # h.LineStyle = 'none'
# # title('Estimated PDF of $\theta_4$', 'Interpreter', 'latex')
# # thetalim([-pi, pi])
#
# #
# # # # theta_5
# #
# # # case3
# # theta = -pi:0.001: pi
# #
# # #  \theta_4 + \theta_1
# # Mu_41 = 0
# # Kappa_41 = 100
# # # Kappa_41 = 0
# # pdf_41 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_41(i,:)= circ_vmpdf(theta, Mu_41 + theta_4(i) + theta_1(i), Kappa_41)
# # end
# #
# # #  \theta_3 + \theta_2
# # Mu_32 = 0
# # Kappa_32 = 100
# # # Kappa_32 = 0
# # pdf_32 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_32(i,:)= circ_vmpdf(theta, Mu_32 + theta_3(i) + theta_2(i), Kappa_32)
# # end
# #
# # #  \theta_3 + \theta_1 + \theta_1
# # Mu_311 = 0
# # Kappa_311 = 1
# # pdf_311 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_311(i,:)= circ_vmpdf(theta, Mu_311 + theta_3(i) + theta_1(i) + theta_1(i), Kappa_311)
# # end
# #
# # #  \theta_2 + \theta_2 + \theta_1
# # Mu_221 = 0
# # Kappa_221 = 100
# # # Kappa_221 = 0
# # pdf_221 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_221(i,:)= circ_vmpdf(theta, Mu_221 + theta_2(i) + theta_2(i) + theta_1(i), Kappa_221)
# # end
# #
# # #  \theta_3 + \theta_3 - \theta_1
# # Mu_33m1 = 0
# # Kappa_33m1 = 100
# # # Kappa_33m1 = 0
# # pdf_33m1 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_33m1(i,:)= circ_vmpdf(theta, Mu_33m1 + theta_3(i) + theta_3(i) - theta_1(i), Kappa_33m1)
# # end
# #
# #
# # #  \theta_4 + \theta_2 - \theta_1
# # Mu_42m1 = 0
# # Kappa_42m1 = 1
# # pdf_42m1 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_42m1(i,:)= circ_vmpdf(theta, Mu_42m1 + theta_4(i) + theta_2(i) - theta_1(i), Kappa_42m1)
# # end
# #
# #
# # #  \theta_4 + \theta_3 - \theta_2
# # Mu_43m2 = 0
# # Kappa_43m2 = 100
# # # Kappa_43m2 = 0
# # pdf_43m2 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_43m2(i,:)= circ_vmpdf(theta, Mu_43m2 + theta_4(i) + theta_3(i) - theta_2(i), Kappa_43m2)
# # end
# #
# #
# # #  \theta_4 + \theta_4 - \theta_3
# # Mu_44m3 = 0
# # Kappa_44m3 = 1
# # pdf_44m3 = np.zeros(numsample, length(theta))
# # for i=1:numsample
# # pdf_44m3(i,:)= circ_vmpdf(theta, Mu_44m3 + theta_4(i) + theta_4(i) - theta_3(i), Kappa_44m3)
# # end
# #
# #
# # theta_5 = np.zeros(numsample, 1)
# # for i=1:numsample
# # theta_5(i, 1) = UserCircRVGen(pdf_41(i,:).*pdf_32(i,:) ...
#                                                          #.*pdf_311(i,:).*pdf_221(i,:)...
#                                                                                       #.*pdf_42m1(i,:).*pdf_33m1(
#     # i,:).*pdf_43m2(i,:).*pdf_44m3(i,:), theta)
# # end
# #
# #
# #
# # figure
# # subplot(3, 3, 1)
# polarhistogram(theta_5 - (theta_4 + theta_1), 100, 'Normalization', 'pdf')
# # title('\theta_5: \beta_{41}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 2)
# polarhistogram(theta_5 - (theta_3 + theta_2), 100, 'Normalization', 'pdf')
# # title('\theta_5: \beta_{32}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 3)
# polarhistogram(theta_5 - (theta_3 + theta_1 + theta_1), 'Normalization', 'pdf')
# # title('\theta_5: \gamma_{311}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 4)
# polarhistogram(theta_5 - (theta_2 + theta_2 + theta_1), 100, 'Normalization', 'pdf')
# # title('\theta_5: \gamma_{221}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 5)
# polarhistogram(theta_5 - (theta_4 + theta_2 - theta_1), 100, 'Normalization', 'pdf')
# # title('\theta_5: \gamma_{42-1}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 6)
# polarhistogram(theta_5 - (theta_3 + theta_3 - theta_1), 100, 'Normalization', 'pdf')
# # title('\theta_5: \gamma_{33-1}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 7)
# polarhistogram(theta_5 - (theta_4 + theta_3 - theta_2), 100, 'Normalization', 'pdf')
# # title('\theta_5: \gamma_{43-2}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 8)
# polarhistogram(theta_5 - (theta_4 + theta_4 - theta_3), 100, 'Normalization', 'pdf')
# # title('\theta_5: \gamma_{44-3}')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # subplot(3, 3, 9)
# polarhistogram(theta_5, 100, 'Normalization', 'pdf')
# # title('\theta_5')
# # ax = gca
# # ax.ThetaAxisUnits = 'radians'
# # ax.ThetaLim = [-pi, pi]
# #
# # save('Data_GenerateBiphaseTriphase_1_1.mat')
#
# # # theta6
# #
# # Mu_321 = 0
# # rho_321 = np.sqrt(0.5)
# # triphase_321 = wcrand(Mu_321, rho_321, numsample, 1)
# # theta_6 = theta_3 + theta_2 + theta_1 + triphase_321
# #
# # figure(6)
# h = polarhistogram(triphase_321, 100, 'Normalization', 'pdf')
# # set(gca, 'ThetaAxisUnits', 'radians')
# # set(gca, 'FontSize', 15, 'FontName', 'Times New Roman')
# # set(gca, 'GridAlpha', 0.5)
# # set(gca, 'Rlim', [0 1])
# # set(gca, 'RTickLabel', {'0.00', '0.20', '0.40', '0.60', '0.80', '1.00'})
# # set(gca, 'RMinorGrid', 'on')
# # set(gca, 'MinorGridLinestyle', ':')
# # set(gca, 'MinorGridAlpha', 0.5)
# # h.FaceColor = [0, 0, 1]
# # h.FaceAlpha = 0.5
# # h.LineStyle = 'none'
# # title('Estimated PDF of $\theta_6 - (\theta_3 \ + \theta_2 + \theta_1)$', 'Interpreter', 'latex')
# # thetalim([-pi, pi])
# #
# #
# # figure(7)
# h = polarhistogram(theta_6, 100, 'Normalization', 'pdf')
# # # title(['cr: ', num2str(circ_r(a_Biphase(:, 11, 11)))])
# # set(gca, 'ThetaAxisUnits', 'radians')
# # set(gca, 'FontSize', 15, 'FontName', 'Times New Roman')
# # set(gca, 'GridAlpha', 0.5)
# # set(gca, 'Rlim', [0 0.2])
# # set(gca, 'RTickLabel', {'0.00', '0.05', '0.10', '0.15', '0.20'})
# # set(gca, 'RMinorGrid', 'on')
# # set(gca, 'MinorGridLinestyle', ':')
# # set(gca, 'MinorGridAlpha', 0.5)
# # h.FaceColor = [1, 0, 0]
# # h.FaceAlpha = 0.5
# # h.LineStyle = 'none'
# # title('Estimated PDF of $\theta_6$', 'Interpreter', 'latex')
# # thetalim([-pi, pi])
#
# # # Input parameters
# norm = 'SRM'
# Beta = 1.0
# T = 1 # Time(1 / T = df)
# m = 400 + 1 # Num.of Discretized Time
# F = 100 # Frequency.(Hz)
# n = 100 + 1 # Num of Discretized Freq.
# Mv = np.zeros(1, m) # Mean vector(m)
# Vv = np.ones(1, m) # Variance vector(m)
# nsamples = numsample # Num.of
# samples
#
# # # Generation
# of
# Input
# Data(Stationary)
# dt = T / (m - 1)
# t = 0:dt: T - dt
# Nt = size(t, 2)
# df = F / (n - 1)
# f = 0:df: (df * (n - 1))
#
# df = f(2) - f(1) # frequency
# step
# Nf = length(f) # original length
#
# # Target PSDF(stationary)
# # case 1(var=20)
# # p = @(wx) 20 * 1 / np.sqrt(2 * pi * 1) * exp(-1 / (2 * 1) * wx. ^ 2)
# # case 2
# # p = @(wx) 20 * 1 / np.sqrt(2 * pi * 1) * exp(-1 / (2 * 1) * (wx - 2). ^ 2)
# # case 3(var=2 * 20 * 0.9772 = 39.0880)
# # p = @(wx) 40 * 1 / np.sqrt(2 * pi * 0.25) * exp(-1 / (2 * 0.25) * (wx - 2). ^ 2)
# # case 4
# # p = @(wx) (wx >= 1). * (wx <= 6). * 14
# # case5
# # p = @(wx)(wx == 10). * 0.25 / df + (wx == 20). * 0.25 / df + (wx == 30). * 0.25 / df + (wx == 40). * 0.25 / df
# # p = @(wx) (wx == 10). * 0.25 / df + (wx == 20). * 0.25 / df + (wx == 30). * 0.25 / df # + (wx == 40). * 0.25 / df
# # p = @(wx) (wx == 10). * 0.25 / df + (wx == 20). * 0.25 / df + 0 * (wx == 30). * 0.25 / df + (wx == 40). * 0.25 / df
# # p = @(wx) (wx == 5). * 0.25 / df + (wx == 10). * 0.25 / df + (wx == 15). * 0.25 / df + (wx == 30). * 0.25 / df
#
# # P = p(f)
#
# plt.figure(10)
# plt.plot(f, P, 'k-')
# xlim([0 50])
# # ylim([0 0.4])
# # title('Power spectrum')
# xlabel('$f$(Hz)', 'Interpreter', 'latex')
# ylabel('$S(f)$', 'Interpreter', 'latex')
# set(h, 'Linewidth', 1.5)
# set(gca, 'LineWidth', 1.5)
# h0 = legend('Target')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# set(h0, 'Interpreter', 'latex')
# saveas(gcf, 'powerspectrum.fig')
# saveas(gcf, 'powerspectrum', 'epsc')
#
# # Generate Samples
# for iii=1:nsamples
# if mod(iii, 10) == 0
#     iii
# end
#
# nn = 1
#
# P3 = np.zeros(nn, Nt)
#
# Coeff1 = 2 * np.sqrt(df * P)
# Coeff1(1) = Coeff1(1) / np.sqrt(2) # # # # # # # # # # # # # # # # # # # # for the zero freqeuncey!!!!!
#
# for i=1:nn
# Phi = 2 * pi * rand(1, Nf)
# # Phi = np.zeros(1, Nf)
# Phi(jumpfreq + 1) = theta_1(iii, 1)
# Phi(2 * jumpfreq + 1) = theta_2(iii, 1)
# Phi(3 * jumpfreq + 1) = theta_3(iii, 1)
# # Phi(6 * jumpfreq + 1) = theta_6(iii, 1)
#
# for j=1:Nt
# P3(i, j) = Coeff1 * cos(2 * pi * f * t(j) - Phi)
# '
# # P14(i, j) = Coeff2 * cos(2 * pi * f * t(j) - Phi)
# '
# end
# end
#
# P33(iii,:)=P3
# end
#
# # # Estimate
# PSDF & BSDF
# # FFT_1D
# Xw = fft(P33, Nt, 2)
# Xw = Xw(:, 1: Nt / 2)
#
# # PowerSpectrum
# m_P = np.zeros(1, Nt / 2)
# a_P = abs(Xw). ^ 2 / Nt / (Nt / (T))
# m_P = mean(a_P, 1)
#
# figure(102)
# hold
# on
# h = plot(f, P, 'k-')
# set(h, 'Linewidth', 1.5)
# h = plot(0:(1 / T): (1 / (2 * dt) - 1 / T), m_P, 'r')
# xlim([0 50])
# # ylim([0 0.4])
# # title('Estimated Powerspectrum')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$f$(Hz)', 'Interpreter', 'latex')
# ylabel('$S$(f)', 'Interpreter', 'latex')
# set(h, 'Linewidth', 1.5)
# legend('Target', 'Estimated')
# saveas(gcf, 'EPowerspectrum.fig')
# saveas(gcf, 'EPowerspectrum', 'epsc')
#
# # Coefficient
# figure
# plot(abs(a_P(:, 11)))
# figure
# plot(abs(a_P(:, 21)))
# figure
# plot(abs(a_P(:, 31)))
#
# # Bispctrum / Biphase
# Distribution
# m_B = np.zeros(Nt / 2, Nt / 2)
# s_B = np.zeros(Nt / 2, Nt / 2)
#
# a_B = np.zeros(nsamples, Nt / 2, Nt / 2)
#
# for k=1:nsamples
# if mod(k, 10) == 0
#     k
# end
# for i1=1:jumpfreq: Nt / 2
# for i2=1:jumpfreq: Nt / 2 - (i1 - 1)
# s_B(i1, i2) = s_B(i1, i2) + (Xw(k, i1) * Xw(k, i2) * conj(Xw(k, i1 + i2 - 1)) / (Nt) ^ 2 / ((Nt) / T)) * T
#
# a_B(k, i1, i2) = (Xw(k, i1) * Xw(k, i2) * conj(Xw(k, i1 + i2 - 1)) / (Nt) ^ 2 / ((Nt) / T)) * T
# end
# end
# end
# m_B = s_B / nsamples
#
# a_B(:, 1,:)=0
# a_B(:,:, 1)=0
#
# a_Biphase = atan2(imag(a_B), real(a_B))
#
# figure(111)
# polarhistogram(a_Biphase(:, 21, 11), 100, 'Normalization', 'pdf')
# # title(['cr: ', num2str(circ_r(a_Biphase(:, 21, 11)))])
# set(gca, 'ThetaAxisUnits', 'radians')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# title('Estimated PDF of $\theta_3 - (\theta_2 \ + \theta_1)$', 'Interpreter', 'latex')
# thetalim([-pi, pi])
#
# figure(112)
# polarhistogram(a_Biphase(:, 11, 11), 100, 'Normalization', 'pdf')
# # title(['cr: ', num2str(circ_r(a_Biphase(:, 11, 11)))])
# set(gca, 'ThetaAxisUnits', 'radians')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# title('Estimated PDF of $ \arctan2 [ X(\omega_1) X(\omega_2) X^{*}(\omega_3) ]$', 'Interpreter', 'latex')
# thetalim([-pi, pi])
#
# # figure(112)
# histogram(a_Biphase(:, 11, 11), 100)
# # title('$\theta_2 - (\theta_1 \ + \theta_1)$', 'Interpreter', 'latex')
# # set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# # set(gca, 'XTick', -pi: pi / 2:pi)
# # set(gca, 'XTickLabel', {'-\pi', '-\pi/2', '0', '\pi/2', '\pi'})
# # xlim([-pi, pi])
#
# # # # Set
# zero
# on
# X & Y
# axis
# m_B(1,:)=0
# m_B(:, 1)=0
# # # #
#
# # amplitude
# m_B_Ampl = abs(m_B)
# # real & imag
# m_B_Real = real(m_B)
# m_B_Imag = imag(m_B)
#
# figure(17)
# h = mesh(0:(1 / T): (1 / (2 * dt) - 1 / T), 0: (1 / T):(1 / (2 * dt) - 1 / T), m_B_Real)
# # title('Estimated Real Bispectrum')
# xlim([0 50])
# ylim([0 50])
# # zlim([-0.1 0.1])
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# zlabel('Estimated $\Re{B}(f_1, f_2)$', 'Interpreter', 'latex')
# set(h, 'Linewidth', 1.5)
# set(gca, 'LineWidth', 1.5)
# view(-38, 16)
# saveas(gcf, 'ERBispectrum.fig')
# saveas(gcf, 'ERBispectrum', 'epsc')
#
# figure(18)
# h = mesh(0:(1 / T): (1 / (2 * dt) - 1 / T), 0: (1 / T):(1 / (2 * dt) - 1 / T), m_B_Imag)
# # title('Estimated Imaginary Bispectrum')
# xlim([0 50])
# ylim([0 50])
# # zlim([-0.1 0.1])
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# zlabel('Estimated $\Im{B}(f_1, f_2)$', 'Interpreter', 'latex')
# set(h, 'Linewidth', 1.5)
# set(gca, 'LineWidth', 1.5)
# view(-38, 16)
# saveas(gcf, 'EIRBispectrum.fig')
# saveas(gcf, 'EIBispectrum', 'epsc')
#
# # D_P11 = (diff(P11')/dt)'
# D_P33 = (diff(P33')/dt)'
#
# # Bicoherence2 (Brillinger and Rosenblatt (1967))
# m_Bicoh2 = np.zeros(Nt / 2, Nt / 2)
# for k = 1:1
#           # if mod(k, 10) == 0
#                # k
#                # end
# for i1=1:jumpfreq: Nt / 2
# for i2=1:jumpfreq: Nt / 2 - (i1 - 1)
# if m_P(i1) > 0.001 & & m_P(i2) > 0.001 & & m_P(i1 + i2 - 1) > 0.001
# m_Bicoh2(i1, i2) = abs(m_B(i1, i2)) ^ 2 / m_P(i1) / m_P(i2) / m_P(i1 + i2 - 1) * df
# end
# end
# end
# end
# figure(19)
# h = mesh(0:(1 / T): (1 / (2 * dt) - 1 / T), 0: (1 / T):(1 / (2 * dt) - 1 / T), m_Bicoh2)
# xlim([0 50])
# ylim([0 50])
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# zlabel('Estimated $b_1^2(f_1, f_2)$', 'Interpreter', 'latex')
# set(h, 'Linewidth', 1.5)
# set(gca, 'LineWidth', 1.5)
# view(-38, 16)
#
# # Bicoherence2(Kim and Powers(1967))
# m_Bicoh2_2 = np.zeros(Nt / 2, Nt / 2)
#
# s_XiXj = np.zeros(Nt / 2, Nt / 2)
# # m_XiXj = np.zeros(Nt / 2, Nt / 2)
# for k = 1:nsamples
# if mod(k, 10) == 0
# k
# end
# for i1=1:jumpfreq: Nt / 2
# for i2=1:jumpfreq: Nt / 2 - (i1 - 1)
# s_XiXj(i1, i2) = s_XiXj(i1, i2) + (abs((Xw(k, i1). * Xw(k, i2))). ^ 2 / Nt / (Nt / (T)) / Nt / (Nt / (T)))
# end
# end
# end
# m_XiXj = s_XiXj / nsamples
#
# for k = 1:1
#           # if mod(k, 10) == 0
#                # k
#                # end
# for i1=1:jumpfreq: Nt / 2
# for i2=1:jumpfreq: Nt / 2 - (i1 - 1)
# if m_P(i1) > 0.001 & & m_P(i2) > 0.001 & & m_P(i1 + i2 - 1) > 0.001
# m_Bicoh2_2(i1, i2) = abs(m_B(i1, i2)) ^ 2 / m_XiXj(i1, i2) / m_P(i1 + i2 - 1) * df
# end
# end
# end
# end
# figure(20)
# h = mesh(0:(1 / T): (1 / (2 * dt) - 1 / T), 0: (1 / T):(1 / (2 * dt) - 1 / T), m_Bicoh2_2)
# xlim([0 50])
# ylim([0 50])
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# zlabel('Estimated $b_2^2(f_1, f_2)$', 'Interpreter', 'latex')
# set(h, 'Linewidth', 1.5)
# set(gca, 'LineWidth', 1.5)
# view(-38, 16)
#
# # samples
# for i=4:4
# figure(14)
# hold
# on
# # h = plot(t, P11(i,:), 'k--')
# # set(h, 'Linewidth', 1.5)
# hold
# on
# h = plot(t, P33(i,:), 'r')
# set(h, 'Linewidth', 1.5)
# # title('Sample process')
# end
# figure(14)
# xlim([0 T / 2])
# # ylim([-4 4])
# # title('Sample process')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$t$(sec.)', 'Interpreter', 'latex')
# ylabel('$f(t)$', 'Interpreter', 'latex')
# h0 = legend('$f_1(t)$')
# set(h0, 'Interpreter', 'latex')
# set(gca, 'LineWidth', 1.5)
#
# # Derivative
# samples
# for i=4:4
# figure(20 + i)
# # h = plot(t(1:size(D_P11, 2)), D_P11(i,:), 'k--')
# set(h, 'Linewidth', 1.5)
# hold
# on
# h = plot(t(1:size(D_P33, 2)), D_P33(i,:), 'r')
# set(h, 'Linewidth', 1.5)
# # title('Derivative of Sample process')
# end
#
# figure(14)
# xlim([0 T / 2])
# # ylim([-4 4])
# # title('Sample process')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$t$(sec.)', 'Interpreter', 'latex')
# ylabel('$f(t)$', 'Interpreter', 'latex')
# h0 = legend('$f_1(t)$')
# set(h0, 'Interpreter', 'latex')
# set(gca, 'LineWidth', 1.5)
#
# figure(24)
# xlim([0 T / 2])
# # ylim([-800 800])
# # title('Sample process of derivative')
# set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
# xlabel('$t$(sec.)', 'Interpreter', 'latex')
# ylabel('$\frac{\partial{f(t)}}{\partial{t}}$', 'Interpreter', 'latex')
# h0 = legend('$\frac{\partial{f_1(t)}}{\partial{t}}$')
# set(h0, 'Interpreter', 'latex')
# set(gca, 'LineWidth', 1.5)
#
# # # # Trispectrum(1
# st & 8
# th
# octants)
# # clearvars
# Xw \
# # sample_size2 = size(P33, 2)
# # sample_maxtime2 = T
# # for i=1:size(P33, 1)
#           # Xw(i,:) = fftshift(fft(P33(i,:)))
# # end
#   #
#   # # Xw = fft(P33, sample_size2, 2)
# # # Xw = fftshift(Xw)
# #
# #
# # freq33 = (-((sample_size2) / 2):(sample_size2) / 2 - 1)'/sample_maxtime2  \
#                                                    # freq_step33 = (freq33(2) - freq33(1))
#                                                          #
#                                                          # Nt = sample_size2
# # T = sample_maxtime2
# # nsamples = size(P33, 1)
# #
# # s_T1234 = np.zeros(Nt, Nt, Nt)
# # s_T12 = np.zeros(Nt, Nt, Nt)
# # s_T34 = np.zeros(Nt, Nt, Nt)
# # s_T31 = np.zeros(Nt, Nt, Nt)
# # s_T24 = np.zeros(Nt, Nt, Nt)
# # s_T23 = np.zeros(Nt, Nt, Nt)
# # s_T14 = np.zeros(Nt, Nt, Nt)
# #
# # # estimate_sample_size2 = Nt
# # estimate_sample_size2 = Nt / 2 + Nt / 4
# #
# # # for k=1:size(P33, 1)
#             #
# for k=1:nsamples
#         # if mod(k, 10) == 0
#              # k
#              # end
#              # # # Full
# octant
# # #
# for i1=Nt / 2 + 1 - Nt / 4:10: estimate_sample_size2
#                                # #
# for i2=Nt / 2 + 1 - Nt / 4:10: estimate_sample_size2
#                                # #
# for i3=Nt / 2 + 1 - Nt / 4:10: estimate_sample_size2
#                                #
#                                # # 1
# st and 8
# th
# octant
# #
# for i1=Nt / 2+1:jumpfreq: estimate_sample_size2
#                           #
# for i2=Nt / 2+1:jumpfreq: estimate_sample_size2 - (i1 - (Nt / 2 + 1))
#                           #
# for i3=1:jumpfreq: estimate_sample_size2 - (i1 - (Nt / 2 + 1) + i2 - (Nt / 2 + 1))
#                    # # if (freq33(i1) + freq33(i2) + freq33(i3)) <= freq33(Nt) & & (
#         freq33(i1) + freq33(i2) + freq33(i3)) >= -freq33(Nt)
#                           #
#                           # # index_i1_i2_i3 = find(
#     round(freq33, 4) == (round(freq33(i1) + freq33(i2) + freq33(i3), 4)))
# #
# index_i1_i2_i3 = (i1 - (Nt / 2 + 1)) + (i2 - (Nt / 2 + 1)) + (i3 - (Nt / 2 + 1)) + Nt / 2 + 1
# # # if index_i1_i2_i3 ~= index_i1_i2_i3_1
# # # index_i1_i2_i3
# # # index_i1_i2_i3_1
# # # end
# # s_T1234(i1, i2, i3)= s_T1234(i1, i2, i3) + ((Xw(k, i1) * Xw(k, i2) * Xw(k, i3) * conj(Xw(k, index_i1_i2_i3))) / (Nt) ^ 3 / ((Nt) / T)) * T ^ 2
# #
# # s_T12(i1, i2, i3)= s_T12(i1, i2, i3) + (Xw(k, i1) * Xw(k, i2)) * np.sqrt(1 / (Nt) ^ 3 / ((Nt) / T) * T ^ 2) # (round(freq(i1), 2) == -round(freq(i2), 2))
# # s_T23(i1, i2, i3)= s_T23(i1, i2, i3) + (Xw(k, i2) * Xw(k, i3)) * np.sqrt(1 / (Nt) ^ 3 / ((Nt) / T) * T ^ 2)
# # s_T31(i1, i2, i3)= s_T31(i1, i2, i3) + (Xw(k, i3) * Xw(k, i1)) * np.sqrt(1 / (Nt) ^ 3 / ((Nt) / T) * T ^ 2)
# # s_T34(i1, i2, i3)= s_T34(i1, i2, i3) + (Xw(k, i3) * conj(Xw(k, index_i1_i2_i3))) * np.sqrt(1 / (Nt) ^ 3 / ((Nt) / T) * T ^ 2)
# # s_T24(i1, i2, i3)= s_T24(i1, i2, i3) + (Xw(k, i2) * conj(Xw(k, index_i1_i2_i3))) * np.sqrt(1 / (Nt) ^ 3 / ((Nt) / T) * T ^ 2)
# # s_T14(i1, i2, i3)= s_T14(i1, i2, i3) + (Xw(k, i1) * conj(Xw(k, index_i1_i2_i3))) * np.sqrt(1 / (Nt) ^ 3 / ((Nt) / T) * T ^ 2)
# #
# #
# # # end
# # end
# # end
# # end
# # end
# # m_T = (s_T1234 - s_T12.* s_T34 / nsamples - s_T23.* s_T14 / nsamples - s_T31.* s_T24 / nsamples) / nsamples
# # clear s_T1234 s_T12 s_T34 s_T23 s_T14 s_T31 s_T24
# #
# # m_T(Nt / 2+1,:,:)=0
# # m_T(:, Nt / 2 + 1,:)=0
# # m_T(:,:, Nt / 2 + 1)=0
# #
# #
# # F = 50
# # critvalue = 0.001
# # radius = 1000
# # # amplitude \
#     # m_T_Ampl = abs(m_T)
# # # real & imag & phase \
#     # m_T_Real = real(m_T)
# # m_T_Imag = imag(m_T)
# # m_T_Phse = atan2(m_T_Imag, m_T_Real)
# #
# # # ef = 0:(1 / T): (1 / (2 * dt) - 1 / T)
# # ef = freq33
# # [X1, X2, X3] = ndgrid(ef, ef, ef)
# #
# # x1 = X1(:)
# # x2 = X2(:)
# # x3 = X3(:)
# # m_t_Real = m_T_Real(:)
# # m_t_Imag = m_T_Imag(:)
# # m_tt_Real = [x1 x2 x3 m_t_Real]
# # m_tt_Imag = [x1 x2 x3 m_t_Imag]
# #
# # # Erasing
# rows
# of
# 4
# D
# trispectrum
# # m_tt_Real(abs(m_t_Real) < critvalue,:,:,:)=[]
# # m_tt_Imag(abs(m_t_Imag) < critvalue,:,:,:)=[]
# #
# # figure(21)
# h = scatter3(m_tt_Real(:, 1), m_tt_Real(:, 2), m_tt_Real(:, 3), radius * abs(
#     m_tt_Real(:, 4)), m_tt_Real(:, 4), 'filled')
# # # xlim([0 F])
#     # # ylim([0 F])
#     # # zlim([0 F])
#     # xlim([-F F])
#     # ylim([-F F])
#     # zlim([-F F])
#     #
#     # set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
#     # xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# # ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# # zlabel('$f_3$(Hz)', 'Interpreter', 'latex')
# # title('Estimated $\Re{T}(f_1, f_2, f_3)$', 'Interpreter', 'latex') \
#   # colorbar
# # set(h, 'Linewidth', 1.5)
# # set(gca, 'LineWidth', 1.5)
# # view(-38, 16)
# # grid
# on
# # saveas(gcf, 'ERTrispectrum_1.fig')
#   # saveas(gcf, 'ERTrispectrum_1', 'jpeg')
#   #
#   # figure(22)
# h = scatter3(m_tt_Imag(:, 1), m_tt_Imag(:, 2), m_tt_Imag(:, 3), radius * abs(
#     m_tt_Imag(:, 4)), m_tt_Imag(:, 4), 'filled')
# # # xlim([0 F])
#     # # ylim([0 F])
#     # # zlim([0 F])
#     # xlim([-F F])
#     # ylim([-F F])
#     # zlim([-F F])
#     # set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
#     # xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# # ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# # zlabel('$f_3$(Hz)', 'Interpreter', 'latex')
# # title('Estimated $\Im{T}(f_1, f_2, f_3)$', 'Interpreter', 'latex') \
#   # colorbar \
#   # set(h, 'Linewidth', 1.5)
# # set(gca, 'LineWidth', 1.5)
# # view(-38, 16)
# # grid
# on
# # saveas(gcf, 'EITrispectrum_1.fig')
#   # saveas(gcf, 'EITrispectrum_1', 'jpeg')
#   #
#   # # Tricoherence2(Brillinger and Rosenblatt(1967))
#   # m_Tricoh2 = np.zeros(Nt, Nt, Nt)
# # for k=1:1
#           # # if mod(k, 10) == 0
#                  # # k
#                  # # end
#                  #
# for i1=Nt / 2+1:jumpfreq: estimate_sample_size2
#                           # i1
#                           #
# for i2=Nt / 2+1:jumpfreq: estimate_sample_size2 - (i1 - (Nt / 2 + 1))
#                           # i2
#                           #
# for i3=1:jumpfreq: estimate_sample_size2 - (i1 - (Nt / 2 + 1) + i2 - (Nt / 2 + 1))
#                    # i3
#                    #
#                    # if abs(m_T(i1, i2, i3)) > 0.001
#                         # index_i1_i2_i3 = (i1 - (Nt / 2 + 1)) + (i2 - (Nt / 2 + 1)) + (i3 - (Nt / 2 + 1)) + Nt / 2 + 1
# #
# # m_Tricoh2(i1, i2, i3) = abs(m_T(i1, i2, i3)) ^ 2 / m_P(i1 - (Nt / 2 + 1) + 1) / m_P(i2 - (Nt / 2 + 1) + 1) / ... \
#                           # m_P((i3 - (Nt / 2 + 1) < 0) * (abs(i3 - (Nt / 2 + 1)) + 1) + (i3 - (Nt / 2 + 1) >= 0) * (
#         abs(i3 - (Nt / 2 + 1)) + 1)) / ... \
#                           # m_P((index_i1_i2_i3 - (Nt / 2 + 1) < 0) * (abs(index_i1_i2_i3 - (Nt / 2 + 1)) + 1) + (
#         index_i1_i2_i3 - (Nt / 2 + 1) >= 0) * (index_i1_i2_i3 - (Nt / 2 + 1) + 1)) * df * df
# # end
#   # end
#   # end
#   # end
#   # end
#   #
#   # F = 50
# # critvalue = 0.001
# # radius = 200
# #
# # # ef = 0:(1 / T): (1 / (2 * dt) - 1 / T)
# # ef = freq33
# # [X1, X2, X3] = ndgrid(ef, ef, ef)
# #
# # x1 = X1(:)
# # x2 = X2(:)
# # x3 = X3(:)
# # m_tricoh2_vec = m_Tricoh2(:)
# # m_ttricoh2_vec = [x1 x2 x3 m_tricoh2_vec]
# #
# # # Erasing
# rows
# of
# 4
# D
# trispectrum
# # m_ttricoh2_vec(abs(m_tricoh2_vec) < critvalue,:,:,:)=[]
# #
# # figure(31)
# h = scatter3(m_ttricoh2_vec(:, 1), m_ttricoh2_vec(:, 2), m_ttricoh2_vec(:, 3), radius * abs(
#     m_ttricoh2_vec(:, 4)), m_ttricoh2_vec(:, 4), 'filled')
# # # xlim([0 F])
#     # # ylim([0 F])
#     # # zlim([0 F])
#     # xlim([0 F])
#     # ylim([0 F])
#     # zlim([-F F])
#     #
#     # set(gca, 'FontSize', 20, 'FontName', 'Times New Roman')
#     # xlabel('$f_1$(Hz)', 'Interpreter', 'latex')
# # ylabel('$f_2$(Hz)', 'Interpreter', 'latex')
# # zlabel('$f_3$(Hz)', 'Interpreter', 'latex')
# # title('Estimated $t_1^2(f_1, f_2, f_3)$', 'Interpreter', 'latex') \
#   # colorbar
# # set(h, 'Linewidth', 1.5)
# # set(gca, 'LineWidth', 1.5)
# # view(140, 16)
# # grid on
#
