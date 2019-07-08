import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt


# Input parameters
T = 20  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
W = 1 / T * nt / 2  # Frequency.(Hz)
nw = 128  # Num of Discretized Freq.

# # Generation of Input Data
dt = T / nt
t = np.linspace(0, T - dt, nt)
dw = W / nw
w = np.linspace(0, W - dw, nw)

samples_SRM = np.load('data_non_stationary/samples_SRM.npy')
samples_BSRM = np.load('data_non_stationary/samples_BSRM.npy')
R_SRM = np.load('data_non_stationary/R_SRM.npy')
R_BSRM = np.load('data_non_stationary/R_BSRM.npy')
P = np.load('data_non_stationary/P.npy')
sum_Bc2 = np.load('data_non_stationary/sum_Bc2.npy')

fig7 = plt.figure()
plt.plot(t, np.mean(samples_SRM, axis=0), label ='SRM')
plt.plot(t, np.mean(samples_BSRM, axis=0), label ='BSRM')
plt.plot(t, np.zeros_like(t), label='Theoretical')
plt.legend()
plt.savefig('plots_non_stationary/non_stationary_mean.eps', dpi=300)

fig1 = plt.figure()
plt.plot(t, np.var(samples_SRM, axis=0), label ='SRM')
plt.plot(t, np.var(samples_BSRM, axis=0), label ='BSRM')
plt.plot(t, 19.5*(T-t), label='Theoretical')
plt.legend()
plt.savefig('plots_non_stationary/non_stationary_variance.eps', dpi=300)

fig2 = plt.figure()
plt.plot(t, skew(samples_SRM, axis=0), label ='SRM')
plt.plot(t, skew(samples_BSRM, axis=0), label ='BSRM')
plt.plot(t, 0.160*np.sqrt(T-t), label='Theoretical')
plt.legend()
plt.savefig('plots_non_stationary/non_stationary_skewness.eps', dpi=300)

fig3 = plt.figure()
plt.plot(t, samples_SRM[0], label ='SRM')
plt.plot(t, samples_BSRM[0], label ='BSRM')
plt.legend()
plt.savefig('plots_non_stationary/non_stationary_samples.eps', dpi=300)

x_list = [t, t]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
xy_list = np.array(xy_list)

fig4 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], R_SRM, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$t+\tau$')
plt.savefig('plots_non_stationary/R_SRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

fig5 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], R_SRM, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$t+\tau$')
plt.savefig('plots_non_stationary/R_BSRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

fig6 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], R_SRM - R_BSRM, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$t+\tau$')
plt.savefig('plots_non_stationary/R_BSRM_SRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

x_list = [t, w]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
xy_list = np.array(xy_list)

fig8 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], P, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$\omega$')
plt.savefig('plots_non_stationary/P.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

fig9 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], sum_Bc2, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$\omega$')
plt.savefig('plots_non_stationary/sum_Bc2.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

