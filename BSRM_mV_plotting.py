import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn')

nsamples = 10000
n = 1  # Number of dimensions
m = 2  # Number of variables

W = 2.0  # Cutoff Frequency
nw = 400  # Number of frequency steps
dw = W / nw  # Length of frequency step
w = np.linspace(dw, W, nw)  # frequency vector
wx, wy = np.meshgrid(w, w)  # Frequency mesh

nt = 800  # Number of time steps
T = 1 / W * nt / 2  # Total Simulation time
dt = T / nt  # Duration of time step
t = np.linspace(dt, T, nt)  # Vector of time

samples_SRM = np.load('data_multi_variate/samples_SRM.npy')
samples_BSRM = np.load('data_multi_variate/samples_BSRM.npy')
# R2_SRM = np.load('data__multi_variate/R2_SRM.npy')
# R2_BSRM = np.load('data__multi_variate/R2_BSRM.npy')
# R3_SRM = np.load('data__multi_variate/R3_SRM.npy')
# R3_BSRM = np.load('data__multi_variate/R3_BSRM.npy')
S = np.load('data_multi_variate/S.npy')
SP = np.load('data_multi_variate/SP.npy')
B = np.load('data_multi_variate/B.npy')
B1 = np.load('data_multi_variate/B.npy')
# estimated_S = np.load('data_multi_variate/estimated_S.npy')
# estimated_B = np.load('data_multi_variate/estimated_B.npy')

fig1 = plt.figure()
plt.plot(t, samples_SRM[0, :, 0], label='SRM')
plt.plot(t, samples_BSRM[0, :, 0], label='BSRM')
plt.grid(b=True, which='major')
plt.xlabel('Time(sec)')
plt.ylabel('$f_{1}(t)$')
plt.legend(loc= 'upper right')
plt.savefig('plots_multi_variate/SRM_BSRM_var_1.eps', dpi=300)

fig2 = plt.figure()
plt.plot(t, samples_SRM[0, :, 1], label='SRM')
plt.plot(t, samples_BSRM[0, :, 1], label='BSRM')
plt.grid(b=True, which='major')
plt.xlabel('Time(sec)')
plt.ylabel('$f_{2}(t)$')
plt.legend(loc= 'upper right')
plt.savefig('plots_multi_variate/SRM_BSRM_var_2.eps', dpi=300)

# fig3 = plt.figure()
# plt.plot(t, samples_SRM[0, :, 2], 'r', '-', label='SRM')
# plt.plot(t, samples_BSRM[0, :, 2], 'b', '-', label='BSRM')
# plt.legend(loc= 'upper right')
# plt.grid(b=True, which='major')
# plt.xlabel('Time(sec)')
# plt.ylabel('$f_{3}(t)$')
# plt.savefig('plots_multi_variate/SRM_BSRM_var_3.eps', dpi=300)

# fig4 = plt.figure()
# plt.plot(t, S[:, 0, 0], 'r', '-', label='True')
# plt.plot(t, estimated_S[:, 0, 0], 'r', '-', label='Estiamted')
# plt.legend(loc= 'upper right')
# plt.grid(b=True, which='major')
# plt.xlabel('$Frequency(\omega)$')
# plt.ylabel('$S_{11}(\omega)$')
# plt.savefig('plots_multi_variate/estimated_S_11.eps', dpi=300)
#
# fig5 = plt.figure()
# plt.plot(t, S[:, 0, 1], 'r', '-', label='True')
# plt.plot(t, estimated_S[:, 0, 1], 'r', '-', label='Estiamted')
# plt.legend(loc= 'upper right')
# plt.grid(b=True, which='major')
# plt.xlabel('$Frequency(\omega)$')
# plt.ylabel('$S_{12}(\omega)$')
# plt.savefig('plots_multi_variate/estimated_S_12.eps', dpi=300)
#
# fig6 = plt.figure()
# plt.plot(t, S[:, 1, 1], 'r', '-', label='True')
# plt.plot(t, estimated_S[:, 1, 1], 'r', '-', label='Estiamted')
# plt.legend(loc= 'upper right')
# plt.grid(b=True, which='major')
# plt.xlabel('$Frequency(\omega)$')
# plt.ylabel('$S_{22}(\omega)$')
# plt.savefig('plots_multi_variate/estimated_S_22.eps', dpi=300)

# x_list = [t, t]
# xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
# xy_list = np.array(xy_list)
#
# fig5 = plt.figure()
# pcm = plt.pcolor(xy_list[0], xy_list[1], R2_BSRM, cmap='jet')
# plt.colorbar(pcm, extend='neither', orientation='vertical')
# plt.xlabel('$t$')
# plt.ylabel(r'$t+\tau$')
# plt.savefig('plots_non_stationary/R2_BSRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)
#
# fig6 = plt.figure()
# pcm = plt.pcolor(xy_list[0], xy_list[1], R2_SRM - R2_BSRM, cmap='jet')
# plt.colorbar(pcm, extend='neither', orientation='vertical')
# plt.xlabel('$t$')
# plt.ylabel(r'$t+\tau$')
# plt.savefig('plots_non_stationary/R2_BSRM_SRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)
#
# x_list = [t, w]
# xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
# xy_list = np.array(xy_list)
#
# fig8 = plt.figure()
# pcm = plt.pcolor(xy_list[0], xy_list[1], P, cmap='jet')
# plt.colorbar(pcm, extend='neither', orientation='vertical')
# plt.xlabel('$t$')
# plt.ylabel(r'$\omega$')
# plt.savefig('plots_non_stationary/P.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)
#
# fig9 = plt.figure()
# pcm = plt.pcolor(xy_list[0], xy_list[1], sum_Bc2, cmap='jet')
# plt.colorbar(pcm, extend='neither', orientation='vertical')
# plt.xlabel('$t$')
# plt.ylabel(r'$\omega$')
# plt.savefig('plots_non_stationary/sum_Bc2.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)
