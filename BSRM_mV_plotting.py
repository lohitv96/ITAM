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
R2_SRM = np.load('data__multi_variate/R2_SRM.npy')
R2_BSRM = np.load('data__multi_variate/R2_BSRM.npy')
R3_SRM = np.load('data__multi_variate/R3_SRM.npy')
R3_BSRM = np.load('data__multi_variate/R3_BSRM.npy')
P = np.load('data__multi_variate/P.npy')
sum_Bc2 = np.load('data__multi_variate/sum_Bc2.npy')

fig7 = plt.figure()
plt.plot(t, np.mean(samples_SRM, axis=0), 'r', '-', label='SRM')
plt.plot(t, np.mean(samples_BSRM, axis=0), 'b', '-', label='BSRM')
plt.plot(t, np.zeros_like(t), 'k', '-.', label='Theoretical')
plt.legend(loc= 'upper right')
plt.grid(b=True, which='major')
plt.savefig('plots_non_stationary/non_stationary_mean.eps', dpi=300)

fig1 = plt.figure()
plt.plot(t, np.var(samples_SRM, axis=0), 'r', '-', label='SRM')
plt.plot(t, np.var(samples_BSRM, axis=0), 'b', '-', label='BSRM')
plt.plot(t, 19.5 * (t[-1] - t), 'k', '-.', label='Theoretical')
plt.legend(loc= 'upper right')
plt.grid(b=True, which='major')
plt.savefig('plots_non_stationary/non_stationary_variance.eps', dpi=300)

fig2 = plt.figure()
plt.plot(t, skew(samples_SRM, axis=0), 'r', '-', label='SRM')
plt.plot(t, skew(samples_BSRM, axis=0), 'b', '-', label='BSRM')
plt.plot(t, 0.160 * np.sqrt(t[-1] - t), 'k', '-.', label='Theoretical')
plt.legend(loc= 'upper right')
plt.grid(b=True, which='major')
plt.savefig('plots_non_stationary/non_stationary_skewness.eps', dpi=300)

fig3 = plt.figure()
plt.plot(t, samples_SRM[0], 'r', '-', label='SRM')
plt.plot(t, samples_BSRM[0], 'b', '-', label='BSRM')
plt.legend(loc= 'upper right')
plt.grid(b=True, which='major')
plt.savefig('plots_non_stationary/non_stationary_samples.eps', dpi=300)

x_list = [t, t]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
xy_list = np.array(xy_list)

fig4 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], R2_SRM, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$t+\tau$')
plt.savefig('plots_non_stationary/R2_SRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

fig5 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], R2_BSRM, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$t+\tau$')
plt.savefig('plots_non_stationary/R2_BSRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

fig6 = plt.figure()
pcm = plt.pcolor(xy_list[0], xy_list[1], R2_SRM - R2_BSRM, cmap='jet')
plt.colorbar(pcm, extend='neither', orientation='vertical')
plt.xlabel('$t$')
plt.ylabel(r'$t+\tau$')
plt.savefig('plots_non_stationary/R2_BSRM_SRM.eps', bbox_inches='tight', pad_inches=0.25, dpi=300)

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

########################################################################################################################

# fig10 = plt.figure(figsize=(6, 6))
# ax = fig10.gca(projection='3d')
# X, Y = np.meshgrid(t, t, indexing='xy')
# Z = deepcopy(R3_SRM)
#
# cmap = 'jet'
# cset = [[], [], [], [], [], []]
# cset[0] = ax.contourf(X, Y, Z[:, :, 0], zdir='z', offset=t[0], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[1] = ax.contourf(X, Y, Z[:, :, -1], zdir='z', offset=t[-1], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[2] = ax.contourf(Z[0, :, :], X, Y, zdir='x', offset=t[0], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[3] = ax.contourf(Z[-1, :, :], X, Y, zdir='x', offset=t[-1], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[4] = ax.contourf(X, Z[:, 0, :], Y, zdir='y', offset=t[0], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[5] = ax.contourf(X, Z[:, -1, :], Y, zdir='y', offset=t[-1], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
#
# # setting 3D-axis-parameters
# ax.set_xlim3d(t[-1] + 1, t[0])
# ax.set_ylim3d(t[0], t[-1] + 1)
# ax.set_zlim3d(t[-1] + 1, t[0])
# ax.set_xlabel('$t_{1}$')
# ax.set_ylabel(r'$t + \tau_{1}$')
# ax.set_zlabel(r'$t + \tau_{2}$')
# # ax.set_aspect('equal')
# ax.grid(True, which='both')
# # ax.set_title('$3-Dimensional\ Power\ Spectrum$', fontsize='large')
#
# fig10.colorbar(cset[0], orientation='vertical', use_gridspec=True, pad=0.05, shrink=0.6)
# plt.savefig('plots_non_stationary/R3_SRM.eps', dpi=300, bbox_inches='tight', pad_inches=0.25)
# plt.show()

########################################################################################################################

# fig11 = plt.figure(figsize=(6, 6))
# ax = fig11.gca(projection='3d')
# X, Y = np.meshgrid(t, t, indexing='xy')
# Z = deepcopy(R3_BSRM)
#
# cmap = 'jet'
# cset = [[], [], [], [], [], []]
# cset[0] = ax.contourf(X, Y, Z[:, :, 0], zdir='z', offset=t[0], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[1] = ax.contourf(X, Y, Z[:, :, -1], zdir='z', offset=t[-1], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[2] = ax.contourf(Z[0, :, :], X, Y, zdir='x', offset=t[0], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[3] = ax.contourf(Z[-1, :, :], X, Y, zdir='x', offset=t[-1], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[4] = ax.contourf(X, Z[:, 0, :], Y, zdir='y', offset=t[0], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
# cset[5] = ax.contourf(X, Z[:, -1, :], Y, zdir='y', offset=t[-1], levels=np.linspace(np.min(Z), np.max(Z), 30), cmap=cmap)
#
# # setting 3D-axis-parameters
# ax.set_xlim3d(t[-1] + 1, t[0])
# ax.set_ylim3d(t[0], t[-1] + 1)
# ax.set_zlim3d(t[-1] + 1, t[0])
# ax.set_xlabel('$t_{1}$')
# ax.set_ylabel(r'$t + \tau_{1}$')
# ax.set_zlabel(r'$t + \tau_{2}$')
# # ax.set_aspect('equal')
# ax.grid(True, which='both')
# # ax.set_title('$3-Dimensional\ Power\ Spectrum$', fontsize='large')
#
# fig11.colorbar(cset[0], orientation='vertical', use_gridspec=True, pad=0.05, shrink=0.6)
# plt.savefig('plots_non_stationary/R3_BSRM.eps', dpi=300, bbox_inches='tight', pad_inches=0.25)
# plt.show()

########################################################################################################################
