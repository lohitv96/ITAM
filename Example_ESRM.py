import numpy as np
from joblib import Parallel, delayed


# Input parameters
T = 20  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
W = 1 / T * nt / 2  # Frequency.(Hz)
nw = 128  # Num of Discretized Freq.
nsamples = 1000  # Num.of samples

# # Generation of Input Data
dt = T / nt
t = np.linspace(0, T - dt, nt)
dw = W / nw
w = np.linspace(0, W - dw, nw)

t_u = 2 * np.pi / (2 * W)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

x_list = [t[::-1], w]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
xy_list = np.array(xy_list)

# Defining the Power Spectrum Density Function
P = 20 * 1 / np.sqrt(2 * np.pi) * xy_list[0] * np.exp(-1 / 2 * xy_list[1] ** 2)
P[:, 0] = 0

# Generating the 2 dimensional mesh grid
x_list = [t[::-1], w, w]
xy_list = np.meshgrid(*x_list, indexing='ij')
xy_list = np.array(xy_list)

b = 20 / (2 * np.pi) * (xy_list[0] ** (4 / 2)) * np.exp(2 * (-1 / 2 * (xy_list[1] ** 2 + xy_list[2] ** 2)))
B_Real = b
B_Imag = b

B_Real[:, 0, :] = 0
B_Real[:, :, 0] = 0
B_Imag[:, 0, :] = 0
B_Imag[:, :, 0] = 0

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

Biphase = np.arctan2(B_Imag, B_Real)
Biphase[np.isnan(Biphase)] = 0

Bc2 = np.zeros_like(B_Real)
PP = np.zeros_like(P)
sum_Bc2 = np.zeros_like(P)

PP[:, 0] = P[:, 0]
PP[:, 1] = P[:, 1]

for k in range(nt):
    for i in range(nw):
        wk = i
        for j in range(int((i + 1) / 2)):
            wj = j
            wi = wk - wj
            if B_Ampl[k, wi, wj] > 0 and PP[k, wi] * PP[k, wj] != 0:
                Bc2[k, wi, wj] = B_Ampl[k, wi, wj] ** 2 / (PP[k, wi] * PP[k, wj] * P[k, wk]) * dw
                sum_Bc2[k, wk] = sum_Bc2[k, wk] + Bc2[k, wi, wj]
            else:
                Bc2[k, wi, wj] = 0
        if sum_Bc2[k, wk] > 1:
            print('came here')
            for j in range(int((i + 1) / 2)):
                wj = j
                wi = wk - wj
                Bc2[k, wi, wj] = Bc2[k, wi, wj] / sum_Bc2[k, wk]
            sum_Bc2[k, wk] = 1
        PP[k, wk] = P[k, wk] * (1 - sum_Bc2[k, wk])

Bc = np.sqrt(Bc2)
num_batches = 100


def simulate():
    samples_SRM = np.zeros(shape=[nsamples, nt])
    samples_BSRM = np.zeros(shape=[nsamples, nt])
    phi = np.random.uniform(size=[nsamples, nw]) * 2 * np.pi
    for k in range(nt):
        for i in range(nw):
            wk = i
            samples_SRM[:, k] = samples_SRM[:, k] + np.sqrt(4 * P[k, wk] * dw) * np.cos(w[wk] * t[k] + phi[:, wk])
            samples_BSRM[:, k] = samples_BSRM[:, k] + np.sqrt(4 * PP[k, wk] * dw) * np.cos(w[wk] * t[k] + phi[:, wk])
            for j in range(int((i + 1) / 2)):
                wj = j
                wi = wk - wj
                samples_BSRM[:, k] = samples_BSRM[:, k] + np.sqrt(4 * P[k, wk] * dw) * Bc[k, wi, wj] * np.cos(
                    w[wk] * t[k] + phi[:, wi] + phi[:, wj] + Biphase[k, wi, wj])
    return [samples_SRM, samples_BSRM]


samples_list = Parallel(n_jobs=4)(delayed(simulate)() for _ in range(num_batches))
samples1 = np.concatenate(samples_list, axis=0)

# np.save('data_non_stationary/samples_SRM.npy', samples_SRM)
# np.save('data_non_stationary/samples_BSRM.npy', samples_BSRM)
# np.save('data_non_stationary/P.npy', P)
# np.save('data_non_stationary/B_Real.npy', B_Real)
# np.save('data_non_stationary/sum_Bc2.npy', sum_Bc2)
#
# R2_SRM = np.zeros(shape=[nt, nt])
# R2_BSRM = np.zeros(shape=[nt, nt])
# for i in range(nt):
#     for j in range(i, nt):
#         R2_SRM[i, j] = np.mean(samples_SRM[:, i] * samples_SRM[:, j])
#         R2_BSRM[i, j] = np.mean(samples_BSRM[:, i] * samples_BSRM[:, j])
#
# np.save('data_non_stationary/R_SRM.npy', R2_SRM)
# np.save('data_non_stationary/R_BSRM.npy', R2_BSRM)
#
# R3_SRM = np.zeros(shape=[nt, nt, nt])
# R3_BSRM = np.zeros(shape=[nt, nt, nt])
# for i in range(nt):
#     print(i)
#     for j in range(i, nt):
#         for k in range(j, nt):
#             R3_SRM[i, j, k] = np.mean(samples_SRM[:, i] * samples_SRM[:, j] * samples_SRM[:, k])
#             R3_BSRM[i, j, k] = np.mean(samples_BSRM[:, i] * samples_BSRM[:, j] * samples_BSRM[:, k])

# t0 = 27
# t1 = 89
# print(np.mean(samples_SRM[:, t0] * samples_SRM[:, t1]))
# print(np.mean(samples_BSRM[:, t0] * samples_BSRM[:, t1]))
# print(2 * np.sum(np.sqrt(P[t0, :] * P[t1, :]) * np.cos(w * (t1 - t0) * dt)) * dw)
# print(2 * np.sum(np.sqrt(PP[t0, :] * PP[t1, :]) * np.cos(w * (t1 - t0) * dt)) * dw)
#
# temp2 = 0
# for i in range(nw):
#     wk = i
#     for j in range(int((i + 1) / 2)):
#         wj = j
#         wi = wk - wj
#         temp2 = temp2 + 2 * np.sqrt(P[t0, wk] * P[t1, wk]) * dw * Bc[t0, wi, wj] * Bc[t1, wi, wj] * np.cos(wk * (t1 - t0) * dt)
