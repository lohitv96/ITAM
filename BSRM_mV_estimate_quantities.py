import numpy as np
import copy
import math
from scipy.stats import skew, kurtosis, moment


def estimate_cross_power_spectrum(samples):
    nsamples, m, nt = samples.shape
    nw = int(nt / 2)
    Xw = np.fft.ifft(samples, axis=2)
    Xw = Xw[:, :, :nw]
    # Initializing the array before hand
    s_P = np.zeros([nsamples, m, m, nw])
    s_P = s_P + 1.0j * s_P
    for i1 in range(nw):
        s_P[..., i1] = s_P[..., i1] + np.einsum('ij, ik-> ijk', Xw[..., i1], np.conj(Xw[..., i1]))
    m_P = np.mean(s_P, axis=0)
    return m_P


def estimate_cross_bispectrum(samples):
    nsamples, m, nt = samples.shape
    nw = int(nt / 2)
    Xw = np.fft.ifft(samples, axis=2)
    Xw = Xw[:, :, :nw]
    # Initializing the array before hand
    s_B = np.zeros([nsamples, m, m, m, nw, nw])
    s_B = s_B + 1.0j * s_B
    for i1 in range(nw):
        for i2 in range(nw - i1):
            s_B[..., i1, i2] = s_B[..., i1, i2] + np.einsum('ij, ik, il-> ijkl', Xw[..., i1], Xw[..., i2],
                                                            np.conj(Xw[..., i1 + i2]))
    m_B = np.mean(s_B, axis=0)
    return m_B


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

########################################################################################################################
# Loading the required data

samples_SRM = np.load('data_multi_variate/samples_SRM.npy')
samples_BSRM = np.load('data_multi_variate/samples_BSRM.npy')
S = np.load('data_multi_variate/S.npy')
SP = np.load('data_multi_variate/SP.npy')
B = np.load('data_multi_variate/B.npy')
B1 = np.load('data_multi_variate/B1.npy')

########################################################################################################################
# Computing the 2nd-order and 3-rd cross spectrum

estimated_S = estimate_cross_power_spectrum(samples_BSRM)
estimated_B = estimate_cross_bispectrum(samples_BSRM)

np.save('data_multi_variate/estimated_S', estimated_S)
np.save('data_multi_variate/estimated_B', estimated_B)

########################################################################################################################
# Computing the 2nd-order and 3rd-order statistics to fill the tables

print('Quantity    Simulation    Theoretical')
print('Var-11', np.var(samples_BSRM[:, :, 0]), np.sum(2 * S[:, 0, 0] * dw))
print('Var-22', np.var(samples_BSRM[:, :, 0]), np.sum(2 * S[:, 1, 1] * dw))
print('Var-12', np.mean(samples_BSRM[:, :, 0] * samples_BSRM[:, :, 1]), np.sum(2 * S[:, 0, 1] * dw))

print('Moment-111', moment(samples_BSRM[:, :, 0], moment=3), np.sum(6 * B1[:, :, 0, 0, 0] * dw**2))
print('Moment-222', moment(samples_BSRM[:, :, 1], moment=3), np.sum(6 * B1[:, :, 1, 1, 1] * dw**2))
print('Moment-112', np.mean(samples_BSRM[:, :, 0]**2 * samples_BSRM[:, :, 1]), np.sum(6 * B1[:, :, 0, 0, 1] * dw**2))
print('Moment-122', np.mean(samples_BSRM[:, :, 1]**2 * samples_BSRM[:, :, 0]), np.sum(6 * B1[:, :, 0, 1, 1] * dw**2))