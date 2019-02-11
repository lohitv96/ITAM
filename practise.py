import matplotlib.pyplot as plt
from scipy.stats import skew, moment, kurtosis
from mpl_toolkits.mplot3d import Axes3D
from BSRM import *
from SRM import *
from joblib import Parallel, delayed


def estimate_power_spectrum(samples):
    nsamples, nt = samples.shape
    nw = int(nt/2)
    Xw = np.fft.ifft(samples, axis=1)
    Xw = Xw[:, :nw]
    # Initializing the array before hand
    s_P = np.zeros([nsamples, nw])
    s_P = s_P + 1.0j * s_P
    for i1 in range(nw):
        s_P[..., i1] = s_P[..., i1] + np.einsum('i, i-> i', Xw[..., i1], np.conj(Xw[..., i1]))
    m_P = np.mean(s_P, axis=0)
    return m_P


def estimate_bispectrum(samples):
    nsamples, nt = samples.shape
    nw = int(nt/2)
    Xw = np.fft.ifft(samples, axis=1)
    Xw = Xw[:, :nw]
    # Initializing the array before hand
    s_B = np.zeros([nsamples, nw, nw])
    s_B = s_B + 1.0j * s_B
    for i1 in range(nw):
        for i2 in range(nw - i1):
            s_B[..., i1, i2] = s_B[..., i1, i2] + np.einsum('i, i, i-> i', Xw[..., i1], Xw[..., i2],
                                                            np.conj(Xw[..., i1 + i2]))
    m_B = np.mean(s_B, axis=0)
    return m_B


def estimate_trispectrum(samples):
    nsamples, nt = samples.shape
    nw = int(nt/2)
    Xw = np.fft.ifft(samples, axis=1)
    Xw = Xw[:, :nw]
    # Initializing the array before hand
    s_T = np.zeros([nsamples, nw, nw, nw])
    s_T = s_T + 1.0j * s_T
    for i1 in range(nw):
        for i2 in range(nw - i1):
            for i3 in range(nw - i1 - i2):
                s_T[..., i1, i2, i3] = s_T[..., i1, i2, i3] + np.einsum('i, i, i, i-> i', Xw[:, i1], Xw[:, i2], Xw[:, i3], np.conj(Xw[:, i1 + i2 + i3]))
    m_T = np.mean(s_T, axis=0)
    return m_T


def estimate_cross_power_spectrum(samples):
    nsamples, m, nt = samples.shape
    nw = int(nt/2)
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
    nw = int(nt/2)
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


def estimate_cross_trispectrum(samples):
    nsamples, m, nt = samples.shape
    nw = int(nt/2)
    Xw = np.fft.ifft(samples, axis=2)
    Xw = Xw[:, :, :nw]
    # Initializing the array before hand
    s_T = np.zeros([nsamples, m, m, m, m, nw, nw, nw])
    s_T = s_T + 1.0j * s_T
    for i1 in range(nw):
        for i2 in range(nw - i1):
            for i3 in range(nw - i1 - i2):
                s_T[..., i1, i2, i3] = s_T[..., i1, i2, i3] + np.einsum('ij, ij, il, im-> ijklm', Xw[:, i1], Xw[:, i2], Xw[:, i3], np.conj(Xw[:, i1 + i2 + i3]))
    m_T = np.mean(s_T, axis=0)
    return m_T


plt.style.use('seaborn')

# Input parameters
T = 20  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128  # Num of Discretized Freq.
nsamples = 100  # Num.of samples
nbatches = 1

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

# Generating the 2 dimensional mesh grid
fx = f
fy = f
Fx, Fy = np.meshgrid(fx, fy)

# Target PSDF(stationary)
P = 125 * (f ** 2) * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)

b = 80 * 2 * (Fx ** 2 + Fy**2) / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
B_Real = b
B_Imag = b

B_Real[0, :] = 0
B_Real[:, 0] = 0
B_Imag[0, :] = 0
B_Imag[:, 0] = 0

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

# samples_list = Parallel(n_jobs=nbatches)(delayed(simulate)() for _ in range(nbatches))
# samples1 = np.concatenate(samples_list, axis=0)

obj = BSRM(nsamples, P, B_Complex, dt, df, nt, nf)
samples1 = obj.samples

Tspectra = np.real(estimate_trispectrum(samples1))
np.save('Tspectra.npy', Tspectra)

# Target PSDF(stationary)
# P = 10 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)
#
# b = 4 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
# B_Real = b
# B_Imag = b
#
# B_Real[0, :] = 0
# B_Real[:, 0] = 0
# B_Imag[0, :] = 0
# B_Imag[:, 0] = 0
#
# B_Complex = B_Real + 1j * B_Imag
# B_Ampl = np.absolute(B_Complex)
#
# # samples_list = Parallel(n_jobs=nbatches)(delayed(simulate)() for _ in range(nbatches))
# # samples2 = np.concatenate(samples_list, axis=0)
#
# obj = BSRM(nsamples, P, B_Complex, dt, df, nt, nf)
# samples2 = obj.samples
#
# samples1 = np.reshape(samples1, [nsamples*nbatches, 1, nt])
# samples2 = np.reshape(samples2, [nsamples*nbatches, 1, nt])
#
# samples = np.concatenate((samples1, samples2), axis=1)
#
# P = np.real(estimate_cross_power_spectrum(samples))*T
# B = np.real(estimate_cross_bispectrum(samples))*T
# B = np.real(estimate_cross_bispectrum(samples))*T
