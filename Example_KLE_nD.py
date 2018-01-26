from KLE import *
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

plt.style.use('seaborn')

m = 400 + 1
T = 1000
dt = T / (m - 1)
t = np.linspace(0, T, m)

# Target Covariance(ACF)
R = np.zeros([m, m])
for i in range(m):
    for j in range(m):
        R[i, j] = 2 * np.exp(-((t[j] - t[i]) / 281) ** 2)  # var = 2

KLE_Object = KLE(10, R)
samples = KLE_Object.samples

plt.figure()
plt.plot(samples[1])
plt.show()

plt.figure()
for i in range(10):
    ecdf = ECDF(samples[i, :])
    plt.plot(np.sort(samples[i, :]), ecdf(np.sort(samples[i, :])), ':')
plt.show()

cov_samples = np.cov(samples.T)

S = 2 / (2 * np.pi) * np.fft.fftn(R[0]*dt)
Sr = np.real(S)

def R_to_S(R, w):
    # Following Fourier Transform Implementation
    S = 2 / (2 * np.pi) * np.fft.fftn(R, s=[len(w)])
    return S

