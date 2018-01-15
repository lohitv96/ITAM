from itam_srm import *
from SRM import *
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import pylab as p

plt.style.use('seaborn')

# Input Data
# Time
m = 400 + 1
T = 1000
dt = T / (m - 1)
t = np.linspace(0, T, m)

# Frequency
n = 500 + 1
W = 0.2
dw = W / (n - 1)
w = np.linspace(0, W, n)

# Defining the Power Spectrum Density Function
S = np.zeros((1, len(w)))
for i in range(1):
    S[i, :] = 2 * 281 / 2 / np.sqrt(np.pi) * np.exp(-78961 / 4 * w ** 2)

# plt.figure()
# plt.title('Power Spectrum Density Function')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.ylim([0, 160])
# plt.xlim([0, 0.2])
# plt.plot(w, S[0])
# plt.show()

# Examples for User Specified Distribution
a = -3
b = 3
c = 3
X = np.arange(a, b + 0.0001, 0.0001)
Y = np.zeros_like(X)
for i in range(len(X)):
    Y[i] = (X[i] - a) ** 2 / (b - a) / (c - a)
Y[-1] = 1
Y[1] = 0

# plt.figure()
# plt.title('Cumulative Distribution Function (CDF)')
# plt.xlabel('X')
# plt.ylabel('Probability')
# plt.ylim([0.0, 1.0])
# plt.xlim([-3.0, 3.0])
# plt.plot(X, Y)
# plt.show()

mu = np.ones(m)
sig = np.sqrt(2) * np.ones(m)
pseudo = 'pseudo'
S_G_Converged, S_NG_Converged = itam_srm(S, 1.0, w, t, 'User', mu, sig, X, Y)

plt.figure()
plt.title('Comparision of Results')
plt.plot(w, S_NG_Converged[0], label='Converged Non-Gaussian')
plt.plot(w, S_G_Converged[0], label='Converged Gaussian')
plt.plot(w, S[0], label='Original')
plt.xlim([0, 0.2])
plt.ylim([0, 160])
plt.legend(loc='upper right')
plt.show()

SRM_object = SRM(10000, S_G_Converged, w, t, 'User', mu, sig, X, Y)
samples = SRM_object.samples

plt.figure()
for i in range(len(samples)):
    ecdf = ECDF(samples[i, :])
    plt.plot(np.sort(samples[i, :]), ecdf(np.sort(samples[i, :])), ':')
plt.show()

cov_samples = np.cov(samples.T)

