from SRM import *
import matplotlib.pyplot as plt

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
S = 125 / 4 * w ** 2 * np.exp(-5 * w)

SRM_object = SRM(10, S, dw, m, n)
samples = SRM_object.samples

#
# plt.figure()
# for i in range(len(samples)):
#     ecdf = ECDF(samples[i, :])
#     plt.plot(np.sort(samples[i, :]), ecdf(np.sort(samples[i, :])), ':')
# plt.show()
#
# cov_samples = np.cov(samples.T)
