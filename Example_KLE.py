import numpy as np

m = 40 + 1
# m = 100 + 1
# T = 2 # stationary
# T = 1 # nonstationary
T = 1000
dt = T / (m - 1)
t = np.linspace(0, T, m)

# Target Covariance(ACF)
R = np.zeros([m, m])
for i in range(m):
    for j in range(m):
        R[i, j] = 2 * np.exp(-((t[j] - t[i]) / 281) ** 2)  # var = 2

a = -3
c = 3
b = 3
X = np.arange(a, b + 0.0001, 0.0001)
Y = (X - a) ** 2 / (b - a) / (c - a)
Y[-1] = 1
Y[0] = 0

mu = np.ones(m)
sig = np.sqrt(np.diag(R))
