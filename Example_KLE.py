from itam_kle import *
from KLE import *
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

m = 40 + 1
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

plt.figure()
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('X')
plt.ylabel('Probability')
plt.ylim([0.0, 1.0])
plt.xlim([-3.0, 3.0])
plt.plot(X, Y)
plt.show()

mu = np.ones(m)
sig = np.sqrt(np.diag(R))

R_G_Converged, R_NG_Canverged = itam_kle(R, t, 'User', mu, sig, X, Y)
KLE_Object = KLE(10000, R_G_Converged, t,'User',mu,sig,X,Y)
samples = KLE_Object.samples
