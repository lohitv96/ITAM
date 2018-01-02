# from SRM_class_develop import *
import numpy as np
import matplotlib.pyplot as plt

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
# plt.xlim([0, 500])
# plt.ylim([0, 160])
# plt.plot(S[0])
# plt.show()

a = -3
b = 3
c = 3
X = np.arange(a, b + 0.0001, 0.0001)
Y = np.zeros_like(X)
for i in range(len(X)):
    Y[i] = (X[i]-a)**2/(b-a)/(c-a)
Y[-1] = 1
Y[1] = 0

mu = np.ones(m)
sig = np.sqrt(2)*np.ones(m)
pseudo = 'pseudo'
