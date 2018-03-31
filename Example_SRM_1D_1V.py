from SRM import *
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

plt.style.use('seaborn')

# Input Data
# Time
nt = 400 + 1
T = 250
dt = T / (nt - 1)
t = np.linspace(0, T, nt)

# Frequency
nw = 100 + 1
W = 3
dw = W / (nw - 1)
w = np.linspace(0, W, nw)

n_sim = 100000

t_u = 2*np.pi/2/W

if dt>t_u:
    print('Error')

# Defining the Power Spectrum Density Function
S = 125 / 4 * w ** 2 * np.exp(-5 * w)

plt.plot(w, S)
plt.show()

SRM_object = SRM(n_sim, S, dw, nt, nw, case='uni')
samples = SRM_object.samples

# Plotting the emperical distribution of the samples
# plt.figure()
# for i in range(len(samples)):
#     ecdf = ECDF(samples[i, :])
#     plt.plot(np.sort(samples[i, :]), ecdf(np.sort(samples[i, :])), ':')
# plt.show()

var_samples = np.var(samples[0])

# plotting individual samples
# plt.figure()
# plt.plot(t, samples[0])
# plt.show()

# Estimating the power spectrum density function
a = []
for i in range(n_sim):
    xw = np.fft.ifft(samples[i], 401)
    xw = xw[:201]
    a1 = np.abs(xw)**2*T/2/np.pi
    a.append(a1)

a = np.array(a)
a_m = np.mean(a, axis=0)
num = int((1/(2*dt) + 1/T)/(1/T))
w1 = np.linspace(0, 1/(2*dt) + 1/T, num)
plt.figure()
plt.plot(w1, a_m)
plt.plot(w, S)
plt.show()
