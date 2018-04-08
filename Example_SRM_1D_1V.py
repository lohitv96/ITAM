from SRM import *
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

plt.style.use('seaborn')

# Input parameters
T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nw = 128 # Num of Discretized Freq.
n_sim = 100000  # Num.of samples

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
dw = F / nw
w = np.linspace(0, F - dw, nw)

t_u = 2*np.pi/2/F

if dt>t_u:
    print('Error')

# Defining the Power Spectrum Density Function
S = 125 / 4 * w ** 2 * np.exp(-5 * w)

plt.plot(w, S)
plt.show()

SRM_object = SRM(n_sim, S, dw, nt, nw, case='uni')
samples = SRM_object.samples
