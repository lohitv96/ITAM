from tools import *
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
plt.style.use('seaborn')

# # Generate a series of theta
nsamples = 1024

########################################################################################################################
# Input parameters

dim = 2

T = 100  # Time(1 / T = dw)
nt = 256  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 128  # Num of Discretized Freq.

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

f_list = [f for _ in range(dim)]
F_P = np.array(np.meshgrid(*f_list, indexing='ij'))
P = 20 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * np.linalg.norm(F_P, axis=0) ** 2)

t_u = 2 * np.pi / (2 * 2 * np.pi * F)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

# Generating the 2 dimensional mesh grid
F_B = np.meshgrid(*[*f_list, *f_list])

b = 20 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * np.linalg.norm(F_B, axis=0)**2))
B_Real = b
B_Imag = b
B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)
Biphase = np.arctan2(B_Imag, B_Real)
Biphase[np.isnan(Biphase)] = 0

Bc2 = np.zeros_like(B_Real)
PP = np.zeros_like(P)
sum_Bc2 = np.zeros_like(P)
PP[0, :] = P[0, :]
PP[1, :] = P[1, :]
PP[:, 0] = P[:, 0]
PP[:, 1] = P[:, 1]

for i in range(nf):
    for j in range(nf):
        wk = [i, j]
        for k in range(int(np.ceil((i + 1) / 2))):
            for l in range(int(np.ceil((j + 1) / 2))):
                wi = [i - k, j - l]
                wj = [k, l]
                if B_Ampl[(*wi, *wj)] > 0 and PP[(*wi, *[])] * PP[(*wj, *[])] != 0:
                    Bc2[(*wi, *wj)] = B_Ampl[(*wi, *wj)] ** 2 / (PP[(*wi, *[])] * PP[(*wj, *[])] * P[(*wk, *[])]) * df
                    sum_Bc2[(*wk, *[])] = sum_Bc2[(*wk, *[])] + Bc2[(*wi, *wj)]
                else:
                    Bc2[(*wi, *wj)] = 0
        if sum_Bc2[i, j] > 1:
            for k in range(int(np.ceil((i + 1) / 2))):
                for l in range(int(np.ceil((i + 1) / 2))):
                    wi = [i - k, j - l]
                    wj = [k, l]
                    Bc2[(*wi, *wj)] = Bc2[(*wi, *wj)] / sum_Bc2[(*wk, *[])]
            sum_Bc2[(*wk, *[])] = 1
        PP[(*wk, *[])] = P[(*wk, *[])] * (1 - sum_Bc2[(*wk, *[])])

Coeff = np.sqrt(2 ** (dim + 1) * df ** dim * P)
Biphase_e = np.exp(Biphase * 1.0j)
Bc = np.sqrt(Bc2)

# save the simulation data
P.tofile('data/P.csv')
B_Complex.tofile('data/B_Complex.csv')
PP.tofile('data/PP.csv')
sum_Bc2.tofile('data/sum_Bc2.csv')
Bc.tofile('data/Bc.csv')
Coeff.tofile('data/Coeff.csv')
Biphase_e.tofile('data/Biphase_e.csv')

# # loading the simulation data
# P = np.fromfile('data/P.csv').reshape([128, 128])
# B_Complex = np.fromfile('data/B_Complex.csv').reshape([128, 128, 128, 128])
# PP = np.fromfile('data/PP.csv').reshape([128, 128])
# sum_Bc2 = np.fromfile('data/sum_Bc2.csv').reshape([128, 128])
# Bc = np.fromfile('data/Bc.csv').reshape([128, 128, 128, 128])
# Coeff = np.fromfile('data/Coeff.csv').reshape([128, 128])
# Biphase_e = np.fromfile('data/Biphase_e.csv').reshape([128, 128, 128, 128])


def simulate():
    Phi = np.random.uniform(size=np.append(nsamples, np.ones(dim, dtype=np.int32) * nf)) * 2 * np.pi
    Phi_e = np.exp(Phi * 1.0j)
    B = np.sqrt(1 - sum_Bc2) * Phi_e

    for i in range(nf):
        for j in range(nf):
            for k in range(1, int(np.ceil((i + 1) / 2))):
                for l in range(1, int(np.ceil((j + 1) / 2))):
                    wi = [i - k, j - l]
                    wj = [k, l]
                    B[:, i, j] = B[:, i, j] + Bc[(*wi, *wj)] * Biphase_e[(*wi, *wj)] * Phi_e[:, i - k, j - l] * Phi_e[:,
                                                                                                                k, l]
    B_temp = B * Coeff
    B_temp[np.isnan(B_temp)] = 0
    samples = np.fft.fftn(B_temp, [nt, nt])
    samples = np.real(samples)
    return samples


samples_list = Parallel(n_jobs=4)(delayed(simulate)() for _ in range(12))
samples1 = np.concatenate(samples_list, axis=0)

# saving the samples data
samples1.tofile('data/samples.csv')

# loading the samples data
samples1 = np.fromfile('data/samples.csv').reshape([12*1024, 256, 256])

print('The estimate of mean is', np.mean(samples1), 'whereas the expected mean is 0.000')
print('The estimate of variance is', np.mean(np.var(samples1, axis=0)), 'whereas the expected variance is',
      np.sum(P) * 4 * df ** 2)

from scipy.stats import skew, moment

print(np.mean(moment(samples1, moment=3, axis=0)))
print('The estimate of skewness is', np.mean(skew(samples1, axis=0)), 'whereas the expected variance is',
      (np.sum(b) * 16 * df ** 4) / (np.sum(P) * 4 * df ** 2) ** (3 / 2))

# # Estimating the 2 dimensional Power spectrum
# B_list = []
# for i in range(12):
#     B1 = np.fft.ifftn(samples1[i*nsamples:(i+1)*nsamples])
#     B2 = B1[:, :, :nf]
#     B2[:, :, 0] = B2[:, :, 0] + B1[:, :, nf]
#     B2[:, :, 1:] = B2[:, :, 1:] + np.conjugate(np.flip(B1[:, :, nf+1:], axis=2))
#     B3 = B2[:, :nf, :]
#     B3[:, 0, :] = B3[:, 0, :] + B2[:, nf, :]
#     B3[:, 1:, :] = B3[:, 1:, :] + np.flip(B2[:, nf+1:, :], axis=1)
#     B_list.append(B3)
#     print(i)
# temp = np.mean(B3, axis=0)
