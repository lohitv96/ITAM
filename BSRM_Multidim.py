import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import skew, moment
from copy import deepcopy
from BSRM import *

plt.style.use('seaborn')

# # Generate a series of theta
nsamples = 1024
num_batches = 12
########################################################################################################################
# Input parameters
dim = 2

T = 20  # Time(1 / T = dw)
nt = 128  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 64  # Num of Discretized Freq.

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

f_list = [f for _ in range(dim)]
F_P = np.array(np.meshgrid(*f_list, indexing='ij'))
P = 20 / np.sqrt(2 * np.pi) * np.exp(-1/2 * np.linalg.norm(F_P, axis=0) ** 2)

t_u = 2 * np.pi / (2 * 2 * np.pi * F)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

F_B = np.meshgrid(*[*f_list, *f_list])
b = 40 / (2 * np.pi) * np.exp(2 * (-1/2 * np.linalg.norm(F_B, axis=0) ** 2))
B_Real = deepcopy(b)
B_Imag = deepcopy(b)

B_Real[0, :, :, :] = 0
B_Real[:, 0, :, :] = 0
B_Real[:, :, 0, :] = 0
B_Real[:, :, :, 0] = 0
B_Imag[0, :, :, :] = 0
B_Imag[:, 0, :, :] = 0
B_Imag[:, :, 0, :] = 0
B_Imag[:, :, :, 0] = 0

B_Complex = B_Real + 1j * B_Imag

BSRM_object = BSRM(nsamples, P, B_Complex, dt, df, nt, nf)
samples = BSRM_object.samples

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

ranges = [range(nf) for _ in range(dim)]

for i in itertools.product(*ranges):
    wk = np.array(i)
    for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
        wj = np.array(j)
        wi = wk - wj
        if B_Ampl[(*wi, *wj)] > 0 and PP[(*wi, *[])] * PP[(*wj, *[])] != 0:
            Bc2[(*wi, *wj)] = B_Ampl[(*wi, *wj)] ** 2 / (PP[(*wi, *[])] * PP[(*wj, *[])] * P[(*wk, *[])]) * df ** dim
            sum_Bc2[(*wk, *[])] = sum_Bc2[(*wk, *[])] + Bc2[(*wi, *wj)]
        else:
            Bc2[(*wi, *wj)] = 0
    if sum_Bc2[(*wk, *[])] > 1:
        print('came here')
        for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
            wj = np.array(j)
            wi = wk - wj
            Bc2[(*wi, *wj)] = Bc2[(*wi, *wj)] / sum_Bc2[(*wk, *[])]
        sum_Bc2[(*wk, *[])] = 1
    PP[(*wk, *[])] = P[(*wk, *[])] * (1 - sum_Bc2[(*wk, *[])])

Coeff = np.sqrt(2 ** (dim + 1) * df ** dim * P)
Biphase_e = np.exp(Biphase * 1.0j)
Bc = np.sqrt(Bc2)

# # save the simulation data
# P.tofile('data/P.csv')
# B_Complex.tofile('data/B_Complex.csv')
# PP.tofile('data/PP.csv')
# sum_Bc2.tofile('data/sum_Bc2.csv')
# Bc.tofile('data/Bc.csv')
# Coeff.tofile('data/Coeff.csv')
# Biphase_e.tofile('data/Biphase_e.csv')

# # loading the simulation data
# P = np.fromfile('data/P.csv').reshape([128, 128])
# B_Complex = np.fromfile('data/B_Complex.csv').reshape([128, 128, 128, 128])
# PP = np.fromfile('data/PP.csv').reshape([128, 128])
# sum_Bc2 = np.fromfile('data/sum_Bc2.csv').reshape([128, 128])
# Bc = np.fromfile('data/Bc.csv').reshape([128, 128, 128, 128])
# Coeff = np.fromfile('data/Coeff.csv').reshape([128, 128])
# Biphase_e = np.fromfile('data/Biphase_e.csv').reshape([128, 128, 128, 128])

# def simulate():
#     Phi = np.random.uniform(size=np.append(nsamples, np.ones(dim, dtype=np.int32) * nf)) * 2 * np.pi
#     Phi_e = np.exp(Phi * 1.0j)
#     B = np.sqrt(1 - sum_Bc2) * Phi_e
#     Phi_e = np.einsum('i...->...i', Phi_e)
#     B = np.einsum('i...->...i', B)
#
#     for i in itertools.product(*ranges):
#         wk = np.array(i)
#         for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
#             wj = np.array(j)
#             wi = wk - wj
#             B[(*wk, *[])] = B[(*wk, *[])] + Bc[(*wi, *wj)] * Biphase_e[(*wi, *wj)] * Phi_e[(*wi, *[])] * \
#                             Phi_e[(*wj, *[])]
#
#     B = np.einsum('...i->i...', B)
#     Phi_e = np.einsum('...i->i...', Phi_e)
#     B_temp = B * Coeff
#     B_temp[np.isnan(B_temp)] = 0
#     samples = np.fft.fftn(B_temp, [nt, nt])
#     samples = np.real(samples)
#     # samples = samples.reshape([nt, nt])
#     return samples

# Plotting of individual samples
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

t_list = [t for _ in range(dim)]
T = np.array(np.meshgrid(*t_list, indexing='ij'))

fig1 = plt.figure()
plt.title('2d random field with a prescribed Power Spectrum and Bispectrum')
pcm = pcolor(T[0], T[1], samples, cmap='RdBu_r', vmin=-30, vmax=30)
plt.colorbar(pcm, extend='both', orientation='vertical')
plt.xlabel('$X_{1}$')
plt.ylabel('$X_{2}$')
plt.savefig('BSRM samples')
plt.show()
#
# fig2 = plt.figure()
# plt.title('2d random field with a prescribed Power Spectrum')
# pcolor(T[0], T[1], samples_SRM, cmap='RdBu_r', vmin=-30, vmax=30)
# plt.colorbar(pcm, extend='both', orientation='vertical')
# plt.xlabel('$X_{1}$')
# plt.ylabel('$X_{2}$')
# plt.savefig('SRM samples')
# plt.show()


samples_list = Parallel(n_jobs=4)(delayed(simulate)() for _ in range(num_batches))
samples1 = np.concatenate(samples_list, axis=0)

# # saving the samples data
# samples1.tofile('data/samples.csv')
#
# # loading the samples data
# samples1 = np.fromfile('data/samples.csv').reshape([num_batches * nsamples, nt, nt])

print('The estimate of mean is', np.mean(samples), 'whereas the expected mean is 0.000')
print('The estimate of variance is', np.var(samples.flatten()), 'whereas the expected variance is',
      np.sum(P) * 4 * df ** 2)
print('The estimate of the third moment is ', moment(samples.flatten(), moment=3, axis=0),
      'whereas the expected third moment is ', np.sum(B_Real) * 9 * df ** 4)
print('The estimate of skewness is', skew(samples.flatten(), axis=0), 'whereas the expected skewness is',
      (np.sum(B_Real) * 9 * df ** 4) / (np.sum(P) * 4 * df ** 2) ** (3 / 2))

# B_list = []
# for i in range(12):
#     B1 = np.fft.fftn(samples1[i * nsamples:(i + 1) * nsamples], axes=[1, 2])
#     B2 = B1[:, :nf, :nf]
#     B2[:, 0, 0] = B2[:, 0, 0]/np.sqrt(2)
#     B_list.append(B2)
#     print(i)
#
# P_est = np.mean(np.mean(np.absolute(B_list) ** 2, axis=0), axis=0)*T**2/nt**4/2
# ratio = P_est/P

# Estimating the bispectrum for the two-dimensional case
# Computationally Intractable
# Xw = np.fft.fft(samples1[:1024])[:, :nf, :nf]
#
# s_B = np.zeros(shape=[nf, nf, nf, nf])
# s_B = s_B + s_B * 1.0j
#
# for i in range(1024):
#     print(i)
#     for i1 in range(nf):
#         for i2 in range(nf):
#             for j1 in range(nf - i1):
#                 for j2 in range(nf - i2):
#                     s_B[i1, i2, j1, j2] = s_B[i1, i2, j1, j2] + (
#                                 Xw[i, i1, i2] * Xw[i, j1, j2] * np.conj(Xw[i, i1 + j1, i2 + j2]))
#
# m_B = s_B/1024
# m_B_Real = np.real(m_B)
# m_B_Real[0, :, :, :] = 0
# m_B_Real[:, 0, :, :] = 0
# m_B_Real[:, :, 0, :] = 0
# m_B_Real[:, :, :, 0] = 0
#
#
# for i in itertools.product(*ranges):
#     wk = np.array(i)
#     for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
#         wj = np.array(j)
#         wi = wk - wj
