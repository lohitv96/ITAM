# import matplotlib.pyplot as plt
from scipy.stats import skew, moment, kurtosis
# from mpl_toolkits.mplot3d import Axes3D
from BSRM import *
from SRM import *
from joblib import Parallel, delayed

# plt.style.use('seaborn')

# Input parameters
T = 20  # Time(1 / T = dw)
nt = 1024  # Num.of Discretized Time
F = 1 / T * nt / 2  # Frequency.(Hz)
nf = 512  # Num of Discretized Freq.
nsamples = 100  # Num.of samples_SRM
nbatches = 400

# # Generation of Input Data(Stationary)
dt = T / nt
t = np.linspace(0, T - dt, nt)
df = F / nf
f = np.linspace(0, F - df, nf)

# Target PSDF(stationary)
P = 20 * 1 / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * f ** 2)
P[0] = 0.1

t_u = 2 * np.pi / (2 * 2 * np.pi * F)
if dt * 0.99 > t_u:
    print('\n')
    print('ERROR:: Condition of delta_t <= 2*np.pi/(2*2*np.pi*F_u) = 1/(2*W_u)')
    print('\n')

# Generating the 2 dimensional mesh grid
fx = f
fy = f
Fx, Fy = np.meshgrid(fx, fy)

B = 40 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2)))
B_Real = B
B_Imag = B

B_Real[0, :] = 0
B_Real[:, 0] = 0
B_Imag[0, :] = 0
B_Imag[:, 0] = 0

B_Complex = B_Real + 1j * B_Imag
B_Ampl = np.absolute(B_Complex)

fx = f
fy = f
fz = f
Fx, Fy, Fz = np.meshgrid(fx, fy, fz)

T = 40 * 2 * 1 / (2 * np.pi) * np.exp(2 * (-1 / 2 * (Fx ** 2 + Fy ** 2 + Fz ** 2)))
T_Real = T
T_Imag = T

T_Real[0, :, :] = 0
T_Real[:, 0, :] = 0
T_Real[:, :, 0] = 0
T_Imag[0, :, :] = 0
T_Imag[:, 0, :] = 0
T_Imag[:, :, 0] = 0

T_Complex = T_Real + 1j * T_Imag
T_Ampl = np.absolute(T_Complex)

Biphase = np.arctan2(B_Imag, B_Real)
Biphase[np.isnan(Biphase)] = 0

Triphase = np.arctan2(T_Imag, T_Real)
Triphase[np.isnan(Triphase)] = 0

Bc2 = np.zeros_like(B_Real)
Tc2 = np.zeros_like(T_Real)

PP = np.zeros_like(P)
sum_Bc2 = np.zeros_like(P)
sum_Tc2 = np.zeros_like(P)
sum_c2 = np.zeros_like(P)

temp_b = np.zeros_like(B_Real)

PP[0, :] = P[0, :]
PP[1, :] = P[1, :]
PP[:, 0] = P[:, 0]
PP[:, 1] = P[:, 1]


for i in range(nf):
    wk = i
    for j in range(int((wk + 1)/ 2)):
        wj = j
        wi = wk - wj
        temp_b[wi, wj] = 1

temp_t = np.zeros_like(T_Real)
for i in range(nf):
    wl = i
    for j in range(int(np.ceil((i+1)/2))):
        wi = i - j
        for k in range(int(np.ceil((j+1)/2))):
            wk = k
            wj = j - k
            temp_t[wi, wj, wk] = 1

    #     if B_Ampl[(*wi, *wj)] > 0 and PP[(*wi, *[])] * PP[(*wj, *[])] != 0:
    #         Bc2[(*wi, *wj)] = B_Ampl[(*wi, *wj)] ** 2 / (PP[(*wi, *[])] * PP[(*wj, *[])] * P[(*wk, *[])]) * df ** dim
    #         sum_Bc2[(*wk, *[])] = sum_Bc2[(*wk, *[])] + Bc2[(*wi, *wj)]
    #     else:
    #         Bc2[(*wi, *wj)] = 0
    # sum_c2[wl] = sum_Bc2[wl] + sum_Tc2[wl]
    # if sum_c2[wl] > 1:
    #     print('check the Bispectrum and Trispectrum')
    #     for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
    #         wj = j
    #         wi = wk - wj
    #         Bc2[wi, wj] = Bc2[wi, wj] / sum_c2[wl]
    #     # write a nested loop for the tricoherence values
    #         Tc2[wi, wj, wk] = Tc2[wi, wj, wk] / sum_c2[wl]
    #     sum_Bc2[(*wk, *[])] = 1
    # PP[wl] = P[wl] * (1 - sum_c2[wl])

Coeff = np.sqrt(2 ** 2 * df * P)
Biphase_e = np.exp(Biphase * 1.0j)
Triphase_e = np.exp(Triphase * 1.0j)
Bc = np.sqrt(Bc2)
Tc = np.sqrt(Tc2)

np.save('data/P.npy', P)
np.save('data/B_Complex.npy', B_Complex)
np.save('data/T_Complex.npy', T_Complex)
np.save('data/sum_Bc2.npy', sum_Bc2)
np.save('data/sum_Tc2.npy', sum_Tc2)
np.save('data/sum_c2.npy', sum_c2)
np.save('data/Bc.npy', Bc)
np.save('data/Tc.npy', Tc)
np.save('data/Coeff.npy', Coeff)
np.save('data/Biphase_e.npy', Biphase_e)
np.save('data/Triphase_e.npy', Triphase_e)


def simulate():
    Phi = np.random.uniform(size=np.append(nsamples, np.ones(dim, dtype=np.int32) * nf)) * 2 * np.pi
    Phi_e = np.exp(Phi * 1.0j)
    B = np.sqrt(1 - sum_c2) * Phi_e
    Phi_e = np.einsum('i...->...i', Phi_e)
    B = np.einsum('i...->...i', B)

    for i in itertools.product(nf):
        wk = np.array(i)
        for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
            wj = np.array(j)
            wi = wk - wj
            B[(*wk, *[])] = B[(*wk, *[])] + Bc[(*wi, *wj)] * Biphase_e[(*wi, *wj)] * Phi_e[(*wi, *[])] * \
                            Phi_e[(*wj, *[])]

    B = np.einsum('...i->i...', B)
    Phi_e = np.einsum('...i->i...', Phi_e)
    B_temp = B * Coeff
    B_temp[np.isnan(B_temp)] = 0
    samples = np.fft.fftn(B_temp, [nt, nt])
    samples = np.real(samples)
    # samples_SRM = samples_SRM.reshape([nt, nt])
    return samples


np.save('data/SP.npy', PP)
