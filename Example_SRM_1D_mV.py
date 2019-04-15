from SRM import *
# Kaimal et al. (1972)
# Example from Deodatis Book Chapter 4

nsamples = 100

T = 10.0
nt = 800
dt = T/nt
dt = np.linspace(0, T-dt, nt)

W = 0.5
nw = 400
dw = W / nw
w = np.linspace(0, W-dw, nw)

S_11 = 38.3 / (1 + 6.19 * w) ** (5 / 3)
S_22 = 43.4 / (1 + 6.98 * w) ** (5 / 3)
S_33 = 135 / (1 + 21.8 * w) ** (5 / 3)

g_12 = np.exp(-0.1757 * w)
g_13 = np.exp(-3.478 * w)
g_23 = np.exp(-3.392 * w)

m = 3
S_list = np.array([S_11, S_22, S_33])
g_list = np.array([g_12, g_13, g_23])

# Assembly of S_jk
S_sqrt = np.sqrt(S_list)
S_jk = np.einsum('i...,j...->ij...', S_sqrt, S_sqrt)
# Assembly of g_jk
g_jk = np.zeros_like(S_jk)
counter = 0
for i in range(m):
    for j in range(i + 1, m):
        g_jk[i, j] = g_list[counter]
        counter = counter + 1
g_jk = np.einsum('ij...->ji...', g_jk) + g_jk

for i in range(m):
    g_jk[i, i] = np.ones_like(S_jk[0, 0])
S = S_jk * g_jk

SRM_object = SRM(nsamples, S, dw, nt, nw, case='multi')
samples = SRM_object.samples
