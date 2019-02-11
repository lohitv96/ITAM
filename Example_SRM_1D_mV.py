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

SRM_object = SRM(nsamples, S_list, dw, nt, nw, case='multi', g=g_list)
samples = SRM_object.samples