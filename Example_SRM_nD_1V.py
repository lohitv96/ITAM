from SRM import *

# Input Data
# Time
dim = 2

T = 10
nt = 200
dt = T/nt
t = np.linspace(0, T-dt, nt)

# Frequency
W = np.array([1.0, 1.0])
nw = 100
dw = W / nw
x_list = [np.linspace(0, W[i] - dw[i], nw) for i in range(dim)]
xy_list = np.array(np.meshgrid(*x_list, indexing='ij'))
S = 125 / 4 * np.linalg.norm(xy_list, axis=0) ** 2 * np.exp(-5 * np.linalg.norm(xy_list, axis=0))

n_sim = 1

SRM_object = SRM(n_sim, S, dw, nt, nw, case='uni')
samples = SRM_object.samples

# Tx, Ty = np.meshgrid(t, t)
#
# # Plotting a sample realisation
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Tx, Ty, samples_SRM[0])
# plt.title('Realisation')
# ax.set_xlabel('$t_1$')
# ax.set_ylabel('$t_2$')
# plt.show()

print(np.var(samples))
print(np.sum(S)*np.prod(dw)*2**dim)