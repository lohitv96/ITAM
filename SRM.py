from tools import *


class SRM:
    def __init__(self, n_sim, S_G, w, t, Dist, mu, sig, parameter1, parameter2):
        # TODO: Error check for all the variables
        self.w = w
        self.t = t
        self.dw = w[1] - w[0]
        self.dt = t[1] - t[0]
        self.Dist = Dist
        self.mu = mu
        self.sig = sig
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.S = S_G
        self.t_u = 2 * np.pi / (2 * self.w[-1])
        if self.dt > self.t_u:
            print('\n')
            print('ERROR:: Condition of delta_t <= 2*pi/(2*W_u)')
            print('\n')
        self.samples = self._simulate(n_sim)

    def _simulate(self, n_sim):
        samples = np.zeros([n_sim, len(self.t)])
        for i in range(n_sim):
            phi = np.random.uniform(size=len(self.w)) * 2 * np.pi
            a_t = np.zeros(len(self.t))
            for j in range(len(self.t)):
                a_t[j] = 2 * np.matrix(np.sqrt(self.dw * self.S)) * np.transpose(np.matrix(np.cos(self.w * self.t[j] + phi)))
            samples[i, :] = a_t
        samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return samples
