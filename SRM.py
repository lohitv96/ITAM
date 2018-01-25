from tools import *


class SRM:
    def __init__(self, n_sim, S, dw, m, n):
        # TODO: Error check for all the variables
        self.S = S
        self.dim = len(S.shape)
        self.dw = dw
        self.m = m
        self.n = n
        self.n_sim = n_sim
        self.samples = self._simulate(self.n_sim)

    def _simulate(self, n_sim):
        samples = []
        for i in range(n_sim):
            phi = np.random.uniform(size=np.ones(self.dim, dtype=np.int32) * self.n) * 2 * np.pi
            B = (2 ** self.dim) * np.exp(phi * 1.0j) * np.sqrt(self.S * np.prod(self.dw))
            sample = np.fft.fftn(B, np.ones(self.dim, dtype=np.int32) * self.m)
            sample = np.real(sample)
            samples.append(sample)
        samples = np.array(samples)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return samples
