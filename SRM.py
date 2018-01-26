from tools import *


class SRM:
    def __init__(self, n_sim, S, dw, nt, nw, case='uni', g=None):
        # TODO: Error check for all the variables
        self.S = S
        self.dw = dw
        self.nt = nt
        self.nw = nw
        self.n_sim = n_sim
        self.case = case
        if self.case == 'uni':
            self.n = len(S.shape)
            self.samples = self._simulate_uni(self.n_sim)
        elif self.case == 'multi':
            self.n = int(len(S.shape) - 2)
            self.g = g
            self.samples = self._simulate_uni(self.n_sim)

    def _simulate_uni(self, n_sim):
        samples = []
        for i in range(n_sim):
            phi = np.random.uniform(size=np.ones(self.n, dtype=np.int32) * self.nw) * 2 * np.pi
            B = (2 ** self.n) * np.exp(phi * 1.0j) * np.sqrt(self.S * np.prod(self.dw))
            sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
            sample = np.real(sample)
            samples.append(sample)
        samples = np.array(samples)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return samples

    def _simulate_multi(self, n_sim):
        self.m = self.S.shape[0]
        # Assembly of S_jk
        S_sqrt = np.sqrt(self.S)
        S_jk = np.einsum('i...,j...->ij...', S_sqrt, S_sqrt)
        # Assembly of g_jk
        g_jk = np.zeros_like(S_jk)
        l = 0
        for i in range(self.m):
            for j in range(i + 1, self.m):
                g_jk[i, j] = self.g[l]
                l = l + 1
        g_jk = np.einsum('ij...,ji...->ij...', g_jk, g_jk)

        for i in range(self.m):
            g_jk[i, i] = np.ones_like(S_jk[0, 0])
        S = S_jk * g_jk

        S = np.einsum('ij...->...ij', S)
        H_jk = np.linalg.cholesky(S)
        H_jk = np.einsum('...ij->ij...', H_jk)

        samples = []
        for i in range(n_sim):
            phi = np.random.uniform(size=np.ones(self.n, dtype=np.int32) * self.nw) * 2 * np.pi
            B = 2 * H_jk[0, 0] * np.sqrt(self.dw[0]) * np.exp(phi * 1.0j)
            sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
            sample = np.real(sample)
            samples.append(sample)
        samples = np.array(samples)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return samples
