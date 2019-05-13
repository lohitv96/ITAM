from tools import *


class SRM:
    def __init__(self, n_sim, S, dw, nt, nw, case='uni'):
        self.S = S
        self.dw = dw
        self.nt = nt
        self.nw = nw
        self.n_sim = n_sim
        self.case = case
        if self.case == 'uni':
            self.n = len(S.shape)
            self.phi = np.random.uniform(
                size=np.append(self.n_sim, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
            self.samples = self._simulate_uni(self.phi)
        elif self.case == 'multi':
            self.m = self.S.shape[0]
            self.n = len(S.shape[2:])
            self.phi = np.random.uniform(
                size=np.append(self.n_sim, np.append(np.ones(self.n, dtype=np.int32) * self.nw, self.m))) * 2 * np.pi
            self.samples = self._simulate_multi(self.phi)

    def _simulate_uni(self, phi):
        B = np.exp(phi * 1.0j) * np.sqrt(2 ** (self.n + 1) * self.S * np.prod(self.dw))
        sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
        samples = np.real(sample)
        return samples

    def _simulate_multi(self, phi):
        S = np.einsum('ij...->...ij', self.S)
        Coeff = np.sqrt(2 ** (self.n + 1)) * np.sqrt(np.prod(self.dw))
        U, s, V = np.linalg.svd(S)
        R = np.einsum('...ij,...j->...ij', U, np.sqrt(s))
        F = Coeff * np.einsum('...ij,n...j -> n...i', R, np.exp(phi * 1.0j))
        F[np.isnan(F)] = 0
        samples = np.real(np.fft.fftn(F, s=[self.nt for _ in range(self.n)], axes=tuple(np.arange(1, 1+self.n))))
        return samples
