from tools import *

class SRM:
    def __init__(self, n_sim, S, dw, nt, nw, case='uni', g=None):
        # TODO: Error check for all the variables
        # TODO: Division by 2 to deal with zero frequency for all cases
        self.S = S
        self.dw = dw
        self.nt = nt
        self.nw = nw
        self.n_sim = n_sim
        self.case = case
        if self.case == 'uni':
            self.n = len(S.shape)
            self.phi = np.random.uniform(size=np.append(self.n_sim, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
            self.samples = self._simulate_uni(self.phi)
        elif self.case == 'multi':
            self.m = self.S.shape[0]
            self.n = len(S.shape[1:])
            self.g = g
            self.phi = np.random.uniform(size=np.append(self.n_sim, np.append(self.m, np.ones(self.n, dtype=np.int32) * self.nw))) * 2 * np.pi
            self.samples = self._simulate_multi(self.phi)

    def _simulate_uni(self, phi):
        B = (2 ** self.n) * np.exp(phi * 1.0j) * np.sqrt(self.S * np.prod(self.dw))
        sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
        samples = np.real(sample)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return samples

    def _simulate_multi(self, phi):
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
        g_jk = np.einsum('ij...->ji...', g_jk) + g_jk

        for i in range(self.m):
            g_jk[i, i] = np.ones_like(S_jk[0, 0])
        S = S_jk * g_jk

        S = np.einsum('ij...->...ij', S)
        S1 = S[..., :, :]
        H_jk = np.zeros_like(S1)
        for i in range(len(S1)):
            try:
                H_jk[i] = np.linalg.cholesky(S1[i])
            except:
                H_jk[i] = np.linalg.cholesky(nearestPD(S1[i]))
        H_jk = H_jk.reshape(S.shape)
        H_jk = np.einsum('...ij->ij...', H_jk)
        samples_list = []
        for i in range(self.m):
            samples = 0
            for j in range(i+1):
                B = 2 * H_jk[i, j] * np.sqrt(np.prod(self.dw)) * np.exp(phi[:, j] * 1.0j)
                sample = np.fft.fftn(B, np.ones(self.n, dtype=np.int32) * self.nt)
                samples += np.real(sample)
            samples_list.append(samples)
        samples_list = np.array(samples_list)
        # samples = translate_process(samples, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return np.einsum('ij...->ji...', samples_list)
