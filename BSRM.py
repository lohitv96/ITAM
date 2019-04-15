from tools import *


class BSRM:
    def __init__(self, n_sim, S, B, dt, dw, nt, nw, case='uni'):
        self.n_sim = n_sim
        self.nw = nw
        self.nt = nt
        self.dw = dw
        self.dt = dt
        self.S = S
        self.B = B
        self.B_Ampl = np.absolute(B)
        self.B_Real = np.real(B)
        self.B_Imag = np.imag(B)
        self.Biphase = np.arctan2(self.B_Imag, self.B_Real)
        self.Biphase[np.isnan(self.Biphase)] = 0
        if case == 'uni':
            self.n = len(S.shape)
            self.phi = np.random.uniform(size=np.append(self.n_sim, np.ones(self.n, dtype=np.int32) * self.nw)) * 2 * np.pi
            self._compute_bicoherence()
            self.samples = self._simulate_bsrm_uni()
        if case == 'multi':
            self.m = S.shape[0]
            self.n = len(S.shape[2:])
            self.phi = np.random.uniform(size=np.append(self.n_sim, np.append(self.nw, self.m))) * 2 * np.pi
            self.samples = self._simulate_bsrm_multi()

    def _compute_bicoherence(self):
        self.Bc2 = np.zeros_like(self.B_Real)
        self.PP = np.zeros_like(self.S)
        self.sum_Bc2 = np.zeros_like(self.S)

        if self.n == 1:
            self.PP[0] = self.S[0]
            self.PP[1] = self.S[1]

        if self.n == 2:
            self.PP[0, :] = self.S[0, :]
            self.PP[1, :] = self.S[1, :]
            self.PP[:, 0] = self.S[:, 0]
            self.PP[:, 1] = self.S[:, 1]

        if self.n == 3:
            self.PP[0, :, :] = self.S[0, :, :]
            self.PP[1, :, :] = self.S[1, :, :]
            self.PP[:, 0, :] = self.S[:, 0, :]
            self.PP[:, 1, :] = self.S[:, 1, :]
            self.PP[:, :, 0] = self.S[:, :, 0]
            self.PP[:, :, 1] = self.S[:, :, 1]

        self.ranges = [range(self.nw) for _ in range(self.n)]

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                wj = np.array(j)
                wi = wk - wj
                if self.B_Ampl[(*wi, *wj)] > 0 and self.PP[(*wi, *[])] * self.PP[(*wj, *[])] != 0:
                    self.Bc2[(*wi, *wj)] = self.B_Ampl[(*wi, *wj)] ** 2 / (
                                self.PP[(*wi, *[])] * self.PP[(*wj, *[])] * self.S[(*wk, *[])]) * self.dw ** self.n
                    self.sum_Bc2[(*wk, *[])] = self.sum_Bc2[(*wk, *[])] + self.Bc2[(*wi, *wj)]
                else:
                    self.Bc2[(*wi, *wj)] = 0
            if self.sum_Bc2[(*wk, *[])] > 1:
                print('Results may not be as expected as sum of partial bicoherences is greater than 1')
                for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                    wj = np.array(j)
                    wi = wk - wj
                    self.Bc2[(*wi, *wj)] = self.Bc2[(*wi, *wj)] / self.sum_Bc2[(*wk, *[])]
                self.sum_Bc2[(*wk, *[])] = 1
            self.PP[(*wk, *[])] = self.S[(*wk, *[])] * (1 - self.sum_Bc2[(*wk, *[])])

    def _simulate_bsrm_uni(self):
        Coeff = np.sqrt((2 ** (self.n + 1)) * self.S * self.dw ** self.n)
        Phi_e = np.exp(self.phi * 1.0j)
        Biphase_e = np.exp(self.Biphase * 1.0j)
        B = np.sqrt(1 - self.sum_Bc2) * Phi_e
        Bc = np.sqrt(self.Bc2)

        Phi_e = np.einsum('i...->...i', Phi_e)
        B = np.einsum('i...->...i', B)

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                wj = np.array(j)
                wi = wk - wj
                B[(*wk, *[])] = B[(*wk, *[])] + Bc[(*wi, *wj)] * Biphase_e[(*wi, *wj)] * Phi_e[(*wi, *[])] * \
                                Phi_e[(*wj, *[])]

        B = np.einsum('...i->i...', B)
        Phi_e = np.einsum('...i->i...', Phi_e)
        B = B * Coeff
        B[np.isnan(B)] = 0
        samples = np.fft.fftn(B, [self.nt for _ in range(self.n)])
        return np.real(samples)

    def _simulate_bsrm_multi(self):
        Coeff = np.sqrt((2 ** (self.n + 1)) * self.S * np.prod(self.dw))
        Phi_e = np.exp(self.phi * 1.0j)
        Biphase_e = np.exp(self.Biphase * 1.0j)
        B = np.sqrt(1 - self.sum_Bc2) * Phi_e
        Bc = np.sqrt(self.Bc2)

        Phi_e = np.einsum('i...->...i', Phi_e)
        B = np.einsum('i...->...i', B)

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                wj = np.array(j)
                wi = wk - wj
                B[(*wk, *[])] = B[(*wk, *[])] + Bc[(*wi, *wj)] * Biphase_e[(*wi, *wj)] * Phi_e[(*wi, *[])] * \
                                Phi_e[(*wj, *[])]

        B = np.einsum('...i->i...', B)
        Phi_e = np.einsum('...i->i...', Phi_e)
        B = B * Coeff
        B[np.isnan(B)] = 0
        samples = np.fft.fftn(B, [self.nt for _ in range(self.n)])
        return np.real(samples)