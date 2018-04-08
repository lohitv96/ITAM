from SRM import *

class BSRM(SRM):
    def __init__(self, n_sim, S, B, dt, dw, nt, nw, case='uni', g=None):
        super().__init__(n_sim, S, dw, nt, nw, case='uni', g=None)
        self.dt = dt
        self.B = B
        self.B_Ampl = np.absolute(B)
        self.B_Real = np.real(B)
        self.B_Imag = np.imag(B)
        self.Biphase = np.arctan2(self.B_Imag, self.B_Real)
        self.Biphase[np.isnan(self.Biphase)] = 0
        self._compute_bicoherence()
        self.samples = self._simulate_bsrm_uni()

    def _compute_bicoherence(self):
        self.Bc2 = np.zeros_like(self.B_Real)
        self.PP = np.zeros_like(self.S)
        self.sum_Bc2 = np.zeros(self.nw)
        self.PP[0] = self.S[0]
        self.PP[1] = self.S[1]

        for i in range(self.nw):
            for j in range(int(np.ceil((i + 1) / 2))):
                w1 = i - j
                w2 = j
                if self.B_Ampl[w2, w1] > 0 and self.PP[w2] * self.PP[w1] != 0:
                    self.Bc2[w2, w1] = self.B_Ampl[w2, w1] ** 2 / (self.PP[w2] * self.PP[w1] * self.S[i]) * self.dw
                    self.sum_Bc2[i] = self.sum_Bc2[i] + self.Bc2[w2, w1]
                else:
                    self.Bc2[w2, w1] = 0
            if self.sum_Bc2[i] > 1:
                for j in range(int(np.ceil((i + 1) / 2))):
                    w1 = i - j
                    w2 = j
                    self.Bc2[w2, w1] = self.Bc2[w2, w1] / self.sum_Bc2[i]
                self.sum_Bc2[i] = 1
            self.PP[i] = self.S[i] * (1 - self.sum_Bc2[i])

    def _simulate_bsrm_uni(self):
        # samples_1 shows the contribution of the Pure Power Spectrum
        obj = SRM(self.n_sim, self.PP, self.dw, self.nt, self.nw, case='uni', g=None)
        samples_1 = obj.samples
        self.phi = obj.phi

        samples_2 = np.zeros_like(samples_1)
        Coeff = (2 ** self.n) * np.sqrt(self.S * np.prod(self.dw))

        for k in range(self.nw):
            if self.sum_Bc2[k] > 0:
                for l in range(int(np.ceil((k + 1) / 2))):
                    f1 = k - l
                    f2 = l
                    if self.Bc2[f2, f1] > 0:
                        for j in range(self.nt):
                            samples_2[:, j] = samples_2[:, j] + Coeff[k] * np.sqrt(self.Bc2[f2, f1]) * np.cos(
                                2 * np.pi * (f2 + f1) * self.dw * j * self.dt - self.phi[:, f2] - self.phi[:, f1] -
                                self.Biphase[f2, f1])
        samples = samples_1 + samples_2
        return samples
