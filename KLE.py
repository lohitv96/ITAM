from tools import *


class KLE():
    def __init__(self, n_sim, R_G, t, Dist, mu, sig, parameter1, parameter2):
        self.t = t
        self.R = R_G
        self.Dist = Dist
        self.mu = mu
        self.sig = sig
        self.parameter1 = parameter1
        self.parameter1 = parameter2
        self.samples = self._simulate(n_sim)

    def _simulate(self, n_sim):
        # Gaussian Samples
        [lam, phi] = np.linalg.eig(self.R)
        dlambda = np.diag(lam)
        nRV = self.R.shape[0]
        xi = np.random.normal(size=(nRV, n_sim))
        Samples_G = phi[:, :nRV] * np.sqrt(lam) * xi
        Samples_G = np.real(Samples_G)
        Samples_NG = translate_process(Samples_G,self.Dist,self.mu,self.sig,self.parameter1,self.parameter2)
        return Samples_NG