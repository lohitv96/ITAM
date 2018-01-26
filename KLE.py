from tools import *
from scipy.linalg import sqrtm


class KLE:
    def __init__(self, n_sim, R):
        self.R = R
        self.samples = self._simulate(n_sim)

    def _simulate(self, n_sim):
        lam, phi = np.linalg.eig(self.R)
        nRV = self.R.shape[0]
        xi = np.random.normal(size=(nRV, n_sim))
        lam = np.diag(lam)
        lam = lam.astype(np.float64)
        samples = np.dot(phi, np.dot(sqrtm(lam), xi))
        samples = np.real(samples)
        samples = samples.T
        # Samples_NG = translate_process(Samples_G, self.Dist, self.mu, self.sig, self.parameter1, self.parameter2)
        return samples
