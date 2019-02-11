import numpy as np
import scipy.stats as stats
from scipy import interpolate
import itertools


def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q * xdiag * Q.T


def _getPs(A, W=None):
    W05 = np.matrix(W ** .5)
    return W05.I * _getAplus(W05 * A * W05) * W05.I


def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)


def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
    # W is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def estimate_PSD(samples, nt, T):
    sample_size = nt
    sample_maxtime = T
    dt = T / (nt - 1)
    Xw = np.fft.fft(samples, sample_size, axis=1)
    Xw = Xw[:, 0: int(sample_size / 2)]
    m_Ps = np.mean(np.absolute(Xw) ** 2 * sample_maxtime / sample_size ** 2, axis=0)
    num = int(T / (2 * dt))
    return np.linspace(0, (1 / (2 * dt) - 1 / T), num), m_Ps


def S_to_R(S, w, t):
    dw = w[1] - w[0]
    fac = np.ones(len(w))
    fac[1: len(w) - 1: 2] = 4
    fac[2: len(w) - 2: 2] = 2
    fac = fac * dw / 3
    R = np.zeros([S.shape[0], len(t)])
    for i in range(S.shape[0]):
        for j in range(len(t)):
            if S.shape[0] == 1:
                # np.array(2 * np.multiply(np.cos(np.matmul(np.transpose(np.matrix(t)), np.matrix(w))), S_NGT[i, :])*np.transpose(np.matrix(fac))).flatten()
                R[i, j] = 2 * np.dot(fac, S[i, :] * np.cos(w * t[j]))
            else:
                R[i, j] = 2 * np.dot(fac, np.sqrt((S[i, :] * S[j, :])) * np.cos(w * (t[i] - t[j])))
    return R


def R_to_r(R):
    # Normalize target non - Gaussian Correlation function to Correlation coefficient
    rho = np.zeros_like(R)
    for i in range(R.shape[0]):
        # Stationary
        if R.shape[0] == 1:
            if R[i, i] != 0:
                rho[i, :] = R[i, :] / R[i, i]
            else:
                rho[i, :] = 0
        # Nonstationary
        else:
            for j in range(R.shape[1]):
                if R[i, i] != 0 and R[j, j] != 0:
                    rho[i, j] = R[i, j] / np.sqrt(R[i, i] * R[j, j])
                else:
                    rho[i, j] = 0
    return rho


def R_to_S(R, w, t):
    dt = t[1] - t[0]
    fac = np.ones(len(t))
    fac[1: len(t) - 1: 2] = 4
    fac[2: len(t) - 2: 2] = 2
    fac = fac * dt / 3

    S = np.zeros([R.shape[0], len(w)])
    for i in range(R.shape[0]):
        for j in range(len(w)):
            S[i, j] = 2 / (2 * np.pi) * np.dot(fac, (R[i, :] * np.cos(t * w[j])))
    S[S < 0] = 0
    return S


def LogN_Var(y1, y2, rx_g, muN1, sigmaN1, muN2, sigmaN2, sy1, sy2, shift1, shift2):
    y1sq = y1 ** 2
    y2sq = y2 ** 2
    y1y2 = y1 * y2

    fg1 = stats.norm.cdf(y1, 0, sy1)
    g1 = stats.lognorm.ppf(fg1, muN1, sigmaN1)
    g1 = g1 + shift1

    fg2 = stats.norm.cdf(y2, 0, sy2)
    g2 = stats.lognorm.ppf(fg2, muN2, sigmaN2)
    g2 = g2 + shift2

    fy = 1 / (2 * np.pi * sy1 * sy2 * (np.sqrt(1 - (rx_g / (sy1 * sy2)) ** 2))) * np.exp(
        -1 / (2 * (1 - (rx_g / (sy1 * sy2)) ** 2)) * (
                y1sq / (sy1 ** 2) + y2sq / (sy2 ** 2) - 2. * (y1y2 * rx_g / (sy1 * sy2)) / (sy1 * sy2)))
    z = fy * (g1 * g2)
    return z


def Beta_Var(y1, y2, rx_g, lo_lim1, stretch1, lo_lim2, stretch2, alpha, beta, sy1, sy2):
    y1sq = y1 ** 2
    y2sq = y2 ** 2
    y1y2 = y1 * y2

    fg1 = stats.norm.cdf(y1, 0, sy1)
    g1 = stats.beta.ppf(fg1, alpha, beta)
    g1 = g1 * stretch1 + lo_lim1

    fg2 = stats.norm.cdf(y2, 0, sy2)
    g2 = stats.beta.ppf(fg2, alpha, beta)
    g2 = g2 * stretch2 + lo_lim2

    fy = 1 / (2 * np.pi * sy1 * sy2 * (np.sqrt(1 - (rx_g / (sy1 * sy2)) ** 2))) * np.exp(
        -1 / (2 * (1 - (rx_g / (sy1 * sy2)) ** 2)) * (
                y1sq / (sy1 ** 2) + y2sq / (sy2 ** 2) - 2. * (y1y2 * rx_g / (sy1 * sy2)) / (sy1 * sy2)))
    z = fy * (g1 * g2)
    return z


def User_Var(y1, y2, rx_g, cdf_x, cdf_y, sy1, sy2):
    y1sq = y1 ** 2
    y2sq = y2 ** 2
    y1y2 = y1 * y2

    fg1 = stats.norm.cdf(y1, 0, sy1)
    g1 = interpolate.interp1d(cdf_y, cdf_x)
    g1 = g1(fg1)

    fg2 = stats.norm.cdf(y2, 0, sy2)
    g2 = interpolate.interp1d(cdf_y, cdf_x)
    g2 = g2(fg2)

    fy = 1 / (2 * np.pi * sy1 * sy2 * (np.sqrt(1 - (rx_g / (sy1 * sy2)) ** 2))) * np.exp(
        -1 / (2 * (1 - (rx_g / (sy1 * sy2)) ** 2)) * (
                y1sq / (sy1 ** 2) + y2sq / (sy2 ** 2) - 2. * (y1y2 * rx_g / (sy1 * sy2)) / (sy1 * sy2)))
    z = fy * (g1 * g2)
    return z


def translate_process(Samples_G, Dist, mu, sig, parameter1, parameter2):
    Samples_NG = np.zeros_like(Samples_G)
    if Dist == 'Lognormal':
        for i in range(len(Samples_G)):
            sy1 = 1
            sigmaN1 = parameter1[i]
            muN1 = 0.5 * np.log(sig[i] ** 2 / (np.exp(sigmaN1 ** 2) - 1)) - 0.5 * sigmaN1 ** 2
            shift1 = -np.exp(muN1 + 0.5 * sigmaN1 ** 2)
            fg1 = stats.norm.cdf(Samples_G[i], 0, sy1)
            g1 = stats.lognorm.ppf(fg1, muN1, sigmaN1)
            g1 = g1 + shift1
            Samples_NG[i, :] = mu[i] + g1
    elif Dist == 'Beta':
        for i in range(len(Samples_G)):
            sy1 = 1
            alpha = parameter1[i]
            beta = parameter2[i]
            lo_lim1 = 0. - sig[i] * np.sqrt(alpha * (alpha + beta + 1) / beta)
            hi_lim1 = 0. + sig[i] * np.sqrt(beta * (alpha + beta + 1) / alpha)
            stretch1 = hi_lim1 - lo_lim1
            fg1 = stats.norm.cdf(Samples_G[i], 0, sy1)
            g1 = stats.beta.ppf(fg1, alpha, beta)
            g1 = g1 * stretch1 + lo_lim1
            Samples_NG[i, :] = mu[i] + g1
    elif Dist == 'User':
        for i in range(len(Samples_G)):
            sy1 = 1
            fg1 = stats.norm.cdf(Samples_G[i], 0, sy1)
            g1 = interpolate.interp1d(parameter2, parameter1)
            g1 = g1(fg1)
            Samples_NG[i, :] = g1
    return Samples_NG
