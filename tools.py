import numpy as np
import scipy.stats as stats
from scipy import interpolate
from scipy.integrate import dblquad


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
            S[i, j] = 2 / (2 * np.pi) * fac * (R[i, :] * np.cos(t * w[j]))
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
