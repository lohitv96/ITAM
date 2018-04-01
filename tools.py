import numpy as np
import scipy.stats as stats
from scipy import interpolate
from scipy.integrate import simps


def estimate_PSD(samples, nt, T):
    sample_size = nt
    sample_maxtime = T
    dt = T / (nt - 1)
    Xw = np.fft.fft(samples, sample_size, axis=1)
    Xw = Xw[:, 0: int(sample_size / 2)]
    m_Ps = np.mean(np.absolute(Xw) ** 2 * sample_maxtime / sample_size ** 2 , axis=0)
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


def translate(R_G, name, pseudo, mu, sig, parameter1, parameter2):
    # TODO: 'dblquad' couldn't be used because of the nature of the function. Simpson's rule was applied rather.
    R_NG = np.zeros_like(R_G)

    if name == 'Lognormal_Distribution':
        for i in range(R_G.shape[0]):
            sigmaN1 = parameter1[i]
            if pseudo == 'pseudo':
                sy1 = np.sqrt(R_G[i, 0])
            else:
                sy1 = np.sqrt(R_G[i, i])
            muN1 = 0.5 * np.log(sig[i] ** 2 / (np.exp(sigmaN1 ** 2) - 1)) - 0.5 * sigmaN1 ** 2
            shift1 = -np.exp(muN1 + 0.5 * sigmaN1 ** 2)
            for j in range(R_G.shape[1]):
                if pseudo == 'pseudo':
                    sy2 = sy1
                    if sy1 != 0 and sy2 != 0:
                        if j != 0:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                                                        shift1, shift2)
                            z = f(x1[:, None], x2)
                            # R_NG[i, j] = dblquad(
                            #     lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                            #                             shift1,
                            #                             shift2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                            #     lambda x: 6 * sy2)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
                else:
                    sigmaN2 = parameter1[j]
                    sy2 = np.sqrt(R_G[j, j])
                    muN2 = 0.5 * np.log(sig[j] ** 2 / (np.exp(sigmaN2 ** 2) - 1)) - 0.5 * sigmaN2 ** 2
                    shift2 = -np.exp(muN2 + 0.5 * sigmaN2 ** 2)
                    if sy1 != 0 and sy2 != 0:
                        if i != j:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                                                        shift1, shift2)
                            z = f(x1[:, None], x2)
                            # R_NG[i, j] = dblquad(
                            #     lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                            #                             shift1,
                            #                             shift2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                            #     lambda x: 6 * sy2)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
    if name == 'Beta_Distribution':
        for i in range(R_G.shape[0]):
            alpha = parameter1[i]
            beta = parameter2[i]
            if pseudo == 'pseudo':
                sy1 = np.sqrt(R_G[i, 0])
            else:
                sy1 = np.sqrt(R_G[i, i])
            lo_lim1 = 0. - sig[i] * np.sqrt(alpha * (alpha + beta + 1) / beta)
            hi_lim1 = 0. + sig[i] * np.sqrt(beta * (alpha + beta + 1) / alpha)
            stretch1 = hi_lim1 - lo_lim1
            for j in range(R_G.shape[1]):
                if pseudo == 'pseudo':
                    sy2 = sy1
                    if sy1 != 0 and sy2 != 0:
                        if j != 0:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: Beta_Var(y1, y2, R_G[i, j], lo_lim1, stretch1, lo_lim2, stretch2, alpha,
                                                        beta, sy1, sy2)
                            z = f(x1[:, None], x2)
                            # R_NG[i, j] = dblquad(
                            #     lambda y1, y2: Beta_Var(y1, y2, R_G[i, j], lo_lim1, stretch1, lo_lim2, stretch2, alpha,
                            #                             beta, sy1, sy2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                            #     lambda x: 6 * sy2)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
                else:
                    alpha = parameter1[j]
                    beta = parameter2[j]
                    sy2 = np.sqrt(R_G[j, j])
                    lo_lim2 = 0. - sig[j] * np.sqrt(alpha * (alpha + beta + 1) / beta)
                    hi_lim2 = 0. + sig[j] * np.sqrt(beta * (alpha + beta + 1) / alpha)
                    stretch2 = hi_lim2 - lo_lim2
                    if sy1 != 0 and sy2 != 0:
                        if i != j:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: Beta_Var(y1, y2, R_G[i, j], lo_lim1, stretch1, lo_lim2, stretch2, alpha,
                                                        beta, sy1, sy2)
                            z = f(x1[:, None], x2)
                            # R_NG[i, j] = dblquad(
                            #     lambda y1, y2: Beta_Var(y1, y2, R_G[i, j], lo_lim1, stretch1, lo_lim2, stretch2, alpha,
                            #                             beta, sy1, sy2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                            #     lambda x: 6 * sy2)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0

    if name == 'User_Distribution':
        cdf_x = parameter1
        cdf_y = parameter2
        for i in range(R_G.shape[0]):
            if pseudo == 'pseudo':
                sy1 = np.sqrt(R_G[i, 0])
            else:
                sy1 = np.sqrt(R_G[i, i])
            for j in range(R_G.shape[1]):
                if pseudo == 'pseudo':
                    sy2 = sy1
                    if sy1 != 0 and sy2 != 0:
                        if j != 0:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: User_Var(y1, y2, R_G[i, j], cdf_x, cdf_y, sy1, sy2)
                            z = f(x1[:, None], x2)
                            # R_NG[i, j] = dblquad(f, -1 * sy1, 1 * sy1, lambda x: -1 * sy2, lambda x: 1 * sy2, epsabs=0.0)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
                else:
                    sy2 = np.sqrt(R_G[j, j])
                    if sy1 != 0 and sy2 != 0:
                        if i != j:
                            x1 = np.linspace(-6 * sy1, 6 * sy1, 1000)
                            x2 = np.linspace(-6 * sy2, 6 * sy2, 1000)
                            f = lambda y1, y2: User_Var(y1, y2, R_G[i, j], cdf_x, cdf_y, sy1, sy2)
                            z = f(x1[:, None], x2)
                            R_NG[i, j] = simps(simps(z, x2), x1)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
    return R_NG


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
