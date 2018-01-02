import numpy as np
from tools import *

########################################################################################################################
########################################################################################################################
# Copyright (C) Shields Uncertainty Research Group (SURG)
# All Rights Reserved
# Johns Hopkins University
# Department of Civil Engineering
# Updated: 31 December 2017
# Lohit Vandanapu
########################################################################################################################
########################################################################################################################


def itam_srm(case, S, beta, w, t, CDF, mu, sig, parameter1, parameter2):
    # Initial Conditions
    nError1 = 0
    m = len(t)
    n = len(w)
    if m == 1:
        m = n
    S_NGT = S
    S_G0 = S
    W = w[-1]  # Upper cutoff frequency
    dw = w[1] - w[0]
    T = t[-1]
    dt = t[1] - t[0]

    t_u = 2 * np.pi / (2 * W)
    if dt > t_u:
        print('ERROR:: Condition of delta_t <= 2*pi/(2*W_u)')
    # Computing the Non-Gaussian parameters
    R_NGT = R_to_r(S_to_R(S_NGT, w, t))

    # Initial underlying Gaussian normalized correlation function
    # Stationary Case
    if S_NGT.shape[0] == 1:
        R_G0 = R_NGT
    # Nonstationary Case
    else:
        # Wiener - Khinchine_Simpson(ES -> pseudo - R)
        # Assuming independent structure
        rho = np.zeros_like(R_NGT)
        for i in range(S_NGT.shape[0]):
            for j in range(m):
                rho[i, j] = 2 * np.dot(fac, S_NGT[i, :] * np.cos(w * t[j]))
        R_G0 = rho
        R_G0 = R_to_r(R_G0)

    iconverge = 0
    Error0 = 100
    maxii = 10
    Error1_time = np.zeros([maxii])

    for ii in range(maxii):
        if ii != 0:
            R_G0 = S_to_R(S_G0, w, t)
            R_G0 = R_to_r(R_G0)

        # Translation the correlation coefficients from Gaussian to Non-Gaussian case
        R_NG0 = np.zeros_like(R_G0)
        if CDF == 'Lognormal':
            R_NG0 = translate(R_G0, 'Lognormal_Distribution', 'pseudo', mu, sig, parameter1, parameter2)
        elif case == 'Beta':
            R_NG0 = translate(R_G0, 'Beta_Distribution', 'pseudo', mu, sig, parameter1, parameter2)
        elif case == 'User':
            R_NG0 = translate(R_G0, 'User_Distribution', 'pseudo', mu, sig, parameter1, parameter2)

        # Unnormalize computed non - Gaussian R
        rho = np.zeros_like(R_NG0)
        for i in range(R_NG0.shape[0]):
            if R_NG0[i, 1] != 0:
                rho[i, :] = (R_NG0[i, :] - mu[i] ** 2)
            else:
                rho[i, :] = 0
        R_NG0_Unnormal_Mean = rho

        # Normalize computed non - Gaussian R(Stationary(1D R) & Nonstatioanry(Pseudo R))
        rho = np.zeros_like(R_NG0)
        for i in range(R_NG0.shape[0]):
            if R_G0[i, 1] != 0:
                rho[i, :] = (R_NG0[i, :] - mu[i] ** 2) / sig[i] ** 2
            else:
                rho[i, :] = 0
        R_NG0 = rho

        S_NG0 = R_to_S(R_NG0_Unnormal_Mean, w, t)

        if S_NG0.shape[0] == 1:
            # compute the relative difference between the computed S_NG0 & the target S_NGT
            Err1 = 0
            Err2 = 0
            for j in range(S_NG0.shape[1]):
                Err1 = Err1 + (S_NG0[1, j] - S_NGT[1, j]) ** 2
                Err2 = Err2 + S_NGT[1, j] ** 2
            Error1 = 100 * np.sqrt(Err1 / Err2)
            convrate = (Error0 - Error1) / Error1
            if convrate < 0.001 or ii == maxii or Error1 < 0.0005:
                iconverge = 1
            Error1_time[ii] = Error1
            nError1 = nError1 + 1

            print('\n')
            print('ITAM-SRM: Number of Iterations = ', nError1)
            print('ITAM-SRM: Value of the relative difference = ', Error1)
            print('ITAM-SRM: Converged rate of the difference = ', convrate)
            print('\n')

        else: # Pristely_Simpson (S_NG0 -> R_NG0)
            R_NG0_Unnormal = S_to_R(S_NG0, w, t)
            # compute the relative difference between the computed NGACF & the target R(Normalized)
            Err1 = 0
            Err2 = 0
            for i in range(R_NG0.shape[0]):
                for j in range(R_NG0.shape[1]):
                    Err1 = Err1 + (R_NG0[i, j] - R_NGT[i, j]) ** 2
                    Err2 = Err2 + R_NGT[i, j] ** 2
            Error1 = 100 * np.sqrt(Err1 / Err2)
            convrate = abs(Error0 - Error1) / Error1

            if convrate < 0.001 or ii == maxii or Error1 < 0.0005:
                iconverge = 1

            Error1_time[ii] = Error1
            nError1 = nError1 + 1

            print('\n')
            print('ITAM-SRM: Number of Iterations = ', nError1)
            print('ITAM-SRM: Value of the realative difference = ', Error1)
            print('ITAM-SRM: Converged rate of the difference = ', convrate)
            print('\n')

        # Upgrade the underlying PSDF or ES
        S_G1 = np.zeros_like(S_G0)
        for i in range(S_NG0.shape[0]):
            for j in range(S_NG0.shape[1]):
                if S_NG0[i, j] != 0:
                    S_G1[i, j] = ((S_NGT[i, j] / S_NG0[i, j])) ** beta * S_G0[i, j]
                else:
                    S_G1[i, j] = 0

        # Normalize the upgraded underlying PSDF or ES: Method1: Wiener - Khinchine_Simpson(S -> R or ES -> R)
        R_SG1 = S_to_R(S_G1, w, t)

        for i in range(S_G1.shape[0]):
            S_G1[i, :] = S_G1[i, :] / R_SG1(i, i)

        if iconverge == 0 and ii != maxii:
            S_G0 = S_G1
            Error0 = Error1
        else:
            convergeIter = nError1
            print('\n')
            print('Job Finished')
            print('\n')
            break
    S_G_Converged = S_G0
    S_NG_Converged = S_NG0
    return S_G_Converged, S_NG_Converged


def translate(R_G, name, pseudo, mu, sig, parameter1, parameter2):
    # TODO: Implementation of quad2d - python - dlquad
    R_NG = np.zeros_like(R_G)

    if name == 'Lognormal_Distribution':
        for i in range(R_G.shape[0]):
            sigmaN1 = parameter1[i]
            if pseudo == 'pseudo':
                sy1 = np.sqrt(R_G[i, 1])
            else:
                sy1 = np.sqrt(R_G[i, i])
            muN1 = 0.5 * np.log(sig[i] ** 2 / (np.exp(sigmaN1 ** 2) - 1)) - 0.5 * sigmaN1 ** 2
            shift1 = -np.exp(muN1 + 0.5 * sigmaN1 ** 2)
            for j in range(R_G.shape[1]):
                if pseudo == 'pseudo':
                    sy2 = sy1
                    if sy1 != 0 and sy2 != 0:
                        if j != 1:
                            R_NG[i, j] = dblquad(
                                lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                                                        shift1,
                                                        shift2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                                lambda x: 6 * sy2)
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
                            R_NG[i, j] = dblquad(
                                lambda y1, y2: LogN_Var(y1, y2, R_G[i, j], muN1, sigmaN1, muN2, sigmaN2, sy1, sy2,
                                                        shift1,
                                                        shift2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                                lambda x: 6 * sy2)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
    if name == 'Beta_Distribution':
        for i in range(R_G.shape[0]):
            alpha = parameter1[i]
            beta = parameter2[i]
            if pseudo == 'pseudo':
                sy1 = np.sqrt(R_G[i, 1])
            else:
                sy1 = np.sqrt(R_G[i, i])
            lo_lim1 = 0. - sig[i] * np.sqrt(alpha * (alpha + beta + 1) / beta)
            hi_lim1 = 0. + sig[i] * np.sqrt(beta * (alpha + beta + 1) / alpha)
            stretch1 = hi_lim1 - lo_lim1
            for j in range(R_G.shape[1]):
                if pseudo == 'pseudo':
                    sy2 = sy1
                    if sy1 != 0 and sy2 != 0:
                        if j != 1:
                            R_NG[i, j] = dblquad(
                                lambda y1, y2: Beta_Var(y1, y2, R_G[i, j], lo_lim1, stretch1, lo_lim2, stretch2, alpha,
                                                        beta, sy1, sy2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                                lambda x: 6 * sy2)
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
                            R_NG[i, j] = dblquad(
                                lambda y1, y2: Beta_Var(y1, y2, R_G[i, j], lo_lim1, stretch1, lo_lim2, stretch2, alpha,
                                                        beta, sy1, sy2), -6 * sy1, 6 * sy1, lambda x: -6 * sy2,
                                lambda x: 6 * sy2)
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
                            R_NG[i, j] = dblquad(lambda y2, y1: User_Var(y1, y2, R_G[i, j], cdf_x, cdf_y, sy1, sy2), -6 * sy1, 6 * sy1, lambda y2: -6 * sy2, lambda y2: 6 * sy2)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
                else:
                    sy2 = np.sqrt(R_G[j, j])
                    if sy1 != 0 and sy2 != 0:
                        if i != j:
                            R_NG[i, j] = dblquad(lambda y1, y2: User_Var(y1, y2, R_G[i, j], cdf_x, cdf_y, sy1, sy2),
                                                 -6 * sy1, 6 * sy1, lambda x: -6 * sy2, lambda x: 6 * sy2)
                        else:
                            R_NG[i, j] = mu[i] * mu[j] + sig[i] * sig[j]
                    else:
                        R_NG[i, j] = 0
    return R_NG

"""
x1 = np.linspace(-6*sy1, 6*sy1, 5000)
x2 = np.linspace(-6*sy2, 6*sy2, 5000)
z = temp(x1[:,None], x2)
simps(simps(z,x2),x1)
"""

from scipy.integrate import simps