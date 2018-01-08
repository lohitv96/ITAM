import numpy as np
from tools import *
from correlation_matrix import *


def itam_kle(R, t, CDF, mu, sig, parameter1, parameter2):
    # Initial condition
    nError1 = 0
    m = len(t)
    dt = t[1] - t[0]
    T = t[-1]
    # Erasing zero values of variations
    R_NGT = R
    TF = [False] * m
    TF = np.array(TF)
    for i in range(m):
        if R[i, i] == 0: TF[i] = True
    t[TF] = []
    mu[TF] = []
    sig[TF] = []
    if CDF == 'User':
        parameter1[TF] = []
        parameter2[TF] = []
    # Normalize the non - stationary and stationary non - Gaussian Covariance to Correlation
    R_NGT = R_to_r(R_NGT)
    # Initial Guess
    R_G0 = R_NGT

    # Iteration Condition
    iconverge = 0
    Error0 = 100
    maxii = 5
    Error1_time = np.zeros(maxii)
    for ii in range(maxii):
        if CDF == 'Lognormal':
            R_NG0 = translate(R_G0, 'Lognormal_Distribution', '', mu, sig, parameter1, parameter2)
        elif CDF == 'Beta':
            R_NG0 = translate(R_G0, 'Beta_Distribution', '', mu, sig, parameter1, parameter2)
        elif CDF == 'User':
            # monotonic increasing CDF
            R_NG0 = translate(R_G0, 'User_Distribution', '', mu, sig, parameter1, parameter2)

        # Normalize the computed non - Gaussian ACF
        rho = np.zeros_like(R_NG0)
        for i in range(R_NG0.shape[0]):
            for j in range(R_NG0.shape[1]):
                if R_NG0[i, i] != 0 and R_NG0[j, j] != 0:
                    rho[i, j] = (R_NG0[i, j] - mu[i] * mu[j]) / (sig[i] * sig[j])
                else:
                    rho[i, j] = 0
        R_NG0 = rho

        # compute the relative difference between the computed NGACF & the target R(Normalized)
        Err1 = 0
        Err2 = 0
        for i in range(R_NG0.shape[0]):
            for j in range(R_NG0.shape[1]):
                Err1 = Err1 + (R_NGT[i, j] - R_NG0[i, j]) ** 2
                Err2 = Err2 + R_NG0[i, j] ** 2
        Error1 = 100 * np.sqrt(Err1 / Err2)
        convrate = abs(Error0 - Error1) / Error1
        if convrate < 0.001 or ii == maxii or Error1 < 0.0005:
            iconverge = 1
        Error1_time[ii] = Error1
        nError1 = nError1 + 1
        print('\n')
        print('ITAM-KL: Number of Iterations =', nError1)
        print('ITAM-KL: Value of the realative difference =', Error1)
        print('ITAM-KL: Converged rate of the difference =', convrate)
        print('\n')
        # Upgrade the underlying Gaussian ACF
        R_G1 = np.zeros_like(R_G0)
        for i in range(R_G0.shape[0]):
            for j in range(R_G0.shape[1]):
                if R_NG0[i, j] != 0:
                    R_G1[i, j] = (R_NGT[i, j] / R_NG0[i, j]) * R_G0[i, j]
                else:
                    R_G1[i, j] = 0
        # Eliminate Numerical error of Upgrading Scheme
        R_G1[R_G1 < -1.0] = -0.99999
        R_G1[R_G1 > 1.0] = 0.99999
        # Normalize the Gaussian ACF
        R_G1 = R_to_r(R_G1)
        # Iteratively finding the nearest PSD(Qi & Sun, 2006)
        R_G1 = correlation_matrix(R_G1)
        R_G1 = R_to_r(R_G1)

        # Eliminate Numerical error of finding the nearest PSD Scheme
        R_G1[R_G1 < -1.0] = -0.99999
        R_G1[R_G1 > 1.0] = 0.99999

        if iconverge == 0 and ii != maxii:
            R_G0 = R_G1
            Error0 = Error1
        else:
            convergeIter = nError1
            print('\n')
            print('[Job Finished]')
            print('\n')
            break

    R_G_Converged = R_G0
    # R_NG_Converged = R_NG0_Unnormal
    return R_G_Converged, R_NG_Converged
