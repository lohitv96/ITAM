from tools import *

def itam_srm(S, beta, w, t, CDF, mu, sig, parameter1, parameter2):
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
        elif CDF == 'Beta':
            R_NG0 = translate(R_G0, 'Beta_Distribution', 'pseudo', mu, sig, parameter1, parameter2)
        elif CDF == 'User':
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
                Err1 = Err1 + (S_NG0[0, j] - S_NGT[0, j]) ** 2
                Err2 = Err2 + S_NGT[0, j] ** 2
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

        else:  # Pristely_Simpson (S_NG0 -> R_NG0)
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
            S_G1[i, :] = S_G1[i, :] / R_SG1[i, i]

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

