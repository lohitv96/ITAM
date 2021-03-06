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



import numpy as np
import scipy.stats as stats


def transform_ng_to_g(corr_norm, dist, dist_params, samples_ng, jacobian=True):

    """
        Description:

            A function that performs transformation of a non-Gaussian random variable to a Gaussian one.

        Input:
            :param corr_norm: Correlation matrix in the standard normal space
            :type corr_norm: ndarray

            :param dist: marginal distributions
            :type dist: list

            :param dist_params: marginal distribution parameters
            :type dist_params: list

            :param samples_ng: non-Gaussian samples_SRM
            :type samples_ng: ndarray

            :param jacobian: The Jacobian of the transformation
            :type jacobian: ndarray

        Output:
            :return: samples_g: Gaussian samples_SRM
            :rtype: samples_g: ndarray

            :return: jacobian: The jacobian
            :rtype: jacobian: ndarray

    """

    from scipy.linalg import cholesky

    a_ = cholesky(corr_norm, lower=True)
    samples_g = np.zeros_like(samples_ng)
    m, n = np.shape(samples_ng)
    for j in range(n):
        cdf = dist[j].cdf
        samples_g[:, j] = stats.norm.ppf(cdf(samples_ng[:, j], dist_params[j]))

    if not jacobian:
        print("UQpy: Done.")
        return samples_g, None
    else:
        temp_ = np.zeros([n, n])
        jacobian = [None] * m
        for i in range(m):
            for j in range(n):
                pdf = dist[j].pdf
                temp_[j, j] = stats.norm.pdf(samples_g[i, j]) / pdf(samples_ng[i, j], dist_params[j])
            jacobian[i] = np.linalg.solve(temp_, a_)

        return samples_g, jacobian


def transform_g_to_ng(corr_norm, dist, dist_params, samples_g, jacobian=True):

    """
        Description:

            A function that performs transformation of a Gaussian random variable to a non-Gaussian one.

        Input:
            :param corr_norm: Correlation matrix in the standard normal space
            :type corr_norm: ndarray

            :param dist: marginal distributions
            :type dist: list

            :param dist_params: marginal distribution parameters
            :type dist_params: list

            :param samples_g: Gaussian samples_SRM
            :type samples_g: ndarray

            :param jacobian: The Jacobian of the transformation
            :type jacobian: ndarray

        Output:
            :return: samples_ng: Gaussian samples_SRM
            :rtype: samples_ng: ndarray

            :return: jacobian: The jacobian
            :rtype: jacobian: ndarray

    """

    from scipy.linalg import cholesky

    samples_ng = np.zeros_like(samples_g)
    m, n = np.shape(samples_g)
    for j in range(n):
        i_cdf = dist[j].icdf
        samples_ng[:, j] = i_cdf(stats.norm.cdf(samples_g[:, j]), dist_params[j])

    if not jacobian:
        print("UQpy: Done.")
        return samples_ng, None
    else:
        a_ = cholesky(corr_norm, lower=True)
        temp_ = np.zeros([n, n])
        jacobian = [None] * m
        for i in range(m):
            for j in range(n):
                pdf = dist[j].pdf
                temp_[j, j] = pdf(samples_ng[i, j], dist_params[j]) / stats.norm.pdf(samples_g[i, j])
            jacobian[i] = np.linalg.solve(a_, temp_)

        return samples_ng, jacobian


def run_corr(samples, corr):

    """
        Description:

            A function which performs the Cholesky decomposition of the correlation matrix and correlates standard
            normal samples_SRM.

        Input:
            :param corr: Correlation matrix
            :type corr: ndarray

            :param samples: Standard normal samples_SRM.
            :type samples: ndarray


        Output:
            :return: samples_corr: Correlated standard normal samples_SRM
            :rtype: samples_corr: ndarray

    """

    from scipy.linalg import cholesky
    c = cholesky(corr, lower=True)
    samples_corr = np.dot(c, samples.T)

    return samples_corr.T


def run_decorr(samples, corr):

    """
        Description:

            A function which performs the Cholesky decomposition of the correlation matrix and de-correlates standard
            normal samples_SRM.

        Input:
            :param corr: Correlation matrix
            :type corr: ndarray

            :param samples: standard normal samples_SRM.
            :type samples: ndarray


        Output:
            :return: samples_uncorr: Uncorrelated standard normal samples_SRM
            :rtype: samples_uncorr: ndarray

    """

    from scipy.linalg import cholesky

    c = cholesky(corr, lower=True)
    inv_corr = np.linalg.inv(c)
    samples_uncorr = np.dot(inv_corr, samples.T)

    return samples_uncorr.T


def correlation_distortion(marginal, params, rho_norm):

    """
        Description:

            A function to solve the double integral equation in order to evaluate the modified correlation
            matrix in the standard normal space given the correlation matrix in the original space. This is achieved
            by a quadratic two-dimensional Gauss-Legendre integration.
            This function is a part of the ERADIST code that can be found in:
            https://www.era.bgu.tum.de/en/software/

        Input:
            :param marginal: marginal distributions
            :type marginal: list

            :param params: marginal distribution parameters.
            :type params: list

            :param rho_norm: Correlation at standard normal space.
            :type rho_norm: ndarray

        Output:
            :return rho: Distorted correlation
            :rtype rho: ndarray

    """

    n = 1024
    z_max = 8
    z_min = -z_max
    points, weights = np.polynomial.legendre.leggauss(n)
    points = - (0.5 * (points + 1) * (z_max - z_min) + z_min)
    weights = weights * (0.5 * (z_max - z_min))

    xi = np.tile(points, [n, 1])
    xi = xi.flatten(order='F')
    eta = np.tile(points, n)

    first = np.tile(weights, n)
    first = np.reshape(first, [n, n])
    second = np.transpose(first)

    weights2d = first * second
    w2d = weights2d.flatten()
    rho = np.ones_like(rho_norm)

    print('UQpy: Computing Nataf correlation distortion...')
    for i in range(len(marginal)):
        i_cdf_i = marginal[i].icdf
        moments_i = marginal[i].moments
        mi = moments_i(params[i])
        if not (np.isfinite(mi[0]) and np.isfinite(mi[1])):
            raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")

        for j in range(i + 1, len(marginal)):
            i_cdf_j = marginal[j].icdf
            moments_j = marginal[j].moments
            mj = moments_j(params[j])
            if not (np.isfinite(mj[0]) and np.isfinite(mj[1])):
                raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")

            tmp_f_xi = ((i_cdf_j(stats.norm.cdf(xi), params[j]) - mj[0]) / np.sqrt(mj[1]))
            tmp_f_eta = ((i_cdf_i(stats.norm.cdf(eta), params[i]) - mi[0]) / np.sqrt(mi[1]))
            coef = tmp_f_xi * tmp_f_eta * w2d

            rho[i, j] = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho_norm[i, j]))
            rho[j, i] = rho[i, j]

    print('UQpy: Done.')
    return rho


def itam(marginal, params, corr, beta, thresh1, thresh2):

    """
        Description:

            A function to perform the  Iterative Translation Approximation Method;  an iterative scheme for
            upgrading the Gaussian power spectral density function.
            [1] Shields M, Deodatis G, Bocchini P. A simple and efficient methodology to approximate a general
            non-Gaussian  stochastic process by a translation process. Probab Eng Mech 2011;26:511–9.


        Input:
            :param marginal: marginal distributions
            :type marginal: list

            :param params: marginal distribution parameters.
            :type params: list

            :param corr: Non-Gaussian Correlation matrix.
            :type corr: ndarray

            :param beta:  A variable selected to optimize convergence speed and desired accuracy.
            :type beta: int

            :param thresh1: Threshold
            :type thresh1: float

            :param thresh2: Threshold
            :type thresh2: float

        Output:
            :return corr_norm: Gaussian correlation matrix
            :rtype corr_norm: ndarray

    """

    if beta is None:
        beta = 1
    if thresh1 is None:
        thresh1 = 0.0001
    if thresh2 is None:
        thresh2 = 0.01

    # Initial Guess
    corr_norm0 = corr
    corr_norm = np.zeros_like(corr_norm0)
    # Iteration Condition
    error0 = 0.1
    error1 = 100.
    max_iter = 50
    iter_ = 0

    print("UQpy: Initializing Iterative Translation Approximation Method (ITAM)")
    while iter_ < max_iter and error1 > thresh1 and abs(error1-error0)/error0 > thresh2:
        error0 = error1
        corr0 = correlation_distortion(marginal, params, corr_norm0)
        error1 = np.linalg.norm(corr - corr0)

        max_ratio = np.amax(np.ones((len(corr), len(corr))) / abs(corr_norm0))

        corr_norm = np.nan_to_num((corr / corr0)**beta * corr_norm0)

        # Do not allow off-diagonal correlations to equal or exceed one
        corr_norm[corr_norm < -1.0] = (max_ratio + 1) / 2 * corr_norm0[corr_norm < -1.0]
        corr_norm[corr_norm > 1.0] = (max_ratio + 1) / 2 * corr_norm0[corr_norm > 1.0]

        # Iteratively finding the nearest PSD(Qi & Sun, 2006)
        corr_norm = np.array(nearest_psd(corr_norm))

        corr_norm0 = corr_norm.copy()

        iter_ = iter_ + 1

        print(["UQpy: ITAM iteration number ", iter_])
        print(["UQpy: Current error, ", error1])

    print("UQpy: ITAM Done.")
    return corr_norm


def bi_variate_normal_pdf(x1, x2, rho):

    """

        Description:

            A function which evaluates the values of the bi-variate normal probability distribution function.

        Input:
            :param x1: value 1
            :type x1: ndarray

            :param x2: value 2
            :type x2: ndarray

            :param rho: correlation between x1, x2
            :type rho: float

        Output:

    """
    return (1 / (2 * np.pi * np.sqrt(1-rho**2)) *
            np.exp(-1/(2*(1-rho**2)) *
                   (x1**2 - 2 * rho * x1 * x2 + x2**2)))


def _get_a_plus(a):

    """
        Description:

            A supporting function for the nearest_pd function

        Input:
            :param a:A general nd array

        Output:
            :return a_plus: A modified nd array
            :rtype:np.ndarray
    """

    eig_val, eig_vec = np.linalg.eig(a)
    q = np.matrix(eig_vec)
    x_diagonal = np.matrix(np.diag(np.maximum(eig_val, 0)))

    return q * x_diagonal * q.T


def _get_ps(a, w=None):

    """
        Description:

            A supporting function for the nearest_pd function

    """

    w05 = np.matrix(w ** .5)

    return w05.I * _get_a_plus(w05 * a * w05) * w05.I


def _get_pu(a, w=None):

    """
        Description:

            A supporting function for the nearest_pd function

    """

    a_ret = np.array(a.copy())
    a_ret[w > 0] = np.array(w)[w > 0]
    return np.matrix(a_ret)


def nearest_psd(a, nit=10):

    """
        Description:
            A function to compute the nearest positive semi definite matrix of a given matrix

         Input:
            :param a: Input matrix
            :param nit: Number of iterations to perform (Default=10)

        Output:
            :return:
    """

    n = a.shape[0]
    w = np.identity(n)
    # w is the matrix used for the norm (assumed to be Identity matrix here)
    # the algorithm should work for any diagonal W
    delta_s = 0
    y_k = a.copy()
    for k in range(nit):

        r_k = y_k - delta_s
        x_k = _get_ps(r_k, w=w)
        delta_s = x_k - r_k
        y_k = _get_pu(x_k, w=w)

    return y_k


def nearest_pd(a):

    """
        Description:

            Find the nearest positive-definite matrix to input
            A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
            credits [2].
            [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
            [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
            matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

        Input:
            :param a:
            :type a:


        Output:

    """

    b = (a + a.T) / 2
    _, s, v = np.linalg.svd(b)

    h = np.dot(v.T, np.dot(np.diag(s), v))

    a2 = (b + h) / 2

    a3 = (a2 + a2.T) / 2

    if is_pd(a3):
        return a3

    spacing = np.spacing(np.linalg.norm(a))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrices with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrices of small dimension, be on
    # other order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    k = 1
    while not is_pd(a3):
        min_eig = np.min(np.real(np.linalg.eigvals(a3)))
        a3 += np.eye(a.shape[0]) * (-min_eig * k**2 + spacing)
        k += 1

    return a3


def is_pd(b):

    """
        Description:

            Returns true when input is positive-definite, via Cholesky decomposition.

        Input:
            :param b: A general matrix

        Output:


    """
    try:
        _ = np.linalg.cholesky(b)
        return True
    except np.linalg.LinAlgError:
        return False


def estimate_psd(samples, nt, t):

    """
        Description: A function to estimate the Power Spectrum of a stochastic process given an ensemble of samples_SRM

        Input:
            :param samples: Samples of the stochastic process
            :param nt: Number of time discretisations in the time domain
            :param t: Total simulation time

        Output:
            :return: Power Spectrum
            :rtype: numpy.ndarray

    """

    sample_size = nt
    sample_max_time = t
    dt = t / (nt - 1)
    x_w = np.fft.fft(samples, sample_size, axis=1)
    x_w = x_w[:, 0: int(sample_size / 2)]
    m_ps = np.mean(np.absolute(x_w) ** 2 * sample_max_time / sample_size ** 2, axis=0)
    num = int(t / (2 * dt))

    return np.linspace(0, (1 / (2 * dt) - 1 / t), num), m_ps


def s_to_r(s, w, t):

    """
        Description:

            A function to transform the power spectrum to an autocorrelation function

        Input:
            :param s: Power Spectrum of the signal
            :param w: Array of frequency discretisations
            :param t: Array of time discretisations

        Output:
            :return r: Autocorrelation function
            :rtype: numpy.ndarray
    """

    dw = w[1] - w[0]
    fac = np.ones(len(w))
    fac[1: len(w) - 1: 2] = 4
    fac[2: len(w) - 2: 2] = 2
    fac = fac * dw / 3
    r = np.zeros(len(t))
    for j in range(len(t)):
        r[j] = 2 * np.dot(fac, s * np.cos(w * t[j]))
    return r


def r_to_s(r, w, t):

    """
        Description: A function to transform the autocorrelation function to a power spectrum


        Input:
            :param r: Autocorrelation function of the signal
            :param w: Array of frequency discretisations
            :param t: Array of time discretisations

        Output:
            :return s: Power Spectrum
            :rtype: numpy.ndarray

    """
    dt = t[1] - t[0]
    fac = np.ones(len(t))
    fac[1: len(t) - 1: 2] = 4
    fac[2: len(t) - 2: 2] = 2
    fac = fac * dt / 3

    s = np.zeros(len(w))
    for j in range(len(w)):
        s[j] = 2 / (2 * np.pi) * np.dot(fac, (r * np.cos(t * w[j])))
    s[s < 0] = 0
    return s

