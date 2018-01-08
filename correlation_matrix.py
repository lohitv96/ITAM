import numpy as np
from numpy.linalg import eig, norm
import copy

# This code is designed to solve
# min 0.5*<X-G, X-G>
# s.t. X_ii =b_i, i=1,2,...,n
# X>=tau*I (symmetric and positive semi-definite)
#
# based on the algorithm  in 'A Quadratically Convergent Newton Method for Computing the Nearest Correlation Matrix'
# By Houduo Qi and Defeng Sun
# SIAM J. Matrix Anal. Appl. 28 (2006) 360--385.
# Last modified date: January 22, 2014
# The  input arguments  G, b>0, tau>=0, and tol (tolerance error)
# For correlation matrix, set b =ones(n,1)
# For a positive definite matrix
# set tau = 1.0e-5 for example
# set tol = 1.0e-6 or lower if no very high accuracy required
# The outputs are the optimal primal and dual solutions Diagonal Preconditioner is added
# Send your comments and suggestions to hdqi@soton.ac.uk  or matsundf@nus.edu.sg
# Warning:  Though the code works extremely well, it is your call to use it or not.


def Mymexeig(X):
    D, P = eig(X)
    D = np.real(D)
    P = np.real(P)
    D = D[np.argsort(D)[::-1]]
    P = P[:, np.argsort(D)[::-1]]
    return P, D


def gradient(y, lam, P, b0, n):
    f = 0.0
    Fy = np.zeros(n)
    P0 = copy.deepcopy(P)
    P0 = P0.T
    for i in range(n):
        P0[i, :] = np.sqrt(max(lam[i], 0))*P0[i,:]
    for i in range(n):
        Fy[i] = np.dot(P0[:,i], P0[:,i])
    for i in range(n):
        f = f + (max(lam[i], 0))**2
    f = 0.5*f - np.dot(b0,y)
    return f, Fy


def omega_mat(lam):
    idp = np.where(lam > 0)[0]
    n = len(lam)
    r = len(idp)
    if idp is not None:
        if r == n:
            Omega12 = np.ones([n, n])
        else:
            s = n - r
            dp = lam[:r]
            dn = lam[r:]
            Omega12 = (np.dot(np.matrix(dp).T,  np.matrix(np.ones(s))))/ (np.dot(np.matrix(abs(dp)).T,  np.matrix(np.ones(s))) + np.dot(np.matrix(np.ones(r)).T, np.matrix(abs(dn))))
    else:
        Omega12 = []
    return np.array(Omega12)


def PCA(X, lam, P, n):
    Ip = np.where(lam>0)[0]
    r = len(Ip)

    if r ==0:
        X = np.zeros([n,n])
    elif r ==n:
        X = X
    elif r<=n/2:
        lam1 = lam[Ip]
        lam1 = np.sqrt(lam1)
        P1 = P[:, :r]
        if r>1:
            P1 = np.dot(P1, np.diag(lam1))
            X = np.dot(P1, P1.T)
        else:
            X = lam1**2*np.dot(P1, P1.T)
    else:
        lam2 = -lam[r+1:]
        lam2 = np.sqrt(lam2)
        P2 = P[:, r+1:]
        P2 = np.dot(P2, np.diag(lam2))
        X = X + np.dot(P2, P2.T)
    return X


def precond_matrix(Omega12, P, n):
    [r, s] = Omega12.shape
    c = np.ones(n)
    if (r > 0):
        if (r < n / 2):
            H = P.T
            H = H*H
            H12 = np.dot(H[:r,:].T, Omega12)
            d = np.ones(r)
            for i in range(n):
                c[i] = sum(H[:r, i])*np.dot(d.T,H[:r,i])
                c[i] = c[i] + 2.0 * np.dot(H12[i,:],H[r:, i])
                c[i] = max(c[i], 1.0e-8)
        elif r < n:
            H = P.T
            H = H*H
            Omega12 = np.ones([r, s])-Omega12
            H12 = np.dot(Omega12, H[r:,:])
            d = np.ones(s)
            dd = np.ones(n)
            for i in range(n):
                c[i] = sum(H[r:, i])*np.dot(d.T,H[r:,i])
                c[i] = c[i] + 2.0 * np.dot(H[:r, i].T,H12[:,i])
                alpha = sum(H[:, i])
                c[i] = alpha * np.dot(H[:, i].T,dd)-c[i]
                c[i] = max(c[i], 1.0e-8)
    return c


def pre_cg(b,tol,maxit,c,Omega12,P,n):
    r = b
    n2b = norm(b)
    tolb = tol * n2b
    p = np.zeros(n)
    flag=1
    iterk =0
    relres=1000
    z =r/c
    rz1 = np.dot(r.T,z)
    rz2 = 1
    d = z
    for k in range(maxit):
        if k > 0:
            beta = rz1/rz2
            d = z + np.dot(beta,d)
        w = Jacobian_matrix(d,Omega12,P,n)
        denom = np.dot(d, w)
        iterk =k
        relres = norm(r)/n2b
        if denom <= 0:
            p = d/norm(d)
            break
        else:
            alpha = rz1/denom
            p = p + alpha*d
            r = r - alpha*w
        z = r/c
        if norm(r) <= tolb:
            iterk =k
            relres = norm(r)/n2b
            flag =0
            break
        rz2 = rz1
        rz1 = np.dot(r, z)
    return p, flag, relres, iterk


def Jacobian_matrix(x, Omega12, P, n):
    Ax = np.zeros(n)
    [r, s] = Omega12.shape
    if r > 0:
        H1 = P[:,:r]
        if r < n / 2:
            for i in range(n):
                H1[i,:] = x[i] * H1[i,:]
                Omega12 = Omega12 * np.dot(H1.T,P[:,r:])
                # H =[np.dot(H1.T,P[:,:r])*(P(:,1:r))'+ Omega12 * (P(:, r+1:n))';Omega12' * (P(:, 1:r))']
            for i in range(n):
                Ax[i] = np.dot(P[i,:],H[:, i])
                Ax[i] = Ax[i] + 1.0e-10 * x[i]
        elif r == n:
                Ax = (1 + 1.0e-10) * x
        else:
            H2 = P[:, r:]
            for i in range(n):
                H2[i,:] = x[i] * H2[i,:]
            Omega12 = np.ones([r, s]) - Omega12
            Omega12 = Omega12* np.dot(P[:,:r].T,H2)
            # H = [Omega12 * (P(:, r+1:n))';Omega12' *P[:,:r].T +( (P(:,r+1:n))' * H2)*(P(:, r+1:n))']

        for i in range(n):
            Ax[i] = -P[i,:]*H[:, i]
            Ax[i] = x[i] + Ax[i] + 1.0e-10 * x[i]
    return


def correlation_matrix(G, b=None, tau=None, tol=None):
    print(' --- Semismooth Newton-CG method starts --- ')
    [n, m] = G.shape
    G = (G + G.T) / 2
    b0 = np.ones(n)
    error_tol = 1.0e-6

    if b is None and tau is None and tol is None:
        tau = 0

    if tau is None and tol is None and b is not None:
        b0 = b
        tau = 0

    if tol is None and tau is not None and b is not None:
        b0 = b - tau * np.ones(n)
        G = G - tau * np.eye(n)

    if G is not None and b is not None and tau is not None and tol is not None:
        b0 = b - tau * np.ones(n)
        G = G - tau * np.eye(n)
        error_tol = max(1.0e-12, tol)

    Res_b = np.zeros(300)
    y = b0 - np.diag(G)
    Fy = np.zeros(n)
    k = 0
    f_eval = 0
    Iter_Whole = 200
    Iter_inner = 20
    maxit = 200
    iterk = 0
    Inner = 0
    tol = 1.0e-2
    sigma_1 = 1.0e-4
    x0 = y
    prec_time = 0
    pcg_time = 0
    eig_time = 0
    c = np.ones(n)
    d = np.zeros(n)
    val_G = sum((G * G).flatten()) / 2
    X = G + np.diag(y)
    X = (X + X.T) / 2

    [P, lam] = Mymexeig(X)
    [f0, Fy] = gradient(y, lam, P, b0, n)
    Initial_f = val_G - f0

    X = PCA(X, lam, P, n)
    val_obj = np.sum(((X - G)*(X - G)).flatten())
    gap = (val_obj - Initial_f) / (1 + abs(Initial_f) + abs(val_obj))

    f = f0
    f_eval = f_eval + 1
    b = b0 - Fy
    norm_b = norm(b)

    print('Newton-CG:  Initial Dual objective function value ======== ', Initial_f)
    print('Newton-CG:  Initial Primal objective function value ====== ', val_obj)
    print('Newton-CG:  Norm of Gradient ================= ', norm_b)
    Omega12 = omega_mat(lam)
    x0 = y

    while gap > error_tol and norm_b > error_tol and k < Iter_Whole:
        c = precond_matrix(Omega12, P, n)
        [d, flag, relres, iterk] = pre_cg(b, tol, maxit, c, Omega12, P, n)
        print('Newton-CG: Number of CG Iterations == ', iterk)
        if (flag!=0):
            print('..... Not a complet Newton-CG step......')
        slope = Fy - np.dot(b0,d)
        y = x0 + d
        x0 + d
        X = G + np.diag(y)
        X = (X + X.T)/2
        [P, lam] = Mymexeig(X)
        [f, Fy] = gradient(y, lam, P, b0, n)
        k_inner = 0
        while k_inner <= Iter_inner and f > f0 + sigma_1 * 0.5 ^ k_inner * slope + 1.0e-6:
            k_inner = k_inner + 1
            y = x0 + (0.5**k_inner) * d
            X = G + np.diag(y)
            X = (X + X.T)/2
            [P, lam] = Mymexeig(X)
            [f, Fy] = gradient(y, lam, P, b0, n)
        f_eval = f_eval + k_inner + 1
        x0 = y
        f0 = f
        val_dual = val_G - f0
        X = PCA(X, lam, P, n)
        Dual_f = val_G - f0
        val_obj = sum(((X - G)*(X - G)).flatten()) / 2
        gap = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
        print('Newton-CG:  The relative duality gap ==================== ', gap)
        print('Newton-CG:  The Dual objective function value =========== ', Dual_f)
        print('Newton-CG:  The primal objective function value ========= ', val_obj)

        k = k + 1
        b = b0 - Fy
        norm_b = norm(b)
        print('Newton-CG: Norm of Gradient == ', norm_b)
        Res_b[k] = norm_b
        Omega12 = omega_mat(lam)
    rank_X = len(np.where(max(0, lam) > 0)[0])
    Final_f = val_G - f
    X = X + tau * np.eye(n)
    print('Newton: Norm of Gradient %d \n', norm_b)
    print('Newton-CG: Number of Iterations == %d \n', k)
    print('Newton-CG: Number of Function Evaluations == %d \n', f_eval)
    print('Newton-CG: Final Dual Objective Function value ========== %d \n', Final_f)
    print('Newton-CG: Final primal Objective Function value ======== %d \n', val_obj)
    print('Newton-CG: The final relative duality gap ===============  %d \n', gap)
    print('Newton-CG: The rank of the Optimal Solution - tau*I ================= %d \n', rank_X)
    return X, y
