import numpy as np


# This code is designed to solve
# min 0.5*<X-G, X-G>
# s.t. X_ii =b_i, i=1,2,...,n
# X>=tau*I (symmetric and positive semi-definite)          %%%%%%%%%%%%%%%
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


def correlation_matrix(G, b=None, tau=None, tol=None):
    print(' --- Semismooth Newton-CG method starts --- ')
    [n, m] = G.shape
    G = (G + G.T)/2
    b0 = np.ones(n)
    error_tol = 1.0e-6
    if b is None and tau is None and tol is None:
        tau = 0
    if tau is None and tol is None and b is not None:
        b0 = b
        tau = 0
    if tol is None and tau is not None and b is not None:
        b0 =  b - tau * np.ones(n)
        G  =  G - tau * np.eye(n)
    if G is not None and b is not None and tau is not None and tol is not None:
        b0 =  b - tau * np.ones(n)
        G  =  G - tau * np.eye(n)
        error_tol = max(1.0e-12, tol)

    Res_b = np.zeros(300)
    y = b0 - np.diag(G)
    Fy = np.zeros(n)
    k=0
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
    eig_time =0
    c = np.ones(n)
    d = np.zeros(n)
    val_G = sum((G*G).flatten()) / 2
    X = G + np.diag(y)
    X = (X + X.T)/2
    [P, lam] = Mymexeig(X)
    [f0, Fy] = gradient(y, lam, P, b0, n)
    Initial_f = val_G - f0
    
    X = PCA(X, lambda, P, b0, n)
    val_obj = sum(sum((X - G).* (X - G))) / 2
    gap = (val_obj - Initial_f) / (1+ abs(Initial_f) + abs(val_obj))

    f = f0
    f_eval = f_eval + 1
    b = b0 - Fy
    norm_b = norm(b)

    print('Newton-CG:  Initial Dual objective function value ======== %d \n', Initial_f)
    print('Newton-CG:  Initial Primal objective function value ====== %d \n', val_obj)
    print('Newton-CG:  Norm of Gradient ================= %d \n', norm_b)
    print('Newton-CG: computing time used so far ==== =====================%d \n', time_used)
    Omega12 = omega_mat(P, lambda, n)
    x0 = y

    while (gap > error_tol & norm_b > error_tol & k < Iter_Whole)

        prec_time0 = clock
        c = precond_matrix(Omega12, P, n) % comment
        this
        line
        for no preconditioning
    prec_time = prec_time + etime(clock, prec_time0)

    pcg_time0 = clock
    [d, flag, relres, iterk] = pre_cg(b, tol, maxit, c, Omega12, P, n)
    pcg_time = pcg_time + etime(clock, pcg_time0)
    % d = b0 - Fy
    gradient
    direction
    print('Newton-CG: Number of CG Iterations == %d \n', iterk)

    if (flag~=0) % if CG is unsuccessful, use the negative gradient direction
    % d = b0 - Fy
    disp('..... Not a completed Newton-CG step......')
    end
    slope = (Fy - b0)
    '*d %%% nabla f d

    y = x0 + d % temporary
    x0 + d

    X = G + diag(y)
    X = (X + X
    ')/2
    eig_time0 = clock
    [P, lambda ] = Mymexeig(X) % Eig-decomposition: X = P * diag(D) * P ^ T
    eig_time = eig_time + etime(clock, eig_time0)
    [f, Fy] = gradient(y, lambda , P, b0, n)

    k_inner = 0
    while (k_inner <= Iter_inner & f > f0 + sigma_1 * 0.5 ^ k_inner * slope + 1.0e-6)
        k_inner = k_inner + 1
        y = x0 + 0.5 ^ k_inner * d % backtracking

        X = G + diag(y)
        X = (X + X
        ')/2

        eig_time0 = clock
        [P, lambda ] = Mymexeig(X) % Eig-decomposition: X = P * diag(D) * P ^ T
        eig_time = eig_time + etime(clock, eig_time0)
        [f, Fy] = gradient(y, lambda , P, b0, n)
        end % loop
        for
            while
    f_eval = f_eval + k_inner + 1
    x0 = y
    f0 = f
    val_dual = val_G - f0
    X = PCA(X, lambda , P, b0, n)
    Dual_f = val_G - f0
    val_obj = sum(sum((X - G). * (X - G))) / 2
    gap = (val_obj - val_dual) / (1 + abs(val_dual) + abs(val_obj))
    print('Newton-CG:  The relative duality gap ====================  %d \n', gap)
    print('Newton-CG:  The Dual objective function value ===========  %d \n', Dual_f)
    print('Newton-CG:  The primal objective function value =========  %d \n', val_obj)

    k = k + 1
    b = b0 - Fy
    norm_b = norm(b)
    time_used = etime(clock, t0)
    print('Newton-CG: Norm of Gradient == %d \n', norm_b)
    print('Newton-CG: computing time used so far ==== =====================%d \n', time_used)
    Res_b(k) = norm_b

    Omega12 = omega_mat(P, lambda , n)

    end % end
    loop
    for while i=1

    rank_X = length(find(max(0, lambda ) > 0))
                                       Final_f = val_G - f

                                       X = X + tau * eye(n)
                                       time_used = etime(clock, t0)
    print('\n')

    print('Newton: Norm of Gradient %d \n', norm_b)
    print('Newton-CG: Number of Iterations == %d \n', k)
    print('Newton-CG: Number of Function Evaluations == %d \n', f_eval)
    print('Newton-CG: Final Dual Objective Function value ========== %d \n', Final_f)
    print('Newton-CG: Final primal Objective Function value ======== %d \n', val_obj)
    print('Newton-CG: The final relative duality gap ===============  %d \n', gap)
    print('Newton-CG: The rank of the Optimal Solution - tau*I ================= %d \n', rank_X)

    print('Newton-CG: computing time for computing preconditioners == %d \n', prec_time)
    print('Newton-CG: computing time for linear systems solving (cgs time) ====%d \n', pcg_time)
    print('Newton-CG: computing time for  eigenvalue decompositions (calling mexeig time)==%d \n', eig_time)
    print('Newton-CG: computing time used for equal weights calibration ==== =====================%d \n', time_used)

    % end of the main program
    % To generate F(y)
