# Code adapted from https://github.com/PythonOptimizers/pykrylov/blob/master/pykrylov/lls/lsqr.py


import numpy as np


# Simple shortcuts---linalg.norm is too slow for small vectors
def normof2(x, y): return np.sqrt(x*x + y*y)
def normof3(x1, x2, x3): return np.sqrt(x1*x1 + x2*x2 + x3*x3)


def lsqr(evaluate, evaluate_T, forward_op, noise_std, b, itnlim=0, N=None, atol=1.0e-9, btol=1.0e-9):
    # Solves forward_op @ evaluate @ evaluate_T @ forward_op.T @ x = b using LSQR.

    n, _ = forward_op.shape

    if itnlim == 0: itnlim = 3 * n

    itn = istop = 0
    Anorm = Acond = 0.
    z = xnorm = xxnorm = ddnorm = res2 = 0.
    cs2 = -1. ; sn2 = 0.

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*M*u = b,  alpha*N*v = A'u.

    x = np.zeros(n)

    u = b.copy()

    alpha = 0.
    beta = np.sqrt(np.dot(u, u))       # norm(u)
    if beta > 0:
        u /= beta

        Nv = forward_op @ evaluate(u[:-n]) + noise_std * u[-n:]
        if N is not None:
            v = N(Nv)
        else:
            v = Nv
        alpha = np.sqrt(np.dot(v,Nv))   # norm(v)

    if alpha > 0:
        v /= alpha
        if N is not None: Nv /= alpha
        w = v.copy()

    x_is_zero = False   # Is x=0 the solution to the least-squares prob?
    Arnorm = alpha * beta
    if abs(Arnorm) < 1e-15:    # Arnorm == 0 up to machine precision
        x_is_zero = True
        istop = 0

    rhobar = alpha ; phibar = beta ; bnorm = beta
    rnorm  = beta

    # ------------------------------------------------------------------
    #     Main iteration loop.
    # ------------------------------------------------------------------
    while itn < itnlim and not x_is_zero:

        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #             beta*M*u  =  A*v   -  alpha*M*u,
        #            alpha*N*v  =  A'*u  -   beta*N*v.

        u = np.concatenate((evaluate_T(forward_op.T @ v), noise_std * v))  - alpha * u
        beta = np.sqrt(np.dot(u, u))   # norm(u)
        if beta > 0:
            u /= beta

            Anorm = normof3(Anorm, alpha, beta)

            Nv = forward_op @ evaluate(u[:-n]) + noise_std * u[-n:] - beta * Nv
            if N is not None:
                v = N(Nv)
            else:
                v = Nv
            alpha = np.sqrt(np.dot(v,Nv))  # norm(v)
            if alpha > 0:
                v /= alpha
                if N is not None: Nv /= alpha

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.

        rho     =   normof2(rhobar, beta)
        cs      =   rhobar / rho
        sn      =   beta    / rho
        theta   =   sn * alpha
        rhobar  = - cs * alpha
        phi     =   cs * phibar
        phibar  =   sn * phibar
        tau     =   sn * phi

        # Update x and w.

        t1      =   phi   / rho
        t2      = - theta / rho
        dk      =   (1.0/rho)*w

        x      += t1*w
        w      *= t2 ; w += v
        ddnorm  = ddnorm + np.linalg.norm(dk)**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).

        delta   =   sn2 * rho
        gambar  = - cs2 * rho
        rhs     =   phi  -  delta * z
        zbar    =   rhs / gambar
        xnorm   =   np.sqrt(xxnorm + zbar**2)
        gamma   =   normof2(gambar, theta)
        cs2     =   gambar / gamma
        sn2     =   theta  / gamma
        z       =   rhs    / gamma
        xxnorm +=   z * z

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.

        Acond   =   Anorm * np.sqrt(ddnorm)
        res1    =   phibar**2
        rnorm   =   np.sqrt(res1 + res2)
        Arnorm  =   alpha * abs(tau)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = rnorm / bnorm
        if Anorm == 0. or rnorm == 0.:
            test2 = np.inf
        else:
            test2 = Arnorm/(Anorm * rnorm)
        t1    = test1 / (1    +  Anorm * xnorm / bnorm)
        rtol  = btol  +  atol *  Anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= itnlim:  istop = 7
        if 1 + test2 <= 1: istop = 5
        if 1 + t1    <= 1: istop = 4

        # Allow for tolerances set by the user.

        if test2 <= atol: istop = 2
        if test1 <= rtol: istop = 1

        if istop > 0: break

        # End of iteration loop.
        # Print the stopping condition.

    if istop == 0: status = 'solution is zero'
    if istop in [1,2,4,5]: status = 'residual small'
    if istop in [3,6]: status = 'ill-conditioned operator'
    if istop == 7: status = 'max iterations'
    if istop == 8: status = 'direct error small'

    info = {"status": status, "optimal": istop in [1,2,4,5,8], "istop": istop, "itn": itn, "nMatvec": 2*itn,
            "Anorm": Anorm, "Acond": Acond, "Arnorm": Arnorm, "xnorm": xnorm}

    return x, info
