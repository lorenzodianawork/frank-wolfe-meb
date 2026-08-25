"""
Frank-Wolfe solvers for the dual (simplex-constrained) formulation of the
Minimum Enclosing Ball problem:

    min_{x in simplex} f(x) = x^T Q x - c^T x

Three variants are implemented, in increasing order of per-iteration cost
and decreasing order of iterations-to-convergence:

- fw_standard          : vanilla Frank-Wolfe (sublinear O(1/t) rate)
- fw_pairwise          : Pairwise Frank-Wolfe (away-step variant, linear
                          rate for strongly convex objectives)
- fw_fully_corrective  : Fully-Corrective Frank-Wolfe (re-optimizes exactly
                          over the active set at every outer iteration)

All three share the same smart initialization and gradient-caching trick
(O(n) per iteration instead of a full O(n^2) matrix-vector product), and
all converge to the same unique MEB by convexity of the dual problem.
"""

import time
import numpy as np

MAXIT = 10_000_000   # hard iteration cap
MAXTIME = 15.0        # time budget per solver (s)
EPS = 1e-10           # duality-gap tolerance
SEED = 42


def smart_initialization(Q, c, seed=SEED):
    """Geometric dual start (Yildirim, 2008): initialize on the midpoint of
    an approximate diameter of the point set, found via two farthest-point
    sweeps starting from a random anchor. Avoids the long transient phase
    caused by a naive random start.
    """
    n = Q.shape[1]
    rng = np.random.default_rng(seed)

    r = rng.integers(n)  # random anchor

    # ||a_i - a_r||^2 = c_i + c_r - 2 Q[i,r]
    a_alpha = np.argmax(c + c[r] - 2 * Q[:, r])              # farthest from anchor
    a_beta = np.argmax(c + c[a_alpha] - 2 * Q[:, a_alpha])   # farthest from a_alpha

    x = np.zeros(n)
    x[a_alpha] += 0.5
    x[a_beta] += 0.5

    Qx = 0.5 * (Q[:, a_alpha] + Q[:, a_beta])  # cached Q @ x, O(n) not O(n^2)
    return x, Qx


def fw_standard(Q, c, maxit=MAXIT, maxtime=MAXTIME, eps=EPS, seed=SEED):
    """Vanilla Frank-Wolfe with exact line search."""
    _, n = Q.shape
    timeVec, gh = np.zeros(maxit), np.zeros(maxit)
    x, Qx = smart_initialization(Q, c, seed)

    it = 1
    tstart = time.time()

    while it <= maxit:
        xQx = x.T @ Qx
        timeVec[it - 1] = time.time() - tstart

        g = 2.0 * Qx - c
        gh[it - 1] = g.T @ x - np.min(g)  # duality gap via LMO

        if timeVec[it - 1] > maxtime or gh[it - 1] <= eps:
            break

        istar = np.argmin(g)  # LMO vertex on the simplex
        dQd = 2.0 * (Q[istar, istar] - 2.0 * Qx[istar] + xQx)
        alpha = 1.0 if dQd <= 1e-10 else max(0.0, min(1.0, gh[it - 1] / dQd))

        x = (1.0 - alpha) * x
        x[istar] += alpha
        Qx = (1.0 - alpha) * Qx + alpha * Q[:, istar]

        it += 1

    f_final = x.T @ Q @ x - c.T @ x
    return x, it - 1, f_final, time.time() - tstart, timeVec[:it], gh[:it]


def fw_pairwise(Q, c, maxit=MAXIT, maxtime=MAXTIME, eps=EPS, seed=SEED):
    """Pairwise Frank-Wolfe: at each step, moves mass from the worst active
    (away) atom directly to the new forward atom, avoiding the zig-zagging
    of the vanilla method near the optimal face.
    """
    _, n = Q.shape
    timeVec, gh = np.zeros(maxit), np.zeros(maxit)
    x, Qx = smart_initialization(Q, c, seed)

    it = 1
    tstart = time.time()

    while it <= maxit:
        timeVec[it - 1] = time.time() - tstart

        g = 2.0 * Qx - c
        istar = np.argmin(g)              # forward (LMO) atom
        active = np.where(x > 1e-14)[0]
        j = active[np.argmax(g[active])]  # away atom

        gh[it - 1] = g[j] - g[istar]      # pairwise duality gap

        if timeVec[it - 1] > maxtime or gh[it - 1] <= eps:
            break

        dQd = 2.0 * (Q[istar, istar] + Q[j, j] - 2.0 * Q[istar, j])

        if dQd < 1e-14:
            alpha = x[j]  # drop step
        else:
            alpha = min(x[j], gh[it - 1] / dQd)

        x[istar] += alpha
        x[j] -= alpha
        Qx += alpha * (Q[:, istar] - Q[:, j])

        it += 1

    return x, it - 1, x.T @ Q @ x - c.T @ x, time.time() - tstart, timeVec[:it], gh[:it]


def fw_fully_corrective(Q, c, maxit=MAXIT, maxtime=MAXTIME, eps=EPS, seed=SEED):
    """Fully-Corrective Frank-Wolfe: after adding the new LMO atom, fully
    re-optimizes the weights over the active set via an inner pairwise
    loop restricted to the (small) active sub-matrix.
    """
    _, n = Q.shape
    timeVec, gh = np.zeros(maxit), np.zeros(maxit)
    x, Qx = smart_initialization(Q, c, seed)

    it = 0
    tstart = time.time()

    while it < maxit:
        elapsed = time.time() - tstart
        if elapsed > maxtime:
            break

        g = 2.0 * Qx - c
        istar = np.argmin(g)
        gap = np.dot(x, g) - g[istar]

        timeVec[it] = elapsed
        gh[it] = gap

        if gap <= eps:
            it += 1
            break

        active_idx = np.where(x > 1e-14)[0]
        if istar not in active_idx:
            active_idx = np.append(active_idx, istar)

        if len(active_idx) > 1:
            x_sub = x[active_idx]
            c_sub = c[active_idx]
            Q_sub = Q[np.ix_(active_idx, active_idx)]
            Qx_sub = Q_sub @ x_sub

            inner_tol = max(eps * 0.1, gap * 1e-3)
            for _ in range(1000):  # inner pairwise correction loop
                g_sub = 2.0 * Qx_sub - c_sub
                i_in = np.argmin(g_sub)

                valid_out = x_sub > 1e-14
                g_sub_masked = np.copy(g_sub)
                g_sub_masked[~valid_out] = -np.inf
                i_out = np.argmax(g_sub_masked)

                inner_gap = g_sub[i_out] - g_sub[i_in]
                if inner_gap <= inner_tol:
                    break

                dQd = 2.0 * (Q_sub[i_in, i_in] + Q_sub[i_out, i_out] - 2.0 * Q_sub[i_in, i_out])
                alpha_max = x_sub[i_out]
                alpha = alpha_max if dQd < 1e-14 else min(alpha_max, inner_gap / dQd)

                x_sub[i_in] += alpha
                x_sub[i_out] -= alpha
                Qx_sub += alpha * (Q_sub[:, i_in] - Q_sub[:, i_out])

            x.fill(0.0)
            x[active_idx] = x_sub
            x /= np.sum(x)

        Qx = Q[:, active_idx] @ x_sub
        it += 1

    f_final = x.T @ Qx - c.T @ x
    return x, it, f_final, time.time() - tstart, timeVec[:it], gh[:it]


SOLVERS = {
    "Standard FW": fw_standard,
    "Pairwise FW": fw_pairwise,
    "Fully-Corrective FW": fw_fully_corrective,
}
