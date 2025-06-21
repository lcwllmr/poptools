import numpy as np
import scipy as sp
from poptools.linalg import posdef_linesearch
from poptools.random import maxcut

if __name__ == "__main__":
    np.random.seed(0)
    sdp = maxcut(10, 0.1)

    n = sdp.n
    m = sdp.m
    vsd = sdp.vsd

    # heuristic initialization
    pscale = n * np.max(
        np.divide(1 + np.abs(sdp.b), 1 + np.sqrt(vsd.frobenius(sdp.a[1:])))
    )
    x = pscale * np.eye(n)
    dscale = 1 + max(
        np.max(np.sqrt(vsd.frobenius(sdp.a[1:]))), np.sqrt(vsd.frobenius(sdp.a[0]))
    ) / np.sqrt(n)
    y = np.zeros(m)
    z = dscale * np.eye(n)

    lz = np.sqrt(dscale) * np.eye(n)  # = sp.linalg.cholesky(z, lower=True)

    it = 0
    mu = vsd.frobenius(vsd.vectorize(x), vsd.vectorize(z)) / (2.0 * n)
    print(f"Initialized solver for an SDP of matrix dimension {n} with {m} constraints")

    def log_state() -> bool:
        pobj = sdp.primal_objective(vsd.project(x))
        dobj = sdp.dual_objective(y)
        relgap = sdp.relative_gap(vsd.project(x), y)
        pinfeas = sdp.primal_infeasibility(vsd.vectorize(x))
        dinfeas = sdp.dual_infeasibility(y, vsd.vectorize(z))
        print(
            f"it={it}, mu={mu:.3g}, relgap={relgap:.3g}, lower={pobj:3g}, upper={dobj:3g}, pinfeas={pinfeas:3g}, dinfeas={dinfeas:3g}"
        )

        if dinfeas < 1e-7 and pinfeas < 1e-7 and relgap < 1e-7:
            return True
        else:
            return False

    log_state()

    while True:
        it += 1

        # precompute inverse of dual variable Z
        lz_inv = sp.linalg.solve_triangular(lz, np.eye(n), lower=True)
        zinv = lz_inv.T @ lz_inv

        # build and factor newton system
        system_matrix = sdp.opA(vsd.matmul_project(zinv, sdp.adA(np.eye(m)), x))
        system_matrix_chol = sp.linalg.cho_factor(system_matrix)

        # predictor step
        fd = vsd.vectorize(z) + sdp.a[0] - sdp.adA(y)
        zinv_fd_x = vsd.matmul_project(zinv, fd, x)
        pred_rhs = sdp.opA(zinv_fd_x) - sdp.b
        dy_pred = sp.linalg.cho_solve(system_matrix_chol, pred_rhs)
        adA_y_pred = sdp.adA(dy_pred)
        dx_pred = zinv_fd_x - vsd.vectorize(x) - vsd.matmul_project(zinv, adA_y_pred, x)
        dz_pred = -fd + adA_y_pred

        # corrector step
        tmp = vsd.project(mu * zinv) - vsd.matmul_project(zinv, dz_pred, dx_pred)
        corr_rhs = sdp.opA(tmp)
        dy_corr = sp.linalg.cho_solve(system_matrix_chol, corr_rhs)
        dz_corr = sdp.adA(dy_corr)
        dx_corr = tmp - vsd.matmul_project(zinv, dz_corr, x)

        # compute final directions
        dx = dx_pred + dx_corr
        dy = dy_pred + dy_corr
        dz = dz_pred + dz_corr

        # line search in primal and dual directions
        palpha = posdef_linesearch(
            x, vsd.unvectorize(dx), precomputed="nothing", min_step=1e-8, step_frac=0.95
        )
        dalpha = posdef_linesearch(
            lz_inv,
            vsd.unvectorize(dz),
            precomputed="cholinv",
            min_step=1e-8,
            step_frac=0.95,
        )

        # apply updates
        x = x + palpha * vsd.unvectorize(dx)
        y = y + dalpha * dy
        z = z + dalpha * vsd.unvectorize(dz)
        lz = sp.linalg.cholesky(z, lower=True)
        mu = vsd.frobenius(vsd.vectorize(x), vsd.vectorize(z)) / (2.0 * n)

        if log_state():
            break

    print("stopping")
    bounds = sdp.bounds(vsd.project(x), y)
    print("objective", (bounds[1] + bounds[0]) / 2.0)
