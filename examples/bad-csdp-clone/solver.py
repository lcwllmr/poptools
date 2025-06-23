import numpy as np
import scipy as sp
from poptools.linalg import (
    posdef_linesearch,
    frobenius,
    BlockMatArray,
    cho_factor,
    cho_solve,
)
from poptools.random import maxcut

if __name__ == "__main__":
    np.random.seed(0)
    sdp = maxcut(100, 0.1)

    n = sdp.n
    m = sdp.m

    # heuristic initialization
    pscale: float = n * np.max(
        np.divide(1 + np.abs(sdp.b), 1 + np.sqrt(frobenius(sdp.a)[0]))
    )
    x = pscale * BlockMatArray.identity(1, sdp.block_structure)
    y = np.zeros(m)
    dscale: float = 1 + max(
        np.max(np.sqrt(frobenius(sdp.a)[0])), np.sqrt(frobenius(sdp.c)[0])
    ) / np.sqrt(n)
    z = dscale * BlockMatArray.identity(1, sdp.block_structure)

    it = 0
    mu: float = frobenius(x, z)[0, 0] / (2.0 * n)
    print(f"Initialized solver for an SDP of matrix dimension {n} with {m} constraints")

    def log_state() -> bool:
        pobj = sdp.primal_objective(x)[0]
        dobj = sdp.dual_objective(y)
        relgap = sdp.relative_gap(x, y)[0]
        pinfeas = sdp.primal_infeasibility(x)
        dinfeas = sdp.dual_infeasibility(y, z)[0]
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
        lz = cho_factor(z)
        lz_inv = cho_solve(lz, BlockMatArray.identity(1, sdp.block_structure))
        zinv = lz_inv.T @ lz_inv

        # build and factor newton system
        assert isinstance(x, BlockMatArray)
        tmp = zinv @ sdp.adA(np.eye(m)) @ x
        tmp = 0.5 * tmp + 0.5 * tmp.T

        system_matrix = sdp.opA(tmp)
        system_matrix_chol = sp.linalg.cho_factor(system_matrix)

        # predictor step
        assert isinstance(z, BlockMatArray)
        assert isinstance(y, np.ndarray)
        fd = z + sdp.c - sdp.adA(y)
        zinv_fd_x = zinv @ fd @ x
        zinv_fd_x = 0.5 * zinv_fd_x + 0.5 * zinv_fd_x.T
        assert zinv_fd_x.shape[0] == 1
        pred_rhs = sdp.opA(zinv_fd_x)[:, 0] - sdp.b
        assert pred_rhs.shape == (m,)
        dy_pred = sp.linalg.cho_solve(system_matrix_chol, pred_rhs)
        adA_y_pred = sdp.adA(dy_pred)
        zinv_adypred_x = zinv @ adA_y_pred @ x
        zinv_adypred_x = 0.5 * zinv_adypred_x + 0.5 * zinv_adypred_x.T
        dx_pred = zinv_fd_x - x - zinv_adypred_x
        dz_pred = -fd + adA_y_pred

        # corrector step
        assert isinstance(mu, float)
        tmp = mu * zinv - zinv @ dz_pred @ dx_pred
        tmp = 0.5 * tmp + 0.5 * tmp.T
        corr_rhs = sdp.opA(tmp)[:, 0]
        dy_corr = sp.linalg.cho_solve(system_matrix_chol, corr_rhs)
        dz_corr = sdp.adA(dy_corr)
        tmp2 = zinv @ dz_corr @ x
        tmp2 = 0.5 * tmp2 + 0.5 * tmp2.T
        dx_corr = tmp - tmp2

        # compute final directions
        dx = dx_pred + dx_corr
        dy = dy_pred + dy_corr
        dz = dz_pred + dz_corr

        # line search in primal and dual directions
        palpha = posdef_linesearch(
            x, dx, precomputed="nothing", min_step=1e-8, step_frac=0.95
        )
        dalpha = posdef_linesearch(
            lz_inv,
            dz,
            precomputed="cholinv",
            min_step=1e-8,
            step_frac=0.95,
        )
        print(palpha, dalpha)

        # apply updates
        x = x + palpha * dx
        y = y + dalpha * dy
        z = z + dalpha * dz
        mu = float(frobenius(x, z)[0, 0] / (2.0 * n))

        if log_state():
            break

    print("stopping")
    bounds = sdp.bounds(x, y)
    print("objective", (bounds[1] + bounds[0])[0] / 2.0)
