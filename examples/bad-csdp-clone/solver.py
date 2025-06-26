import time
import argparse
import logging
from enum import Enum
import numpy as np
import scipy as sp
from poptools.io import read_sdpa_sparse_from_string
from poptools.opt import SemidefiniteProgram
from poptools.linalg import (
    posdef_linesearch,
    frobenius,
    BlockMatArray,
    cho_factor,
    cho_solve,
    symmetric_part,
)


class SolverExitCode(Enum):
    OPTIMAL = 1
    REDUCED_PRECISION = 2
    INFEASIBLE = 3
    UNBOUNDED = 4
    SINGULAR = 5
    MAX_ITER = 6
    STUCK = 7


class SolverState:
    def __init__(self, sdp: SemidefiniteProgram):
        self.sdp = sdp
        self.it: int = 0
        self.is_feasible: bool = False
        self.exit_code: SolverExitCode | None = None
        self.metrics_history: dict[str, list[float]] = {
            "pobj": [],
            "dobj": [],
            "relgap": [],
            "pinfeas": [],
            "dinfeas": [],
            "mu": [],
            "palpha": [],
            "dalpha": [],
        }

    def gather_iteration_metrics(
        self,
        x: BlockMatArray,
        y: np.ndarray,
        z: BlockMatArray,
        mu: float,
        palpha: float,
        dalpha: float,
    ) -> None:
        self.metrics_history["pobj"].append(self.sdp.primal_objective(x)[0])
        self.metrics_history["dobj"].append(float(self.sdp.dual_objective(y)))
        self.metrics_history["relgap"].append(self.sdp.relative_gap(x, y)[0])
        self.metrics_history["pinfeas"].append(self.sdp.primal_infeasibility(x))
        self.metrics_history["dinfeas"].append(self.sdp.dual_infeasibility(y, z)[0])
        self.metrics_history["mu"].append(mu)
        self.metrics_history["palpha"].append(palpha)
        self.metrics_history["dalpha"].append(dalpha)
        self.it += 1

    def early_exit(self, exit_code: SolverExitCode, explanation: str) -> None:
        self.exit_code = exit_code
        logging.debug(f"early exit: {explanation}")

    def decide_whether_to_stop(self) -> bool:
        if self.exit_code is not None:
            return True

        pobj = self.metrics_history["pobj"][-1]
        dobj = self.metrics_history["dobj"][-1]
        relgap = self.metrics_history["relgap"][-1]
        pinfeas = self.metrics_history["pinfeas"][-1]
        dinfeas = self.metrics_history["dinfeas"][-1]

        if dinfeas < 1e-6 and pinfeas < 1e-6:
            if not self.is_feasible:
                logging.debug("declared problem feasible")
                self.is_feasible = True
            if relgap < 1e-6:
                self.exit_code = SolverExitCode.OPTIMAL
                return True

        if self.it >= 100:
            self.exit_code = SolverExitCode.MAX_ITER
            return True

        if self.it > 20:
            if pinfeas > 1e8:
                logging.debug("declared primal infeasible")
                self.exit_code = SolverExitCode.INFEASIBLE
                return True
            if dinfeas > 1e8:
                logging.debug("declared dual infeasible")
                self.exit_code = SolverExitCode.INFEASIBLE
                return True
            if np.abs(dobj) > 1e8:
                logging.debug("dual unbounded")
                self.exit_code = SolverExitCode.UNBOUNDED
                return True
            if np.abs(pobj) > 1e8:
                logging.debug("primal unbounded")
                self.exit_code = SolverExitCode.UNBOUNDED
                return True

        if self.it >= 20 and self.it % 5 == 0:
            if pinfeas > 1e-3:
                pinfeas_hist = self.metrics_history["pinfeas"][-20:]
                pfeas_progress = np.min(pinfeas_hist) - np.max(pinfeas_hist)
                if pfeas_progress > 0:
                    logging.debug("no progress in the primal feasibility")
                    self.exit_code = SolverExitCode.STUCK
                    return True

            if dinfeas > 1e-3:
                dinfeas_hist = self.metrics_history["dinfeas"][-20:]
                dfeas_progress = np.min(dinfeas_hist) - np.max(dinfeas_hist)
                if dfeas_progress > 0:
                    logging.debug("no progress in the dual feasibility")
                    self.exit_code = SolverExitCode.STUCK
                    return True

        return False


_tiktok_tasks: dict[str, float] = {}
_tiktok_start: float = 0.0


def tik() -> None:
    global _tiktok_start
    _tiktok_start = time.time()


def tok(task_name: str) -> None:
    global _tiktok_tasks, _tiktok_start
    elapsed = time.time() - _tiktok_start
    _tiktok_start = time.time()
    _tiktok_tasks[task_name] = _tiktok_tasks.get(task_name, 0.0) + elapsed


def tiktok_report() -> None:
    global _tiktok_tasks
    print("Time spent per task (in seconds):")
    for task, elapsed in _tiktok_tasks.items():
        print(f"{task}: {elapsed:.3f}")
    _tiktok_tasks.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("dats_file", type=str, help="Path to an SDPA .dat-s file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s]: %(message)s")

    tik()
    sdp = read_sdpa_sparse_from_string(open(args.dats_file, "r").read())
    tok("read sdp from file")

    tik()
    x = BlockMatArray.identity(1, sdp.block_structure)
    y = np.zeros(sdp.m)
    z = BlockMatArray.identity(1, sdp.block_structure)
    mu: float = frobenius(x, z)[0, 0] / (2.0 * sdp.n)
    exit_code: SolverExitCode | None = None
    state = SolverState(sdp)
    state.gather_iteration_metrics(x, y, z, mu, 0.0, 0.0)

    def log_state(state: SolverState):
        print(
            f"it={state.it}, mu={state.metrics_history['mu'][-1]:.3g}, relgap={state.metrics_history['relgap'][-1]:.3g}, lower={state.metrics_history['pobj'][-1]:3g}, upper={state.metrics_history['dobj'][-1]:3g}, pinfeas={state.metrics_history['pinfeas'][-1]:3g}, dinfeas={state.metrics_history['dinfeas'][-1]:3g}"
        )

    print(
        f"Initialized solver for an SDP with block structure {[b[1] for b in sdp.block_structure]} and {sdp.m} constraints"
    )
    log_state(state)
    tok("main loop preparation")

    while True:
        # invert dual variable Z
        tik()
        try:
            lz = cho_factor(z)
            lz_inv = cho_solve(lz, BlockMatArray.identity(1, sdp.block_structure))
            zinv = lz_inv.T @ lz_inv
        except np.linalg.LinAlgError:
            state.early_exit(SolverExitCode.SINGULAR, "dual variable Z became singular")
            break
        tok("invert dual variable Z")

        # build newton system
        tik()
        system_matrix = sdp.opA(zinv @ sdp.adA(np.eye(sdp.m)) @ x)
        tok("build newton system")

        # factor newton system
        tik()
        try:
            system_matrix_chol = sp.linalg.cho_factor(system_matrix)
        except np.linalg.LinAlgError:
            tok("factor newton system")
            state.early_exit(SolverExitCode.SINGULAR, "system matrix became singular")
            break
        tok("factor newton system")

        # predictor step
        tik()
        fd = z + sdp.c - sdp.adA(y)
        zinv_fd_x = zinv @ fd @ x
        pred_rhs = sdp.opA(zinv_fd_x)[:, 0] - sdp.b
        dy_pred = sp.linalg.cho_solve(system_matrix_chol, pred_rhs)
        adA_y_pred = sdp.adA(dy_pred)
        dx_pred = symmetric_part(zinv_fd_x - x - zinv @ adA_y_pred @ x)
        dz_pred = -fd + adA_y_pred
        tok("predictor step")

        # corrector step
        tik()
        tmp = mu * zinv - zinv @ dz_pred @ dx_pred
        corr_rhs = sdp.opA(tmp)[:, 0]
        dy_corr = sp.linalg.cho_solve(system_matrix_chol, corr_rhs)
        dz_corr = sdp.adA(dy_corr)
        dx_corr = symmetric_part(tmp - zinv @ dz_corr @ x)
        dx = dx_pred + dx_corr
        dy = dy_pred + dy_corr
        dz = dz_pred + dz_corr
        tok("corrector step")

        # primal and dual line search
        tik()
        lsparams = {"min_step": 1e-8, "step_frac": 0.95}
        try:
            palpha = posdef_linesearch(x, dx, precomputed="nothing", **lsparams)
        except np.linalg.LinAlgError:
            state.early_exit(SolverExitCode.STUCK, "primal line search failed")
            break
        try:
            dalpha = posdef_linesearch(lz_inv, dz, precomputed="cholinv", **lsparams)
        except np.linalg.LinAlgError:
            state.early_exit(SolverExitCode.STUCK, "dual line search failed")
            break
        tok("primal and dual line search")

        # apply updates and set new target mu
        tik()
        x = x + palpha * dx
        y = y + dalpha * dy
        z = z + dalpha * dz
        mu = float(frobenius(x, z)[0, 0] / (2.0 * sdp.n))
        if palpha + dalpha > 1.8:
            # fast-forward on central path if we were on a good track
            mu /= 2.0
        tok("apply updates")

        state.gather_iteration_metrics(x, y, z, mu, palpha, dalpha)
        log_state(state)
        if state.decide_whether_to_stop():
            break

    assert state.exit_code is not None
    print("exit code", state.exit_code.name)

    tiktok_report()
