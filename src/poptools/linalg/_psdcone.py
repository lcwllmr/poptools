import numpy as np
import scipy as sp
from typing import Literal


def posdef_linesearch(
    x: np.ndarray,
    dx: np.ndarray,
    precomputed: Literal["nothing", "chol", "cholinv"] = "nothing",
    min_step: float = 1e-8,
    step_frac: float = 1.0,
) -> float:
    """
    Given a symmetric positive definite matrix `x` and a symmetric matrix direction `dx`, this function computes the maximum step size `alpha` such that `x + alpha * dx` remains positive definite.
    It will need to use the inverse of the lower Cholesky factor of `x`. Depending on the value of `precomputed`, it will:

    - `'nothing'`: compute the Cholesky factor of `x` internally and invert it,
    - `'chol'`: assume `x` is already the lower Cholesky factor of `x` but still needs to be inverted, or
    - `'cholinv'`: assume `x` is already the inverse of the lower Cholesky factor of `x`.

    If the maximum possible step size is less than `min_step`, it will raise `numpy.linalg.LinAlgError`.
    The found step size is optionally multiplied by `step_frac` (for making sure to stay in the interior of the PSD cone).

    >>> I = np.eye(5)
    >>> round(posdef_linesearch(I, I), 4) # whole line is PD and moving away from boundary
    1.0
    >>> round(posdef_linesearch(4 * I, -4 * I), 4) # whole line is PD but ends up at boundary
    1.0
    >>> round(posdef_linesearch(0.1 * I, -I), 4) # only 10% is PD
    0.1
    >>> round(posdef_linesearch(I, -1.25 * I), 4) # 80% is PD
    0.8
    >>> posdef_linesearch(1e-7 * I, -I, min_step=1e-6)
    Traceback (most recent call last):
        ...
    numpy.linalg.LinAlgError: Best possible step size is insufficient.
    """
    if precomputed not in ["nothing", "chol", "cholinv"]:
        raise ValueError("precomputed must be one of 'nothing', 'chol', or 'cholinv'.")
    if not x.ndim == 2 or x.shape[0] != x.shape[1]:
        raise ValueError("x must be a square matrix.")
    if not dx.ndim == 2 or dx.shape[0] != dx.shape[1] or dx.shape != x.shape:
        raise ValueError("dx must be a square matrix of the same shape as x.")
    if min_step < 0.0:
        raise ValueError("min_step must be non-negative.")
    if not 0 < step_frac <= 1.0:
        raise ValueError("step_frac must be in the range (0, 1].")

    if precomputed == "nothing":
        x = sp.linalg.cholesky(x, lower=True)
        precomputed = "chol"

    if precomputed == "chol":
        x = sp.linalg.solve_triangular(x, np.eye(x.shape[0]), lower=True)
        precomputed = "cholinv"

    x = x @ dx @ x.T
    maxeig = float(
        sp.sparse.linalg.eigsh(-x, k=1, which="LA", return_eigenvectors=False)[0]
    )
    best_step = 1.0 if maxeig < 0.0 else float(min(1.0, 1.0 / maxeig))

    if best_step < min_step:
        raise np.linalg.LinAlgError("Best possible step size is insufficient.")

    return step_frac * best_step
