import numpy as np
from poptools.linalg import VecSymDomain


class SemidefiniteProgram:
    """
    Stores the data associated with a semidefinite program and
    provides various operations on it.

    Arguments:

    - `vsd` : a `VecSymDomain` object representing the domain of symmetric matrices,
    - `a` : array of vectorized symmetric matrices of shape (m + 1, n * (n + 1) / 2),
    - `b` : vector of shape (m,), and

    The primal SDP reads as follows:
    ```text
    maximize    < a[0], x >
    over        x symmetric, PSD matrix of size n x n
    subject to  < a[i], x > = b[i] for i = 1, ..., m
    ```
    where < ., . > denotes the Frobenius inner product of square matrices.

    The corresponding dual SDP reads is:
    ```text
    minimize    b^T y
    over        y vector of size m
    subject to  z = a[0] - sum(y[i] * a[i] for i = 1, ..., m) is PSD
    ```
    """

    def __init__(self, vsd: VecSymDomain, a: np.ndarray, b: np.ndarray):
        self.vsd = vsd
        if a.ndim != 2 or a.shape[1] != self.vsd.dim:
            raise ValueError(
                "a must be a 2D array with shape (m + 1, n * (n + 1) / 2) where m is the number of constraints and n is the order of the matrices."
            )
        self.m = (
            a.shape[0] - 1
        )  # number of constraints (note that a[0] defines the objective functional)
        self.n = (
            self.vsd.n
        )  # matrix order, i.e. their real size is n x n (but vectorized)
        self.a = a
        if b.ndim != 1 or b.shape[0] != self.m:
            raise ValueError(
                "b must be a 1D array with shape (m,) where m is the number of constraints."
            )
        self.b = b

    def primal_objective(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the primal objective function value `< a[0], x >` for a given vectorized symmetric matrix x (in the same `vsd`).
        """
        return self.vsd.frobenius(self.a[0], x)

    def dual_objective(self, y: np.ndarray) -> np.ndarray:
        """
        Computes the dual objective function value `b^T y` for a given vector y.
        """
        if y.ndim != 1 or y.shape[0] != self.m:
            raise ValueError(
                "y must be a 1D array with shape (m,) where m is the number of constraints."
            )
        return np.dot(self.b, y)

    def bounds(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = self.primal_objective(x)
        upper = self.dual_objective(y)
        return lower, upper

    def opA(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the constraint operator A, which maps a vectorized symmetric matrix `x` to an `m`-vector.
        ```
        opA(X) = [< A[i], X > for i = 1, ..., m]
        ```
        This will also work for a stack of vectorized symmetric matrices.
        """
        return self.vsd.frobenius(self.a[1:], x)

    def adA(self, y: np.ndarray) -> np.ndarray:
        """
        Evaluates the adjoint of the constraint operator A, which is defined as:
        ```
        adA(y) = sum(y[i] * A[i] for i = 1, ..., m)
        ```
        at a given `m`-vector y (or an array of vectors).
        The result is again a vectorized symmetric matrix compatible with `vsd`.
        """
        if y.shape[-1] != self.m:
            raise ValueError(
                "y must be a array with last dimension equal to m - the number of constraints."
            )
        return np.tensordot(
            y,
            self.a[1:],
            axes=(
                [
                    0,
                ],
                [
                    0,
                ],
            ),
        )

    def primal_infeasibility(self, X: np.ndarray) -> np.floating:
        """
        Computes the primal infeasibility, which is defined as the 2-norm of the residual of the constraints:
        ```
        pinfeas(X) = || opA(X) - b ||
        ```
        """
        return np.linalg.norm(self.opA(X) - self.b) / (1.0 + np.linalg.norm(self.b))

    def dual_infeasibility(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Computes the dual infeasibility, which is defined as the Frobenius norm of the residual of the dual constraint:
        ```
        dinfeas(y, z) = || adA(y) - a[0] - z ||
        ```
        where `adA(y)` is the evaluation of the adjoint of the constraint operator A at y.
        """
        return np.sqrt(self.vsd.frobenius(self.adA(y) - self.a[0] - z)) / (
            1.0 + np.sqrt(self.vsd.frobenius(self.a[0]))
        )

    def relative_gap(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        dobj = self.dual_objective(y)
        return np.abs(self.primal_objective(x) - dobj) / (1.0 + np.abs(dobj))
