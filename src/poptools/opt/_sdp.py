import numpy as np
from poptools.linalg import BlockMatArray, frobenius, MatrixBlockType


class SemidefiniteProgram:
    """
    Stores the data associated with a semidefinite program and
    provides various operations on it.

    Arguments:

    - `a` : array of block matrices of shape (m, n, n),
    - `b` : vector of shape (m,), and
    - `c` : single block matrix of shape (1, n, n) with same block structure as `a`

    The primal SDP reads as follows:
    ```text
    maximize    < c, x >
    over        x symmetric, PSD matrix of size n x n
    subject to  < a[i], x > = b[i] for i = 1, ..., m
    ```
    where < ., . > denotes the Frobenius inner product of square matrices.

    The corresponding dual SDP reads is:
    ```text
    minimize    b^T y
    over        y vector of size m
    subject to  z = c - sum(y[i] * a[i] for i = 1, ..., m) is PSD
    ```
    """

    def __init__(self, a: BlockMatArray, b: np.ndarray, c: BlockMatArray):
        # TODO: check dimensions and block structure
        self.m = a.shape[
            0
        ]  # number of constraints (note that a[0] defines the objective functional)
        self.n = a.shape[
            1
        ]  # matrix order, i.e. their real size is n x n (but vectorized)
        self.a = a
        self.block_structure = self.a.structure
        if b.ndim != 1 or b.shape[0] != self.m:
            raise ValueError(
                "b must be a 1D array with shape (m,) where m is the number of constraints."
            )
        self.b = b
        self.c = c

    def primal_objective(self, x: BlockMatArray) -> np.ndarray:
        """
        Computes the primal objective function value `< a[0], x >` for a given vectorized symmetric matrix x (in the same `vsd`).
        """
        return frobenius(self.c, x)[0]

    def dual_objective(self, y: np.ndarray) -> np.ndarray:
        """
        Computes the dual objective function value `b^T y` for a given vector y.
        """
        if y.ndim != 1 or y.shape[0] != self.m:
            raise ValueError(
                "y must be a 1D array with shape (m,) where m is the number of constraints."
            )
        return np.dot(self.b, y)

    def bounds(self, x: BlockMatArray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = self.primal_objective(x)
        upper = self.dual_objective(y)
        return lower, upper

    def opA(self, x: BlockMatArray) -> np.ndarray:
        """
        Evaluates the constraint operator A, which maps a vectorized symmetric matrix `x` to an `m`-vector.
        ```
        opA(X) = [< A[i], X > for i = 1, ..., m]
        ```
        This will also work for a stack of vectorized symmetric matrices.
        """
        return frobenius(self.a, x)

    def adA(self, y: np.ndarray) -> BlockMatArray:
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

        blocks: list[MatrixBlockType] = []
        for block in self.a.blocks:
            blocks.append(np.tensordot(y, block, axes=([0], [0])))
        return BlockMatArray(self.block_structure, blocks)

    def primal_infeasibility(self, x: BlockMatArray) -> np.floating:
        """
        Computes the primal infeasibility, which is defined as the 2-norm of the residual of the constraints:
        ```
        pinfeas(X) = || opA(X) - b ||
        ```
        """
        return np.linalg.norm(self.opA(x) - self.b) / (1.0 + np.linalg.norm(self.b))

    def dual_infeasibility(self, y: np.ndarray, z: BlockMatArray) -> np.ndarray:
        """
        Computes the dual infeasibility, which is defined as the Frobenius norm of the residual of the dual constraint:
        ```
        dinfeas(y, z) = || adA(y) - a[0] - z ||
        ```
        where `adA(y)` is the evaluation of the adjoint of the constraint operator A at y.
        """
        return np.sqrt(frobenius(self.adA(y) - self.c - z)) / (
            1.0 + np.sqrt(frobenius(self.c))
        )

    def relative_gap(self, x: BlockMatArray, y: np.ndarray) -> np.ndarray:
        dobj = self.dual_objective(y)
        return np.abs(self.primal_objective(x) - dobj) / (1.0 + np.abs(dobj))
