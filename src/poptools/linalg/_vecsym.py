import numpy as np


class VecSymDomain:
    """
    Handles vectorization of symmetric matrices by storing only the upper triangular part.
    Thus, it reduces the number of parameters needed to represent a symmetric matrix of size `n` from `n*n` to `n*(n+1)/2`.
    The entries corresponding to the diagonal are scaled by `sqrt(2)` so that the Frobenius inner product after vectorization becomes twice the Euclidean inner product.

    >>> n = 10
    >>> vecsym = VecSymDomain(n)
    >>> vecsym.dim
    55
    >>> a = np.random.normal(size=(n, n))
    >>> vec = vecsym.project(a)
    >>> vec.shape
    (55,)
    >>> assert np.isclose(vecsym.frobenius(vec), np.sum((0.5 * (a + a.T))**2))
    >>> assert np.allclose(vecsym.unvectorize(vec), 0.5 * (a + a.T))
    """

    def __init__(self, n: int):
        self.n = n  # TODO: do quick check on n
        self.dim = n * (n + 1) // 2
        self.mat_diag = np.diag_indices(n)
        self.ut = np.triu_indices(n)
        self.vec_diag = self.ut[0] == self.ut[1]
        self.slt = np.tril_indices(n, k=-1)

        # TODO: this can probably be done much faster
        # build an index array that extracts upper triangular indices (ut) to match the corresponding indices in the strictly lower triangular part (slt)
        reorder_ut_to_slt: list[int] = []
        for i, j in zip(*self.slt):
            if i != j:
                assert i > j
                # find the index in ut that corresponds to (i, j)
                idx = ((self.ut[0] == j) & (self.ut[1] == i)).nonzero()[0][0]
                reorder_ut_to_slt.append(idx)
        self.reorder_ut_to_slt = np.array(reorder_ut_to_slt)

    def vectorize(self, a: np.ndarray) -> np.ndarray:
        """
        Vectorizes symmetric matrices stored in the last two dimensions of a into a vector containing only the upper (resp. lower) triangular part.
        The diagonal entries are scaled by `1/sqrt(2)` for faster computation of the Frobenius inner product.
        """
        # TODO: do checks
        vec = a[..., *self.ut].reshape(a.shape[:-2] + (self.dim,))
        vec[..., self.vec_diag] /= np.sqrt(2)
        return vec

    def project(self, a: np.ndarray) -> np.ndarray:
        """
        Symmetrizes the input matrix (or stack of matrices) `a` by averaging it with its transpose, and returns it vectorized.
        """
        # TODO: do checks
        return self.vectorize(0.5 * (a + a.transpose(*range(len(a.shape) - 2), -1, -2)))

    def unvectorize(self, vec: np.ndarray) -> np.ndarray:
        """
        Exactly reverts the action of `vectorize`, reconstructing the symmetric matrices from vectors.
        """
        # TODO: do checks
        a = np.zeros(vec.shape[:-1] + (self.n, self.n), dtype=vec.dtype)
        a[..., *self.ut] = vec
        a[..., *self.slt] = vec[..., self.reorder_ut_to_slt]
        a[..., *self.mat_diag] *= np.sqrt(2)
        return a

    def frobenius(self, veca: np.ndarray, vecb: np.ndarray | None = None) -> np.ndarray:
        """
        Computes the Frobenius inner product of two stacks of vectorized symmetric matrices.
        If `vecb` is None, it computes the squared Frobenius norm of `veca`.
        If `vecb` is provided, it computes the Frobenius inner product between all vectorized matrices in `veca` and `vecb` stored in the respective last dimension.
        """
        # TODO: do checks
        if vecb is None:
            return 2.0 * np.sum(veca**2, axis=-1)
        else:
            return 2.0 * np.tensordot(veca, vecb, axes=(-1, -1))

    def matmul_project(self, *vecsymmats_or_mats: np.ndarray) -> np.ndarray:
        """
        Multiplies the matrices using standard matrix multiplication, and then projects the result to the vectorized symmetric matrix space.
        The matrices can be either vectorized symmetric matrices or regular matrices, and the decision is made based on their shapes.
        This is somewhat similar to the Jordan product `0.5 * (a @ b + b @ a)` but for multiple matrices.
        """
        # TODO: do checks
        accum = vecsymmats_or_mats[0]
        for a in vecsymmats_or_mats[1:]:
            if a.shape[-2:] == (self.n, self.n):
                # a is a regular matrix
                accum = np.matmul(accum, a)
            else:
                # a is a vectorized symmetric matrix
                accum = np.matmul(accum, self.unvectorize(a))
        return self.project(accum)
