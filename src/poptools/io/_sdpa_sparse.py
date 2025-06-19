from typing import List
import numpy as np
import scipy as sp

from poptools.linalg import VecSymDomain
from poptools.opt import SemidefiniteProgram


def read_sdpa_sparse_from_string(dats: str) -> SemidefiniteProgram:
    """
    Reads a semidefinite program from a string in SDPA sparse format (`.dat-s`).
    See, for instance, the [docs of `sdpa-python`](https://sdpa-python.github.io/docs/formats/sdpa.html) for details.
    The objective sense of the primal as defined by `SemidefiniteProgram` is 'max' by SDPA convention.

    Diagonal blocks (indicated by negative dimensions in the format), and comments are not yet implemented.
    Also, the block structure is totally ignored and the function produces fully dense matrices.
    Will all be fixed soon.
    """

    # first line is the number of constraints
    # FIXME: there could be comments here (and also after each line)
    it = iter(dats.split())
    m = int(next(it))

    # block structure of all matrices (objective and constraints)
    n_blocks = int(next(it))
    block_dims: List[int] = []
    for _ in range(n_blocks):
        block_dims.append(int(next(it)))

    # cost vector `b`
    b = np.zeros(m)
    for i in range(m):
        b[i] = float(next(it))

    a_by_block = [np.zeros((m + 1, d, d)) for d in block_dims]
    while True:
        try:
            matrix = int(next(it))
        except StopIteration:
            break

        block = int(next(it)) - 1
        i = int(next(it)) - 1
        j = int(next(it)) - 1
        value = float(next(it))

        a_by_block[block][matrix, i, j] = value
        a_by_block[block][matrix, j, i] = value

    n = sum(block_dims)  # total order of the matrices
    a = np.zeros((m + 1, n, n))
    for i in range(m + 1):
        a[i] = sp.linalg.block_diag(*(a_by_block[j][i] for j in range(n_blocks)))

    vsd = VecSymDomain(n)
    return SemidefiniteProgram(vsd, vsd.vectorize(a), b, opt_sense="max")
