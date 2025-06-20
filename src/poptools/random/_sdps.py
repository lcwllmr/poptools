import numpy as np
from poptools.linalg import VecSymDomain
from poptools.opt import SemidefiniteProgram


def maxcut(num_verts: int, connectivity: float):
    assert num_verts > 0
    assert 0.0 < connectivity < 1.0

    ut = np.triu(np.random.rand(num_verts, num_verts) < connectivity, k=1).astype(
        np.float64
    )
    G = ut + ut.T
    e = np.ones(num_verts)
    L = np.diag(G @ e) - G

    a = np.zeros((num_verts + 1, num_verts, num_verts))
    a[0, :, :] = L  # objective: graph laplacian
    for i in range(num_verts):
        a[i + 1, i, i] = 1.0
    b = 0.25 * np.ones(num_verts)

    vsd = VecSymDomain(num_verts)
    return SemidefiniteProgram(vsd, vsd.vectorize(a), b)
