import numpy as np
from poptools.linalg import BlockStructure, BlockMatArray
from poptools.opt import SemidefiniteProgram


def maxcut(num_verts: int, connectivity: float) -> SemidefiniteProgram:
    """
    Generates a simple MaxCut relaxation of a random graph with the given number of vertices `num_verts` and `connectivity` which is the probability of two vertices being connected by an edge.

    >>> sdp = maxcut(10, 0.5)
    """
    assert num_verts > 0
    assert 0.0 < connectivity < 1.0

    block_structure: BlockStructure = [("dense", num_verts)]

    a = BlockMatArray.zeros(num_verts, block_structure)
    for i in range(num_verts):  # one constraint per vertex
        a.blocks[0][i, i, i] = 1.0  # one 1 at position (i,i) for each constraint i

    b = 0.25 * np.ones(num_verts)

    c = BlockMatArray.zeros(1, block_structure)
    ut = np.triu(np.random.rand(num_verts, num_verts) < connectivity, k=1).astype(
        np.float64
    )
    G = ut + ut.T
    e = np.ones(num_verts)
    L = np.diag(G @ e) - G  # graph laplacian
    c.blocks[0][:] = L

    return SemidefiniteProgram(a, b, c)
