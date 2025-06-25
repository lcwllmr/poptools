import pytest
import numpy as np
from poptools.linalg import BlockMatArray, BlockStructure
from poptools.opt import SemidefiniteProgram


@pytest.fixture
def sdp():
    # compute the maximal eigenvalue of a diagonal matrix
    bs: BlockStructure = [("diagonal", 10)]
    a = BlockMatArray.identity(1, bs)
    b = np.array([1.0])
    c = BlockMatArray(bs, [np.arange(1, 10 + 1).reshape(1, 10)])
    return SemidefiniteProgram(a, b, c)


def test_primal_infeasibility(sdp: SemidefiniteProgram):
    # primal feasible iff trace equal 1
    x1 = BlockMatArray.zeros(1, sdp.block_structure)
    x1.blocks[0][0, 3] = 0.3
    assert not np.isclose(sdp.primal_infeasibility(x1), 0)

    x2 = BlockMatArray.zeros(1, sdp.block_structure)
    x2.blocks[0][0, 8] = 0.7
    assert not np.isclose(sdp.primal_infeasibility(x2), 0)

    assert np.isclose(sdp.primal_infeasibility(x1 + x2), 0)


def test_primal_objective(sdp: SemidefiniteProgram):
    x = BlockMatArray.zeros(1, sdp.block_structure)
    x.blocks[0][0, 9] = 1.0
    pobj = sdp.primal_objective(x)
    assert np.isclose(pobj, 10.0)


def test_dual_infeasibility(sdp: SemidefiniteProgram):
    # dual feasible has no particular meaning but we need
    #   z = y * id - m
    # where m is the target matrix
    y = np.array([10.0])
    z = BlockMatArray.diagonal(1, sdp.block_structure, np.arange(0, 10)[::-1])
    assert np.isclose(sdp.dual_infeasibility(y, z), 0)

    z.blocks[0][0, 0] += 1
    assert not np.isclose(sdp.dual_infeasibility(y, z), 0)


def test_dual_objective(sdp: SemidefiniteProgram):
    # this is just the dot product between b=1 and the dual variable y,
    # so should send y to y
    for y in range(1, 10 + 1):
        assert np.isclose(sdp.dual_objective(np.array([y])), y)


def test_relative_gap(sdp: SemidefiniteProgram):
    # re-use findings from other tests
    x = BlockMatArray.zeros(1, sdp.block_structure)
    x.blocks[0][0, 9] = 1.0
    y = np.array([10])
    assert np.isclose(sdp.relative_gap(x, y), 0)
    y[0] = 2
    assert not np.isclose(sdp.relative_gap(x, y), 0)
