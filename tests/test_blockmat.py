import numpy as np
from poptools.linalg import BlockMatArray, BlockStructure, frobenius


def test_zero():
    s: BlockStructure = [("diagonal", 2), ("dense", 2), ("diagonal", 3)]
    a = BlockMatArray.zeros(5, s)
    assert a.structure == s
    assert a.blocks[0].shape == (
        5,
        2,
    )
    assert a.blocks[1].shape == (5, 2, 2)
    assert a.blocks[2].shape == (
        5,
        3,
    )
    assert np.allclose(a.dense(), np.zeros((5, 7, 7)))


def test_stack():
    s: BlockStructure = [("dense", 2), ("dense", 3)]
    a = BlockMatArray.zeros(5, s)
    b = BlockMatArray.zeros(3, s)
    c = BlockMatArray.stack(a, b)
    assert c.shape == (8, 5, 5)
    assert np.allclose(c.dense(), np.zeros((8, 5, 5)))


def test_identity():
    s: BlockStructure = [("diagonal", 2), ("diagonal", 3), ("dense", 4)]
    a = BlockMatArray.identity(3, s)
    assert a.structure == s
    assert a.shape == (3, 9, 9)
    d = a.dense()
    assert np.allclose(d[0], np.eye(9))
    assert np.allclose(d[1], np.eye(9))
    assert np.allclose(d[2], np.eye(9))


def test_add():
    s: BlockStructure = [("dense", 2), ("dense", 3)]
    a = BlockMatArray.identity(2, s) + BlockMatArray.identity(2, s)
    assert np.allclose(a.dense(), np.eye(5) * 2)


def test_sub():
    s: BlockStructure = [("diagonal", 2), ("dense", 3)]
    a = BlockMatArray.identity(2, s) - BlockMatArray.identity(2, s)
    assert np.allclose(a.dense(), np.zeros((5, 5)))


def test_mul():
    s: BlockStructure = [("dense", 2), ("diagonal", 3)]
    a = -2.0 * BlockMatArray.identity(2, s) * 3
    assert np.allclose(a.dense(), -6 * np.eye(5))


def test_matmul_zeros():
    s: BlockStructure = [("diagonal", 4), ("dense", 3), ("diagonal", 1)]
    a = BlockMatArray.zeros(2, s)
    b = BlockMatArray.zeros(2, s)
    c = a @ b
    assert c.shape == (2, 8, 8)
    assert np.allclose(c.dense(), np.zeros((8, 8)))


def test_matmul_identity():
    s: BlockStructure = [("diagonal", 1), ("dense", 4), ("dense", 3)]
    a = BlockMatArray.identity(3, s)
    b = BlockMatArray.identity(3, s)
    c = a @ b
    assert c.shape == (3, 8, 8)
    assert np.allclose(c.dense(), np.eye(8))


def test_matmul_threefold_broadcast():
    s: BlockStructure = [("dense", 3), ("diagonal", 2), ("dense", 1)]
    a = 2 * BlockMatArray.identity(1, s)
    b = 2 * BlockMatArray.identity(5, s)
    c = 5 * BlockMatArray.identity(1, s)
    d = a @ b @ c
    assert d.shape == (5, 6, 6)
    assert np.allclose(d.dense(), 20 * np.eye(6))


def test_frobenius_norm():
    s: BlockStructure = [("dense", 2), ("diagonal", 4)]
    a = BlockMatArray.identity(4, s)
    f = np.sqrt(frobenius(a))
    assert f.shape == (4,)
    assert np.allclose(f, np.sqrt(6))


def test_frobenius_inner_product():
    s: BlockStructure = [("diagonal", 2), ("dense", 4)]
    a = BlockMatArray.identity(4, s)
    b = BlockMatArray.identity(2, s)
    f = frobenius(a, b)
    assert f.shape == (4, 2)
    assert np.allclose(f, 6)
