import numpy as np
from poptools.io import read_sdpa_sparse_from_string


def test_example_1_from_docs():
    # from
    dats = """
    "Example 1: mDim = 3, nBLOCK = 1, {2}"
      3  =  mDIM
      1  =  nBLOCK
      2  = bLOCKsTRUCT
    48, -8, 20
    0 1 1 1 -11
    0 1 2 2 23
    1 1 1 1 10
    1 1 1 2 4
    2 1 2 2 -8
    3 1 1 2 -8
    3 1 2 2 -2
    """

    sdp = read_sdpa_sparse_from_string(dats)
    assert sdp.m == 3
    assert sdp.n == 2
    assert len(sdp.block_structure) == 1
    assert sdp.block_structure[0][0] == "dense"
    assert sdp.block_structure[0][1] == 2
    assert sdp.a.blocks[0].shape == (3, 2, 2)


def test_example_from_scipopt():
    dats = """
    3 = number of variables
    3 = number of blocks
    2 2 -2 = blocksizes (negative sign for LP-block, size of LP-block equals the number of LP-constraints)
    * the next line gives the objective values in the order of the variables
    1 -2 -1
    * the remaining lines give the nonzeroes of the constraints with variable (0 meaning the constant part) block row column value
    1 1 1 1 1 * first variable in block one, row one, column one has coefficient one
    2 1 1 2 1 * variable two in block one, row one, column two has coefficient one (note that because we expect the matrix to be symmetric, we don't need to give the entry for row two and column one)
    3 1 2 2 1
    1 2 1 2 1
    3 2 1 1 1
    0 2 2 2 -2.1 * the constant part (variable zero) in block two, row two, column two equals -2.1 (which we are substracting from the A_i, so in the combined matrix it will have a positive sign)
    1 3 1 1 1 * block three is the LP block, the LP constraints appear as diagonal entries in this block
    2 3 1 1 1
    3 3 1 1 1
    0 3 1 1 1
    1 3 2 2 -1
    2 3 2 2 -1
    3 3 2 2 -1
    0 3 2 2 -8
    """
    sdp = read_sdpa_sparse_from_string(dats)
    assert sdp.m == 3
    assert sdp.n == 6
    assert len(sdp.block_structure) == 3
    assert sdp.block_structure[0][0] == "dense"
    assert sdp.block_structure[0][1] == 2
    assert sdp.a.blocks[0].shape == (3, 2, 2)
    assert sdp.block_structure[1][0] == "dense"
    assert sdp.block_structure[1][1] == 2
    assert sdp.a.blocks[1].shape == (3, 2, 2)
    assert sdp.block_structure[2][0] == "diagonal"
    assert sdp.block_structure[2][1] == 2
    assert sdp.a.blocks[2].shape == (3, 2)
    assert np.isclose(sdp.a.blocks[1][2, 0, 0], 1)
    assert np.isclose(sdp.c.blocks[2][0, 1], -8)
