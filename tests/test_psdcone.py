import numpy as np
from poptools.linalg import posdef_linesearch, BlockMatArray, BlockStructure


def test_nparr_whole_line_pd():
    id = np.eye(4)
    assert np.isclose(posdef_linesearch(id, id), 1.0)


def test_nparr_line_ends_at_boundary():
    id = np.eye(4)
    assert np.isclose(posdef_linesearch(4 * id, -4 * id), 1.0)


def test_nparr_only_10_percent_pd():
    id = np.eye(4)
    assert np.isclose(posdef_linesearch(0.1 * id, -id), 0.1)


def test_nparr_80_percent_pd():
    id = np.eye(4)
    assert np.isclose(posdef_linesearch(id, -1.25 * id), 0.8)


def test_blockarr_whole_line_pd():
    s: BlockStructure = [("dense", 3), ("diagonal", 2)]
    id = BlockMatArray.identity(1, s)
    assert np.isclose(posdef_linesearch(id, id), 1.0)


def test_blockarr_line_ends_at_boundary():
    s: BlockStructure = [("diagonal", 3), ("dense", 2)]
    id = BlockMatArray.identity(1, s)
    assert np.isclose(posdef_linesearch(4 * id, -4 * id), 1.0)


def test_blockarr_only_10_percent_pd():
    s: BlockStructure = [("dense", 3), ("dense", 2)]
    id = BlockMatArray.identity(1, s)
    assert np.isclose(posdef_linesearch(0.1 * id, -id), 0.1)


def test_blockarr_80_percent_pd():
    s: BlockStructure = [("dense", 3), ("diagonal", 2)]
    id = BlockMatArray.identity(1, s)
    assert np.isclose(posdef_linesearch(id, -1.25 * id), 0.8)


def test_nparr_step_too_small():
    id = np.eye(4)
    try:
        posdef_linesearch(0.2 * id, -id, min_step=0.5)
    except np.linalg.LinAlgError as e:
        assert "insufficient" in str(e)
    else:
        assert False, "Expected LinAlgError not raised."
