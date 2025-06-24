from typing import Iterator
import numpy as np

from poptools.linalg import BlockMatArray, BlockStructure
from poptools.opt import SemidefiniteProgram


def _float_finder(input_str: str) -> Iterator[list[float]]:
    lineit = iter(input_str.splitlines())

    while True:
        try:
            line = next(lineit).strip()
        except StopIteration:
            break

        delims = [" ", ","]
        tokens: list[str] = [line]
        for d in delims:
            tmp_tokens: list[str] = []
            for token in tokens:
                tmp_tokens.extend(token.split(d))
            tokens = tmp_tokens
        tokens = [token.strip() for token in tokens if token.strip()]

        out: list[float] = []
        for token in tokens:
            try:
                out.append(float(token))
            except ValueError:
                # if one token is not numeric, the whole rest of the line is a comment
                break
        yield out


def read_sdpa_sparse_from_string(dats: str) -> SemidefiniteProgram:
    """
    Reads a semidefinite program from a string in SDPA sparse format (`.dat-s`).
    See, for instance, the [docs of `sdpa-python`](https://sdpa-python.github.io/docs/formats/sdpa.html) for details.
    The objective sense of the primal as defined by `SemidefiniteProgram` is 'max' by SDPA convention.
    """

    lineit = iter(_float_finder(dats))
    line: list[float] = []
    lineno = 1

    def _skip_until_content_line(search_goal: str) -> None:
        nonlocal lineit, line, lineno
        try:
            while len(line := next(lineit)) == 0:
                lineno += 1
        except StopIteration:
            raise ValueError(f"Unexpected EOF in line {lineno} before {search_goal}.")

    _skip_until_content_line("number of constraints")
    if len(line) != 1:
        raise ValueError(
            f"Expected a single number in line {lineno} for the number of constraints but found {len(line)} values:\n {line}"
        )
    num_constraints = int(round(line[0]))

    _skip_until_content_line("number of blocks")
    if len(line) != 1:
        raise ValueError(
            f"Expected a single number in line {lineno} for the number of blocks but found {len(line)} values:\n {line}"
        )
    num_blocks = int(round(line[0]))

    _skip_until_content_line("block dimensions")
    if len(line) != num_blocks:
        raise ValueError(
            f"Expected {num_blocks} block dimensions in line {lineno} but found {len(line)} values:\n {line}"
        )
    block_structure: BlockStructure = []
    c_by_block: list[np.ndarray] = []
    a_by_block: list[np.ndarray] = []
    for d in line:
        d = int(round(d))
        if d < 0:
            d = -d
            block_structure.append(("diagonal", d))
            c_by_block.append(np.zeros((1, d)))
            a_by_block.append(np.zeros((num_constraints, d)))
        else:
            block_structure.append(("dense", d))
            c_by_block.append(np.zeros((1, d, d)))
            a_by_block.append(np.zeros((num_constraints, d, d)))

    _skip_until_content_line("constraint RHS vector")
    if len(line) != num_constraints:
        raise ValueError(
            f"Expected {num_constraints} constraint RHS values in line {lineno} but found {len(line)} values:\n {line}"
        )
    b = np.array(line)

    while True:
        try:
            _skip_until_content_line("matrix index")
            if len(line) != 5:
                raise ValueError(
                    f"Expected 5 values in line {lineno} for matrix index, block, row, column, and value but found {len(line)} values:\n {line}"
                )
            # note the adjustment for 1-based indexing in the SDPA format
            matrix_index = int(round(line[0]))
            block_index = int(round(line[1])) - 1
            i = int(round(line[2])) - 1
            j = int(round(line[3])) - 1
            value = float(line[4])

            target = c_by_block if matrix_index == 0 else a_by_block
            matrix_index = 0 if matrix_index == 0 else matrix_index - 1
            if block_structure[block_index][0] == "diagonal":
                if i != j:
                    raise ValueError(
                        f"Expected diagonal entry for block {block_index + 1}, but got indices {i + 1} and {j + 1}."
                    )
                target[block_index][matrix_index, i] = value
            else:
                target[block_index][matrix_index, i, j] = value
                target[block_index][matrix_index, j, i] = value
        except ValueError:
            break

    return SemidefiniteProgram(
        BlockMatArray(block_structure, a_by_block),
        b,
        BlockMatArray(block_structure, c_by_block),
    )
