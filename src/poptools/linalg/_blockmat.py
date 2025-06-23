from typing import Literal
import numpy as np
import scipy as sp

Scalar = float | int | np.floating | np.integer
BlockKind = Literal["dense", "diagonal"]
BlockStructure = list[tuple[BlockKind, int]]
MatrixBlockType = np.ndarray


def are_block_structures_compatible(a: BlockStructure, b: BlockStructure) -> bool:
    """
    Checks if two block structures are compatible for operations like addition or multiplication.
    Two block structures are compatible if they have the same number of blocks and each block has the same type and size.
    """
    if len(a) != len(b):
        return False
    return all(t1 == t2 and n1 == n2 for (t1, n1), (t2, n2) in zip(a, b))


class BlockMatArray:
    """
    Represents an array of block-diagonal matrices with a fixed structure.
    """

    def __init__(self, structure: BlockStructure, blocks: list[MatrixBlockType]):
        self.structure = structure
        self.blocks = blocks

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns the shape of the block matrix array.
        The shape is determined by the sum of the sizes of each block in the structure.
        """
        total_matrix_size = sum(n for _, n in self.structure)
        stack_size = self.blocks[0].shape[0]
        return (stack_size, total_matrix_size, total_matrix_size)

    def copy(self) -> "BlockMatArray":
        """
        Returns a copy of the block matrix array.
        """
        return BlockMatArray(self.structure, [block.copy() for block in self.blocks])

    def dense(self) -> np.ndarray:
        """
        Converts the block matrix array to an array of dense matrices.
        """
        out = np.zeros(self.shape)
        for i in range(out.shape[0]):
            blocks: list[np.ndarray] = []
            for (t, _), block in zip(self.structure, self.blocks):
                if t == "dense":
                    blocks.append(block[i, :, :])
                elif t == "diagonal":
                    blocks.append(np.diag(block[i, :]))
                # else:
                #    raise ValueError(f"Unknown block type: {t}")
            out[i] = sp.linalg.block_diag(*blocks)
        return out

    @property
    def T(self) -> "BlockMatArray":
        """
        Transposes the block matrix array.
        The structure remains the same, but the blocks are transposed.
        """
        out_blocks: list[MatrixBlockType] = []
        for (t, _), block in zip(self.structure, self.blocks):
            if t == "dense":
                out_blocks.append(block.transpose((0, 2, 1)))
            elif t == "diagonal":
                out_blocks.append(block)
        return BlockMatArray(self.structure, out_blocks)

    def __add__(self, other: "BlockMatArray") -> "BlockMatArray":
        """
        Adds two block matrices with compatible structures.
        """
        if not are_block_structures_compatible(self.structure, other.structure):
            raise ValueError("Incompatible block structures for addition.")
        out_blocks: list[MatrixBlockType] = []
        for a_block, b_block in zip(self.blocks, other.blocks):
            out_blocks.append(a_block + b_block)
        return BlockMatArray(self.structure, out_blocks)

    def __neg__(self) -> "BlockMatArray":
        """
        Negates the block matrix array.
        """
        out_blocks: list[MatrixBlockType] = [-block for block in self.blocks]
        return BlockMatArray(self.structure, out_blocks)

    def __sub__(self, other: "BlockMatArray") -> "BlockMatArray":
        """
        Subtracts two block matrices with compatible structures.
        """
        if not are_block_structures_compatible(self.structure, other.structure):
            raise ValueError("Incompatible block structures for subtraction.")
        out_blocks: list[MatrixBlockType] = []
        for a_block, b_block in zip(self.blocks, other.blocks):
            out_blocks.append(a_block - b_block)
        return BlockMatArray(self.structure, out_blocks)

    def __mul__(self, other: Scalar) -> "BlockMatArray":
        """
        Multiplies each block of the block matrix by a scalar.
        """
        out_blocks = [other * block for block in self.blocks]
        return BlockMatArray(self.structure, out_blocks)

    def __rmul__(self, other: Scalar) -> "BlockMatArray":
        """
        Multiplies each block of the block matrix by a scalar (right multiplication).
        """
        return self.__mul__(other)

    def __div__(self, other: Scalar) -> "BlockMatArray":
        """
        Divides each block of the block matrix by a scalar.
        """
        if other == 0:
            raise ValueError("Division by zero is not allowed.")
        out_blocks = [block / other for block in self.blocks]
        return BlockMatArray(self.structure, out_blocks)

    def __matmul__(self, other: "BlockMatArray") -> "BlockMatArray":
        """
        Multiplies the block matrix array with another block matrix array by block-by-block matrix multiplication.
        """
        if not are_block_structures_compatible(self.structure, other.structure):
            raise ValueError(
                "Incompatible block structures for block-wise matrix multiplication."
            )
        out_blocks: list[MatrixBlockType] = []
        for (t1, _), (t2, _), a_block, b_block in zip(
            self.structure, other.structure, self.blocks, other.blocks
        ):
            if t1 == "dense":
                assert t2 == "dense"
                out_blocks.append(a_block @ b_block)
            elif t1 == "diagonal":
                assert t2 == "diagonal"
                out_blocks.append(a_block * b_block)
            else:
                raise ValueError(
                    f"Incompatible block types for matmul: '{t1}' and '{t2}'"
                )
        return BlockMatArray(self.structure, out_blocks)

    @staticmethod
    def zeros(stack_size: int, s: BlockStructure) -> "BlockMatArray":
        """
        Creates a block diagonal matrix with zero matrices in the blocks.
        """
        out_blocks: list[MatrixBlockType] = []
        for t, n in s:
            if t == "dense":
                out_blocks.append(np.zeros((stack_size, n, n)))
            else:
                assert t == "diagonal"
                out_blocks.append(np.zeros((stack_size, n)))
        return BlockMatArray(s, out_blocks)

    @staticmethod
    def identity(stack_size: int, s: BlockStructure) -> "BlockMatArray":
        """
        Creates a block diagonal matrix with identity matrices in the blocks.
        """
        out_blocks: list[MatrixBlockType] = []
        for t, n in s:
            if t == "dense":
                out_blocks.append(
                    np.eye(n, dtype=np.float64)[None, :, :]
                    * np.ones((stack_size, 1, 1))
                )
            else:
                assert t == "diagonal"
                out_blocks.append(np.ones((stack_size, n)))
        return BlockMatArray(s, out_blocks)

    @staticmethod
    def stack(*bmas: "BlockMatArray") -> "BlockMatArray":
        s = bmas[0].structure
        blocks = [
            np.concatenate([bma.blocks[i] for bma in bmas]) for i in range(len(s))
        ]
        return BlockMatArray(s, blocks)


def frobenius(a: BlockMatArray, b: BlockMatArray | None = None) -> np.ndarray:
    """
    Computes the Frobenius inner product of two block matrix arrays.
    If `b` is None, it computes the squared Frobenius norms of each matrix in `a` and returns them in a 1D array.
    If `b` is provided, it computes the Frobenius inner products between all matrices of `a` and `b` and returns them in a 2D array.
    """
    if b is None:
        accum = np.zeros(a.shape[0])
        for block in a.blocks:
            if block.ndim == 2:
                accum += np.sum(block**2, axis=1)
            elif block.ndim == 3:
                accum += np.sum(block**2, axis=(1, 2))
        assert accum.ndim <= 1
        return accum
    else:
        if not are_block_structures_compatible(a.structure, b.structure):
            raise ValueError(
                "Incompatible block structures for Frobenius inner product."
            )
        accum = np.zeros((a.shape[0], b.shape[0]))
        for (t1, _), (t2, _), a_block, b_block in zip(
            a.structure, b.structure, a.blocks, b.blocks
        ):
            if t1 == "dense":
                assert t2 == "dense"
                accum += np.tensordot(a_block, b_block, axes=([1, 2], [1, 2]))
            elif t1 == "diagonal":
                assert t2 == "diagonal"
                accum += np.tensordot(a_block, b_block, axes=([1], [1]))
            else:
                raise ValueError(
                    f"Incompatible block types for Frobenius inner product: '{t1}' and '{t2}'"
                )
        assert accum.ndim <= 2
        return accum


def cho_factor(x: BlockMatArray) -> BlockMatArray:
    assert x.shape[0] == 1, (
        "Cholesky factorization is currently only implemented for single matrices in the block array."
    )
    out_blocks: list[MatrixBlockType] = []
    for (kind, _), block in zip(x.structure, x.blocks):
        if kind == "dense":
            out_blocks.append(
                sp.linalg.cholesky(block[0, :, :], lower=True)[None, :, :]
            )
        else:
            assert kind == "diagonal"
            if np.any(block < 0):
                raise np.linalg.LinAlgError(
                    "Cannot compute Cholesky factorization of negative diagonal blocks."
                )
            out_blocks.append(np.sqrt(block))
    return BlockMatArray(x.structure, out_blocks)


def cho_solve(chol: BlockMatArray, b: BlockMatArray) -> BlockMatArray:
    assert chol.shape[0] == 1, (
        "Cholesky solve is currently only implemented for single matrices in the block array."
    )
    out_blocks: list[MatrixBlockType] = []
    for (kind, _), c_block, b_block in zip(chol.structure, chol.blocks, b.blocks):
        if kind == "dense":
            block_sol: np.ndarray = sp.linalg.solve_triangular(
                c_block[0], b_block[0], lower=True
            )[None, :, :]  # type: ignore
            assert isinstance(block_sol, np.ndarray)
            out_blocks.append(block_sol)
        else:
            assert kind == "diagonal"
            out_blocks.append(b_block / c_block)
    return BlockMatArray(chol.structure, out_blocks)


def maxeigsh(x: BlockMatArray) -> float:
    assert x.shape[0] == 1, "x must be a single block matrix"
    curmax: float = -np.inf
    for (kind, _), block in zip(x.structure, x.blocks):
        if kind == "dense":
            block_max = float(
                sp.sparse.linalg.eigsh(
                    block[0, :, :], k=1, which="LA", return_eigenvectors=False
                )[0]
            )
        else:
            assert kind == "diagonal"
            block_max = np.max(np.abs(block[0, :]))
        curmax = max(curmax, block_max)
    assert curmax != -np.inf
    return curmax
