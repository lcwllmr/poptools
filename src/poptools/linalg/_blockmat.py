from typing import Literal
import numpy as np
import scipy as sp

Scalar = float | int | np.floating | np.integer
BlockType = Literal["dense", "diagonal"]
BlockStructure = list[tuple[BlockType, int]]
BlockMatrixType = np.ndarray


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

    def __init__(self, structure: BlockStructure, blocks: list[BlockMatrixType]):
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
                else:
                    raise ValueError(f"Unknown block type: {t}")
            out[i] = sp.linalg.block_diag(*blocks)
        return out

    def __add__(self, other: "BlockMatArray") -> "BlockMatArray":
        """
        Adds two block matrices with compatible structures.
        """
        if not are_block_structures_compatible(self.structure, other.structure):
            raise ValueError("Incompatible block structures for addition.")
        out_blocks: list[BlockMatrixType] = []
        for a_block, b_block in zip(self.blocks, other.blocks):
            out_blocks.append(a_block + b_block)
        return BlockMatArray(self.structure, out_blocks)

    def __neg__(self) -> "BlockMatArray":
        """
        Negates the block matrix array.
        """
        out_blocks: list[BlockMatrixType] = [-block for block in self.blocks]
        return BlockMatArray(self.structure, out_blocks)

    def __sub__(self, other: "BlockMatArray") -> "BlockMatArray":
        """
        Subtracts two block matrices with compatible structures.
        """
        if not are_block_structures_compatible(self.structure, other.structure):
            raise ValueError("Incompatible block structures for subtraction.")
        out_blocks: list[BlockMatrixType] = []
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
        out_blocks: list[BlockMatrixType] = []
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
        out_blocks: list[BlockMatrixType] = []
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
        out_blocks: list[BlockMatrixType] = []
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
    If `b` is None, it computes the squared Frobenius norm of `a`.
    If `b` is provided, it computes the Frobenius inner product between `a` and `b`.
    """
    if b is None:
        accum = np.zeros(a.shape[0])
        for block in a.blocks:
            if block.ndim == 2:
                accum += np.sum(block**2, axis=1)
            elif block.ndim == 3:
                accum += np.sum(block**2, axis=(1, 2))
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
        return accum
