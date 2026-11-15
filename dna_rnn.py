# DNA-RNN - DNA codec tuned for recurrent weight matrices
# Part of the Kapildhinwa2010-dev Toolkit.

"""DNA-Net - encode/decode codec for weight matrices ("genes").

Lineage: the DNA repository. A float matrix is encoded into a compact gene
record (quantized codes + scale metadata) and decoded back; the codec is
deterministic and supports lossless round-trips within quantization error.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "encode", "decode", "GeneRecord", "codec_stats",
    "encode_gates", "decode_gates",
]


class GeneRecord:
    """Compact encoded representation of one weight matrix."""

    __slots__ = ("codes", "lo", "scale", "shape", "bits")

    def __init__(self, codes: np.ndarray, lo: float, scale: float, shape, bits: int) -> None:
        self.codes = codes
        self.lo = lo
        self.scale = scale
        self.shape = shape
        self.bits = bits

    def nbytes(self) -> int:
        return int(self.codes.nbytes + 16)

    def to_dict(self) -> dict:
        return {
            "codes": self.codes,
            "lo": self.lo,
            "scale": self.scale,
            "shape": self.shape,
            "bits": self.bits,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneRecord":
        return cls(data["codes"], float(data["lo"]), float(data["scale"]),
                   tuple(data["shape"]), int(data["bits"]))


def encode(matrix: np.ndarray, bits: int = 8) -> GeneRecord:
    """Quantize a matrix into a GeneRecord."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo = float(matrix.min())
    hi = float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    dtype = np.uint16 if bits > 8 else np.uint8
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels).astype(dtype)
    return GeneRecord(codes, lo, scale, matrix.shape, bits)


def decode(record: GeneRecord) -> np.ndarray:
    """Reconstruct the matrix from a GeneRecord."""
    if not isinstance(record, GeneRecord):
        raise TypeError("decode expects a GeneRecord")
    values = np.asarray(record.codes, dtype=np.float64) * record.scale + record.lo
    return values.reshape(record.shape)


def codec_stats(matrix: np.ndarray, bits: int = 8) -> dict:
    """Report compression statistics for encoding ``matrix``."""
    record = encode(matrix, bits=bits)
    original_bytes = matrix.size * 4  # float32
    return {
        "original_bytes": int(original_bytes),
        "encoded_bytes": record.nbytes(),
        "ratio": original_bytes / record.nbytes(),
        "max_abs_error": float(np.abs(matrix - decode(record)).max()),
        "bits": bits,
    }


GATES = {"lstm": 4, "gru": 3, "rhnn": 1}


def encode_gates(matrix: np.ndarray, cell: str = "lstm", bits: int = 8) -> list:
    """Encode a recurrent gate weight matrix gate-by-gate.

    LSTM/GRU layers stack their gates (input, forget, cell, output for
    LSTM; reset, update, candidate for GRU) along the row dimension.
    Gates have different value ranges, so encoding each gate block with
    its own scale gives lower reconstruction error than one global scale.

    Args:
        matrix: 2-D weight matrix whose row count is divisible by the
            gate count (LSTM=4, GRU=3, vanilla RNN=1).
        cell: "lstm", "gru" or "rhnn".

    Returns: List of GeneRecord, one per gate.
    """
    key = cell.lower()
    if key not in GATES:
        raise ValueError("cell must be one of: " + ", ".join(sorted(GATES)))
    return encode_heads_compat(matrix, GATES[key], bits=bits)


def decode_gates(records: list) -> np.ndarray:
    """Reconstruct a gate weight matrix from encode_gates output."""
    if not records:
        raise ValueError("records must be non-empty")
    return np.vstack([decode(rec) for rec in records])


def encode_heads_compat(matrix: np.ndarray, num_blocks: int, bits: int = 8) -> list:
    """Split rows into equal blocks and encode each with its own scale."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if num_blocks < 1 or matrix.shape[0] % num_blocks != 0:
        raise ValueError("block count must divide the number of rows")
    rows = matrix.shape[0] // num_blocks
    return [
        encode(matrix[i * rows:(i + 1) * rows, :], bits=bits)
        for i in range(num_blocks)
    ]