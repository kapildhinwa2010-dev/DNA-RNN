import numpy as np
import pytest

from dna_rnn import encode_gates, decode_gates


def test_lstm_gates_roundtrip():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 16))
    records = encode_gates(W, cell="lstm", bits=8)
    assert len(records) == 4
    out = decode_gates(records)
    assert out.shape == W.shape
    assert np.abs(W - out).max() < 0.05


def test_gru_gates_count():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((9, 4))
    records = encode_gates(W, cell="gru", bits=8)
    assert len(records) == 3


def test_gates_mixed_ranges_low_error():
    rng = np.random.default_rng(2)
    blocks = [rng.standard_normal((4, 8)) * s for s in (0.01, 1.0, 100.0, 10000.0)]
    W = np.vstack(blocks)
    out = decode_gates(encode_gates(W, cell="lstm", bits=8))
    assert np.abs(W[12:] - out[12:]).max() < 10000.0 * 0.05


def test_gates_rejects_unknown_cell():
    with pytest.raises(ValueError):
        encode_gates(np.zeros((8, 8)), cell="transformer")


def test_gates_rejects_indivisible_rows():
    with pytest.raises(ValueError):
        encode_gates(np.zeros((7, 8)), cell="lstm")


def test_decode_gates_rejects_empty():
    with pytest.raises(ValueError):
        decode_gates([])
