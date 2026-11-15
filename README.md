# DNA-RNN

[![CI](https://github.com/kapildhinwa2010-dev/DNA-RNN/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-RNN/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/DNA-RNN/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/DNA-RNN/actions/workflows/codeql.yml)


DNA quantized codec tuned for recurrent weights: per-gate encoding (LSTM/GRU/vanilla).

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

## What it does

DNA-RNN is the recurrent-specialized member of the DNA codec family
(lineage: the private DNA repository). encode()/decode() quantize a
float matrix into a GeneRecord; the RNN-specific part is
encode_gates()/decode_gates(): stacked gate rows (LSTM=4, GRU=3,
vanilla RNN=1) are split and encoded gate-by-gate, each with its own
scale - lower error than one global scale because gates have
different value ranges.

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from dna_rnn import encode_gates, decode_gates

records = encode_gates(W_ih, cell="lstm", bits=8)  # rows divisible by 4
restored = decode_gates(records)
```

## API

- encode(matrix, bits) / decode(record) / codec_stats(matrix, bits)
- encode_gates(matrix, cell, bits) - cell: "lstm", "gru" or "rhnn"
- decode_gates(records) - vstack back to the full matrix
- encode_heads_compat(matrix, num_blocks, bits) - generic row split

## Honest results and limitations

- Lossy: error equals the quantization step of each gate block.
- Row count must be divisible by the gate count; gate order is the
  PyTorch stacking order and is not verified against your layer.
- "Ratio" in codec_stats() is float32-bytes / encoded-bytes
  ARITHMETIC - it counts stored codes, not a real serialized file.
- No per-layer fitting: the reserv.dean-part-2 log records that
  generated-weight reconstructions of trained models were not usable
  without per-layer fitting.

## License

MIT - see [LICENSE](LICENSE).
