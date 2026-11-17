# DNA-RNN

DNA-RNN is a small NumPy-based, lossy quantization codec for two-dimensional weight matrices. It converts matrix values into integer codes plus scale metadata and reconstructs an approximation. Its gate helpers split recurrent-style matrices into equal row blocks and encode each block independently. The repository is an early, minimal implementation (version 0.1.0), not a recurrent neural-network implementation or a model-training toolkit.

## Implemented features

- `encode(matrix, bits=8)` maps a non-empty 2-D matrix to a `GeneRecord`; `bits` must be from 1 to 16. Code arrays use `uint8` for up to 8 bits and `uint16` above 8 bits.
- `decode(record)` reconstructs an approximation from a `GeneRecord`.
- `encode_gates(matrix, cell="lstm", bits=8)` divides the rows into equal contiguous blocks and encodes each block separately. The accepted `cell` values and block counts are `lstm` (4), `gru` (3), and `rhnn` (1). The code does not check that row blocks correspond to a particular model or gate ordering.
- `decode_gates(records)` decodes and vertically stacks the encoded blocks.
- `GeneRecord.to_dict()` and `GeneRecord.from_dict()` provide an in-memory dictionary conversion. The code array remains a NumPy array; this is not a disk or JSON serialization format.
- `codec_stats(matrix, bits=8)` reports a reconstruction error and an estimated size ratio. Its byte calculation compares a float32-size baseline with code-array bytes plus a fixed 16-byte metadata estimate; it does not measure a serialized file or actual end-to-end storage.

## Install

The project metadata declares Python 3.9 or newer and NumPy 1.24 or newer. From the repository root:

```bash
python -m pip install .
```

For running the test suite, install the listed development/test dependencies and invoke pytest:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Example

```python
import numpy as np
from dna_rnn import decode_gates, encode_gates

weights = np.random.default_rng(0).standard_normal((16, 8))
records = encode_gates(weights, cell="lstm", bits=8)
reconstructed = decode_gates(records)
```

`weights` must have a row count divisible by the selected block count. Reconstruction is lossy; each block gets its own min/max range and quantization scale. For a single matrix without block splitting, use `encode(matrix, bits=8)` and `decode(record)`.

## Tests and CI

On the checked-out commit `53be5bc1ddfee47664bf62181464463ffca69da0`, local verification with Python 3.12.3 completed successfully: `python3 -m pytest -q` reported **18 passed**, and the workflow's compile and import/encode/decode smoke checks passed. An editable installation from the project metadata also succeeded in that environment. These are local results; this README review did not verify a GitHub Actions run for the commit.

The CI workflow is configured to run pytest and smoke checks on Python 3.10, 3.11, and 3.12. The test suite checks basic round-trips, input validation, record conversion, estimated statistics, and gate splitting; those tests do not establish general model accuracy or real-world compression/performance.

## Scope and limitations

This package only encodes and decodes matrices. It does not implement LSTM, GRU, or vanilla RNN computations, integrate with a framework, train or fit a model, or validate a matrix against a model's gate layout. Quantization is lossy, and the global min/max range within each encoded block can affect reconstruction error. No broad accuracy, benchmark, or production-readiness claim is established by the included tests. `codec_stats` is an arithmetic estimate rather than a measurement of a persisted representation.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its terms and the copyright notice (`Copyright (c) 2026 Kapildhinwa2010-dev`).
