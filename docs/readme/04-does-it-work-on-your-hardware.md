<p align="center"><img src="docs/anim/hardware.gif" alt="animation: replayed token positions per machine, mismatch counter stays at 0" width="100%"></p>

An exactness proof in float64 is necessary, not sufficient. Running the same suite on different GPUs found two bugs a simulation could never see: bf16 rounding of Qwen's `1/√128` query scale at 16 query rows,
and cuBLAS split-K adding partials in bf16 on an L4 (traces of 288 and 320 tokens came back wrong; the A100 never showed it). Both are fixed and guarded by tests (see the write-up).

| device | architecture | models | arrays replayed | positions replayed | mismatches | suite runs clean |
|---|---|---|---|---|---|---|
| A100 40GB | Ampere sm80 | 0.8B, 2B, 4B | 9 | 705,024 | **0** | 18/18 |
| A10G 24GB | Ampere sm86 | 0.8B, 2B, 4B | 8 | 273,024 | **0** | 16/16 |
| L4 24GB | Ada sm89 | 0.8B, 2B, 4B | 8 | 211,584 | **0** | 16/16 |
| T4 16GB | Turing sm75 | 0.8B, 2B, 4B | 6 | 148,608 | **0** | 9/12 |
| x86 Xeon (c6i, Ice Lake) | x86-64 | 0.8B, 2B, 4B | 8 | 23,040 | **0** | 14/14 |

The suite is 10 tests per (model, array) for the runs in the table (an 11th, restore on the full network, was added afterwards and passes on the A100, 0.8B 8×8): exact inside Qwen · exact after text · pure-text KL · text after a chip run · every trace length (up to 320 tokens, the lengths that broke the L4) · mutation test (zero the TPU layers and answers must break) ·
real 2-layer MLPs · chip-only path bit-identical · CUDA-graph executor bit-identical · restore invisible. Replay compares every predicted token with NumPy (`bench.py`).
The T4 passes all 0.8B suite runs and the three smallest 2B ones, then runs out of its 16 GB; its replay (0.8B and 2B, all six arrays, 4B on four arrays) is exact, but 4B 32×32 did not fit and has no data, and there is no 4B suite on the T4. On the 24 GB cards the 4B suite was run for four arrays (8×8 u3s3, 8×8 u5s5, 16×16 u3s2, 32×32 u3s3), not all six; 4B replay covers the same four on the A10G and three of them on the L4 (32×32 batch 8 was cut off). The 8×8 u3s3 4B run has 11 tests; the later ones have 12 (the final tests/test_merged.py, with VRAM parking, see below). Raw logs for every machine are in [`results/`](results/).
