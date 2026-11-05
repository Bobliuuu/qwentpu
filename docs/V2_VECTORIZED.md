# v2 vectorized: several neurons per token

The v2 sequencer (`docs/V2_SEQUENCER.md`) emits two tokens per neuron: `READ(k)` (read column k) and the `ACT(row; y)` that
the model itself spells from the value it just computed. A chain of `depth` layers of width `n` therefore takes
`2 * depth * n` emitted tokens, and every emitted token is one full forward pass through the chip layers.

This document describes the vectorized chain: **`vec` neurons per token**, `2 * depth * n / vec` emitted tokens.

## What can carry several rows

The circuit already has the packing (`qwentpu/circuit.py`, `configure(kr, nc, ab, wb, g, h)`):

| token | scalar (g = h = 1) | packed (g = h = vec) |
|---|---|---|
| `ACT(s; y_0..y_{g-1})` | one activation, row s | `g` activations, rows `s*g .. s*g+g-1` |
| `READ(k)` | one column | `h` columns `k*h .. k*h+h-1`, one carry-save adder tree per column |
| `OUTK(j; y_0..y_{h-1})` | one output | `h` outputs |

v1 used this packing for the host-driven matvec. The v2 sequencer asserted `g = h = 1`. The vectorized sequencer sets
`g = h = vec` and keeps the same chain, written in token slots instead of rows: block b owns slots
`[b n/vec, (b+1) n/vec)`, `READ(k)` of block b feeds the `ACT` slot `k + n/vec`, and the successor table is
`Schedule.chain(depth, n / vec)`. Nothing else about the schedule, the block-diagonal packing, the program-in-prompt
mechanism (`ProgramMachine`: depth chosen by the entry block, widths by zero padding) or the host loop changes.
`vec` must divide the width.

### The feedback code

After `READ(k)` the next token must spell `ACT(slot; y_0..y_{vec-1})`, whose id is

    id = ACT0 + slot * NOUT^vec + sum_t y_t * NOUT^t .

Bits `t*AB + a` of the id (the low `vec*AB` bits) are bit `a` of the requantised value of output slot `t`; the higher bits
are the slot, a token feature. The scalar sequencer needs carry terms because `ACT0 mod NOUT` is not 0. For `vec > 1`
`circuit.configure(..., align_act=True)` pads the CELL range so that `ACT0` (and the `OUTK` base) is a multiple of
`NOUT^vec`; the next-token id is then `base + digits` with no carry and the code stage is the v1 one: per bit, the
`terms_ge / terms_eq` indicators of slot t's adder tree gated by `is_read`, plus the token feature for the high bits.
No extra layer for the feedback itself; `vec = 1` is the unchanged scalar path.

## Exactness

* **Arithmetic.** Each column's sum is the same exact bf16 count-and-Horner computation as in the scalar array (the same
  `horner_plan`, clamp bounds and 5-bit saturation); the `g` copies of a column's heads hold disjoint rows and their
  counts add to the same total over the `KR` rows. Nothing about the numeric range changes with `vec`.
* **Query normalisation.** `KR / g` must be a power of two (the reference-column normalisation); `configure` asserts it.
* **Checked.** The circuit IR evaluator (float64, all exactness conditions asserted) reproduces every token of the
  chain for `vec = 2` and `4`; the real model is compared token by token with the numpy golden (below).

## Costs and where it stops

| | scalar | vec |
|---|---|---|
| emitted tokens | `2 depth n` | `2 depth n / vec` |
| weight / accumulator heads | `NC` per layer-group | `g * NC` (copy j holds rows `j mod g`) |
| chip layers | `L` | grows with `vec` (16 key heads per layer) |
| `OUT` ids, `ACT` ids | `NOUT`, `KS * NOUT` | `NOUT^vec`, `KS * NOUT^vec` |

The vocabulary is the hard limit: `NOUT^vec` grows exponentially. At 3-bit activations `vec = 4` is 4096 ids per slot
(vocab 24 577 to 36 865, fine); at 5-bit activations `vec = 4` is 2^20 ids per slot (vocab 9.4 M to 15.7 M, not buildable),
so 5-bit stops at `vec = 2` (vocab 11 265 to 60 417).

## Measured (Qwen3.5-0.8B, A100, bf16, host loop = `argmax; append`)

Source: `results/a100/v2_vec.jsonl` and `results/a100/v2_vec.txt` (`bench_vec.py`, `results/summarize_vec.py`).

**Exactness.** One compiled model per (array, vec); random programs of every depth 1..Dmax, both readouts (code dims and
lm_head); emitted tokens equal to the numpy golden and logits equal to the plain network. 24/24 (4 blocks of 4, 3-bit,
vec 1, 2, 4), 12/12 (2 blocks of 8, 3-bit, vec 1, 2, 4), 12/12 (2 blocks of 8, 5-bit, vec 1, 2) and 32/32 (2 blocks of 16,
5-bit, vec 1, 2) including the 15-15-10 digit classifier as a depth-2 program, 20/20 at vec 1 and at vec 2. 0 mismatches.

**Speed.** The deepest program, the same input every time, 3 warm-up and 20 timed inferences, all 20 exact. `loop` is the
emitted-token loop (one forward pass per token); `graph` is the CUDA-graph chip path (`GraphedChip`) and its per-inference
time includes capturing the graph for each new session, so compare the loop columns.

| array | vec | chip layers | vocab | emitted tokens | eager loop | eager per inference | graph loop |
|---|---|---|---|---|---|---|---|
| 4x4 blocks x4, u3s3 | 1 | 3 | 2,234 | 32 | 150.6 ms | 159.4 ms | 39.2 ms |
| | 2 | 5 | 2,881 | 16 | 124.9 ms (x1.21) | 139.3 ms (x1.14) | 43.4 ms (x0.90) |
| | 4 | 9 | 32,769 | 8 | 104.0 ms (x1.45) | 127.4 ms (x1.25) | 59.3 ms (x0.66) |
| 8x8 blocks x2, u3s3 | 1 | 3 | 2,266 | 32 | 152.6 ms | 161.5 ms | 39.6 ms |
| | 2 | 5 | 3,009 | 16 | 119.4 ms (x1.28) | 133.1 ms (x1.21) | 42.6 ms (x0.93) |
| | 4 | 9 | 36,865 | 8 | 107.0 ms (x1.43) | 131.2 ms (x1.23) | 60.0 ms (x0.66) |
| 8x8 blocks x2, u5s5 | 1 | 10 | 9,026 | 32 | 474.9 ms | 501.8 ms | 118.1 ms |
| | 2 | 12 | 23,553 | 16 | 283.5 ms (x1.67) | 316.3 ms (x1.59) | 104.5 ms (x1.13) |
| **16x16 blocks x2, u5s5** (digit MLP) | 1 | 20 | 34,370 | 64 | 2,183.6 ms | 2,248.4 ms | 425.2 ms |
| | 2 | 24 | 60,417 | 32 | 1,106.0 ms (x1.97) | 1,170.5 ms (x1.92) | 350.8 ms (x1.21) |

Reading it:

* The emitted-token count falls by exactly `vec`. The cost of one token rises, because the chip has more layers and a
  wider hidden state (`g` copies of every head): 3 to 5 to 9 layers at 3-bit, 20 to 24 layers at 5-bit and 16x16.
* The eager chip path is launch-bound (cost grows with the number of kernels, not with their size), so fewer tokens win
  clearly: x1.2 to x1.45 at 3-bit, x1.6 to x1.97 on the bigger 5-bit arrays.
* The CUDA-graph path has no launch cost, so a token is as expensive as the layers and hidden size it runs. There
  vectorization pays only where the circuit is already deep (5-bit: x1.13 and x1.21 on the loop) and loses at 3-bit
  (x0.90 to x0.66): the vec-4 chip has 3 times the layers and 1.7 times the hidden size of the scalar one for a quarter of the tokens.
* The vocabulary limits `vec`: `NOUT^vec` ids per slot. 5-bit at vec 4 would be 9.4 to 15.7 million tokens and is not built.
* Prompt processing is unchanged (the weights are the same `KR x NC` CELL tokens), so it caps the per-inference gain.
