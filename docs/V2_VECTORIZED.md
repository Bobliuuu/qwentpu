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

Predicted gain: the emitted loop is `vec` times shorter, but each token passes through more chip layers. The measured
numbers are in the next section.
