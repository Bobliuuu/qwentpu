# qwenGPU, qwenCPU, qwenTPU, and TinyTPU

All three "qwen___" projects share one trick: **compile a machine into the weights of a hybrid
Gated-DeltaNet transformer instead of training anything, and let autoregressive decoding be the clock.**
TinyTPU is the real silicon that qwenTPU imitates.

This repo is an independent implementation. It uses qwenputer's *design* (ballast-normalised residual,
DeltaNet state rows as RAM, silu-as-relu gadgets, tokens as the execution trace) but none of its code:
the compiler (`qwentpu/compiler/`), merge, muting and driver were written here, re-deriving the numerics
from Hugging Face's `modeling_qwen3_5.py`.

## 1. The substrate

Qwen3.5 mixes full attention with Gated DeltaNet (GDN) layers. A GDN head keeps a state `S` (dk x dv):

```
S <- alpha * S                         # gate (decay)
S <- S + k (beta * (v - S^T k))^T      # delta rule
o  = S^T q                             # read
```

* **RAM.** `alpha = 1`, `beta = 1`, `k` one-hot: the update overwrites row i with v exactly. A one-hot `q`
  reads it back. `alpha = 0` wipes the head. sigmoid/softplus saturate to exactly 1/0 in fp32.
* **Making bf16 exact.** The HF layer runs in bf16, so every stored or computed value must be a bf16
  number.
  * *Ballast:* a chip token carries 2^40 in `m` residual dims. If `m*B^2/hidden = 4^k`, every RMSNorm divides
    by exactly 2^k, so a residual value x reaches a projection as x*2^-k and the constant 1 is the normed
    ballast. All other values vanish next to it in the fp32 sum of squares (|x| <= 2^28).
  * *State rows* carry two reference columns (2^24) next to data bits (2^7). The per-head gated RMSNorm
    divides by a known multiple of the reference, so reads come back exactly rescaled.
  * *FFN gadgets:* with gate scale 128, `silu(128 n) = 128 relu(n)` exactly for integers n (n >= 1 rounds
    sigmoid to 1, n <= -1 underflows it to 0). Each SwiGLU unit is `coef * a * relu(b)`: AND, compare,
    multiply-by-a-bit, clamp, one-hot decode.
  * `qwentpu/compiler/evaluate.py` runs every circuit in float64 and asserts that bf16 hardware would
    reproduce each intermediate bit for bit; then HF is checked against it.
* **Tokens are the trace.** State only moves between tokens through the DeltaNet states, and a value
  only leaves a forward pass as a sampled token fed back in. One token = one micro-operation.

## 2. How each project uses it

| | qwenCPU (Qwenputer) | qwenGPU | qwenTPU (this repo) |
|---|---|---|---|
| Base | Qwen3.5-4B | Qwen3.5-2B | Qwen3.5-0.8B, 2B and 4B |
| Machine | RV32IM CPU, 11 layers | SIMT GPU, 45 layers: 4-lane warps, 4-stage pipe, 15 regs/lane, 126 words | weight-stationary MAC array, 2 to 7 layers |
| State holds | registers, PC, code, data | registers + global memory | **the weight matrix** |
| Token = | an instruction (`ADDR`,`VAL`,`STEP`) | a warp-launch record | a dataflow step (`CELL`,`ACT`,`READ`) |
| Control | fetch/decode, a PC | host-compiled kernel (PTX) | none: the layer stack *is* the datapath |
| Strength | general: C, MicroPython | runs real CUDA | one thing, matmul + requant, natively |

qwenGPU is described only from its tweet (the thread was not fetchable). Qwenputer's README already has an "NPU" branch (VDOT/SMAQA instructions on the CPU, weights as data) and a
sibling "QwenPU" vector unit (https://github.com/divagr18/QwenPU).

## 3. qwenTPU datapath

```
CELL(r,c,w)  -> head c, row r  <-  w's bits                            weights, stationary in DeltaNet state
                key = AND(row r, col c) = silu(K (row + col - 1))       one-hot for free in conv + silu
ACT(i,v)     -> read row i of every column head (one-hot query)         weight bits W[i][c]
             -> FFN: AB*WB AND-products  act_bit & weight_bit / column  the multipliers
             -> write them into row i of that column's accumulator head partial sums, one row per input
READ(c)      -> multi-hot query on accumulator c: sum of all KR rows    adder tree, one head read
             -> cnt[j][b] -> s_j -> saturating Horner steps -> code     requantiser (ReLU + clamp)
             -> the code FFN writes the next token's code dims          lm_head samples OUT(v)
```

Every vector rewrites all KR accumulator rows, so there is no flush token. Bias is a constant-1 activation row.
Cost: `KR + NC` tokens per matvec once weights are resident, `KR * NC` tokens to load.

### The adder tree

`cnt[j][b]` (rows where activation bit j and weight bit b are both 1) comes back from the accumulator read.
`s_j = sum_b wt_b cnt[j][b]` with `wt_b = 2^b`, top bit `-2^(WB-1)`; `y = sum_j 2^j s_j`. That is a free linear form,
but its range (e.g. -960..840 for 8x8 u4s4) exceeds what a SwiGLU gate can compare exactly (positive inputs <= 256).

Fold it top-down. Let `p` be the weight of `u`'s lsb, so `y = 2^p u + R` with `R in [Rmin(p), Rmax(p)]`,
`Rmin = smin (2^p - 1)`, `Rmax = smax (2^p - 1)` and `smin/smax = -2^(WB-1) KR, (2^(WB-1) - 1) KR`.
If `u >= Uhi(p) = ceil((AMAX - Rmin) / 2^p)` then `y >= AMAX` whatever the lower digits are, and if
`u <= Ulo(p) = floor(-Rmax / 2^p)` then `y <= 0`. So `u` may be saturated to `[Ulo, Uhi]` after every step with the
final `relu + clamp(y)` unchanged (induction: a saturated `u` stays saturated after absorbing the next digit).
At `p = 0` the bounds are `[0, AMAX]`: `u` already is the output.

One step absorbs `q` digits, `Z = 2^q u + sum_i 2^i s_(p-q+i)`, and clamps with two units,
`u = Z + relu(lo - Z) - relu(Z - hi)` (the `Z` term is a free linear form; only the corrections are FFN units, and each
unit's positive input must be <= 256). `circuit.horner_plan()` picks the largest `q` that fits the window and, for the
last step, checks whether the output code can be read straight off the unclamped `Z` (`fits_code`). Then the
code FFN is one layer: per code bit, `[Z >= AMAX]` and `[Z == v]` units, the not-a-read NOP constant, and the
rewrite of the code dim. Windows allow `KR * 2^(WB-1)` up to about 128; beyond that the planner raises.

| array | Horner steps (q) | chip layers |
|---|---|---|
| 8x8 u3s3 | 1 (merged into the code FFN) | 2 |
| 8x8 u4s4 | 2, 1, 1 | 4 |
| 8x8 u5s5 | 1 x 5 | 7 |
| 16x16 u4s4 | 1 x 4 | 5 |
| 32x32 u3s3 | 1 x 3 | 6 |

### A numerics trap: the query scale

Qwen3.5 l2-normalises q and k and scales q by `1/sqrt(128)`. For a multi-hot query with `n` equal entries the
resulting factor is `1/sqrt(128 n)`. All reads in the state are later divided by the (known) reference column, so a
common factor cancels, but only if every `o` value is rounded the same way: `o` is cast to bf16 *before* the gated norm.
If `128 n` is a power of 4 the factor is a power of two and the cast is exact. Otherwise `bf16(c * s)` has a
different relative error for each count `c`, and the count 7 can come back as 6.96875 (observed on an A100 at KR=16, 3 of 6,144 outputs,
invisible in float64). The fix pads the accumulator query with `KR` extra rows that are never written (their state is
zero, they only change the l2 norm) whenever `log2 KR` is even. One-hot reads (the weight heads) are immune since
every value they return has a single set bit. `evaluate.py` now asserts `128 n` is a power of 4 for every multi-hot query.

TinyTPU mapping:

| TinyTPU (SystemVerilog) | qwenTPU |
|---|---|
| PE `psum + x*w`, weight held in the PE | head c, row r holds w; FFN multiplies act bits by weight bits |
| systolic array, inputs staggered in time | one `ACT` token per element, serial in token time |
| vertical partial-sum flow | per-row accumulator writes + one multi-hot read |
| VPU: bias, ReLU | constant-1 row, saturating-Horner ReLU/clamp |
| unified buffer / 94-bit ISA / control unit | token vocabulary is the ISA; the driver is the control unit |
| on-chip inference **and training** | inference only |

### Do we need TinyTPU?

No. It supplies the architecture to imitate (weight-stationary PEs, streamed activations, integer accumulate, a
post-processing unit) and nothing else: no code, RTL, netlist or test is shared. The honest delta: this array is not
systolic in space. A systolic array forwards operands PE to PE with a skew; here every column head sees the same
`ACT` token and the rows are serialised by token time, with the reduction done by one multi-hot read instead of a
column of adders. "TPU" here means "weight-stationary dataflow matrix unit with a requantising output stage", which is
the claim to keep. A future conformance check against TinyTPU's XOR/MLP examples would be a fair way to tie them together;
it is not done.

## 4. What is new here (and what is not)

**Not new:** hardware compiled into Qwen's weights, and neural networks running on it. Qwenputer has a CPU;
its NPU branch runs stories260K. **QwenPU** (same author) adds a vector unit inside Qwen3.5-4B: 64-lane
int8 x int4 matrix-vector rows streamed through tokens, weights and caches held in DeltaNet state memories,
12-13 compiled layers, a GPU executor. Its matvec is much faster per token than this repo's 8x8 array
(64 lanes/token at the op level versus 4 to 16 MACs/token here). Don't read the 53k tokens-per-story-token
figure against my per-matvec numbers: they are end-to-end versus op-level.

**New, to my knowledge (one search, not exhaustive): the architecture.** QwenPU is an instruction-driven
vector processor, a CPU with a coprocessor. qwenTPU is a TPU: no program counter, no fetch/decode, no
instruction stream.

1. **Weight-stationary dataflow.** Weights are loaded once into DeltaNet state and stay; activations stream
   through; the layer stack *is* the datapath. This is the TPU idea, not a CPU idea.
2. **The DeltaNet read as a reduction.** One multi-hot query returns the exact sum of the selected state
   rows, so the contraction over the inner dimension is a single head read.
3. **Partial sums live in state too.** Stationary-weight heads plus per-input-row accumulator heads;
   every vector rewrites all rows, so nothing is flushed.
4. **Bit-plane counting.** Products are bit ANDs and accumulators hold *counts*, so every stored and read
   value is a small integer that bf16 represents exactly; the weighted sum is a free linear form.
5. **Layers chain through the model's own tokens**, with ReLU and requantisation inside the model and no
   host arithmetic.
6. **Cost model.** `KR+NC` tokens for `KR*NC` MACs once weights are resident: O(N) tokens for O(N^2)
   MACs. That is a scaling argument. It does not beat QwenPU today.

Also here: an independent compiler with its own bf16 exactness evaluator, `tau`, and a cached driver.
Not claimed: first accelerator in a model's weights, first NN in weights, higher throughput than QwenPU.

## 5. Merging into real Qwen3.5 (`merge.py`, `mute.py`)

1. **Widen** every matrix touching the residual to `hidden = m * 4^j` (0.8B: 1024 -> 1280), zero-padded;
   RMSNorm gains over the wider residual are rescaled by sqrt(d0/hidden).
2. **Append** the 2 to 7 TPU layers (2 for 8x8 u3s3) after Qwen's 24/24/32.
3. **Tokens:** 593 new embedding rows at 8x8 (CELL, ACT, READ, OUT, NOP), and an *untied* `lm_head`.
4. **Final norm:** gain 0 on every chip dim except the output code.
5. **tau:** a residual dim holding 2^-40 on text tokens; the final norm amplifies it 2^60 and chip rows of
   the lm_head read it with weight -1, so chip tokens score about -10^6 at text positions: never sampled in
   prose. At chip positions it is 0.
6. **Muting:** on a chip token every ballast RMSNorm output is the same constant, and 0 on text, so a
   weight on one ballast dim acts on chip tokens only, bit for bit on text.
   * GDN layers: `in_proj_a = -1024` (softplus -> 0, no decay), `in_proj_b = -1024` (beta 0, no write),
     `in_proj_z = -1024` (silu -> -0, output exactly 0).
   * Full-attention layers: the output-gate half of `q_proj` = -1024 (sigmoid 0); `k_proj` makes chip
     positions an *anti-sink* key opposite to the mean text query so they take no softmax mass.
   Qwen's state therefore passes through a chip run untouched. What remains is the conv1d window: the
   first 3 text tokens after a chip run see zero inputs in it.

## 6. Results

See the README for the full tables. Summary (A100, bf16, stock `Qwen3_5ForCausalLM`; checkpoints 0.8B / 2B / 4B):

* **Exactness:** 393,984 replayed outputs, 0 mismatches against NumPy, six arrays from 8x8 u3s3 up to 8x8 u5s5 and 32x32 u3s3,
  on all three model sizes; the 7-test model suite passes for all six arrays on 0.8B and for three arrays on 2B and 4B.
* **Text:** pure-text KL vs base 2.5e-4 to 8.2e-4 (the widening noise floor). Chip tokens never sampled in prose.
* **Invisibility:** `HybridSession` restores Qwen's own cache around a chip run: text after a TPU run matches the
  no-run logits at every position (max KL 0.0017 to 0.0047, first token included).
* **Scaling:** 64, 256 and 1,024 MACs per matvec at 16, 32 and 64 tokens: 4, 8, 16 MACs/token.
* **Speed:** up to 45k tok/s (0.8B, 16x16 u3s2, batch 16) and 12k to 14k on 4B. Teacher-forced prefill numbers; a matvec's
  tokens do not depend on each other's outputs, only the hop between layers does.

## 7. Honest limits

* Window: `KR * 2^(WB-1)` must stay near 128 (16x16 u5s5 and 32x32 u4s4 are rejected by the planner). A second adder-tree level
  with digit-wise requantisation would lift it; not built.
* Muting alone leaves the first ~3 text tokens after a chip run perturbed (Qwen's conv window; KL ~1.2). The driver's
  cache restore (`HybridSession`) removes it, but it is a host-side cache operation, not a weight-level fix.
* Text-only checkpoints (vision tower dropped). Inference only; no backprop (TinyTPU's other half).
* The driver streams tokens and relabels an output as the next activation; it does no arithmetic.
* GPU numbers use HF + flash-linear-attention, batched, on a shared A100. There is no fused custom executor, so a
  free-running single stream is not faster than the batch-1 numbers.
* The tweets are on X and the threads were not fetchable; qwenGPU details come from tweet text.

## References
* Qwenputer: https://github.com/divagr18/qwenputer (design reference; tweet x.com/divagr1925/status/2115394340357116169)
* qwenGPU: x.com/TheVixhal/status/2116938733815738516
* TinyTPU: https://github.com/tiny-tpu-v2/tiny-tpu (tweet x.com/suryasure05/status/1957518913648095376)
* HF `modeling_qwen3_5.py` (the semantics everything is derived from)
