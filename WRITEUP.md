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
unit's positive input must be <= 256). If `Z`'s range sticks out of `[lo, hi]` by more than 256 the clamp is done in
several **lossless passes**, each saturating to a partial bound at most 256 beyond the current range edge (clamping to
`[lo-256, hi+256]` and then to `[lo, hi]` equals clamping to `[lo, hi]`); every pass is one FFN level. `circuit.horner_plan()`
picks the largest `q` that needs at most one pass (else one digit and as many passes as it takes) and, for the last step,
checks whether the output code can be read straight off the unclamped `Z` (`fits_code`). The code FFN is then one layer: per
code bit, `[Z >= AMAX]` and `[Z == v]` units, the not-a-read NOP constant, and the rewrite of the code dim.
There is no longer a width limit, only a depth cost: 16x16 u5s5 and 32x32 u4s4 take 10 chip layers, 64x64 u3s3 takes 12.

| array | Horner steps (q) | chip layers |
|---|---|---|
| 8x8 u3s3 | 1 (merged into the code FFN) | 2 |
| 8x8 u4s4 | 2, 1, 1 | 4 |
| 8x8 u5s5 | 1 x 5 | 7 |
| 16x16 u4s4 | 1 x 4 | 5 |
| 32x32 u3s3 | 1 x 3 | 6 |
| 16x16 u5s5 | 1 x 5, passes 1,2,2,2,1 | 10 |
| 32x32 u4s4 | 1 x 4, passes 1,2,2,1 | 10 |

### A numerics trap: the query scale

Qwen3.5 l2-normalises q and k and scales q by `1/sqrt(128)`. For a multi-hot query with `n` equal entries the
resulting factor is `1/sqrt(128 n)`. All reads in the state are later divided by the (known) reference column, so a
common factor cancels, but only if every `o` value is rounded the same way: `o` is cast to bf16 *before* the gated norm.
If `128 n` is a power of 4 the factor is a power of two and the cast is exact. Otherwise `bf16(c * s)` has a
different relative error for each count `c`, and some counts come back one rounding step off (observed on an A100 at KR=16, invisible in float64). The fix pads the accumulator query with `KR` extra rows that are never written (their state is
zero, they only change the l2 norm) whenever `log2 KR` is even. One-hot reads (the weight heads) are immune since
every value they return has a single set bit. `evaluate.py` now asserts `128 n` is a power of 4 for every multi-hot query.

### A second numerics trap: bf16 split-K

cuBLAS may split a GEMM's K dimension across thread blocks and add the partials *in bf16* when
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` is on (the default). Whether it does depends on the
matrix shape and on the GPU. On an L4, traces of 288 and 320 tokens came back wrong; the same code never failed on an A100 or the CPU.
Switching the flag off fixed every length. `merge.instantiate` now turns it off for all models it builds, and
`test_exact_for_every_trace_length` guards it. Lesson: an exactness proof in float64 is necessary, not sufficient;
run on more than one GPU.

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

**New: the architecture.** QwenPU is an instruction-driven
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

## 5b. Optimizations

### Optimization 1: tokens are the bottleneck → pack them

<p align="center"><img src="docs/packing.svg" alt="token packing" width="100%"></p>

One `ACT` token can carry **g** activations, and one `READ` token can read **h** columns and emit **h** outputs in a single `OUT` token.
Every column gets `g` copies of its weight head and `g` accumulator heads; each output slot gets its own adder chain.
Tokens per matvec fall from `KR + NC` to `KR/g + NC/h`.

| array | tokens / matvec | MACs / token | chip layers |
|---|---|---|---|
| 8×8 | 16 | 4 | 2 |
| 8×8 `g2h2` | **8** | **8** | **2** *(still!)* |
| 8×8 `g4h4` | 4 | 16 | 4 |
| 16×16 u3s2 `g4h4` | 8 | **32** | 8 |
| 32×32 `g4h4` | 16 | **64** | 18 |

Honest result: packing is **not** a free lunch. It trades tokens for heads, layers and vocabulary (`g4h4` at 32×32 has a 45,000-token vocabulary).
The sweet spot we measured is small and cheap: **`g2h2` on 8×8 is +47% MACs/s on an A100 (batch 16) and +48% on a Xeon (batch 1)**
and costs *zero* extra layers. On bigger arrays it loses (numbers under *What we observed*).

### Optimization 2: skip the layers that do nothing

<p align="center"><img src="docs/chip-path.svg" alt="chip-only path" width="100%"></p>

Qwen's own layers are **muted** for chip tokens: their mixer output is exactly 0 and their MLP sees a zero vector, so each one is the *identity* on the chip dims.
So `ChipPath` runs `embed → only the TPU layers → norm → next token` and skips the other 24-32 layers. **Bit-identical** (it's a test), and it never touches
Qwen's cache, so a TPU run leaves no trace in Qwen's context *by construction*, no restore needed.

| 0.8B · 8×8 · tokens/s | full network | **TPU layers only** |
|---|---|---|
| A100, batch 1 (same trace, 1,728 positions) | 6,629 | **74,462** (11×) |
| Xeon, batch 1 (same replay) | 171.5 | **2,128** (12×) |
| Xeon, free-running matvec/s | 1.59 | **25.47** (16×) |

*The A100 rows are one trace (1,728 positions, `results/a100/ab_same_trace.jsonl`); the Xeon rows are the same trace both ways.*

### Optimization 3: read the answer off the sign bits

The next token is spelled in the sign of ~14 "code" dims. We read those directly instead of multiplying by a chip-vocabulary `lm_head`
(`hidden × |vocab|` MACs per token, more than the entire TPU once the vocabulary is big). Same argmax, and a test checks the readouts agree (`full == lm_head == code`). We did not keep a timing for this one, so no speedup is claimed.

### Optimization 4: CUDA graphs

A free-running chain (layer *n+1* needs layer *n*'s token) is bound by Python and kernel-launch overhead, not arithmetic. HF's linear-attention cache
already copies into fixed buffers, so once the states exist every chunk of a given length is static. We capture one CUDA graph per chunk length
and replay it: **one launch per chunk**, state carried between graphs exactly as between eager calls (bit-identical, tested).

| A100 · 0.8B · free-running · TPU layers + CUDA graphs | matvec / s |
|---|---|
| 8×8 u3s3 | **705** |
| 8×8 `g2h2` | 684 |
| 16×16 u3s2 | 629 |
| 16×16 u4s4 | 281 |
| 32×32 u3s3 | 215 |
| 16×16 u3s2 `g4h4` | 205 |
| 32×32 u3s3 `g4h4` | 65 |

Every row matched NumPy. An eager A100 baseline for this loop was not saved, so we do not quote a speedup; on the Xeon the same loop is 1.59 matvec/s on the full network and 25.47 with TPU layers only.

## 6. Results

See the README for the tables. Everything below is read from the raw logs in `results/` (replays `*.jsonl`, pytest outcomes `*.suite`).

* **Exactness:** 1,361,280 replayed outputs, 0 mismatches against NumPy, on five machines (A100, A10G, L4, T4 and a Xeon), across 0.8B, 2B and 4B.
  Suite runs clean: A100 18 of 18, A10G 16 of 16, L4 16 of 16, Xeon 14 of 14, T4 9 of 12 (the three failures are out-of-memory errors in 2B runs on its 16 GB; the replay there still ran).
  A passing suite is 10 tests on the A100, the T4 and the 0.8B and 2B runs on the 24 GB cards, 11 or 12 for the 4B runs on those cards (tests were added later), and 7 on the Xeon (an earlier suite).
* **Text:** pure-text KL vs base is asserted below 5e-3 and chip tokens below -1e3 logit at text positions. `results/a100/kl.out` records the values: 3.2e-4 to 7.8e-4 across three prompts on 0.8B, 2B and 4B, with argmax agreement 1.00 on all but one 4B prompt (0.95).
* **Invisibility:** the chip-only path never touches Qwen's cache and is asserted bit-identical (KL = 0.0). With muting alone, the per-position KL after a chip run is 1.1 to 1.3 at the first text token and a mean of 0.014 to 0.059 from the fourth on (`kl.out`). `HybridSession`'s cache restore brings it to 0.0 against a run with no chip activity (asserted below 1e-4).
* **Scaling:** 64, 256 and 1,024 MACs per matvec at 16, 32 and 64 tokens: 4, 8, 16 MACs/token.

### What we observed

* **Devices.** Same 8×8 u3s3 0.8B replay, same trace, full network: batch 1 is 2,410 tok/s on an L4, 2,147 on an A10G, 2,036 on an A100, 145 on the Xeon; batch 8 is 19,477 / 15,519 / 16,228 on the three GPUs. The cheaper GPUs were not slower at this size.
* **Packing.** On an A100 at batch 16, 8×8 `g2h2` reaches 2.54M MACs/s against 1.73M unpacked (+47%) with no extra layers; at batch 1 it is +1%. `g4h4` on 8×8 is +24% at batch 16 for 2 extra layers and a 12,803-token vocabulary. 16×16 u3s2 loses (-15% `g2h2`, -36% `g4h4` at batch 16) and so does 32×32 at batch 1 (-27%, -56%). On the Xeon (batch 1): 8×8 `g2h2` +48%, 16×16 `g2h2` -30%. Packing is a cost trade, not a free lunch.
* **Depth vs width.** A100, batch 16, 16×16: u3s2 (2 chip layers) 3.05M MACs/s, u4s4 (5 layers) 1.35M, u5s5 (10 layers) 0.65M.
* **Chip-only path.** 8×8, 0.8B, batch 1: 6,629 to 74,462 tok/s on the A100 (11×, same trace) and 171.5 to 2,128 on the Xeon (12×, same trace). Free-running on the Xeon: 1.59 to 25.47 matvec/s.
* **CUDA graphs.** A100 free-running chained matvecs, TPU layers only: 705 matvec/s on 8×8 u3s3, 684 on 8×8 `g2h2`, 629 on 16×16 u3s2, 281 on 16×16 u4s4, 215 on 32×32 u3s3, 205 on 16×16 u3s2 `g4h4`, 65 on 32×32 `g4h4`; all matched NumPy. No eager A100 baseline for this loop was saved, so no speedup is quoted.
* **torch.compile (tried and rejected).** `results/a100/compile_experiments.jsonl`, 8×8 chip-only trace, 20 cases: eager 74,630 tok/s, CUDA graph 351,039; `compile:default` 96,190 and `compile:max-autotune-no-cudagraphs` 107,703 keep the hidden state bit-identical but are far behind the graph; `compile:reduce-overhead` (97,997) and `compile:max-autotune` (98,095) change the hidden state. CUDA graphs stay the fast path (the experiment script, `exp_fuse.py`, is in the git history: `git log -- exp_fuse.py`).
* **Batching and model size.** 8×8 TPU layers on the A100: 107k tok/s at batch 1, 487k at batch 16. Full-network 8×8 u3s3 at batch 16: 33,074 (0.8B), 28,401 (2B), 13,773 (4B).
* **Memory.** The first fixtures held base, merged and mutated models at once and ran the A10G and L4 (24 GB) and the 40 GB A100 out of memory in the mutation, XOR/MLP and restore tests. The base model is now rebuilt on demand and freed after each forward pass, so 2B fits 24 GB GPUs and 16 GB hosts. Recorded 2B suite passes on the 24 GB cards: all six arrays on the A10G and on the L4.
* **Two bugs only a second GPU could find:** the irrational query scale and cuBLAS bf16 split-K (both in section 3).

## 7. Honest limits

* Depth: wide operands cost chip layers (multi-pass clamps): 16×16 u5s5 and 32×32 u4s4 are 10 layers, 64×64 u3s3 is 12 (run on the A100 on 0.8B, 2B and 4B: 13,824 positions at batch 1 and 110,592 at batch 8 per model, 0 mismatches; `results/a100/a100.jsonl`).
* Muting alone leaves the first ~3 text tokens after a chip run perturbed (Qwen's conv window). The driver's cache restore (`HybridSession`) removes it, but it is a host-side cache operation, not a weight-level fix.
* Text-only checkpoints (vision tower dropped). Inference only; no backprop (TinyTPU's other half).
* The driver streams tokens and relabels an output as the next activation; it does no arithmetic.
* GPU numbers use HF + flash-linear-attention. There is no fused custom executor: CUDA graphs remove launch overhead, not the cost of the HF kernels.
* Teacher-forced numbers are prefill throughput: a matvec's tokens do not depend on each other, only the hop between layers does.

## 8. v2: a TPU that runs a program

v1 is a datapath that the host drives: it picks every CELL, ACT and READ token and relabels each OUT as the next layer's ACT. v2 keeps that datapath and moves the sequencing into the model. Files: `qwentpu/isa.py`, `program.py`, `sequencer.py`, `sequencer2.py`, `compiler/nonlin.py`; specs in `docs/V2_ISA.md` and `docs/V2_SEQUENCER.md`; diagrams in `docs/v2-*.svg` (`docs/tools/gen_svgs_v2.py`); runnable example `examples/mlp/run_v2.py`.

![v2 architecture](docs/v2-arch.svg)

### Design

* **The host builds a prompt and runs `argmax; append` until HALT.** The prompt is data only: the weights as `CELL` tokens, the input as `ACT` tokens, and nothing else (no READ, no OUT, no op token). After it, every `READ(k)`, `ACT(row; y)` and `OUTK(j; y)` token is the model's own argmax. The host does not know the layer count, widths, or which row or column comes next.
* **Nonlinearity in the datapath.** The adder tree already ends in a saturating requantiser. Thresholding its result gives `clip(relu(sum), 0, AMAX)` for free, and the same thresholds write the value into the id bits of the next token. A layer's output therefore is the next layer's `ACT` token, at zero extra chip layers.
* **Self-sequencing: the program counter is the current token.** A token-only successor table (a token feature baked into the embedding, a ROM indexed by the current token) gives `ACT(s;.) -> READ(s+1-n)`, `OUTK(j;.) -> READ | HALT`, and for `READ(k)` the next token is `base(k) + y`. Layers chain block-diagonally in one array (layer b owns rows and columns `[b n, (b+1) n)`), so there are no weight banks and no mid-run reload.

![token timeline](docs/v2-token-timeline.svg)

### The two sequencer variants, merged into one feedback path

Two designs were built independently: `nonlin.feedback_code` (a single layer's READ emits the next ACT token) and `Sequencer` (the full chain with a successor table). They are the same mechanism. `Sequencer` with the one-layer feedback schedule (`READ(k) -> ACT(k; y)` and nothing else) emits the same next token as `feedback_code` at every position and compiles to the same number of chip layers; `tests/test_sequencer2.py::test_one_feedback_path_sequencer_equals_nonlin_feedback_code` asserts it. The repo therefore has one feedback path, and `nonlin.feedback_code` remains as the reference it is checked against.

![relu feedback](docs/v2-relu-feedback.svg)

### Program in the prompt

`ProgramMachine` (subclass of `Sequencer`, no new circuit) makes depth a prompt choice. The prompt writes layer weights into blocks `Dmax-d .. Dmax-1` and the other blocks stay 0. The input is written into the entry block `s = Dmax-d`; its last token `ACT(s n + n-1)` has the compiled successor `READ(s n)`, so the model walks blocks `s .. Dmax-1` and halts. Width is a zero-padding choice (zero columns requantise to 0). Weights, biases, depth, widths and input are prompt data. Cost scales with the program: `2*d*n` emitted tokens, not `2*Dmax*n`.

![program in prompt](docs/v2-program-in-prompt.svg)

![fixed vs prompt](docs/v2-fixed-vs-prompt.svg)

### Results (Qwen3.5-0.8B, A100, bf16; `results/a100/v2_prog.txt`, `v2_feedback.txt`)

| test | result |
|---|---|
| one compiled model (Dmax 4, 4-wide, u3s3, 3 chip layers), depths 1 to 4 chosen by the prompt | 24 of 24 programs exact (emitted lengths 8, 16, 24, 32) |
| Dmax 2, 8-wide, u3s3 | 12 of 12 exact |
| Dmax 2, 16-wide, u5s5, 20 chip layers | 12 of 12 exact (emitted lengths 32, 64) |
| golden digit MLP (15 to 15 to 10) as a depth-2 program | logits equal the integer golden on 20 of 20 digits |
| the same model, hidden layer alone as a depth-1 program | 10 of 10 exact |
| ReLU feedback token, real model (8, 8, 5, 5) | 128 of 128 feedback tokens exact, 21 distinct values |

The CPU path is checked too: `python examples/mlp/run_v2.py` runs the golden MLP through a numpy simulator of the compiled model, and `--ir N` through the float64 circuit IR (all exactness conditions asserted); both give the golden logits.

The `.txt` logs also contain two pytest summary lines with failures (`4 failed, 1 passed, 36 deselected` in `v2_prog.txt`, `2 failed, 17 passed` in `v2_feedback.txt`) that name no tests. They are recorded as they are, are not cited above, and have not been attributed to a cause in this write-up.

### Accuracy: 87.6% vs 91.1%

The classifier scores 87.6% on 540 held-out digits as an integer network (the one the model runs), against 91.1% for a float MLP of the same shape on the same 15 pooled features. The model reproduces the integer network exactly, so the 3.5 point gap is not hardware error. It comes from the arithmetic the datapath imposes: weights are 5-bit integers, inputs are quantised to 0..12, and there is no requantisation shift (the scale is fixed, so every product must stay inside the 0..31 activation window). `examples/mlp/train.py` trains straight through that arithmetic (round and clamp with straight-through gradients) and picks the seed by training accuracy; an input range sweep of 3 to 31 gave 83.5% to 88.9% test accuracy.

### Vectorized v2: several neurons per token

The circuit's `g` and `h` packing (section 5b, optimization 1) carries over to the sequenced chain. With `vec = g = h` an `ACT` token carries `vec`
activations and a `READ` / `OUTK` token `vec` outputs, so a chain of `depth` blocks of width `n` emits `2 depth n / vec` tokens. The chain is the
same chain written in token slots (`Schedule.chain(depth, n / vec)`); the feedback code is the v1 code stage, because the circuit pads `ACT0` to a
multiple of `NOUT^vec` and the next-token id is `base + digits` with no carry (design: `docs/V2_VECTORIZED.md`).

Measured on Qwen3.5-0.8B (A100, bf16; `results/a100/v2_vec.txt`): every program is exact for vec 1, 2 and 4 on 3-bit arrays and for vec 1 and 2 on
5-bit arrays, 0 mismatches, both readouts, including the digit classifier at vec 2 (20/20). The emitted-token loop of the 16-wide 5-bit classifier
falls from 2,184 ms to 1,106 ms on the eager chip path (1.97x, 1.92x per inference) and from 425 ms to 351 ms with CUDA graphs (1.21x). Each token
costs more (24 chip layers instead of 20, hidden 4864 instead of 3072), so the gain is below `vec`. On 3-bit arrays the eager loop gains 1.2x to 1.45x
and the CUDA-graph loop loses (0.90x at vec 2, 0.66x at vec 4): at 5-bit, vec 4 would need 2^20 ids per slot and is not built. This matches section 5b:
packing trades tokens for heads, layers and vocabulary.

### Honest limits of v2

* **Chain-only topology.** Layer b feeds layer b+1. No skip connections, no fan-out, no branches, loops or early exit.
* **No signed logits.** Every output is relu-clamped to 0..31, so negative scores read as 0 and ties between equal outputs are possible.
* **The schedule is compiled in.** Array size, number format, the READ, value, READ cycle and the successor table are weights. Only depth (through the entry block), widths, weights, biases and input are prompt data. A prompt-loaded instruction memory (design B in `docs/V2_SEQUENCER.md`) is designed, not built.
* **Square block layout.** `Dmax * n` must fit the array: the 20-chip-layer 16-wide model is a 32 by 32 array at 5-bit. Usable features per layer are `n - 1` (one row carries the bias constant).
* **The host still builds the prompt** and runs the argmax loop; the v1 `HybridSession` text path is not part of v2.
* **GPU evidence is one machine.** The v2 results are from Qwen3.5-0.8B on one A100. The 2B and 4B models and the other GPUs have not been run for v2.

## References
* Qwenputer: https://github.com/divagr18/qwenputer (design reference; tweet x.com/divagr1925/status/2115394340357116169)
* qwenGPU: x.com/TheVixhal/status/2116938733815738516
* TinyTPU: https://github.com/tiny-tpu-v2/tiny-tpu (tweet x.com/suryasure05/status/1957518913648095376)
* HF `modeling_qwen3_5.py` (the semantics everything is derived from)
