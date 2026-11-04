# qwenTPU v2 ISA (v2.0, draft-1)

Goal: a recognizable TPU programming model on top of the v1 datapath, where the **host feeds only the
prompt** (weights + input + program text) and **the model emits everything after it** (every ACT/READ/OUT
token of every matrix op, layer to layer, up to HALT). Reference implementation: `qwentpu/isa.py`
(assembler, token codec, bit-exact numpy interpreter), `qwentpu/program.py` (program container, MLP builder).
All token ids below are *symbolic*; `isa.py` computes real ids from `circuit.configure(...)` (they depend on the array).

## 1. Machine model

* Array: `KR x NC` weight-stationary matvec array, activations unsigned `AB` bits (0..AMAX, AMAX=2^AB-1),
  weights two's complement `WB` bits (-2^(WB-1)..2^(WB-1)-1), exact integer accumulate (no overflow possible).
  Defaults for v2 programs: **16x16, AB=WB=5, g=h=1** (10 chip layers, validated on real models).
  8x8 u5s5 (7 layers) is the cheap fallback; 32x32 is only validated at 3-bit, so don't plan on 32x32 u5s5.
* **Weight banks** `B0..B{NB-1}` (NB = 2 for the MLP example, max 4): NB independent weight arrays (extra
  weight/accumulator heads, selected by bank id). Needed because a layer-2 `LOAD_WEIGHTS` cannot be host-fed
  mid-run. (Fallback if banks are too expensive: NB=1 and the host injects `LOADW`+CELL tokens between layers; this
  breaks "no host per matrix op" and is not the v2 target.)
* Activation vector `x[0..KR-1]`, each 0..AMAX. **Bias convention:** row `KR-1` of every activation vector is the
  constant 1, and row `KR-1` of the weight tile is the bias. A previous layer produces the 1 by itself: its column
  `NC-1` has all-zero weights and bias weight 1, so `relu(0+1)=1`. Hence usable inputs/outputs per layer = `KR-1`.
* Square arrays only for chaining: layer output column `c` feeds next layer row `c` (NC == KR).

## 2. Instructions (architectural semantics)

| mnemonic | operands | semantics |
|---|---|---|
| `LOAD_WEIGHTS b, W` | b: bank, W: KR x NC ints in `[wlo,whi]` | `bank[b] = W` (zero padded). Cells not written stay 0. |
| `LOAD_X x` | x: KR ints in `[0,AMAX]` (x[KR-1]=1) | `acc_in = x` (input activation vector) |
| `MATMUL b` | b: bank | `a[c] = sum_r bank[b][r][c] * v[r]` (int, exact) for all NC columns; `v` is the current vector |
| `RELU` | - | `a = max(a, 0)` |
| `REQUANT s` | s: shift 0..15 | `v = clip(a >> s, 0, AMAX)` (arithmetic shift = floor) ; becomes the current vector |
| `OUT` | - | emit the current vector as the program result (first NC values; result tokens, see below) |
| `HALT` | - | end of program |

Canonical fused form: `MATMUL b ; RELU ; REQUANT 0` is **one machine instruction** in v2.0 (that is what the
datapath's READ stage does: saturating-Horner ReLU + clamp, no shift). `MATMUL.R b` is its assembler alias.
The interpreter implements the general `RELU`/`REQUANT s>0`/no-RELU forms, but `encode()` (token lowering) rejects them
(`ValueError: not supported by v2.0 datapath`): hardware can only do fused relu+clamp, shift 0.
Quantization therefore folds the scale into the integer weights (`w = round(Wr * s_in / s_out)`), not into a shift.
Final layer: the same relu+clamp applies, so logits are in `0..AMAX` (ties possible; argmax = lowest index on ties,
`np.argmax` semantics). Linear (signed) logits are NOT available in v2.0.

Program shape: `LOAD_WEIGHTS b0 ; ... ; LOAD_X ; MATMUL.R b0 ; MATMUL.R b1 ; ... ; OUT ; HALT`.
A `MATMUL.R` followed by another `MATMUL.R` chains (output vector becomes the next input vector, in the model's own
tokens); followed by `OUT` it is the final one.

## 3. Token encoding

Reused unchanged from `circuit.py` (ids from `C.tok_*`): `CELL(r,c,w)`, `ACT(s; v_0..v_{g-1})`, `READ(k)`, `OUT(v_0..v_{h-1})`, `NOP`.
**New tokens** appended after `C.VOCAB` (ids `VBASE = len(C.VOCAB)` upward, order fixed):

| id | token | meaning |
|---|---|---|
| `VBASE + b`, b<NB | `OP_LOADW(b)` | next `KR*NC` tokens are `CELL(r,c,w)` in row-major (r,c) order, all of them, written to bank b |
| `VBASE + NB + b` | `OP_MATMUL(b)` | queue a fused MATMUL.R on bank b |
| `VBASE + 2NB` | `OP_LOADX` | next `KS = KR/g` tokens are `ACT(s; ...)`, s = 0..KS-1 ascending: the input vector |
| `VBASE + 2NB + 1` | `OP_OUT` | the last queued MATMUL emits `OUT` tokens, not `ACT` |
| `VBASE + 2NB + 2` | `OP_HALT` | end of program text; **last prompt token**; also the last emitted token |

Prompt (the ONLY host-fed tokens), exactly:

```
OP_LOADW(b0) CELL*KR*NC  [OP_LOADW(b1) CELL*KR*NC ...]  OP_LOADX ACT*KS  OP_MATMUL(b0) [OP_MATMUL(b1) ...] OP_OUT OP_HALT
```

Model-emitted continuation (autoregressive, argmax; each emitted token is fed back as the next input), for a program
with matmuls on banks `m_0 .. m_{L-1}` and `n = NC/h` read groups:

```
for l in 0..L-1:
  for k in 0..n-1:
      emit READ(k)                 # control: sequencer, bank m_l is implied by the queue position
      emit Y                       # ACT(k; y_{k*h..k*h+h-1})   if l < L-1   (chained: row k of next layer)
                                   # OUT(y_{k*h..k*h+h-1})      if l == L-1   (the result)
emit OP_HALT
```

`ACT(k; digits)` and `OUT(digits)` carry the same base-2^AB digits; with **g == h** the chained token is the OUT id
relabelled by `ACT0 + k*NOUT^G` (the model itself emits the ACT id; no host relabel). **Require g == h** (g=h=1 baseline,
g=h=2 allowed). Total emitted tokens = `2*L*n + 1`. Result of the program = digits of the `n` `OUT` tokens of the last
matmul (`isa.decode_result`). Everything the host does after the prompt: nothing (reads emitted ids, compares to golden).

Control state the datapath must hold (not specified further, design freedom for the compiler agent): program memory
(the queued `OP_MATMUL`/`OP_OUT` list in a DeltaNet row set), a step counter (matmul index l, group k), the active
bank. Weights/accumulators of all banks are written by `OP_LOADW` + CELL during prefill.

## 4. Operand ranges / constraints summary

* `AB, WB` in 1..6 (circuit asserts); recommended 5/5. Weights `[-16,15]`, activations/outputs `[0,31]` at 5 bits.
* `KR/g` power of two, `<= 128`, `KR/g = 64` unsupported. 16x16 (KS=16, bias row 15, 15 usable features).
* Per-layer: `n_in <= KR-1`, `n_out <= KR-1` when another layer follows (column KR-1 is the constant-1 generator).
* NB banks: one matmul = one bank; `NB <= 4`. Same bank may be used twice (recurrent reuse) since banks are not consumed.
* Result tokens: `NC/h` OUT tokens; padding columns are 0.
* A chip run must not leave Qwen text context changed (existing `HybridSession.chip()` guarantee still applies).

## 5. Golden files (`examples/mlp/`)

`golden.npz` holds, per test digit i: `prompt_ids[i]` (host-fed), `expected_emitted_ids[i]` (what the model must
emit), `logits[i]` (the NC output values), `label[i]`, plus the array config. A
conforming model's emitted ids == `expected_emitted_ids[i]` exactly.

Current golden (16x16 u5s5, g=h=1, NB=2): prompt = 535 tokens (2x(1+256) CELL blocks, LOADX+16 ACT, 2 MATMUL, OUT, HALT), emitted
= 65 tokens (2 layers x 16 groups x (READ,Y) + HALT). Network: 15 pooled-pixel features (inputs quantized to 0..12) -> 15 ReLU units
-> 10 logits, integer weights in [-16,15], bias on row 15. Integer test accuracy 0.876 vs float 0.911 on 540 held-out digits
(`examples/mlp/train.py`). No-shift costs accuracy: with `REQUANT` fixed at shift 0 the scale cannot adapt, so activations must stay
small. A per-layer output shift (`REQUANT s`, floor-divide before the clamp) in the datapath would let normal PTQ work; it is
the single most valuable v2.1 datapath feature if the Horner chain can drop low digits (note floor needs the carries from them).
