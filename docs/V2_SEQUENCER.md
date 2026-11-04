# v2 sequencer: the model emits every chip token

Question: after the prompt (weights + input), how does the compiled network pick the next chip token with no host logic?

## 1. Feasibility

The model's next token is read off the `code` dims (sign bits of the token id) at every position. In v1 those dims
are `OUT(v)` at a READ and `NOP` elsewhere. They are an ordinary function of the **current token and the DeltaNet
state**, so nothing stops them from spelling a useful next token. Two ingredients, both already in the circuit:

* a **token-only successor table** costs nothing: a token feature (a lambda of the token id, baked into the
  embedding) that holds bit *i* of `succ(token)`. This is a ROM indexed by the current token.
* a **state-dependent part**: the requantised output `u` that a READ already computes and reads out as bits.
  Next token = `base(READ k) + u`. Because `OUT(u)`, `ACT(row; u)` and the final result tokens are all "id base + value",
  the model can emit the *next layer's ACT token directly* (no host relabel) or a result token.

Options considered:

| design | what holds the schedule | cost | verdict |
|---|---|---|---|
| **A. fixed token cycle, control ROM in the embedding** | successor table in token features; program counter = the current token itself (READ(k), ACT(row), OUTK(j) each name their own step) | +1 token family, code FFNs get ~3x the terms, **0 extra layers** for the default array | **chosen, implemented, bit-exact** |
| B. instruction fetch from a DeltaNet program memory | prompt writes `PROG(slot, target)` tokens into a head keyed by the current step; each step reads its successor template | one more head + latch, ~slots x targets write tokens, more layers; gives programmable topology | designed (section 5), not built; A's `Schedule` table is exactly the content B would load |
| C. counter in a DeltaNet state row | a head that counts steps | needs a state-dependent decode of a count into token bits; A already encodes the step in the token | rejected: more machinery for nothing A cannot do |

**Honesty.** In A the *schedule is not in the prompt and not in the host*: it is a hardwired control ROM, compiled into the
weights, exactly like the datapath wiring (an accelerator's control FSM). What the prompt carries is weights and input.
The network's shape (depth x width, i.e. which output feeds which row) is fixed at compile time, like the array size
already is. Changing topology = recompiling; making it a prompt-time choice is design B.

## 2. Mechanics

Layers chain **block-diagonally in the one array** (no weight banks, no mid-run weight reload):

* layer *b* owns rows `[b n, (b+1) n)` and columns `[b n, (b+1) n)`; cross-block weights are 0.
* READ(k), k in block *b*, sums over all rows, but only block-*b* rows have non-zero weight in that column, so rows that hold
  other layers' activations are irrelevant. ACT(row s; v) writes `v * W[s][.]` into row s, which is 0 outside the block.
* array is `(depth*width)^2` (power of two): 2x4 on 8x8, 2x8 or 4x4 on 16x16.
* bias: row `width-1` of each block is the constant 1 (as in `TpuSession.mlp`); column `width-1` of a non-final block carries
  bias weight 1 plus 0 elsewhere so it emits `relu(1)=1` for the next block. Usable features per layer = `width-1`.
* every accumulator row must hold a write before the first READ (the A head normalises by written rows), so the prompt first writes
  `ACT(s; 0)` for the rows of layers 1.., then the real input rows `0..n-1` last.

Token cycle (depth L, width n, k = column index, rows/cols indexed over the whole array):

```
prompt:  CELL*(KR*NC)  ACT(n..KR-1;0)  ACT(0;x0) .. ACT(n-1;x_{n-1})      <- host-fed (weights + input; no READ, no OUT, no op)
emitted: READ(0)  ACT(n;y0)  READ(1)  ACT(n+1;y1) ...  READ(n-1) ACT(2n-1;y_{n-1})      layer 0 -> layer 1
         READ(n) ...                                                                     (each READ(k) is followed by the layer's value token)
         last layer: READ(k) OUTK(j;y_j) ... OUTK(n-1;y_{n-1}) HALT
```

Successors: `ACT(s;.)` (s >= n-1) -> `READ(s+1-n)`; `READ(k)` -> `base(k)+u` (`ACT(k+n;u)` for non-final layers, `OUTK(j;u)` for the last);
`OUTK(j;.)` -> `READ(next col)` or `HALT`; `HALT` -> `HALT`. Emitted tokens per run: `2*L*n` (+ the HALT that stops the loop).
Same shape as the ISA draft's `READ(k), Y` pairs (docs/V2_ISA.md section 3), minus the `OP_*` tokens.

Carry detail: `ACT0 mod NOUT != 0`, so `base + u` can carry into the row bits. `OUTK0` is placed with the same residue so the low bits
are `(u + low) mod NOUT` for every family; the high bits are a token feature plus `carry_feature * [u >= NOUT-low]`.
`u` is the unclamped Z where the adder tree ends, so value 0 is `u <= 0` and value AMAX is `u >= AMAX` (fixed from first run).

Circuit changes (`qwentpu/circuit.py`, tiny, v1 unaffected: all 17 circuit-IR tests pass): `SEQ` hook that supplies the code-bit
terms, and `is_read` bounded to the READ ids (previously `t >= READ0`, which would have caught the new tokens).
`Sequencer.install()` configures the packed array and extends `C.VOCAB`/`C.CODE`; `Sequencer.uninstall()` restores v1.

## 3. What the host still does

```python
def generate(feed, prompt, halt, max_new=4096):      # qwentpu.sequencer.generate
    nxt = feed(prompt)[-1]                           # feed = TpuSession.feed: prefill, return argmax per position
    out = []
    while nxt != halt and len(out) < max_new:
        out.append(nxt); nxt = feed([nxt])[-1]       # argmax; append
    return out
```

* builds the prompt (weights as CELL tokens, input as ACT tokens: this is data, like quantising and tokenising an input);
* argmax, append, stop at the end-of-sequence token HALT (and a safety cap);
* afterwards reads the result: `OUTK` token ids -> values (`Sequencer.logits`: `(id - outk0) % NOUT`), which is detokenisation.

It does **not**: choose or order any ACT/READ token, relabel OUT as ACT, know the layer count, widths or which row/column is next, or touch
any matrix. HALT only means end-of-sequence. (`fast=True` chip path skips Qwen's own muted layers; that is a v1 optimisation, not host logic.)

## 4. Results

* Circuit IR (float64, every bf16/silu/state exactness condition asserted), CPU, teacher-forced = deterministic argmax: model-emitted
  sequence == numpy golden and the output values == `ref_chain` (relu+clamp numpy network) for 2x4 u3s3, 4x4 u3s3, 2x8 u3s3, 2x4 u4s4, 2x8 u5s5.
  `pytest tests/test_sequencer.py`.
* ISA agent's golden MLP (`examples/mlp/golden.npz`, 15 -> 15 -> 10, u5s5): run as depth 2 x width 16 on a **32x32** u5s5 array (20 chip layers,
  1235 FFN units, 64 heads; IR-exact): logits and predictions equal golden for the digits tested (`test_ir_isa_golden_mlp_logits`; all 20 in a
  scratch run). The prompt is 1024 CELL + 32 ACT tokens, 65 tokens are emitted (2*2*16+1), the same count as the ISA draft.
  Token ids differ from `golden.expected_emitted_ids` (no OP_* tokens, no banks); the *logits* are identical.
* Real merged Qwen3.5-0.8B (A100, bf16): see section 7.

**Vs the ISA draft (banks + program text in the prompt).** The draft needs NB weight banks selected by control state and a program memory
(queued MATMUL list, step counter, active bank). Block-diagonal packing replaces banks with a bigger single array (32x32 instead of
2 x 16x16: ~2x the layers, 20 vs ~10) and needs no control state beyond the current token. Banks + program text = design B; see section 5.
For the ISA token format to be honoured literally, B is required; the logits are the same either way.

## 5. Design B sketch (programmable topology), for the ISA agent

Replace the `succ`/`base` token features by a program memory: a key head keyed by one-hot(current step id) (READ col, ACT row, OUTK index:
<= 128 keys) whose value columns hold the bits of the successor's high part and carry flag; the prompt fills it with `PROG(slot, class)`
tokens (slots x classes ~ 24 x 24 tokens). The `u`-dependent low bits and the carry stay as built here. Cost: one head + its read FFN,
no extra weight array. `Schedule` (routes / act_next / out_next) is the table such a prompt would encode. Not needed for fixed MLP shapes.

## 7. GPU result

Merged Qwen3.5-0.8B, A100, bf16, host = `generate()` (argmax; append) only, both readouts (code bits and lm_head), 6 random nets each:

| config | KR | chip layers | emitted/run | result |
|---|---|---|---|---|
| 2x4 u3s3 | 8 | 2 | 16 | exact |
| 4x4 u3s3 | 16 | 3 | 32 | exact |
| 2x8 u3s3 | 16 | 3 | 32 | exact |
| 2x8 u5s5 | 16 | 10 | 32 | exact |
| 2x16 u5s5 | 32 | 20 | 64 | exact; ISA golden MLP 20/20 digits identical logits |

0 mismatches in 60 runs (emitted ids equal the numpy golden, output values equal `ref_chain`). Script: `seq_run.py` (scratchpad copy at `~/v2_seq` on the box).

## 6. Limits

* g = h = 1 only. The final value is relu+clamp (no linear logits), as in the ISA draft.
* depth*width must be a power of two (array rule); layers are all `width` wide.
* ReLU/REQUANT gadgets from the nonlin agent plug in at `ctx["slots"]` (the final value expression/range) in `Sequencer.nxt_terms`:
  anything that yields `(u, lo, hi)` with value = clamp(u, 0, AMAX) works unchanged.

## 8. Program as a prompt choice (`qwentpu/sequencer2.py`)

Critique: "the host still runs the chip", and the schedule is hardwired. What is now prompt data and what is not:

**One compiled model, many programs.** `ProgramMachine(Dmax, n, ab, wb)` is the section-2 circuit unchanged (no new head, no new
layer, same chip-layer count as `Sequencer(Dmax, n)`). The program lives in DeltaNet state:

| program part | where it lives | chosen by |
|---|---|---|
| layer weights, biases, per-layer widths (<= n-1, zero padded) | weight-head state (block-diagonal CELLs) | prompt |
| depth d in 1..Dmax | **initial program counter**: the input is written into block `Dmax-d`, so the last prompt token `ACT(s n + n-1)` has the compiled successor `READ(s n)` | prompt |
| network (e.g. 15-15-10 MLP vs its hidden layer alone vs random nets) | all of the above | prompt |

The program counter is the current token itself (as in section 1, design A); the prompt picks its initial value by *where* it writes
the input, and the model walks blocks `s .. Dmax-1` then emits `HALT`. Emitted tokens = `2 d n + 1` (cost scales with the program,
not with Dmax); a shallower program skips the earlier blocks entirely (their weights are 0, their rows are zero-written).

**Still fixed in the weights (compiled):** array size and number format (Dmax, n, ab, wb); the token cycle `READ -> ACT -> READ`; the
successor table (a layer's outputs feed the next block's rows; the chain *ends* after block Dmax-1 with OUTK + HALT, so a program
always runs to the last block: depth is chosen by the entry point, not by an early exit); relu+clamp requantisation. Not built:
a state-fetched instruction memory (design B, section 5), i.e. early HALT, loops, branches or non-chain topologies need it.
Weights outside the block diagonal are not claimed/tested.

**Host (unchanged):** builds the prompt (weights + input + the entry block = loading a program image), `argmax; append` until the
model emits `HALT`, detokenises the OUTK ids. It does not choose, order or relabel any ACT/READ/OUT token and does not know the depth.

**Reconciling `nonlin.feedback_code`.** There is one feedback path: `Sequencer.nxt_terms` (token ROM + `base(READ k) + u` with carry).
`nonlin.feedback_code`/`act_code` is its special case `Schedule([("act", k)...], {}, ...)` (READ(k) -> ACT(k; y), nothing else).
`test_one_feedback_path_sequencer_equals_nonlin_feedback_code` shows both give identical next-token bits at every position and the same
chip-layer count (both are zero extra layers; they replace the same code stage). Keep `act_code` as the independent cross-check and for
the single-matvec fused op; anything with chaining/OUT/HALT uses the Sequencer path.

Tests: `tests/test_sequencer2.py` (IR, CPU): Dmax=4 n=4 and Dmax=2 n=8 u3s3, every depth 1..Dmax with random widths, one circuit per
config; the golden 15-15-10 MLP as a depth-2 prompt on Dmax=2 n=16 u5s5 (logits and predictions equal golden); the hidden layer alone as
a depth-1 prompt on the same model; feedback equivalence.

GPU (real merged Qwen3.5-0.8B, A100, bf16, `~/v2_prog/prog_run.py`, host = `generate()` argmax/append, both readouts code + lm_head, emitted ids and
logits equal the numpy golden): Dmax=4 n=4 u3s3 (3 chip layers, ONE model, depths 1-4, emitted 8/16/24/32): 24/24 exact; Dmax=2 n=8 u3s3: 12/12 exact;
Dmax=2 n=16 u5s5 (20 chip layers): 12/12 random programs exact, golden MLP as depth-2 prompt 20/20 identical logits, hidden-layer-only depth-1 prompt 10/10.
