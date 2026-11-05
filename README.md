<p align="center"><img src="docs/banner.svg" alt="qwenTPU: a TPU compiled into the weights of Qwen3.5" width="100%"></p>

<h1 align="center">qwenTPU</h1>
<p align="center"><b>A TPU compiled into the weights of a stock Qwen3.5. No TPU hardware, no training.<br>A language model that does exact matrix multiplication.</b></p>

<p align="center">
<a href="https://bobliuuu.github.io/qwentpu/"><b>🎮 Watch it run</b></a> ·
<a href="WRITEUP.md"><b>📖 Write-up</b></a> ·
<a href="#-what-is-actually-new"><b>🧭 What's new</b></a> ·
<a href="#-does-it-work-on-your-hardware"><b>🔬 Hardware</b></a> ·
<a href="#-v2-a-tpu-that-runs-a-program"><b>🧬 v2</b></a> ·
<a href="#-quickstart"><b>🚀 Quickstart</b></a>
</p>

---

We took a stock **Qwen3.5** (0.8B, 2B or 4B), compiled a few new layers into it, and it now contains a **weight-stationary matrix-multiply array**:
up to **32×32 (1,024 MACs per matvec)** with up to **5-bit operands**. Every new weight is computed; Qwen's own weights are untouched. Output is integer-exact against NumPy
(0 mismatches in 1,139,712 replayed token positions on four machines), plain text generation is unchanged (KL 3e-4 to 8e-4), and a TPU run leaves no trace in Qwen's context.

<p align="center"><a href="docs/qwentpu_demo.mp4"><img src="docs/demo.gif" alt="qwenTPU v1 running a matvec inside Qwen3.5, token by token" width="100%"></a><br><sub>v1 running inside Qwen3.5-0.8B on real tokens. <a href="docs/qwentpu_demo.mp4">Full 70 s video (mp4)</a></sub></p>

<p align="center"><img src="docs/arch.svg" alt="architecture" width="100%"></p>

## 🧭 What is actually new

Putting a machine into Qwen3.5's weights is **not** new: [Qwenputer](https://github.com/divagr18/qwenputer) did a RISC-V CPU, [QwenPU](https://github.com/divagr18/QwenPU) a CPU plus vector unit, qwenGPU a SIMT GPU.
Their design vocabulary (DeltaNet rows as RAM, tokens as the trace) is what we build on. Four things here are ours:

| | the new part | why it matters |
|---|---|---|
| 1 | **A dataflow array, not a CPU.** No program counter, no instruction stream: the layer stack *is* the datapath, and layers chain through the model's own sampled tokens. | The others emulate a processor. This is the first one that is the accelerator's *structure*. |
| 2 | **One multi-hot read is a whole reduction.** A Gated DeltaNet head returns the exact sum of the rows you select, so the adder tree costs one read. | A TPU is mostly RAM plus adders, and a DeltaNet head already is both. |
| 3 | **Exact integers in bf16, up to 5 bits.** Bit-plane counting, then a top-down fold with saturating clamps that is provably lossless. | bf16 only compares small integers exactly. This is what makes 16×16 and 32×32 possible. |
| 4 | **Zero trace.** Qwen's own layers are muted for chip tokens, so a chip-only path skips them and a TPU run leaves Qwen's context bit-identical (KL = 0.0). | The TPU and the language model share one set of weights without disturbing each other. |

| | machine | control | state holds | a token is |
|---|---|---|---|---|
| Qwenputer | RV32IM CPU | PC, fetch, decode | registers, code, data | an instruction |
| QwenPU | CPU + vector unit | the CPU's program | vector regs, weight rows, KV | a vector-op step |
| qwenGPU | SIMT GPU | host-compiled kernel | registers, global memory | a warp-launch record |
| **qwenTPU** | **weight-stationary array** | **none** | **weights *and* partial sums** | **a dataflow step** |

Not faster than QwenPU per op (it streams 64 lanes per token, we do 4 to 64 MACs per token).
[TinyTPU](https://github.com/tiny-tpu-v2/tiny-tpu) is the architecture reference only: this repo shares no code with it, and our array is not systolic in space.

## 🧠 How it works

A Gated DeltaNet head (Qwen3.5's linear-attention layer) keeps a state matrix `S`. That is exactly the memory and adder a TPU needs:

| Gated DeltaNet does… | …which, for a TPU, is |
|---|---|
| a **one-hot write** replaces row *i* exactly | store a weight row / a partial sum |
| a **one-hot read** returns row *i* | feed row *i* of the weights to the multipliers |
| a **multi-hot read** returns the exact **sum** of the chosen rows | **the adder tree. A whole reduction in one read.** |
| the sampled token is fed back in | layer *n*'s output **is** layer *n+1*'s input |

Weights live in state 🗄️, activations arrive as tokens 🎟️, products are bit-ANDs in SwiGLU layers ✖️, the reduction is one head read ➕, and the answer is sampled as the next token 🔁.

<p align="center"><img src="docs/life-of-a-matvec.svg" alt="life of a matvec" width="100%"></p>

<p align="center"><img src="docs/anim/matvec.gif" alt="animation: weights written, activations streamed, rows summed in one read" width="100%"></p>

The accumulator returns *counts* of rows with activation bit *a* and weight bit *b* set, and `y = Σ 2^a · Σ_b wt_b · cnt[a][b]` is far too wide to compare exactly in bf16. So the sum is folded top-down,
`u ← clamp(2^q·u + next digits)`, and saturating early is lossless: once `u` is past what the remaining digits could undo, the answer is already pinned. Wider ranges clamp in several lossless passes.

<p align="center"><img src="docs/adder-tree.svg" alt="carry-save adder tree" width="100%"></p>

<p align="center"><img src="docs/anim/adder.gif" alt="animation: bit-plane counts folded through saturating clamps" width="100%"></p>

| array | chip layers |
|---|---|
| 8×8, 3-bit | **2** |
| 8×8, **5-bit** | 7 |
| 16×16, **5-bit** | 10 |
| 32×32, 3-bit | 6 |

## ⚡ Speed

Depth is the price of width, and the biggest wins are in skipping work, not adding it (details and the packing trade-off in the [write-up](WRITEUP.md)):

| A100 · 0.8B · 8×8 u3s3 | |
|---|---|
| whole network, batch 1, same trace | 6,629 tok/s |
| **TPU layers only** (Qwen's muted layers skipped, bit-identical) | **74,462 tok/s** (11×) |
| TPU layers only, **one CUDA graph per trace** (`bench.py --graph`, bit-identical) | **~341,000 tok/s** (mean of 3 runs ×2 traces, range 299k-377k; 4.6× over the eager TPU-layers rerun at 75,207) |
| free-running chained matvecs, TPU layers + **CUDA graphs** | **705 matvec/s** |

<p align="center"><img src="docs/chip-path.svg" alt="chip-only path" width="100%"></p>

<p align="center"><img src="docs/anim/stack.gif" alt="animation: chip tokens skip the muted Qwen layers" width="100%"></p>

## 🔬 Does it work on your hardware?

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

## 🧬 v2: a TPU that runs a program

<p align="center"><a href="docs/qwentpu_v2_demo.mp4"><img src="docs/anim/v2_demo.gif" alt="v2 classifying digits: the model emits every token itself" width="100%"></a><br><sub>v2 classifying held-out digits through Qwen3.5-0.8B, animated from the real traces. <a href="docs/qwentpu_v2_demo.mp4">Full 75 s video (mp4)</a></sub></p>

Two versions live in this repo. v1 is tagged `v1.0`; v2 is on `main` as new files and leaves v1 bit-identical.

| | **v1: matvec array** | **v2: a TPU that runs a program** |
|---|---|---|
| What it is | weight-stationary matrix-multiply array in Qwen's weights | the same array plus a tiny instruction set, exact ReLU and clamp, and a sequencer |
| Who picks the chip tokens | **the host** (CELL, ACT, READ, OUT) | **the model**: after the prompt it emits every token itself |
| What the host does | drives the machine token by token | builds the prompt (weights, input, entry point) and runs `argmax; append` until HALT |
| Nonlinearity | none | `clip(relu(x@W), 0, max)` in the datapath, at zero extra layers |
| What the prompt chooses | data only | the network: depth, per-layer widths, weights, input |
| Proven on | A100, A10G, L4, x86 Xeon (1,139,712 replayed positions, 0 mismatches) | Qwen3.5-0.8B on an A100 (below) |

**v2 results** (`results/a100/v2_prog.txt`, `results/a100/v2_feedback.txt`; bf16, real Qwen3.5-0.8B on an A100):

- One compiled model, depths 1 to 4 chosen by the prompt: 24 of 24 programs exact. 8-wide, depths 1 to 2: 12 of 12. 16-wide at 5-bit (20 chip layers): 12 of 12.
- A digit classifier (15 to 15 to 10, 5-bit integer weights) runs as a depth-2 program: logits match the integer golden on **20 of 20** digits. As a depth-1 program (hidden layer alone): 10 of 10.
- The model's next token after a READ spells the next layer's ACT token: 128 of 128 feedback tokens exact (8×8, 5-bit).
- Accuracy: 87.6% on 540 held-out digits for the integer network the model runs, against 91.1% for a float MLP of the same shape. The model reproduces the integer network exactly; the gap is the cost of 5-bit weights, 0..12 inputs and a fixed scale with no requantisation shift (details in the [write-up](WRITEUP.md#8-v2-a-tpu-that-runs-a-program)).

### How v2 works

<p align="center"><img src="docs/v2-arch.svg" alt="v2 architecture: the prompt carries weights, input and entry point; the model emits every token; the host does argmax and append" width="100%"></p>

1. **The prompt is the program.** Weights as `CELL` tokens, the input as `ACT` tokens, nothing else. Where the input is written sets the entry point.
2. **The model sequences itself.** A successor table compiled into the embedding maps each token to the next one, and a READ emits `base + y`, where `y` is the layer's value. The host runs `argmax; append` until HALT and never inspects a token.
3. **ReLU and clamp are free.** The adder tree already ends in a saturating requantiser, so the sum is thresholded straight into the id bits of the next layer's `ACT` token, at zero extra layers.

<p align="center"><img src="docs/v2-token-timeline.svg" alt="v2 token timeline for one digit: 1056 host-fed prompt tokens, then 64 model-emitted tokens, then HALT" width="100%"></p>
<p align="center"><img src="docs/v2-relu-feedback.svg" alt="relu and clamp of the sum become the next ACT token" width="100%"></p>
<p align="center"><img src="docs/v2-program-in-prompt.svg" alt="the entry block selects the depth" width="100%"></p>
<p align="center"><img src="docs/v2-fixed-vs-prompt.svg" alt="what is compiled into the weights and what the prompt chooses" width="100%"></p>

Animated versions: <a href="docs/anim/v2_self_sequencing.gif">self-sequencing</a> · <a href="docs/anim/v2_program_in_prompt.gif">program in prompt</a> · <a href="docs/anim/v2_relu_feedback.gif">relu feedback</a> · <a href="docs/anim/v2_whats_fixed.gif">fixed vs prompt</a>. The [interactive site](docs/site/index.html) has a token-by-token player (section "v2") driven by the real traces.

### Run it

```bash
python examples/mlp/run_v2.py -n 20            # CPU simulator, all 20 golden digits, ~1 s, numpy only
python examples/mlp/run_v2.py -n 1 -v          # also print every emitted token
python examples/mlp/run_v2.py -n 1 --ir 1      # also check against the compiled circuit IR (float64, ~1 min)
QWEN_DIR=q08 python examples/mlp/run_v2.py --cuda -n 5     # optional: the real Qwen3.5-0.8B with the chip layers merged in (CUDA)
```

```
digit  0: prompt 1056 tokens, model emitted 64 tokens, logits [0, 31, 0, 0, 0, 0, 0, 0, 0, 0] -> predicted 1, label 1  match
```

The same in code (this is what `examples/mlp/run_v2.py` does; with a model, `feed` is `TpuSession(model, v0, fast=True, readout="lm_head").feed`):

```python
import numpy as np
from qwentpu.sequencer2 import ProgramMachine, run_program

z = np.load("examples/mlp/golden.npz")
seq = ProgramMachine(2, 16, 5, 5).install()            # Dmax=2 blocks, 16 wide, 5-bit: 20 chip layers
emitted, logits = run_program(feed, seq, [z["W1"], z["W2"]], z["x_test"][0])   # host: build prompt; argmax; append until HALT
print(int(np.argmax(logits)), z["label"][0])
```

`feed` is any function that returns the model's next token at each fed position (`examples/mlp/run_v2.py` has a numpy one, `SimFeed`). Pass one weight matrix instead of two to run a depth-1 program on the same compiled model.

### Vectorized v2: several neurons per token

`Sequencer(depth, width, ab, wb, vec=2)` (and `ProgramMachine`) packs `vec` neurons into every token: one `ACT` token carries `vec` activations
and one `READ` / `OUTK` token `vec` outputs, so the model emits `2 * depth * width / vec` tokens instead of `2 * depth * width`. The model still
emits every token, the host still runs `argmax; append`, and `vec=1` (the default) is the scalar chain, unchanged ([design](docs/V2_VECTORIZED.md)).
On Qwen3.5-0.8B (A100) every program is exact for vec 1, 2 and 4 (0 mismatches), including the 15-15-10 digit classifier at vec 2 (20 of 20).
Speed on the classifier's array (two 16-wide blocks, 5-bit, 64 tokens become 32; the loop is the emitted-token loop):

| | vec 1 | vec 2 |
|---|---|---|
| chip layers / vocabulary | 20 / 34,370 | 24 / 60,417 |
| eager: loop, per inference | 2,184 ms, 2,248 ms | 1,106 ms, 1,170 ms (**1.92x**) |
| CUDA graph: loop | 425 ms | 351 ms (1.21x) |

At 3-bit the eager loop gains 1.2x (vec 2) to 1.45x (vec 4), and the CUDA-graph loop loses (0.90x and 0.66x): the vectorized chip has more layers
and a wider hidden state, and the graph path already pays no launch cost. The vocabulary grows as `2^(AB * vec)` per slot, so 5-bit stops at vec 2.
Full table: [`results/a100/v2_vec.txt`](results/a100/v2_vec.txt).

### What v2 does not do

- **Chain-only topology:** layer b feeds layer b+1. No branches, loops, skips or early exit.
- **No signed logits:** outputs are relu-clamped to 0..31, so negative scores read as 0.
- **The schedule is compiled in:** array size, number format, the READ, value, READ cycle and the successor table are weights. The prompt chooses depth, widths, weights, biases and input. A prompt-loaded instruction memory is designed (`docs/V2_SEQUENCER.md`) and not built.
- **Evidence:** Qwen3.5-0.8B on one A100 (plus the CPU circuit IR). 2B, 4B and other GPUs are not run for v2.

Details: `docs/V2_ISA.md` (instruction set, token layout), `docs/V2_SEQUENCER.md` (how the model sequences itself), `examples/mlp/` (the classifier, golden outputs and `run_v2.py`), `WRITEUP.md` section 8.

## 🚀 Quickstart

```bash
pip install -e ".[dev]"
python demo.py                                         # float64 sim of the datapath, no model needed
python -m pytest -q tests/test_qwentpu.py tests/test_circuit_ir.py tests/test_circuit_hf.py

hf download Qwen/Qwen3.5-0.8B --local-dir q08          # or 2B / 4B
QWEN_DIR=q08 python -m pytest -q tests/test_merged.py -s
QWENTPU_CFG=8,8,3,3,2,2 QWEN_DIR=q08 python bench.py 24 --device cuda --batch 16 --vectors 32
QWEN_DIR=q08 python bench_driver.py --device cuda --mode graph      # free-running chained matvecs
python -m qwentpu.merge q08 out/qwen35-0.8b-tpu        # write a plain HF checkpoint
```

```python
from qwentpu.merge import build_merged, instantiate
from qwentpu.driver import HybridSession

tc, sd, v0, lay, g = build_merged("q08"); model = instantiate(tc, sd).cuda()
s = HybridSession(model, v0, fast="graph")             # fast=False runs the whole network (slow, same answers)
s.text(tokenizer("What is 2+2?").input_ids)            # ordinary Qwen text
with s.chip():                                         # the TPU, in the same model, leaving no trace
    s.load_weights(W)                                  # KR·NC tokens, once
    y = s.matvec(x)                                    # AMAX-bounded activations in, out
```

## 🧰 Repo map

| path | what |
|---|---|
| `qwentpu/circuit.py` | the TPU in our IR: weight/accumulator heads, AND products, adder-tree planner, packing |
| `qwentpu/compiler/` | **own compiler**: IR, bf16 exactness evaluator, weight emitter, integer gadgets |
| `qwentpu/merge.py`, `mute.py` | widen Qwen, append the TPU layers, `tau`, muting, split-K guard |
| `qwentpu/isa.py`, `program.py`, `sequencer.py`, `sequencer2.py`, `compiler/nonlin.py` | v2: instruction set, golden interpreter, self-sequencing, prompt-chosen programs, exact ReLU and feedback |
| `qwentpu/driver.py` | `ChipPath`, `GraphedChip` / `GraphedTrace` (CUDA graphs), `TpuSession`, `HybridSession` |
| `bench.py`, `bench_driver.py` | mass replay vs NumPy · free-running chained matvecs |
| `run_matrix.sh` | every array on every model |
| `docs/site/` | 🎮 interactive walkthrough · `docs/tools/gen_svgs.py` regenerates the diagrams |
| [`WRITEUP.md`](WRITEUP.md) | the long version |

## ⚠️ Limits, honestly

- **v1 is host-driven.** The host chooses every chip token and there is no in-datapath nonlinearity. v2 moves the sequencing into the model (see [v1 and v2](#-v2-a-tpu-that-runs-a-program)), but its schedule, array size and chain topology are still compiled into the weights, and the host still builds the prompt and runs the argmax loop.
- **Teacher-forced numbers are prefill throughput.** The free-running figure is the honest chained number.
- **The split-K guard is a process-wide torch flag** set by `merge.instantiate`. Load a saved checkpoint with `qwentpu.merge.load_checkpoint(path)`, which sets it and casts the rotary buffers the way the tests do (plain `from_pretrained` does neither).

## 🙏 Credits

Design vocabulary: **Qwenputer** (Divyansh Agrawal), with which this repo shares no code. Architecture reference: **TinyTPU** (Surya Sure et al.), likewise no code. Base models: **Qwen3.5**. MIT licensed.
