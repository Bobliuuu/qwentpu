<p align="center"><img src="docs/banner.svg" alt="qwenTPU: a TPU compiled into the weights of Qwen3.5" width="100%"></p>

<h1 align="center">✈️ qwenTPU</h1>
<p align="center"><b>Airplanes don't flap their wings. This TPU doesn't have a chip.<br>It's a language model that does exact matrix multiplication, with no training.</b></p>

<p align="center">
<a href="https://bobliuuu.github.io/qwentpu/"><b>🎮 Watch it run</b></a> ·
<a href="WRITEUP.md"><b>📖 Write-up</b></a> ·
<a href="#-what-is-actually-new"><b>🧭 What's new</b></a> ·
<a href="#-does-it-work-on-your-hardware"><b>🔬 Hardware</b></a> ·
<a href="#-quickstart"><b>🚀 Quickstart</b></a>
</p>

---

We took a stock **Qwen3.5** (0.8B, 2B or 4B), compiled a few new layers into it, and it now contains a **weight-stationary matrix-multiply array**:
up to **32×32 (1,024 MACs per matvec)** with up to **5-bit operands**. Every new weight is computed; Qwen's own weights are untouched. Output is integer-exact against NumPy
(0 mismatches in 1,139,712 replayed token positions on four machines), plain text generation is unchanged (KL 3e-4 to 8e-4), and a TPU run leaves no trace in Qwen's context.

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

Not faster than QwenPU per op (it streams 64 lanes per token, we do 4 to 64 MACs per token), and "to my knowledge" rests on one search, not an exhaustive one.
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

The accumulator returns *counts* of rows with activation bit *a* and weight bit *b* set, and `y = Σ 2^a · Σ_b wt_b · cnt[a][b]` is far too wide to compare exactly in bf16. So the sum is folded top-down,
`u ← clamp(2^q·u + next digits)`, and saturating early is lossless: once `u` is past what the remaining digits could undo, the answer is already pinned. Wider ranges clamp in several lossless passes.

<p align="center"><img src="docs/adder-tree.svg" alt="carry-save adder tree" width="100%"></p>

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
| free-running chained matvecs, TPU layers + **CUDA graphs** | **705 matvec/s** |

<p align="center"><img src="docs/chip-path.svg" alt="chip-only path" width="100%"></p>

## 🔬 Does it work on your hardware?

An exactness proof in float64 is necessary, not sufficient. Running the same suite on different GPUs found two bugs a simulation could never see: bf16 rounding of Qwen's `1/√128` query scale at 16 query rows,
and cuBLAS split-K adding partials in bf16 on an L4 (traces of 288 and 320 tokens came back wrong; the A100 never showed it). Both are fixed and guarded by tests (see the write-up).

| device | architecture | models | arrays replayed | positions replayed | mismatches | suite runs clean |
|---|---|---|---|---|---|---|
| A100 40GB | Ampere sm80 | 0.8B, 2B, 4B | 9 | 705,024 | **0** | 18/18 |
| A10G 24GB | Ampere sm86 | 0.8B, 2B | 8 | 221,184 | **0** | 12/12 |
| L4 24GB | Ada sm89 | 0.8B, 2B | 8 | 190,464 | **0** | 12/12 |
| T4 16GB | Turing sm75 | 0.8B, 2B | n/a | n/a | n/a | 9/12 |
| x86 Xeon (c6i, Ice Lake) | x86-64 | 0.8B, 2B, 4B | 8 | 23,040 | **0** | 14/14 |

The suite is 10 tests per (model, array) for the runs in the table (an 11th, restore on the full network, was added afterwards and passes on the A100, 0.8B 8×8): exact inside Qwen · exact after text · pure-text KL · text after a chip run · every trace length (up to 320 tokens, the lengths that broke the L4) · mutation test (zero the TPU layers and answers must break) ·
real 2-layer MLPs · chip-only path bit-identical · CUDA-graph executor bit-identical · restore invisible. Replay compares every predicted token with NumPy (`bench.py`).
The T4 passes all 0.8B runs and the three smallest 2B ones, then runs out of its 16 GB; it has suites only, no replay. Raw logs for every machine are in [`results/`](results/).

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
| `qwentpu/driver.py` | `ChipPath`, `GraphedChip` (CUDA graphs), `TpuSession`, `HybridSession` |
| `bench.py`, `bench_driver.py` | mass replay vs NumPy · free-running chained matvecs |
| `run_matrix.sh` | every array on every model |
| `docs/site/` | 🎮 interactive walkthrough · `docs/tools/gen_svgs.py` regenerates the diagrams |
| [`WRITEUP.md`](WRITEUP.md) | the long version |

## ⚠️ Limits, honestly

- **Depth is the price of width.** Wide operands cost chip layers (16×16 u5s5 is 10) and packing costs layers and vocabulary; only `g2h2` on 8×8 was a clear win.
- **No fused kernel.** CUDA graphs remove launch overhead, not the cost of the HF kernels. A purpose-built executor could go further.
- **Teacher-forced numbers are prefill throughput.** The free-running figure is the honest chained number.
- **Memory:** a 16 GB T4 runs out of VRAM on the larger 2B arrays. 4B was run on the A100 and the Xeon, not on the 24 GB cards.
- **Coverage is what `results/` says.** An earlier L40S, Graviton3 and Apple-silicon run was lost in a crash and is not claimed.
- **The split-K guard is a process-wide torch flag** set by `merge.instantiate`. A saved checkpoint loaded with plain HF does not get it: set `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` yourself.
- Text-only checkpoints, inference only (no backprop, unlike TinyTPU). Merged checkpoints are built on the fly, not uploaded.

## 🙏 Credits

Design vocabulary: **Qwenputer** (Divyansh Agrawal), with which this repo shares no code. Architecture reference: **TinyTPU** (Surya Sure et al.), likewise no code. Base models: **Qwen3.5**. MIT licensed.
