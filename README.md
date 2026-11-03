<p align="center"><img src="docs/banner.svg" alt="qwenTPU: a TPU compiled into the weights of Qwen3.5" width="100%"></p>

<h1 align="center">✈️ qwenTPU</h1>
<p align="center"><b>Airplanes don't flap their wings. This TPU doesn't have a chip.<br>It's a language model that learned to do matrix multiplication, without learning anything.</b></p>

<p align="center">
<a href="docs/site/index.html"><b>🎮 Watch it run</b></a> ·
<a href="WRITEUP.md"><b>📖 Write-up</b></a> ·
<a href="#-does-it-work-on-your-hardware"><b>🔬 Hardware</b></a> ·
<a href="#-quickstart"><b>🚀 Quickstart</b></a>
</p>

---

## 🛬 TL;DR

We took a stock **Qwen3.5** (0.8B, 2B or 4B), wrote a few new layers into it with a compiler, and it now contains a
**weight-stationary matrix-multiply array**, a TPU, up to **32×32 (1,024 MACs per matvec)** with up to **5-bit operands**.

- 🧠 **No training.** Every new weight is computed. Qwen's own weights are not touched.
- 🎯 **Exact.** Integer-exact against NumPy: **1,139,712 outputs checked, 0 mismatches**, on **5 different machines** (NVIDIA A100, A10G and L4 GPUs across two architectures, plus an x86 Xeon).
- 🤐 **Silent.** Plain text generation stays within KL < 5e-3 of the base model (asserted by the suite), and a chip-only TPU run leaves *no trace* in Qwen's context (KL = 0.0, bit-identical).
- ⚡ **Fast enough to be fun.** Free-running chained matvecs: **705 matvec/s** on an A100 (0.8B, 8×8) with CUDA graphs, through four optimizations (below).

If you read nothing else: a Gated DeltaNet head is a little RAM with a built-in adder. A TPU is mostly RAM and adders.
So it was sitting there the whole time. 🪽

---

## 🧠 The one idea

A **TPU** is a dataflow machine: *weights stay put, activations flow past them, partial sums pile up in place.*
A **Gated DeltaNet head** (the linear-attention layer in Qwen3.5) keeps a state matrix `S`:

| Gated DeltaNet does… | …which, for a TPU, is |
|---|---|
| a **one-hot write** replaces row *i* exactly | store a weight row / a partial sum |
| a **one-hot read** returns row *i* | feed row *i* of the weights to the multipliers |
| a **multi-hot read** returns the exact **sum** of the chosen rows | **the adder tree. A whole reduction in one read.** |
| the sampled token is fed back in | layer *n*'s output **is** layer *n+1*'s input |

So: weights live in state 🗄️ · activations arrive as tokens 🎟️ · products are bit-ANDs in SwiGLU layers ✖️ · the reduction is one head read ➕ · and the answer is *sampled as the next token*, which is the next layer's input 🔁.

<p align="center"><img src="docs/arch.svg" alt="architecture" width="100%"></p>

## 🎬 Life of a matvec

<p align="center"><img src="docs/life-of-a-matvec.svg" alt="life of a matvec" width="100%"></p>

```text
CELL(r,c,w)   × KR·NC    📥 load the weights, once       one token writes one weight into DeltaNet state
ACT(s; v…)    × KR/g     🌊 stream a vector              g activations per token → AND-products → accumulator rows
READ(k)       × NC/h     📤 read h columns at once       multi-hot sum of rows → adder tree → OUT(v…)
                                                          OUT is the model's own next token = the next layer's input
```

Nothing here is a plug-in or a tool call. Every step is a forward pass of the stock Qwen3.5 architecture.
The host does no arithmetic: it relabels one output token as the next input.

---

## 🧮 The adder tree (why it can be 5-bit)

<p align="center"><img src="docs/adder-tree.svg" alt="carry-save adder tree" width="100%"></p>

The accumulator returns *counts*: how many rows have activation bit *a* **and** weight bit *b* set. The exact answer is
`y = Σ 2^a · Σ_b wt_b · cnt[a][b]`, but `y` is far too wide for a bf16-exact comparator (a SwiGLU gate only compares small integers exactly).

So we fold it **top-down**, saturating as we go: `u ← clamp(2^q·u + next digits)`. The trick is that saturating early is **lossless**:
once `u` is past the value the remaining digits could ever undo, the answer is already pinned at 0 or max.
A clamp costs two SwiGLU units (`Z + relu(lo−Z) − relu(Z−hi)`). If a range is wider than bf16's exact window (256), the clamp runs in
**several lossless passes**. The last step is fused into the output-code layer.

| array | chip layers |
|---|---|
| 8×8, 3-bit | **2** |
| 8×8, 4-bit | 4 |
| 8×8, **5-bit** | 7 |
| 16×16, 4-bit | 5 |
| 16×16, **5-bit** | 10 |
| 32×32, 3-bit | 6 |
| 64×64, 3-bit | 12 |

---

## 📦 Optimization #1: tokens are the bottleneck → pack them

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

## ⏩ Optimization #2: skip the layers that do nothing

<p align="center"><img src="docs/chip-path.svg" alt="chip-only path" width="100%"></p>

Qwen's own layers are **muted** for chip tokens: their mixer output is exactly 0 and their MLP sees a zero vector, so each one is the *identity* on the chip dims.
So `ChipPath` runs `embed → only the TPU layers → norm → next token` and skips the other 24-32 layers. **Bit-identical** (it's a test), and it never touches
Qwen's cache, so a TPU run leaves no trace in Qwen's context *by construction*, no restore needed.

| 0.8B · 8×8 · tokens/s | full network | **TPU layers only** |
|---|---|---|
| A100, batch 1 | 1,997 | **24,030** (12×) |
| Xeon, batch 1 (same replay) | 171.5 | **2,128** (12×) |
| Xeon, free-running matvec/s | 1.59 | **25.47** (16×) |

*A100 rows come from two replays with different trace lengths (768 vs 3,072 tokens); the Xeon rows are the same trace both ways.*

## 🔢 Optimization #3: read the answer off the sign bits

The next token is spelled in the sign of ~14 "code" dims. We read those directly instead of multiplying by a chip-vocabulary `lm_head`
(`hidden × |vocab|` MACs per token, more than the entire TPU once the vocabulary is big). Same argmax, and a test checks the readouts agree (`full == lm_head == code`). We did not keep a timing for this one, so no speedup is claimed.

## 📹 Optimization #4: CUDA graphs

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

---

## 🔬 Does it work on your hardware?

"Exact on an A100" is a claim about an A100. So we ran the same suite everywhere we could get a machine.

| device | architecture | models | arrays replayed | positions replayed | mismatches | suite runs clean |
|---|---|---|---|---|---|---|
| A100 40GB | Ampere sm80 | 0.8B, 2B, 4B | 9 | 705,024 | **0** | 18/18 |
| A10G 24GB | Ampere sm86 | 0.8B, 2B | 8 | 221,184 | **0** | 12/12 |
| L4 24GB | Ada sm89 | 0.8B, 2B | 8 | 190,464 | **0** | 12/12 |
| T4 16GB | Turing sm75 | 0.8B, 2B | n/a | n/a | n/a | 9/12 |
| x86 Xeon (c6i, Ice Lake) | x86-64 | 0.8B, 2B, 4B | 8 | 23,040 | **0** | 14/14 |

The suite is 10 tests per (model, array), 9 on CPU (the CUDA-graph test is skipped): TPU exact inside Qwen · exact with text already in the cache · pure-text KL · text after a chip run ·
exact for every trace length · mutation test (zero the TPU layers and the answers must break) · XOR + random 2-layer MLPs through the real model ·
chip-only path bit-identical · CUDA-graph executor bit-identical · restore invisible (KL < 1e-4 after a cache restore, 0.0 on the chip-only path). Replay = `bench.py` comparing every predicted token with NumPy.

---

## 👀 What we observed

Everything here comes from the raw logs in `results/` (replays `*.jsonl`, pytest outcomes `*.suite`). Throughput is tokens/s unless noted.

- **Exact everywhere we looked.** 1,139,712 replayed outputs, 0 mismatches against NumPy, on every device in the table above. Every A100, A10G and L4 suite run is `10 passed`. The T4 passes all twelve 0.8B and the three smallest 2B runs, and runs out of its 16 GB on the bigger 2B arrays (9 of 12 clean). The Xeon logs are from an earlier 7-test suite, `7 passed`. The T4 has suites only, no replay.
- **Cheaper GPUs were not slower here.** Same 8×8 u3s3 replay on 0.8B, same trace, full network: batch 1 is 2,410 tok/s on an L4, 2,147 on an A10G, 2,036 on an A100; batch 8 is 19,477 / 15,519 / 16,228. That fits launch overhead, not FLOPs, dominating at this size. The Xeon does 145 at batch 1.
- **Packing wins on small arrays only.** `g2h2` on 8×8 is +47% MACs/s on an A100 at batch 16 (2.54M vs 1.73M) and +1% at batch 1; `g4h4` is +24% at batch 16 but adds 2 layers and a 12,803-token vocabulary. On 16×16 u3s2 `g2h2` is -15% and `g4h4` -36% at batch 16; on 32×32 at batch 1 they are -27% and -56%. The Xeon agrees: 8×8 `g2h2` +48%, 16×16 `g2h2` -30%.
- **Depth is what wide operands cost.** On an A100 at batch 16, a 16×16 array does 3.05M MACs/s with 3-bit activations and 2-bit weights (2 chip layers), 1.35M with 4-bit (5 layers) and 0.65M with 5-bit (10 layers).
- **Most of Qwen's layers are dead weight for chip tokens.** Skipping them is 12× on an A100 at batch 1 (1,997 → 24,030 tok/s) and 12× on the Xeon on the same replay (171.5 → 2,128).
- **Free-running chains are far slower than batched replay.** With CUDA graphs an A100 does 705 matvec/s on 8×8 u3s3, 629 on 16×16 u3s2, 215 on 32×32 u3s3 and 65 on 32×32 `g4h4`. Each layer waits for the previous layer's token, so these are not comparable to the batched replay numbers in *Results*.
- **Batching pays.** 8×8 on an A100, TPU layers only: 107k tok/s at batch 1, 487k at batch 16.
- **Bigger base models are slower.** Full-network 8×8 u3s3 at batch 16 on an A100: 33,074 (0.8B), 28,401 (2B), 13,773 (4B).
- **Memory was the boring bug.** The first test fixtures held the base model, the merged model and a mutated copy at once. That ran 24 GB cards (A10G, L4) out of memory in the mutation, XOR/MLP and restore tests, and it hit the 40 GB A100 too. The base model is now rebuilt on demand and freed after each forward pass, so the 2B suites run on 24 GB GPUs and on 16 GB hosts. Recorded 2B passes on 24 GB cards: A10G on 8×8 u3s3, 8×8 u5s5 and 16×16 u3s2; L4 on 8×8 u3s3 and 16×16 u3s2. The other 2B suite runs on those cards have no outcome recorded.

## 🪲 Bug museum

Cross-hardware testing found two bugs that float64 simulation could never see.

<p align="center"><img src="docs/bug-museum.svg" alt="bug museum" width="100%"></p>

**🐛 #1 The irrational scale.** Qwen l2-normalises the query and scales it by `1/√128`. For a multi-hot query of `n` rows that factor is exact only if `n·128` is a power of 4.
At 16 rows it isn't, so bf16 rounds `count × scale` differently for each count and some counts came back one rounding step off. float64 simulation never saw it.
Fix: pad the query with unused all-zero state rows. The exactness evaluator now rejects any query that breaks the rule.

**🐛 #2 cuBLAS bf16 split-K.** With `allow_bf16_reduced_precision_reduction` on (PyTorch's default), cuBLAS may split a GEMM's K dimension and add the partials *in bf16*.
Whether it does depends on the matrix shape **and the GPU**. On an **L4**, traces of 288 and 320 tokens came back wrong. The A100 and the CPUs never showed it. Fix: switch the flag off in `merge.instantiate`; `test_exact_for_every_trace_length` guards it.
Lesson: an exactness proof in float64 is necessary, not sufficient. **Run on more than one GPU.**

---

## 🧭 What's new, what's not (honestly)

**Not new:** putting a machine into Qwen3.5's weights, or running a neural net on it.
[Qwenputer](https://github.com/divagr18/qwenputer) put a RISC-V CPU there and ran stories260K on an in-weights NPU; [QwenPU](https://github.com/divagr18/QwenPU) added a 64-lane int8×int4 vector unit; qwenGPU did a SIMT GPU.
The design vocabulary (ballast-normalised RMSNorm, DeltaNet rows as RAM, tokens as the trace) is theirs.

**New (to my knowledge; one search, not exhaustive): the architecture.**

| | machine | control | state holds | a token is |
|---|---|---|---|---|
| Qwenputer | RV32IM CPU | PC, fetch, decode | registers, code, data | an instruction |
| QwenPU | CPU + vector unit | the CPU's program | vector regs, weight rows, KV | a vector-op step |
| qwenGPU | SIMT GPU | host-compiled kernel | registers, global memory | a warp-launch record |
| **qwenTPU** | **weight-stationary array** | **none** | **weights *and* partial sums** | **a dataflow step** |

No program counter, no instruction stream: the layer stack *is* the datapath. One multi-hot read is the whole reduction. Bit-plane counting keeps every value bf16-exact.
Layers chain through the model's own tokens. It is **not faster than QwenPU at the op level**: its matvec streams 64 lanes per token, we do 4 to 64 MACs per token.

### 🤔 Do we need TinyTPU?

No. [TinyTPU](https://github.com/tiny-tpu-v2/tiny-tpu) is real RTL and this repo shares **no code, netlist or test** with it. It's the *reference for the architecture*:
weights held in the PEs, activations streamed, integer accumulate, a post-processing unit. Two honest differences: this array is not *systolic in space*
(no PE-to-PE forwarding: every column sees the same `ACT` token), and it's inference-only. "TPU" here means *weight-stationary dataflow matrix unit with a requantising output stage*.

---

## 📊 Results

**Arrays** (`QWENTPU_CFG=rows,cols,abits,wbits[,g,h]`; weights two's complement, activations unsigned, output requantised to `abits`).

| array | MACs / matvec | tokens / matvec | MACs / token | chip layers |
|---|---|---|---|---|
| 8×8 u3·s3 | 64 | 16 | 4 | **2** |
| 8×8 u4·s4 | 64 | 16 | 4 | 4 |
| 8×8 **u5·s5** | 64 | 16 | 4 | 7 |
| 16×16 u3·s2 | 256 | 32 | 8 | 2 |
| 16×16 u4·s4 | 256 | 32 | 8 | 5 |
| 16×16 **u5·s5** | 256 | 32 | 8 | 10 |
| **32×32** u3·s3 | **1,024** | 64 | 16 | 6 |
| 32×32 u4·s4 | 1,024 | 64 | 16 | 10 |
| 64×64 u3·s3 *(compiled, not run on a model)* | 4,096 | 128 | 32 | 12 |

**Per model** (8×8 u3s3; KL measured on the A100, three prompts, `results/a100/kl.out`).

| | 0.8B | 2B | 4B |
|---|---|---|---|
| layers (Qwen + TPU) | 24 + 2 | 24 + 2 | 32 + 2 |
| hidden (base → merged) | 1024 → 1280 | 2048 → 2304 | 2560 → 2816 |
| value / key heads | 16 / 16 | 16 / 16 | **32 / 16** |
| pure-text KL vs base | 4.4e-4 – 6.9e-4 | 3.2e-4 – 7.8e-4 | 3.3e-4 – 5.9e-4 |

**Invisible to the text path.** Chip tokens are muted inside Qwen's own layers (GDN `a=b=z=−1024`, attention output gate, an anti-sink key), `tau` keeps them from being sampled in prose,
and with the chip-only path (or `HybridSession`'s cache restore) a TPU run changes **nothing** in Qwen's context:

| text after a TPU run vs the same model with no run | KL |
|---|---|
| muting only | not zero: first token KL 1.1 – 1.3, because Qwen's conv window sees zeros for ~3 tokens; mean KL from token 4 on is 0.014 – 0.059 (0.8B / 2B / 4B) |
| `HybridSession` cache restore | **0.0** vs a run with no chip; max 0.0018 – 0.0022 vs the unmodified base |
| chip-only path | **0.0 (asserted bit-identical)** |
| (for reference) merged vs the unmodified base, pure text | 3.2e-4 – 7.8e-4 |

**Throughput** (A100, 0.8B, teacher-forced replay of 32 vectors per weight load, TPU layers only, `results/a100/sweep2.jsonl`). Tokens/s, batch 16 unless noted:

| array | tok/s | MACs/s |
|---|---|---|
| 8×8 | 487k | 1.7M |
| 8×8 `g2h2` | 398k | **2.5M** |
| 16×16 u3s2 | 476k | **3.0M** |
| 16×16 u4s4 | 211k | 1.3M |
| 32×32 u3s3 | 155k (batch 1) | 1.7M |

---

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

- **Depth is the price of width.** Wide operands cost chip layers (16×16 u5s5 is 10; 64×64 is 12) and packing costs layers and vocabulary. Only `g2h2` on 8×8 was a clear win.
- **No fused kernel.** The CUDA-graph executor removes launch overhead, not the cost of the HF kernels themselves. A purpose-built fused executor (as Qwenputer has) could go further.
- **Teacher-forced numbers are prefill throughput.** A matvec's tokens don't depend on each other, so they batch; only the hop between layers waits. The free-running table is the honest "chained" number.
- **Memory:** the old 24 GB OOM is fixed (the base model is rebuilt on demand, so every 2B suite fits on the 24 GB A10G and L4 and on 16 GB hosts). A 16 GB T4 still runs out of VRAM on the larger 2B arrays (16×16 u5s5 and the 32×32 ones). 4B has been run on the A100 and the Xeon; not on the 24 GB cards.
- **Coverage is what `results/` says.** Every machine claimed here has raw logs there. An earlier L40S, Graviton3 and Apple-silicon run was lost in a crash and is not claimed.
- Text-only checkpoints. Inference only (no backprop, unlike TinyTPU). Merged checkpoints are built on the fly, not uploaded.

## 🙏 Credits

Design vocabulary: **Qwenputer** (Divyansh Agrawal), with which this repo shares no code. Architecture reference: **TinyTPU** (Surya Sure et al.), likewise no code. Base models: **Qwen3.5**. MIT licensed.
