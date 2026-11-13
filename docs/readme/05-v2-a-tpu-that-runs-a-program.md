<p align="center"><a href="docs/qwentpu_v2_demo.mp4"><img src="docs/anim/v2_demo.gif" alt="v2 classifying digits: the model emits every token itself" width="100%"></a><br><sub>v2 classifying held-out digits through Qwen3.5-0.8B, animated from the real traces. <a href="docs/qwentpu_v2_demo.mp4">Full 79 s video (mp4)</a></sub></p>

Two versions live in this repo. v1 is tagged `v1.0`; v2 is on `main` as new files; the v1 test suite still passes unchanged on it (v1 and v2 share `circuit.py`, `driver.py` and `merge.py`, which gained small hooks, and no test compares v1 output against the `v1.0` tag).

| | **v1: matvec array** | **v2: a TPU that runs a program** |
|---|---|---|
| What it is | weight-stationary matrix-multiply array in Qwen's weights | the same array plus a tiny instruction set, exact ReLU and clamp, and a sequencer |
| Who picks the chip tokens | **the host** (CELL, ACT, READ, OUT) | **the model**: after the prompt it emits every token itself |
| What the host does | drives the machine token by token | builds the prompt (weights, input, entry point) and runs `argmax; append` until HALT |
| Nonlinearity | none | `clip(relu(x@W), 0, max)` in the datapath, at zero extra layers |
| What the prompt chooses | data only | the network: depth, per-layer widths, weights, input |
| Proven on | A100, A10G, L4, T4, x86 Xeon (1,361,280 replayed positions, 0 mismatches) | Qwen3.5-0.8B on an A100 (below) |

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
