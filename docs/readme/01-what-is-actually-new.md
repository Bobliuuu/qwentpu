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
