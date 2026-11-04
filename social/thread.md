# qwenTPU: Twitter thread draft

Images are in `social/images/` (regenerate with `social/render_images.sh`). Video: `social/qwentpu_demo.mp4`. Every number is from `results/`.

**1/** (video + banner.png)
Airplanes don't flap their wings. This TPU doesn't have a chip.
We compiled a matrix-multiply array into the weights of a stock Qwen3.5. No training. Integer-exact. 🧵
github.com/Bobliuuu/qwentpu · bobliuuu.github.io/qwentpu

**2/** (arch.png)
It's a weight-stationary TPU, up to 32×32 (1,024 MACs per matvec), 5-bit operands. Every new weight is computed; Qwen's own weights are untouched.
No program counter, no instruction stream. The layer stack is the datapath.

**3/** (life-of-a-matvec.png)
Why it works: a Gated DeltaNet head is a little RAM with a built-in adder.
• one-hot write = store a weight
• multi-hot read = the exact SUM of the chosen rows, a whole reduction in one read
• the sampled token feeds the next layer
A TPU is mostly RAM and adders. It was sitting there the whole time.

**4/** (adder-tree.png)
The hard part: bf16 only compares small integers exactly.
Bit-plane counting, then a top-down fold with saturating clamps. Saturating early is provably lossless, which is what makes 5-bit and 16×16 / 32×32 possible.

**5/** (chip-path.png)
Qwen's own layers are muted for chip tokens, so we skip them: 6,629 → 74,462 tok/s at batch 1 on an A100 (11×, same trace), bit-identical. With CUDA graphs, free-running chained matvecs hit 705/s.
A TPU run leaves Qwen's context untouched (KL = 0.0).

**6/** (hardware.png)
"Exact on an A100" is a claim about an A100. 1,139,712 replayed token positions, 0 mismatches, on A100, A10G, L4 and an x86 Xeon. (T4 ran the test suites only; it runs out of 16 GB on the bigger 2B arrays.)

**7/** (bug-museum.png)
Cross-hardware testing found two bugs float64 simulation could never see: an irrational 1/√128 query scale, and cuBLAS split-K adding partials in bf16 on an L4. Lesson: run on more than one GPU.

**8/** 
Honest bit: putting a machine in Qwen3.5's weights isn't new. Qwenputer did a RISC-V CPU, QwenPU a vector unit, qwenGPU a SIMT GPU. What we think is new (one search, not exhaustive): a dataflow array with no control, one-read reductions, exact bf16 integers to 5 bits, and zero trace.
Not faster than QwenPU per op. Details + limits in the write-up.
