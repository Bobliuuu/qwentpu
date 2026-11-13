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
| 64×64, 3-bit | 12 |
