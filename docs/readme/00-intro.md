We took a stock **Qwen3.5** (0.8B, 2B or 4B), compiled a few new layers into it, and it now contains a **weight-stationary matrix-multiply array**,
up to **64×64 (4,096 MACs per matvec, at 3-bit)**, with operands up to **5 bits** (16×16 u5s5). Every new weight is computed; Qwen's own weights are untouched. Output is integer-exact against NumPy
(0 mismatches in 1,361,280 replayed token positions on five machines), plain text generation is unchanged (KL 3e-4 to 8e-4), and on the chip-only path (or with the driver's cache restore) a TPU run leaves no trace in Qwen's context.

<p align="center"><a href="docs/qwentpu_demo.mp4"><img src="docs/demo.gif" alt="qwenTPU v1 running a matvec inside Qwen3.5, token by token" width="100%"></a><br><sub>v1 running inside Qwen3.5-0.8B on real tokens. <a href="docs/qwentpu_demo.mp4">Full 70 s video (mp4)</a></sub></p>

<p align="center"><img src="docs/arch.svg" alt="architecture" width="100%"></p>
