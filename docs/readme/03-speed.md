Depth is the price of width, and the biggest wins are in skipping work, not adding it (details and the packing trade-off in the [write-up](WRITEUP.md)):

| A100 · 0.8B · 8×8 u3s3 | |
|---|---|
| whole network, batch 1, same trace | 6,629 tok/s |
| **TPU layers only** (Qwen's muted layers skipped, bit-identical) | **74,462 tok/s** (11×) |
| TPU layers only, **one CUDA graph per trace** (`bench.py --graph`, bit-identical) | **~341,000 tok/s** (mean of 3 runs ×2 traces, range 299k-377k; 4.5× over the eager TPU-layers rerun at 75,207) |
| free-running chained matvecs, TPU layers + **CUDA graphs** | **705 matvec/s** |

<p align="center"><img src="docs/chip-path.svg" alt="chip-only path" width="100%"></p>

<p align="center"><img src="docs/anim/stack.gif" alt="animation: chip tokens skip the muted Qwen layers" width="100%"></p>
