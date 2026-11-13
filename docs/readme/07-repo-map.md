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
