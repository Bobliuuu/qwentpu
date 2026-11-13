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
