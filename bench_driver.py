"""End-to-end driver throughput: chained matvecs through a merged Qwen3.5 with a real cache, one hop at a time.

Unlike bench.py (teacher-forced: a whole trace is one prefill), here layer n+1's activations are the model's own
sampled tokens from layer n, so every hop waits for the model.  A matvec is two forward calls (KR ACT tokens,
then NC READ tokens), so per-token launch cost is amortised over a chunk.  Verified against NumPy.
    QWEN_DIR=q08 python bench_driver.py [--device cuda] [--layers 4] [--vectors 8]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "tests")
import test_merged as T
from qwentpu import circuit as C
from qwentpu.driver import TpuSession, ref_mlp
from qwentpu.merge import build_merged, instantiate
from qwentpu.mute import fit_anti_sink
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu"); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--vectors", type=int, default=8)
ap.add_argument("--mode", default="tpu-layers", choices=["full", "tpu-layers", "graph"])
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(T.SRC); base = T.base_model().to(a.device)
sinks = fit_anti_sink(base, tok, T.PROMPTS); del base
tc, sd, v0, lay, g = build_merged(T.SRC, sinks=sinks); model = instantiate(tc, sd).to(a.device)

rng = np.random.default_rng(7)
n = min(C.KR, C.NC) - 1
lo, hi = max(C.wlo(), -1), min(C.whi(), 2)
Ws = [rng.integers(lo, hi + 1, (n + 1, n)) for _ in range(a.layers)]
X = rng.integers(0, min(C.AMAX, 3) + 1, (a.vectors, n))
fast = {"full": False, "tpu-layers": True, "graph": "graph"}[a.mode]
s = TpuSession(model, v0, fast=fast)
s.feed(C.load_trace(np.zeros((C.KR, C.NC), int)))                      # warmup
for _ in range(2): s.matvec(np.zeros(C.KR, int))
for ncol in {n, C.NC}: s.matvec(np.zeros(C.KR, int), ncol)
if a.device.startswith("cuda"): torch.cuda.synchronize()
t0 = time.time(); tokens0 = s.n_tokens
got = s.mlp(Ws, X)
if a.device.startswith("cuda"): torch.cuda.synchronize()
dt = time.time() - t0
want = ref_mlp(Ws, X)
toks = s.n_tokens - tokens0
print(json.dumps({"array": f"{C.KR}x{C.NC} u{C.AB}s{C.WB}" + (f" g{C.G}h{C.H}" if C.G * C.H > 1 else ""), "model": os.path.basename(T.SRC.rstrip("/")), "device": a.device,
                  "mode": a.mode, "chain_layers": a.layers, "vectors": a.vectors, "tokens": toks, "seconds": round(dt, 2),
                  "tok_per_s": round(toks / dt, 1), "matvecs_per_s": round(a.layers * a.vectors / dt, 2),
                  "match_numpy": bool((got == want).all())}))
