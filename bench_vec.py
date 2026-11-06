"""v2 vectorization on the real merged Qwen3.5: exactness and speed, scalar (vec=1) vs vec neurons per token.

usage: python bench_vec.py   (env: QWEN_DIR checkpoint, QWEN_DEVICE=cuda; one compiled model per (config, vec))
For every config and vec: (1) exactness, the host loop is `argmax; append` until HALT, both readouts (code dims, lm_head),
programs of every depth 1..Dmax with random weights, emitted tokens == the numpy golden and logits == the plain network;
the golden MLP (15-15-10, u5s5) when n = 16.  (2) speed, the deepest program, 3 warm-up + 20 timed inferences on the eager
chip path and on the CUDA-graph chip path: prefill ms, emitted-token loop ms, tokens/s, ms per inference."""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tests"))
from qwentpu import circuit as C                                   # noqa: E402
from qwentpu.driver import TpuSession                               # noqa: E402
from qwentpu.merge import build_merged, instantiate                 # noqa: E402
from qwentpu.sequencer2 import ProgramMachine, generate, ref_chain  # noqa: E402
from test_sequencer2 import rand_prog                               # noqa: E402

SRC = os.environ.get("QWEN_DIR") or sys.exit("set QWEN_DIR to a Qwen3.5 checkpoint directory")
CFGS = [(4, 4, 3, 3, (1, 2, 4)), (2, 8, 3, 3, (1, 2, 4)), (2, 8, 5, 5, (1, 2)), (2, 16, 5, 5, (1, 2))]
if os.environ.get("CFGS"):
    CFGS = [c for i, c in enumerate(CFGS) if str(i) in os.environ["CFGS"].split(",")]
OUT = os.environ.get("OUT", "")
sync = torch.cuda.synchronize
bad = 0


def emit(rec):
    print(json.dumps(rec), flush=True)
    if OUT:
        open(OUT, "a").write(json.dumps(rec) + "\n")


for dmax, n, ab, wb, vecs in CFGS:
    for vec in vecs:
        seq = ProgramMachine(dmax, n, ab, wb, vec=vec).install()
        t0 = time.time()
        tc, sd, v0, lay, g = build_merged(SRC, mute_qwen=False)
        model = instantiate(tc, sd, "cuda")
        info = dict(dmax=dmax, n=n, ab=ab, wb=wb, vec=vec, KR=C.KR, chip_layers=g.n_layers(), vocab=len(C.VOCAB),
                    hidden=tc["hidden_size"], build_s=round(time.time() - t0, 1))
        rng = np.random.default_rng(dmax * 7 + n + 1000 * ab)
        runs = ok = 0
        # ---- exactness -------------------------------------------------------------------------------------------
        for ro in ("code", "lm_head"):
            for depth in range(1, dmax + 1):
                for trial in range(3):
                    widths = [int(rng.integers(1, n)) for _ in range(depth + 1)]
                    Ws, x = rand_prog(rng, depth, widths, n)
                    p = seq.program_prompt(Ws, x)
                    sess = TpuSession(model, v0, fast=True, readout=ro)
                    em = generate(sess.feed, p, seq.halt)
                    good = em == seq.expected(p) and seq.program_logits(em, Ws[-1].shape[1]) == list(ref_chain(Ws, x))
                    runs += 1; ok += good
                    if not good: print("MISMATCH", info, ro, depth, widths, flush=True)
        golden = None
        if (n, ab) == (16, 5) and os.path.exists(os.path.join(HERE, "examples", "mlp", "golden.npz")):
            z = np.load(os.path.join(HERE, "examples", "mlp", "golden.npz")); gok = 0
            for i in range(20):
                p = seq.program_prompt([z["W1"], z["W2"]], z["x_test"][i]); sess = TpuSession(model, v0, fast=True)
                em = generate(sess.feed, p, seq.halt)
                gok += seq.program_logits(em, 10) == list(z["logits"][i][:10]) and em == seq.expected(p)
            golden = f"{gok}/20"; runs += 20; ok += gok
        bad += runs - ok
        emit(dict(kind="exact", **info, runs=runs, exact=ok, golden_mlp=golden))
        # ---- speed: deepest program, the same input every time ---------------------------------------------------
        widths = [n - 1] * (dmax + 1)
        Ws, x = rand_prog(np.random.default_rng(99), dmax, widths, n)
        p = seq.program_prompt(Ws, x); want = seq.expected(p); ref = list(ref_chain(Ws, x))
        for name, fast in (("eager", True), ("graph", "graph")):
            def infer():
                sess = TpuSession(model, v0, fast=fast, readout="code")
                sync(); a = time.perf_counter()
                nxt = sess.feed(list(p))[-1]; sync(); b = time.perf_counter()
                em = []
                while nxt != seq.halt:
                    em.append(nxt); nxt = sess.feed([nxt])[-1]
                sync(); c = time.perf_counter()
                return em == want and seq.program_logits(em, Ws[-1].shape[1]) == ref, len(em), b - a, c - b
            try:
                for _ in range(3): infer()
                rs = [infer() for _ in range(20)]
            except Exception as e:
                emit(dict(kind="speed", mode=name, **info, error=f"{type(e).__name__}: {str(e)[:200]}")); continue
            good = sum(r[0] for r in rs); bad += len(rs) - good
            pre, dec = np.mean([r[2] for r in rs]), np.mean([r[3] for r in rs]); ne = rs[0][1]
            emit(dict(kind="speed", mode=name, **info, prompt_tokens=len(p), emitted=ne, exact=f"{good}/{len(rs)}",
                      prefill_ms=round(pre * 1e3, 2), loop_ms=round(dec * 1e3, 2), tok_per_s=round(ne / dec, 1),
                      ms_per_token=round(dec / ne * 1e3, 3), ms_per_inference=round((pre + dec) * 1e3, 2),
                      inferences_per_s=round(1 / (pre + dec), 2)))
        del model, sd; torch.cuda.empty_cache(); ProgramMachine.uninstall()
print("TOTAL BAD", bad, flush=True)
