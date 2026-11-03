"""Mass replay: random weight matrices and vectors through a merged Qwen3.5, every predicted token compared with NumPy.
    QWEN_DIR=q08 python bench.py [n_matrices] [--device cuda] [--batch 8]
With --batch B, B independent TPU programs run as one batch.  The lm_head is evaluated on the chip rows only
(text rows are irrelevant at chip positions: tau and the code decide), which saves ~all of the head's cost."""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, "tests")
import test_merged as T
from qwentpu import circuit as C
from qwentpu.merge import build_merged, instantiate
from qwentpu.mute import fit_anti_sink

ap = argparse.ArgumentParser(); ap.add_argument("n_mat", type=int, nargs="?", default=20)
ap.add_argument("--device", default="cpu"); ap.add_argument("--readout", default="code", choices=["code", "lm_head"]); ap.add_argument("--vectors", type=int, default=4, help="vectors per weight load"); ap.add_argument("--full", action="store_true", help="run all layers (default: TPU layers only)"); ap.add_argument("--batch", type=int, default=1)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(T.SRC); base = T.base_model().to(a.device)
sinks = fit_anti_sink(base, tok, T.PROMPTS); del base
tc, sd, v0, lay, g = build_merged(T.SRC, sinks=sinks); model = instantiate(tc, sd).to(a.device)
head = model.lm_head.weight[v0: v0 + len(C.VOCAB)]               # chip rows
from qwentpu.driver import ChipPath
chip = None if a.full else ChipPath(model, v0, readout=a.readout)
rng = np.random.default_rng(123); pos = bad = mid = ntok = nvec = 0; elapsed = 0.0
_W, _X = C.rand_case(np.random.default_rng(0), True); _t, _ = T.trace(_W, _X)       # warmup: kernel compile / autotune
with torch.no_grad():
    (chip.tokens(torch.tensor([[v0 + x for x in _t]] * a.batch, device=a.device), None) if chip else model.model(input_ids=torch.tensor([[v0 + x for x in _t]] * a.batch, device=a.device)))
if a.device.startswith("cuda"): torch.cuda.synchronize()
done = 0
while done < a.n_mat:
    B = min(a.batch, a.n_mat - done); seqs, wants = [], []
    for _ in range(B):
        W, X = C.rand_case(rng, rng.random() < 0.5, a.vectors); t, w = T.trace(W, X); seqs.append(t); wants.append(w)
    ids = torch.tensor([[v0 + x for x in t] for t in seqs], device=a.device)
    if a.device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        pred = (chip.tokens(ids, None) if chip else (model.model(input_ids=ids).last_hidden_state @ head.T).argmax(-1)).cpu().numpy()
    if a.device.startswith("cuda"): torch.cuda.synchronize()
    elapsed += time.time() - t0; ntok += ids.numel(); nvec += B * a.vectors
    for b in range(B):
        for t, w in enumerate(wants[b]):
            if pred[b, t] != w and os.environ.get('BENCH_VERBOSE'): print('MISMATCH', done + b, t, int(pred[b, t]), w, flush=True)
            pos += 1; bad += int(pred[b, t] != w); mid += sum(1 <= v <= C.AMAX - 1 for v in C.out_values(w - C.OUT0)) if w != C.NOP else 0
    done += B
res = {"array": f"{C.KR}x{C.NC} u{C.AB}s{C.WB}" + (f" g{C.G}h{C.H}" if C.G * C.H > 1 else ""), "model": os.path.basename(T.SRC.rstrip("/")), "device": a.device, "batch": a.batch, "vectors": a.vectors,
       "hidden": lay.hidden, "layers": tc["num_hidden_layers"], "positions": pos, "mismatches": bad, "path": "full" if a.full else "tpu-layers",
       "mid_range_outputs": mid, "tokens": ntok, "tok_per_s": round(ntok / elapsed, 1), "macs_per_s": round(nvec * C.KR * C.NC / elapsed)}
print(json.dumps(res))
