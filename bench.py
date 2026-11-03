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
ap.add_argument("--device", default="cpu"); ap.add_argument("--batch", type=int, default=1)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(T.SRC); base = T.base_model().to(a.device)
sinks = fit_anti_sink(base, tok, T.PROMPTS); del base
tc, sd, v0, lay, g = build_merged(T.SRC, sinks=sinks); model = instantiate(tc, sd).to(a.device)
head = model.lm_head.weight[v0: v0 + len(C.VOCAB)]               # chip rows
rng = np.random.default_rng(123); pos = bad = mid = ntok = 0; elapsed = 0.0
_W, _X = C.rand_case(np.random.default_rng(0), True); _t, _ = T.trace(_W, _X)       # warmup: kernel compile / autotune
with torch.no_grad():
    model.model(input_ids=torch.tensor([[v0 + x for x in _t]] * a.batch, device=a.device))
if a.device.startswith("cuda"): torch.cuda.synchronize()
done = 0
while done < a.n_mat:
    B = min(a.batch, a.n_mat - done); seqs, wants = [], []
    for _ in range(B):
        W, X = C.rand_case(rng, rng.random() < 0.5); t, w = T.trace(W, X); seqs.append(t); wants.append(w)
    ids = torch.tensor([[v0 + x for x in t] for t in seqs], device=a.device)
    if a.device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state
        pred = (h @ head.T).argmax(-1).cpu().numpy()
    if a.device.startswith("cuda"): torch.cuda.synchronize()
    elapsed += time.time() - t0; ntok += ids.numel()
    for b in range(B):
        for t, w in enumerate(wants[b]):
            if pred[b, t] != w and os.environ.get('BENCH_VERBOSE'): print('MISMATCH', done + b, t, int(pred[b, t]), w, flush=True)
            pos += 1; bad += int(pred[b, t] != w); mid += int(w != C.NOP and 1 <= w - C.OUT0 <= C.AMAX - 1)
    done += B
res = {"array": f"{C.KR}x{C.NC} u{C.AB}s{C.WB}", "model": os.path.basename(T.SRC.rstrip("/")), "device": a.device, "batch": a.batch,
       "hidden": lay.hidden, "layers": tc["num_hidden_layers"], "positions": pos, "mismatches": bad,
       "mid_range_outputs": mid, "tokens": ntok, "tok_per_s": round(ntok / elapsed, 1)}
print(json.dumps(res))
