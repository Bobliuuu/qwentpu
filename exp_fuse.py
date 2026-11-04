"""Fused-kernel experiments on the chip path: eager vs torch.compile variants, bit-exactness checked on hidden states.
    QWEN_DIR=q08 QWEN_DEVICE=cuda python exp_fuse.py eager|default|max-autotune|max-autotune-no-cudagraphs|reduce-overhead"""
import sys, time, json
import numpy as np, torch
sys.path.insert(0, "tests")
import test_merged as T
from qwentpu import circuit as C
from qwentpu.merge import build_merged, instantiate
from qwentpu.mute import fit_anti_sink
from qwentpu.driver import ChipPath
from transformers import AutoTokenizer
dev = "cuda"
tok = AutoTokenizer.from_pretrained(T.SRC); base = T._build_base().to(dev)
sinks = fit_anti_sink(base, tok, T.PROMPTS); del base
tc, sd, v0, lay, g = build_merged(T.SRC, sinks=sinks); model = instantiate(tc, sd).to(dev)
assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is False
chip = ChipPath(model, v0)
rng = np.random.default_rng(123)
cases = []
for _ in range(20):
    W, X = C.rand_case(rng, rng.random() < 0.5, 32); t, w = T.trace(W, X); cases.append(torch.tensor([[v0 + x for x in t]], device=dev))


def run(fn, reps=1):
    outs = [fn(ids, None) for ids in cases]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(reps):
        for ids in cases: fn(ids, None)
    torch.cuda.synchronize(); dt = time.time() - t0
    return outs, sum(i.numel() for i in cases) * reps / dt


def hid(ids, past): return chip.hidden(ids, past)


ref_h = [hid(i, None) for i in cases]

def graphed(fn):
    """one CUDA graph per distinct trace length (static input), whole chip forward"""
    gs = {}
    def f(ids, past):
        T = ids.shape[1]
        if T not in gs:
            st = ids.clone(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): fn(st, None)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): o = fn(st, None)
            gs[T] = (g, st, o)
        g, st, o = gs[T]; st.copy_(ids); g.replay(); return o.clone()
    return f

def prof():
    from torch.profiler import profile, ProfilerActivity
    for ids in cases[:2]: hid(ids, None)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for ids in cases[:5]: hid(ids, None)
        torch.cuda.synchronize()
    ev = p.key_averages()
    gpu = sum(e.device_time_total for e in ev if e.device_type.name == "CUDA") / 5
    print(json.dumps({"variant": "profile", "gpu_kernel_us_per_case": round(gpu, 1), "n_kernels_per_case": sum(e.count for e in ev if e.device_type.name == "CUDA") // 5}), flush=True)
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60), flush=True)

def go(mode):
    fn = hid
    if mode == "profile": return prof()
    if mode == "graph": fn = graphed(hid)
    elif mode.startswith("compile:"):
        m = mode.split(":", 1)[1]
        kw = {"default": {}, "max-autotune": {"mode": "max-autotune"}, "max-autotune-no-cudagraphs": {"mode": "max-autotune-no-cudagraphs"}, "reduce-overhead": {"mode": "reduce-overhead"}}[m]
        fn = torch.compile(hid, dynamic=False, **kw)
    t0 = time.time()
    try:
        outs, _ = run(fn)
    except Exception as e:
        print(json.dumps({"variant": mode, "error": repr(e)[:600]}), flush=True); return
    ct = time.time() - t0
    outs, tps = run(fn, reps=5)
    print(json.dumps({"variant": mode, "tok_per_s": round(tps, 1), "hidden_bit_identical": all(torch.equal(a, b) for a, b in zip(outs, ref_h)), "setup_s": round(ct, 1), "n_cases": len(cases), "tokens": [c.shape[1] for c in cases][:3]}), flush=True)

for mode in (sys.argv[1:] or ["eager"]): go(mode)
