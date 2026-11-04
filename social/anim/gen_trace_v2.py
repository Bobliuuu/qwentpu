#!/usr/bin/env python3
"""Generate the REAL v2 traces that v2_dashboard.html and the v2_* clips step through (trace_v2.js / trace_v2.json).

For the 20 digits of examples/mlp/golden.npz (the first 20 held-out digits of examples/mlp/train.py's split, rebuilt here
from sklearn load_digits so the 8x8 images are available) it
  1. runs the ISA path: qwentpu.program.mlp_program -> isa.run / isa.encode / isa.emitted_trace and asserts the prompt ids,
     every emitted token id, the logits and the prediction equal golden.npz and the independent numpy network,
  2. runs the model path: the repo's CPU circuit-IR sequencer (qwentpu.sequencer2.ProgramMachine, Dmax=2 n=16 u5s5, 20 chip layers,
     float64 IR with all exactness conditions asserted), feeds it the program prompt and asserts that the IR's next-token
     prediction at every emitted position equals the expected stream and that its logits equal golden logits,
  3. runs one compiled small model (Dmax=4 n=4 u3s3) with programs of depth 1, 2 and 4 chosen only by the prompt (as tests/test_sequencer2.py).
Nothing is written unless every assertion holds.

    cd social/anim && python3 gen_trace_v2.py        (needs numpy + scikit-learn; ~10 min on CPU for the IR runs, cached in .v2_ir_cache.json)
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); sys.path.insert(0, os.path.join(ROOT, "examples", "mlp"))
from qwentpu import circuit as C, isa, program
from qwentpu.compiler import evaluate as ir
from qwentpu.sequencer import ref_chain
from qwentpu.sequencer2 import ProgramMachine
import train as T
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

CACHE = os.path.join(HERE, ".v2_ir_cache.json")
z = np.load(os.path.join(ROOT, "examples", "mlp", "golden.npz"))
a = isa.Arr(*[int(v) for v in z["cfg"]])
W1, W2 = z["W1"], z["W2"]; N = len(z["label"])

# ---- images: rebuild the split of train.py (random_state=0, stratified, test_size .3) and check it is the golden set -------------
d = load_digits(); I = d.images / 16.0
Zall = I.reshape(-1, 4, 2, 4, 2).mean((2, 4)).reshape(len(I), 16)[:, 1:]
_, ite = train_test_split(np.arange(len(I)), test_size=0.3, random_state=0, stratify=d.target)
assert (T.quant_in(Zall[ite][:N], 12) == z["x_test"]).all() and (d.target[ite][:N] == z["label"]).all()
imgs = [d.images[ite[i]].astype(int).tolist() for i in range(N)]               # 8x8, 0..16

# ---- 1. ISA path ------------------------------------------------------------------------------------------------------------------
digits = []
for i in range(N):
    x = z["x_test"][i]
    prog = program.mlp_program(a, [W1, W2], x)
    ref = isa.run(a, prog); ref_np = program.run_mlp_np(a, [W1, W2], x)
    prompt = isa.encode(a, prog); em = isa.emitted_trace(a, prompt)
    assert (ref == ref_np).all() and (ref == z["logits"][i]).all()
    assert prompt == z["prompt_ids"][i].tolist() and em == z["expected_emitted_ids"][i].tolist()
    assert (isa.decode_result(a, em) == z["logits"][i]).all() and int(ref[:10].argmax()) == int(z["pred"][i])
    # layer values (for the animation): pre-activation sums, relu, clamp
    v = np.zeros(a.kr, dtype=np.int64); v[:15] = x; v[15] = 1
    T1 = program.tile(a, W1, True); T2 = program.tile(a, W2, False)
    raw1 = v @ T1; y1 = np.clip(np.maximum(raw1, 0), 0, a.amax); v2 = y1.copy()
    raw2 = v2 @ T2; y2 = np.clip(np.maximum(raw2, 0), 0, a.amax)
    assert (y2 == ref).all()
    # emitted token stream: (READ k, Y) x16 per layer, then HALT; decode each Y id back to its value
    toks = []
    for j in range(0, 64, 2):
        l, k = divmod(j // 2, 16)
        assert em[j] == a.tok_read(k)
        y = em[j + 1]
        if l == 0: assert y == a.tok_act(k, int(y1[k])); kind = "ACT"; val = int(y1[k])
        else: assert y == a.tok_out(int(y2[k])); kind = "OUT"; val = int(y2[k])
        toks.append({"layer": l, "k": k, "read": int(em[j]), "y": int(y), "kind": kind, "val": val})
    assert em[64] == a.op_halt
    digits.append({"i": i, "label": int(z["label"][i]), "pred": int(z["pred"][i]), "img": imgs[i], "x": x.tolist(),
                   "raw1": raw1.tolist(), "y1": y1.tolist(), "raw2": raw2.tolist(), "logits": [int(t) for t in ref[:10]],
                   "toks": toks, "ids_ok": True})
print("ISA path: 20/20 prompts, 65 emitted ids, logits and predictions equal golden + numpy")

# ---- 2. model path: circuit IR sequencer (CPU) -----------------------------------------------------------------------------------
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
seq = ProgramMachine(2, 16, 5, 5).install(); g, dd = C.build(); chip_layers = g.n_layers(); assert chip_layers == 20
t0 = time.time()
for i in range(N):
    p = seq.program_prompt([W1, W2], z["x_test"][i]); em = seq.expected(p)
    assert len(p) == 1056 and len(em) == 64
    if str(i) not in cache:
        m = ir.evaluate(g, list(p) + list(em))
        pred = [sum((1 << b) * int(ir.ev(dd["code"][b], m[t:t + 1])[0] > 0) for b in range(C.CODE)) for t in range(len(p) - 1, len(p) + len(em))]
        cache[str(i)] = pred; json.dump(cache, open(CACHE, "w"))
        print(" IR digit", i, "%.0fs" % (time.time() - t0), flush=True)
    pred = cache[str(i)]
    assert pred == em + [seq.halt], i                                              # the model's own next token at every position
    lg = seq.program_logits(em, 10)
    assert lg == [int(t) for t in z["logits"][i][:10]] and int(np.argmax(lg)) == int(z["pred"][i])
    digits[i]["ir_ok"] = True
ProgramMachine.uninstall()
print("IR sequencer: 20/20 digits, model-emitted stream == expected, logits == golden; chip layers", chip_layers)

# ---- 3. one compiled model, programs of depth 1, 2, 4 chosen by the prompt -------------------------------------------------------
def rand_prog(rng, depth, widths):
    Ws, n_in = [], widths[0]
    for w in widths[1:depth + 1]:
        W = rng.integers(-1, 3, (n_in + 1, w)); W[-1] = rng.integers(0, 3, w); Ws.append(W); n_in = w
    return Ws, rng.integers(0, 4, widths[0])

DM, NW = 4, 4
sq = ProgramMachine(DM, NW, 3, 3).install(); g2, d2 = C.build(); layers2 = g2.n_layers()
progs = []
rng = np.random.default_rng(2024)
for depth in (1, 2, 4):
    for trial in range(50):
        Ws, x = rand_prog(rng, depth, [3] * (depth + 1))
        ys = []; v = list(x) + [0] * (NW - 1 - len(x)) + [1]
        want = ref_chain(Ws, x)
        # want a run where layers differ so the picture shows a real computation
        if len(set(int(t) for t in want)) > 1 and sum(1 for t in want if 0 < t < C.AMAX) >= 2: break
    p = sq.program_prompt(Ws, x); em = sq.expected(p)
    m = ir.evaluate(g2, list(p) + list(em))
    pred = [sum((1 << b) * int(ir.ev(d2["code"][b], m[t:t + 1])[0] > 0) for b in range(C.CODE)) for t in range(len(p) - 1, len(p) + len(em))]
    assert pred == em + [sq.halt] and len(em) == 2 * depth * NW
    assert sq.program_logits(em, Ws[-1].shape[1]) == [int(t) for t in ref_chain(Ws, x)]
    # per-layer values for the picture (clamp(relu(.)) chain, block layout identical to the prompt)
    layers = []; vin = [int(t) for t in x]
    for W in Ws:
        raw = [int(t) for t in (np.array(vin + [1]) @ W)]; y = [int(min(max(t, 0), C.AMAX)) for t in raw]; layers.append({"raw": raw, "y": y}); vin = y
    progs.append({"depth": depth, "entry": DM - depth, "prompt_len": len(p), "emitted": len(em), "x": [int(t) for t in x],
                  "layers": layers, "out": sq.program_logits(em, Ws[-1].shape[1]), "W": [np.asarray(W).tolist() for W in Ws]})
    print("depth", depth, "entry block", DM - depth, "emitted", len(em), "== expected, logits == numpy", progs[-1]["out"])
ProgramMachine.uninstall()

out = {"cfg": [int(t) for t in z["cfg"]], "float_acc": float(z["float_acc"]), "int_acc": float(z["int_acc"]), "prompt_len": len(z["prompt_ids"][0]),
       "emitted_len": 65, "chip_layers": chip_layers, "digits": digits, "W1": W1.tolist(), "W2": W2.tolist(), "T1": T1.tolist(), "T2": T2.tolist(),
       "depth_demo": {"dmax": DM, "n": NW, "ab": 3, "wb": 3, "chip_layers": layers2, "progs": progs},
       "program": ["LOAD weights", "LOAD x", "MATMUL", "RELU", "MATMUL", "OUT", "HALT"]}
blob = json.dumps(out, default=int)
open(os.path.join(HERE, "trace_v2.json"), "w").write(blob)
open(os.path.join(HERE, "trace_v2.js"), "w").write("window.TRACE2 = " + blob + ";\n")
print("wrote trace_v2.json/js; all assertions passed")
