#!/usr/bin/env python3
"""Generate the REAL token traces that dashboard.html steps through.

For each job we configure the repo's own circuit (qwentpu.circuit), build the compiled IR, feed it the exact token
stream the driver would (CELL tokens for W, ACT tokens for x, READ tokens), evaluate it with the repo's exact float64
evaluator (qwentpu.compiler.evaluate, which asserts bf16/silu exactness) and decode the OUT token the circuit emits at
every READ position.  Each OUT is checked against NumPy (circuit.ref_matvec) before it is written to trace.json.
Chained job: the second stage's ACT tokens are built from the OUT tokens the circuit produced for stage 1.

    cd social/anim && python3 gen_trace.py        (needs numpy; run from anywhere, it finds the repo)
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from qwentpu import circuit as C
from qwentpu.compiler import evaluate as ir


def decode(d, m, t):
    return sum((1 << i) * int(ir.ev(d["code"][i], m[t:t + 1])[0] > 0) for i in range(C.CODE))


def pick(rng_seed, kr, nc, wrange, xmax, density=1.0, tries=400):
    """Search seeds for a W, x whose outputs are mostly mid-range (not stuck on the 0 / max rails)."""
    best = None
    for s in range(tries):
        rng = np.random.default_rng(rng_seed + s)
        W = rng.integers(wrange[0], wrange[1] + 1, (kr, nc)); x = rng.integers(0, xmax + 1, kr) * (rng.random(kr) < density)
        y = C.ref_matvec(x, W); mid = int(((y >= 1) & (y <= C.AMAX - 1)).sum())
        if best is None or mid > best[0]: best = (mid, W, x)
    return best[1], best[2]


def run_job(name, cfg, stages, prompt, mode_note):
    """stages: list of (W, x or None).  x None = take the previous stage's circuit output."""
    C.configure(*cfg); g, d = C.build()
    toks, meta, marks = [], [], []
    prev = None
    for si, (W, x) in enumerate(stages):
        if x is None: x = prev              # chain: previous OUT values (decoded from the circuit's own tokens)
        for t in C.load_trace(W):
            toks.append(t); r, c = divmod((t - C.CELL0) // C.NW, C.NC)
            meta.append({"k": "CELL", "r": r, "c": c, "w": int(W[r, c]), "stage": si})
        for s, t in enumerate(C.matvec_trace(x)):
            toks.append(t)
            if s < C.KS: meta.append({"k": "ACT", "s": s, "v": [int(v) for v in x[s * C.G:(s + 1) * C.G]], "stage": si})
            else:
                k = s - C.KS; meta.append({"k": "READ", "kk": k, "stage": si, "pos": len(toks) - 1,
                                           "raw": [int(v) for v in (np.asarray(x) @ W)[k * C.H:(k + 1) * C.H]]})
        # the circuit's own answer goes in below; for chaining we need it now, so evaluate the stream so far
        m = ir.evaluate(g, toks)
        y = []
        for mt in meta:
            if mt["k"] == "READ" and mt["stage"] == si:
                out = decode(d, m, mt["pos"]); mt["out_id"] = out; mt["out"] = C.out_values(out - C.OUT0)
                y += mt["out"]
        want = [int(v) for v in C.ref_matvec(x, W)]
        assert y == want, (name, si, y, want)
        stages_info = {"W": W.tolist(), "x": [int(v) for v in x], "y": y, "numpy": want}
        marks.append(stages_info); prev = np.array(y)
    # whole stream must also evaluate exactly (one pass over every token, incl. weight rewrite for chains)
    m = ir.evaluate(g, toks)
    for mt in meta:
        if mt["k"] == "READ": assert decode(d, m, mt["pos"]) == mt["out_id"]
    for t, mt in zip(toks, meta): mt["id"] = int(t)
    return {"name": name, "cfg": list(cfg), "KR": C.KR, "NC": C.NC, "AB": C.AB, "WB": C.WB, "G": C.G, "H": C.H, "KS": C.KS,
            "chip_layers": g.n_layers(), "prompt": prompt, "mode": mode_note, "stages": marks, "tokens": meta, "ntok": len(toks),
            "vocab": len(C.VOCAB), "match_numpy": True}


jobs = []
W1, x1 = pick(10, 8, 8, (-1, 2), 5)
jobs.append(run_job("small", (8, 8, 3, 3), [(W1, x1)], "Multiply this 8x8 matrix by x", "full"))
W2, x2 = pick(500, 16, 16, (-1, 1), 3, .5)
jobs.append(run_job("bigger", (16, 16, 3, 2), [(W2, x2)], "Multiply this 16x16 matrix by x", "full"))
Wa, xa = pick(900, 8, 8, (-1, 2), 4)
Wb, _ = pick(1500, 8, 8, (-1, 2), 4)
# W2 for the chain: pick so the first stage's output is a good input (search a few seeds, keep mid-range outputs)
best = None
ya = C.ref_matvec(xa, Wa) if (C.configure(8, 8, 3, 3) or True) else None
for s in range(300):
    rng = np.random.default_rng(2000 + s); Wc = rng.integers(-1, 3, (8, 8)); yc = C.ref_matvec(ya, Wc); mid = int(((yc >= 1) & (yc <= 6)).sum())
    if best is None or mid > best[0]: best = (mid, Wc)
jobs.append(run_job("chain", (8, 8, 3, 3), [(Wa, xa), (best[1], None)], "Chain two layers: y2 = f(W2 . f(W1 . x))", "chip"))
blob = json.dumps({"jobs": jobs})
open(os.path.join(HERE, "trace.json"), "w").write(blob)
open(os.path.join(HERE, "trace.js"), "w").write("window.TRACE = " + blob + ";\n")   # trace.js: loadable from file://
for j in jobs: print(j["name"], j["cfg"], "tokens", j["ntok"], "chip layers", j["chip_layers"], [s["y"] for s in j["stages"]], "numpy ok")
