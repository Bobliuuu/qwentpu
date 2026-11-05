"""results/a100/v2_vec.jsonl -> results/a100/v2_vec.txt (the table in docs/V2_VECTORIZED.md)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "a100", "v2_vec.jsonl"))]
cfgs = []
for r in rows:
    k = (r["dmax"], r["n"], r["ab"], r["wb"])
    if k not in cfgs: cfgs.append(k)
out = ["# qwenTPU v2 vectorized (vec neurons per token), A100, Qwen3.5-0.8B bf16, host loop = argmax; append",
       "# source: results/a100/v2_vec.jsonl (bench_vec.py).  exact = emitted tokens == numpy golden and logits == plain network,",
       "# random programs of every depth 1..Dmax x both readouts (+ golden MLP for n=16).  Speed: deepest program, 3 warm-up + 20 timed",
       "# inferences, same input; 'graph' prefill includes the one-off CUDA-graph capture of each new session, so compare loop_ms.", ""]
for d, n, ab, wb in cfgs:
    out.append(f"Dmax={d} n={n} u{ab}s{wb}")
    out.append(f"  {'vec':>3} {'layers':>6} {'vocab':>7} {'hidden':>6} {'emitted':>7} {'exact':>9} | "
               f"{'eager loop ms':>13} {'ms/inf':>8} | {'graph loop ms':>13} {'ms/inf':>8}")
    base = {}
    for vec in sorted({r["vec"] for r in rows if (r["dmax"], r["n"], r["ab"], r["wb"]) == (d, n, ab, wb)}):
        sel = [r for r in rows if (r["dmax"], r["n"], r["ab"], r["wb"], r["vec"]) == (d, n, ab, wb, vec)]
        ex = next(r for r in sel if r["kind"] == "exact"); e = next(r for r in sel if r.get("mode") == "eager"); g = next(r for r in sel if r.get("mode") == "graph")
        if vec == 1: base = dict(e=e["loop_ms"], ei=e["ms_per_inference"], g=g["loop_ms"], gi=g["ms_per_inference"])
        out.append(f"  {vec:>3} {ex['chip_layers']:>6} {ex['vocab']:>7} {ex['hidden']:>6} {e['emitted']:>7} {ex['exact']:>3}/{ex['runs']:<5}| "
                   f"{e['loop_ms']:>13} {e['ms_per_inference']:>8} | {g['loop_ms']:>13} {g['ms_per_inference']:>8}"
                   + (f"   loop x{base['e'] / e['loop_ms']:.2f} eager, x{base['g'] / g['loop_ms']:.2f} graph; inference x{base['ei'] / e['ms_per_inference']:.2f} eager"
                      if vec > 1 else "") + (f"   golden MLP {ex['golden_mlp']}" if ex.get("golden_mlp") else ""))
    out.append("")
open(os.path.join(HERE, "a100", "v2_vec.txt"), "w").write("\n".join(out))
print("\n".join(out))
