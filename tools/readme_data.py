#!/usr/bin/env python3
"""Derive results/derived/readme_data.json (used by README.spec.yml) from the committed result files:
results/<device>/*.jsonl (replay), results/hw.md (suite table written by results/finalize.py), results/a100/*.jsonl (speed), v2_vec.jsonl.
Nothing is typed by hand."""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
J = lambda p: [json.loads(l) for l in open(os.path.join(RES, p)) if l.strip().startswith("{")]
DEV_FILES = {"A100 40GB": ["a100/a100.jsonl"], "A10G 24GB": ["a10g/a10g.jsonl", "a10g/a10g_4b.jsonl"], "L4 24GB": ["l4/l4.jsonl", "l4/l4_4b.jsonl"],
             "T4 16GB": ["t4/t4_replay.jsonl", "t4/t4_q4_replay.jsonl"], "x86 Xeon (c6i, Ice Lake)": ["xeon/cpu_final.jsonl"]}


def main():
    hw = {}
    for l in open(os.path.join(RES, "hw.md")):
        c = [x.strip().strip("*") for x in l.strip().strip("|").split("|")]
        if len(c) == 7 and c[0] != "device" and not c[0].startswith("-"):
            hw[c[0]] = dict(device=c[0], arch=c[1], models=c[2], suite=c[6])
    devices = []
    for dev, files in DEV_FILES.items():
        pos = bad = 0; arrays = set()
        for f in files:
            for r in J(f):
                if "positions" in r:
                    pos += r["positions"]; bad += r["mismatches"]; arrays.add(r["array"])
        h = hw[dev]
        ok, tot = h["suite"].split("/")
        devices.append(dict(key=re.sub(r"\W+", "_", dev.split(" ")[0]).lower(), device=dev, arch=h["arch"], models=h["models"], arrays=len(arrays), positions=pos, mismatches=bad,
                            suite_clean=int(ok), suite_total=int(tot), suite=h["suite"]))
    ab = J("a100/ab_same_trace.jsonl")
    cg = [r for r in J("a100/chip_graph_ab.jsonl") if r["array"] == "8x8 u3s3"][:12]
    mean = lambda xs: sum(xs) / len(xs)
    full = next(r for r in ab if r["path"] == "full")["tok_per_s"]
    tpu = next(r for r in ab if r["path"] == "tpu-layers")["tok_per_s"]
    eager = mean([r["tok_per_s"] for r in cg if r["path"] == "tpu-layers"])
    graph = mean([r["tok_per_s"] for r in cg if r["path"] == "tpu-layers+graph"])
    chain = next(r for r in J("a100/driver_sweep.jsonl") if r.get("mode") == "graph" and r["array"] == "8x8 u3s3")
    speed = [dict(path="whole network, eager", tok_s=round(full)), dict(path="TPU layers only", tok_s=round(tpu)), dict(path="TPU layers + CUDA graph (mean of 3 runs x 2 traces)", tok_s=round(graph))]
    v2 = [r for r in J("a100/v2_vec.jsonl") if r.get("kind") == "exact"]
    out = dict(devices=devices, speed=speed,
               totals=dict(positions=sum(d["positions"] for d in devices), mismatches=sum(d["mismatches"] for d in devices), machines=len(devices), eager_tpu_mean=round(eager),
                           graph_mean=round(graph), full=round(full), tpu=round(tpu), speedup_tpu=round(tpu / full, 1), speedup_graph=round(graph / eager, 1), chain_matvec_s=chain["matvecs_per_s"],
                           v2_exact_runs=sum(r["exact"] for r in v2), v2_total_runs=sum(r["runs"] for r in v2)))
    json.dump(out, open(os.path.join(RES, "derived/readme_data.json"), "w"), indent=1)
    print(out["totals"]); [print(d) for d in devices]


if __name__ == "__main__":
    sys.exit(main())
