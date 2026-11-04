import json, re, glob, os
R = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(R, "../qwentpu")
DEVS = [  # key, label, arch, kind, suite files, replay files
 ("A100", "A100 40GB", "Ampere sm80", "gpu", ["a100/a100.suite", "a100/a100q08.suite"], ["a100/a100.jsonl"]),
 ("A10G", "A10G 24GB", "Ampere sm86", "gpu", ["a10g/a10g_final.suite", "a10g/a10g_4b.suite"], ["a10g/a10g.jsonl", "a10g/a10g_4b.jsonl"]),
 ("L4", "L4 24GB", "Ada sm89", "gpu", ["l4/l4_final.suite", "l4/l4_4b.suite"], ["l4/l4.jsonl", "l4/l4_4b.jsonl"]),
 ("T4", "T4 16GB", "Turing sm75", "gpu", ["t4/t4_final.suite"], ["t4/t4_replay.jsonl", "t4/t4_q4_replay.jsonl"]),
 ("Xeon", "x86 Xeon (c6i, Ice Lake)", "x86-64", "cpu", ["xeon/cpu_final.log"], ["xeon/cpu_final.jsonl"]),
]
def suites(files):
    ok = tot = 0; seen = {}; models = set()
    for f in files:
        p = f"{R}/{f}"
        if not os.path.exists(p): continue
        cur = None
        for l in open(p):
            m = re.match(r"=== (\w+) (\S+)", l)
            if m: cur = (m.group(1), m.group(2)); continue
            if cur and re.search(r"\d+ (passed|failed)", l):
                good = "failed" not in l
                seen[cur] = seen.get(cur, False) or good; models.add(cur[0])
    return sum(seen.values()), len(seen), models
def replay(files):
    n = bad = 0; arrs = set(); mods = set()
    for f in files:
        for l in open(f"{R}/{f}"):
            if l.startswith("{") and "positions" in l:
                j = json.loads(l); n += j["positions"]; bad += j["mismatches"]; arrs.add(j["array"]); mods.add(j["model"])
    return n, bad, arrs, mods
NAME = {"q08": "0.8B", "q2": "2B", "q4": "4B"}
hw = []; rows = []; total = totbad = 0; machines = 0
for key, label, arch, kind, sf, rf in DEVS:
    ok, tot, sm = suites(sf); n, bad, arrs, rm = replay(rf)
    machines += 1; total += n; totbad += bad
    models = ", ".join(NAME[m] for m in ("q08", "q2", "q4") if m in sm | rm)
    rep = f"{n:,}" if rf else "n/a"; mm = f"**{bad}**" if rf else "n/a"
    rows.append(f"| {label} | {arch} | {models} | {len(arrs) or 'n/a'} | {rep} | {mm} | {ok}/{tot} |")
    hw.append({"device": label, "kind": kind, "arch": arch, "models_tested": [NAME[m] for m in ("q08","q2","q4") if m in sm|rm],
               "arrays_tested": sorted(arrs), "replay_outputs": n if rf else None, "mismatches": bad if rf else None,
               "suite_note": f"{ok} of {tot} test-suite runs fully passed"})
table = ("| device | architecture | models | arrays replayed | positions replayed | mismatches | suite runs clean |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows))
open(f"{R}/hw.md", "w").write(table); open(f"{R}/outputs.txt", "w").write(f"{total:,}")
words = {5: "5"}
for fn in ("README.md", "WRITEUP.md"):
    p = f"{REPO}/{fn}"; s = open(p).read()
    s = s.replace("@OUTPUTS@", f"{total:,}").replace("@HW@", table).replace("@NMACH@", str(machines))
    open(p, "w").write(s)
rp = f"{REPO}/docs/site/data/results.json"; r = json.load(open(rp))
r["hardware"] = hw; r["headline"]["outputs_replayed"] = total; r["headline"]["mismatches"] = totbad
r["chip_path"] = {"model": "0.8B", "array": "8×8", "b1_full": 1997, "b1_chip": 24030, "b16_full": 33074, "b16_chip": 125208,
                  "source": "A100: results/a100/final.jsonl (full) and sweep.jsonl (tpu-layers)"}
json.dump(r, open(rp, "w"), indent=1, ensure_ascii=False)
print(table); print(total, totbad, machines)
