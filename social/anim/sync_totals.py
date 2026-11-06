#!/usr/bin/env python3
"""Patch the hardware numbers in the animation sources from results/hw.md and results/outputs.txt
(written by results/finalize.py), so the clips never carry a hand-typed total.

    python3 results/finalize.py     # not needed if hw.md / outputs.txt are current
    python3 social/anim/sync_totals.py

Touches: social/anim/{hardware,dashboard,titles}.html, social/video/data.json, docs/site/og.svg.
Re-render the clips afterwards (see social/video/README.md).
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
R = lambda *p: os.path.join(ROOT, *p)

TOTAL_S = open(R("results", "outputs.txt")).read().strip()          # "1,361,280"
TOTAL = int(TOTAL_S.replace(",", ""))

# hw.md rows: | device | arch | models | arrays | positions | mismatches | suites |
ROWS = []
for line in open(R("results", "hw.md")):
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) == 7 and c[0] not in ("device", "---"):
        ROWS.append(dict(label=c[0], arch=c[1], models=c[2], arrays=int(c[3]), pos=int(c[4].replace(",", "")),
                         mism=int(c[5].strip("*")), suite=c[6]))
assert sum(r["pos"] for r in ROWS) == TOTAL and len(ROWS) == 5, (ROWS, TOTAL)
assert all(r["mism"] == 0 for r in ROWS)
NMACH = {5: "five"}[len(ROWS)]

SHORT = {"A100 40GB": "A100 40GB", "A10G 24GB": "A10G 24GB", "L4 24GB": "L4 24GB", "T4 16GB": "T4 16GB",
         "x86 Xeon (c6i, Ice Lake)": "x86 Xeon"}
ARCH = {"x86 Xeon (c6i, Ice Lake)": "Ice Lake CPU (c6i)"}
COL = {"A100 40GB": "C.blue", "A10G 24GB": "C.gr", "L4 24GB": "C.or", "T4 16GB": "C.pk", "x86 Xeon (c6i, Ice Lake)": "C.pu"}
LCOL = {"A100 40GB": "C.bluel", "A10G 24GB": "C.grl", "L4 24GB": "C.orl", "T4 16GB": "C.pkl", "x86 Xeon (c6i, Ice Lake)": "C.pul"}


def sub(path, pairs, regex=False):
    s = open(path).read()
    for a, b in pairs:
        if regex:
            s, n = re.subn(a, b, s, flags=re.S)
            assert n, (path, a[:60])
        else:
            assert a in s, (path, a[:60])
            s = s.replace(a, b)
    open(path, "w").write(s)


# --- hardware.html ---------------------------------------------------------
dev_lines = []
for r in ROWS:
    k = r["label"]
    dev_lines.append("{n:'%s',a:'%s',m:'%s',pos:%d,arr:%d,suite:'%s',c:%s,l:%s}" % (
        SHORT[k], r["arch"].replace("Xeon", "Xeon") if k not in ARCH else ARCH[k], r["models"].replace(", ", " · "),
        r["pos"], r["arrays"], r["suite"], COL[k], "'#ffffff'" if False else LCOL[k]))
dev_js = "const DEV=[" + ",\n ".join(dev_lines) + "];"
H = R("social", "anim", "hardware.html")
sub(H, [(r"const DEV=\[.*?\];\nconst TOTAL=\d+,", dev_js + "\nconst TOTAL=%d," % TOTAL)], regex=True)
sub(H, [
    (r"so we replayed the same checks on [a-z]+ different machines", "so we replayed the same checks on %s different machines" % NMACH),
    (r"replayed token positions on an A100, an A10G, an L4 and an x86 Xeon", "replayed token positions on an A100, an A10G, an L4, a T4 and an x86 Xeon"),
    (r"'[\d,]+ positions checked\. 0 mismatches\.','[^']*'",
     "'%s positions checked. 0 mismatches.','The T4 passed 9 of 12 suite runs (the larger 2B arrays do not fit in 16 GB); its replay is exact'" % TOTAL_S),
    (r"const CW=430,CG=34,CX0=\(W-4\*CW-3\*CG\)/2", "const CW=340,CG=24,CX0=(W-5*CW-4*CG)/2"),
    (r"tx\('models '\+d\.m,x\+26,y\+132,\{size:28", "tx('models '+d.m,x+26,y+132,{size:25"),
    (r"\{size:30,weight:800,align:'center',color:done\?d\.c:C\.mut\}", "{size:27,weight:800,align:'center',color:done?d.c:C.mut}"),
], regex=True)

# --- dashboard / titles ------------------------------------------------------
sub(R("social", "anim", "dashboard.html"), [
    (r"\['[\d,]+','token positions replayed'\],\['0','mismatches'\],\['\d','A100 \\u00B7 A10G \\u00B7 L4 \\u00B7 Xeon'\]",
     "['%s','token positions replayed'],['0','mismatches'],['5','A100 \\\\u00B7 A10G \\\\u00B7 L4 \\\\u00B7 T4 \\\\u00B7 Xeon']" % TOTAL_S)], regex=True)
sub(R("social", "anim", "titles.html"), [
    (r"0 MISMATCHES IN [\d,]+ POSITIONS", "0 MISMATCHES IN %s POSITIONS" % TOTAL_S)], regex=True)

# --- social/video/data.json --------------------------------------------------
P = R("social", "video", "data.json")
d = json.load(open(P))
SRC = {"A100 40GB": ("results/a100/a100.jsonl", "results/a100/a100.suite, a100q08.suite"),
       "A10G 24GB": ("results/a10g/a10g.jsonl, a10g_4b.jsonl", "results/a10g/a10g_final.suite, a10g_4b.suite"),
       "L4 24GB": ("results/l4/l4.jsonl, l4_4b.jsonl", "results/l4/l4_final.suite, l4_4b.suite"),
       "T4 16GB": ("results/t4/t4_replay.jsonl, t4_q4_replay.jsonl", "results/t4/t4_final.suite (3 of 12 runs ran out of 16 GB)"),
       "x86 Xeon (c6i, Ice Lake)": ("results/xeon/cpu_final.jsonl", "results/xeon/cpu_final.log")}
ID = {"A100 40GB": "a100", "A10G 24GB": "a10g", "L4 24GB": "l4", "T4 16GB": "t4", "x86 Xeon (c6i, Ice Lake)": "xeon"}
hw = []
for r in ROWS:
    k = r["label"]; ok, tot = (int(x) for x in r["suite"].split("/"))
    mk = lambda v, s: {"value": v, "unit": "", "source": s}
    hw.append({"id": ID[k], "label": k if k != "x86 Xeon (c6i, Ice Lake)" else "x86 Xeon (c6i)", "arch": r["arch"],
               "models": r["models"], "positions": mk(r["pos"], SRC[k][0]), "arrays": mk(r["arrays"], SRC[k][0]),
               "mismatches": mk(0, SRC[k][0]), "clean": mk(ok, SRC[k][1]), "total": mk(tot, SRC[k][1])})
d["hardware"] = hw
d["total_outputs"]["value"] = TOTAL
s = json.dumps(d, indent=1, ensure_ascii=False).replace("on four machines", "on %s machines" % NMACH)
open(P, "w").write(s + "\n")
# the legacy video page special-cased the T4 as suites-only; it has a replay now
VI = R("social", "video", "index.html")
sub(VI, [(r"t4=d\.id==='t4';", "t4=d.clean.value<d.total.value;")], regex=True)
sub(VI, [(r"\$\{t4\?`<div style=\"font-size:30px;font-weight:800;color:\$\{C\.am\};margin-top:44px;letter-spacing:\.08em\">SUITES ONLY</div>.*?\n\s*:`", "${`")], regex=True)
sub(VI, [(r"const cnt=t4\?0:Math\.round", "const cnt=Math.round")], regex=True)

# --- docs/site/og.svg --------------------------------------------------------
sub(R("docs", "site", "og.svg"), [(r">[\d,]+</text><text x=\"82\" y=\"540\"", ">%s</text><text x=\"82\" y=\"540\"" % TOTAL_S)], regex=True)

print("synced", TOTAL_S, NMACH, "machines")
