#!/usr/bin/env python3
"""Check that every relative link and image in the READMEs and docs exists: python docs/tools/check_links.py"""
import os, re, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FILES = ["README.md", "WRITEUP.md"] + [os.path.join("docs", f) for f in sorted(os.listdir(os.path.join(ROOT, "docs"))) if f.endswith(".md")]
bad = []
for f in FILES:
    base = os.path.dirname(os.path.join(ROOT, f))
    s = open(os.path.join(ROOT, f)).read()
    refs = set(re.findall(r'(?:src|href)="([^"#]+)"', s)) | set(re.findall(r"\]\(([^)#\s]+)", s))
    for r in sorted(refs):
        if r.startswith(("http://", "https://", "mailto:")):
            continue
        if not os.path.exists(os.path.normpath(os.path.join(base, r))):
            bad.append((f, r))
for f, r in bad:
    print(f"missing: {f} -> {r}")
print(f"{len(FILES)} files checked, {len(bad)} missing")
sys.exit(1 if bad else 0)
