# social/video: qwenTPU demo (71 s, 1920x1080, 30 fps)

One data-driven HTML/SVG/CSS page (`index.html`) rendered frame by frame by `render.js` (headless system Chrome via puppeteer-core, piped to ffmpeg, libx264 crf 16, one segment per scene in parallel, then concatenated). Run `npm install` once, then `node render.js` to write `out/qwentpu_demo.mp4` (copy it to `../qwentpu_demo.mp4`); `node render.js --scene speed` re-renders one segment and `node render.js --stills 3,26,47` writes PNGs at global times into `out/stills/` for a visual check. Needs node and ffmpeg on PATH (`CHROME=/path` for another browser). Every number comes from `data.json`, where each value carries a `source` under `results/` or `docs/site/data/results.json`; the lost-log "7 -> 89 matvec/s" step is deliberately absent. The 4x4 matvec animation is an illustration with made-up small integers, labelled as such. Scene timing and burned-in captions live under `scenes` in `data.json`.


## Dashboard master demo and animated explainers (`../anim/`)

`../qwentpu_demo.mp4` is the light three-column dashboard (`social/anim/dashboard.html`): left the prompt, job code, live "N tok . X tok/s" and status; middle the Qwen3.5-0.8B layer stack (muted Qwen layers, TPU layers, a token going in and the model's output coming out); right the TPU in Gated DeltaNet state (weight cells filling as CELL tokens arrive, the activation vector, per-READ accumulators, CELL > ACT > READ > OUT chips, a state strip). It runs three jobs in sequence, a small 8x8 matvec, a 16x16 matvec and a chained pair of 8x8 matvecs, preceded by the "First a CPU. Then a GPU. Then a filesystem. Then another GPU. What's next? qwenTPU." hook. The chained job starts with the mode switch from the whole network (6,629 tok/s) to TPU layers only (74,462 tok/s, 11x).

Nothing in it is invented: `social/anim/gen_trace.py` configures the repo's own circuit (`qwentpu/circuit.py`), builds the real token streams (CELL, ACT, READ), evaluates them with the exact float64 evaluator in `qwentpu/compiler/evaluate.py`, decodes the OUT token at every READ, asserts it equals NumPy, and writes `trace.json` / `trace.js`. The dashboard steps through those tokens. For the chained job the second matvec's ACT tokens are the first one's decoded OUT tokens, and the weights are rewritten in place by the second load. tok/s values come from results/: 8x8 u3s3 6,628.6 and 74,462 (`results/a100/ab_same_trace.jsonl`), 16x16 u3s2 6,149.1 (`results/a100/a100.jsonl`). The ms readout is tokens divided by that rate. The run is slowed down on screen; the CELL, ACT and READ tokens are all shown.

The same folder holds four short explainers in the same light style (`matvec.html`, `adder.html`, `stack.html`, `hardware.html`, sharing `light.js`; `titles.html` and `lib.js` are the older dark intro/outro cards). Each has a one-sentence plain-language caption with the precise terms underneath, and runs on real data: `matvec` steps through the real 8x8 trace (every CELL, ACT and READ token and the circuit's OUT values), `adder` shows the real bit-plane counts, digit sums, Horner Z values and clamp windows of an 8x8 4-bit job (exported by `gen_trace.py`), `stack` races the whole network (6,629 tok/s) against TPU layers only (74,462 tok/s) on the same 1,728-token trace (time slowed 40x, 1 dot = 36 tokens), and `hardware` fills the four replay counters from `results/hw.md` while a ticker shows live circuit-vs-numpy comparisons. Every page exposes `window.DUR` and `window.render(t)`, a pure function of time, so frames are deterministic. `render.js` drives headless Chrome through them frame by frame (1920x1080, 30 fps) into ffmpeg and reuses `puppeteer-core` from `social/video/node_modules` (run `npm install` in `social/video` once). Final GIFs (960 px, 15 fps, each under 5 MB) are also copied to `docs/anim/<name>.gif` for the README.

```
cd social/anim
python3 sync_totals.py              # patch the replay total and hardware cards from results/hw.md + outputs.txt, then re-render
python3 gen_trace.py                 # regenerate trace.json / trace.js (needs numpy)
node render.js                       # every page -> out/<name>.mp4 (dashboard, matvec, adder, stack, hardware, intro, outro)
node render.js dashboard             # just the master -> out/dashboard.mp4
node render.js matvec adder --gif    # chosen ones, plus out/<name>.gif (960 px, 15 fps)
node render.js --master              # also concat the explainers into out/explainers_reel.mp4
node render.js dashboard --stills 14,28,59   # PNGs at those times in out/stills/ for a visual check
cp out/dashboard.mp4 ../qwentpu_demo.mp4
```

Needs node, ffmpeg and Google Chrome (`CHROME=/path` for another browser). 
