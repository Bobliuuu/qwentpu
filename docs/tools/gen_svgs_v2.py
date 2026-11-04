"""Generates the static v2 diagrams into docs/: v2-arch, v2-token-timeline, v2-program-in-prompt, v2-relu-feedback, v2-fixed-vs-prompt (.svg + .png via rsvg-convert if installed).
Run:  python3 docs/tools/gen_svgs_v2.py      Pure stdlib. Same light/dark style and helpers as gen_svgs.py (its helper section is reused, nothing of v1 is rewritten).
Numbers come from social/anim/trace_v2.json (the real traces) and docs/site/data/results.json["v2"]."""
import json, os, shutil, subprocess
_here = os.path.dirname(os.path.abspath(__file__))
_src = open(_here + '/gen_svgs.py').read().split('# ---------------- arch.svg')[0]
__file__ = _here + '/gen_svgs.py'
exec(compile(_src, 'gen_svgs_helpers', 'exec'))          # head, t, box, arrow, path, STYLE, OUT, DOCS
TR = json.load(open(_here + '/../../social/anim/trace_v2.json'))
OUTS = []
def save(name, s):
    s += '</svg>'; open(OUT + name, 'w').write(s); OUTS.append(name)

def pill(x, y, w, label, col, fill, size=12, h=26, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="var(--{fill})" stroke="var(--{col})" stroke-width="1.6"{d}/>'
            + t(x + w / 2, y + h / 2 + size * 0.35, label, size, 'mono', 'middle', f'var(--{col})'))

# ---------------- v2-arch ----------------
W, H = 1200, 470
s = head(W, H)
s += t(40, 40, 'v2 architecture: the prompt is the program, the model sequences itself', 22, 'b')
s += t(40, 62, 'The host builds one prompt and runs  argmax; append  until HALT. Every READ, ACT and OUTK token after the prompt is the model\'s own output.', 13, 'm')
s += box(30, 100, 230, 150, 'amber', 'afill', 'Host (the only thing outside)', ['1. build the prompt:', 'CELL tokens = weights,', 'ACT tokens = input x,', 'entry block = depth', '2. loop: argmax; append', '3. stop at HALT'], 13)
s += f'<rect x="300" y="100" width="660" height="150" rx="12" fill="var(--vfill)" stroke="var(--violet)" stroke-width="2"/>'
s += t(630, 124, 'Qwen3.5 + appended chip layers (one compiled model)', 14, 'b', 'middle', 'var(--violet)')
inner = [('Qwen layers', 'mute', 'soft', ['muted for chip', 'tokens (as v1)']), ('Weight heads', 'violet', 'vfill', ['prompt CELLs', 'write them']),
         ('Accumulator', 'teal', 'tfill', ['READ(k): exact sum,', 'then relu + clamp']), ('Code FFN + ROM', 'rose', 'rfill', ['successor of the', 'current token, plus y'])]
for i, (a_, c, f, ls) in enumerate(inner):
    x = 314 + i * 160
    s += box(x, 142, 140, 84, c, f, a_, ls, 11)
    if i: s += arrow(x - 18, 184, x - 1, 184)
s += t(630, 242, 'weights stationary in DeltaNet state; schedule and array shape compiled in', 11.5, 'm', 'middle')
s += arrow(260, 175, 300, 175, True, 'amber'); s += t(280, 165, 'prompt', 11, 'm', 'middle')
s += arrow(960, 175, 1000, 175, True, 'teal')
s += box(1000, 100, 180, 150, 'teal', 'tfill', 'Next token', ['READ(k), ACT(row; y),', 'OUTK(j; y) or HALT', 'chosen by the model;', 'the host does not look'], 12)
s += path('M1090 250 V330 H145 V252', True, 'teal', 2.5)
s += f'<rect x="330" y="316" width="500" height="28" rx="14" fill="var(--panel)" stroke="var(--teal)" stroke-width="1.5"/>'
s += t(580, 335, 'argmax; append  (the sampled token is the next input; no relabelling)', 13, 'b', 'middle', 'var(--teal)')
s += t(40, 390, 'v1: the host chose each CELL, ACT, READ and relabelled OUT as ACT. v2: the same datapath, plus a nonlinearity (relu then clamp) and a successor table, so the host does neither.', 12.5, 'm')
s += t(40, 412, 'Prompt contents: only data (no READ, no OUT, no op token). What it chooses: depth (entry block), per-layer widths, all weights and biases, the input.', 12.5, 'm')
s += t(40, 434, f'Measured: Qwen3.5-0.8B on an A100, {TR["chip_layers"]} chip layers at 16-wide 5-bit, prompt {TR["digits"][0]["toks"] and 1056} tokens, 64 emitted tokens per digit.', 12.5, 'm')
save('v2-arch.svg', s)

# ---------------- v2-token-timeline ----------------
d0 = TR['digits'][0]; tk = d0['toks']
W, H = 1200, 420
s = head(W, H)
s += t(40, 40, 'v2 token timeline: a digit classified by a depth-2 program', 22, 'b')
s += t(40, 62, f'Digit 0 of the held-out set (label {d0["label"]}, predicted {d0["pred"]}). Teal tokens are fed by the host; violet and rose tokens are emitted by the model.', 13, 'm')
y = 100
s += t(40, y - 8, 'prompt (host-fed, 1056 tokens)', 13, 'b')
x = 40
for lab, w in [('CELL × 1024', 120), ('ACT(r;0) × 16', 120), ('ACT(0;x0)', 90), ('ACT(1;x1)', 90), ('…', 40), ('ACT(15;1)', 90)]:
    s += pill(x, y, w, lab, 'teal', 'tfill', 11.5); x += w + 8
s += t(x + 6, y + 18, 'weights W1, W2, then x; the last ACT sets the entry point', 12, 'm')
y = 190
s += t(40, y - 8, 'emitted (model, 64 tokens): READ(k), then the layer\'s value token', 13, 'b')
x = 40
def pair(x, k, v, kind, y):
    col, fill = ('violet', 'vfill') if kind == 'ACT' else ('rose', 'rfill')
    s_ = pill(x, y, 78, f'READ({k})', 'amber', 'afill', 11)
    s_ += pill(x + 82, y, 92, (f'ACT({16 + k};{v})' if kind == 'ACT' else f'OUTK({k};{v})'), col, fill, 11)
    return s_
for j in (0, 1, 2):
    s += pair(x, tk[j]['k'], tk[j]['val'], 'ACT', 190); x += 184
s += t(x, y + 18, '…  16 hidden units  …', 12, 'm'); x += 150
s += pair(x, tk[15]['k'], tk[15]['val'], 'ACT', 190); x += 184
y2 = 280
x = 40
s += t(40, y2 - 8, 'layer 2 reads the activations layer 1 wrote; its values are the logits (OUTK)', 13, 'b')
for j in (16, 17, 18):
    s += pair(x, tk[j]['k'], tk[j]['val'], 'OUT', y2); x += 184
s += t(x, y2 + 18, '…  10 logits, 6 zero-padded  …', 12, 'm'); x += 210
s += pair(x, tk[31]['k'], tk[31]['val'], 'OUT', y2); x += 184
s += pill(x, y2, 70, 'HALT', 'mute', 'soft', 11.5)
s += t(40, 346, f'logits (ReLU-clamped, 0..31): {d0["logits"]}   argmax = {d0["pred"]}   label = {d0["label"]}', 13, 'b')
s += t(40, 370, 'Each emitted token is the model\'s argmax at the previous position. The host appends it and never inspects it, except to compare with HALT.', 12.5, 'm')
s += t(40, 392, 'Drawn from social/anim/trace_v2.json (layer 1 shows the first and last of 16 hidden units; layer 2 the first and last of its 16 outputs).', 12, 'm')
save('v2-token-timeline.svg', s)

# ---------------- v2-program-in-prompt ----------------
dd = TR['depth_demo']; D = dd['dmax']
W, H = 1200, 500
s = head(W, H)
s += t(40, 40, 'Program in the prompt: the entry block selects the depth', 22, 'b')
s += t(40, 62, f'One compiled model ({dd["chip_layers"]} chip layers, Dmax = {D}, width {dd["n"]}). Layers are block-diagonal in one array; the prompt writes weights into the last d blocks and puts x in block {D}-d.', 13, 'm')
y = 100
for pr in dd['progs']:
    d = pr['depth']; e = pr['entry']
    s += t(40, y + 30, f'depth {d}', 16, 'b')
    s += t(40, y + 50, f'entry block {e}', 12, 'm')
    s += t(40, y + 68, f'{pr["emitted"]} tokens emitted', 12, 'm')
    for b in range(D):
        bx = 190 + b * 150
        active = b >= e
        if active:
            s += f'<rect x="{bx}" y="{y}" width="138" height="84" rx="10" fill="var(--vfill)" stroke="var(--violet)" stroke-width="2"/>'
            s += t(bx + 69, y + 34, f'W{b - e + 1}', 16, 'b', 'middle', 'var(--violet)')
            s += t(bx + 69, y + 56, 'layer ' + str(b - e + 1) + (' (last)' if b == D - 1 else ''), 12, 'm', 'middle')
        else:
            s += f'<rect x="{bx}" y="{y}" width="138" height="84" rx="10" fill="none" stroke="var(--line)" stroke-width="1.5" stroke-dasharray="4 4"/>'
            s += t(bx + 69, y + 48, 'zeros', 12, 'm', 'middle')
        if b == e:
            s += f'<path d="M{bx + 8} {y - 12} L{bx + 8} {y - 2}" stroke="var(--amber)" stroke-width="2"/>'
            s += t(bx + 14, y - 3, 'x written here: start', 11.5, 'b', 'start', 'var(--amber)')
        if b < D - 1 and active and b + 1 >= e: s += arrow(bx + 139, y + 42, bx + 149, y + 42, False)
    s += pill(190 + D * 150, y + 29, 62, 'HALT', 'mute', 'soft', 11.5)
    y += 118
s += t(40, y + 6, 'The program counter is the current token: the compiled successor table walks blocks s … Dmax-1 and halts. The prompt sets the start by choosing where x is written.', 12.5, 'm')
s += t(40, y + 26, 'Measured: depths 1 to 4 on one compiled model, 24 of 24 programs exact on Qwen3.5-0.8B (A100). Cost scales with the program (2·d·n emitted tokens), not with Dmax.', 12.5, 'm')
save('v2-program-in-prompt.svg', s)

# ---------------- v2-relu-feedback ----------------
r1 = d0['raw1']; y1 = d0['y1']
W, H = 1200, 420
s = head(W, H)
s += t(40, 40, 'ReLU feedback: the model\'s next token is the next layer\'s ACT token', 22, 'b')
s += t(40, 62, 'The adder tree already ends in a saturating requantiser, so relu and clamp are free there: the sum is thresholded straight into the next token\'s id bits.', 13, 'm')
stages = [('READ(k)', 'amber', 'afill', ['current token names', 'the column k']), ('Σ rows (exact)', 'teal', 'tfill', ['accumulator head', 'multi-hot read']), ('relu + clamp', 'rose', 'rfill', ['0 if sum ≤ 0', 'AMAX if sum ≥ AMAX']),
          ('code bits', 'violet', 'vfill', ['threshold ladder on', 'the clamped sum']), ('ACT(row; y)', 'teal', 'tfill', ['= base(READ k) + y', 'next layer\'s input'])]
for i, (a, c, f, ls) in enumerate(stages):
    x = 40 + i * 226
    s += box(x, 100, 196, 90, c, f, a, ls, 13)
    if i: s += arrow(x - 28, 145, x - 2, 145)
s += path('M1100 190 V250 H138 V192', True, 'teal', 2.2)
s += t(620, 244, 'sampled and appended: it is the input of the next READ', 12.5, 'b', 'middle', 'var(--teal)')
s += t(40, 290, 'Example (digit 0, layer 1, first four hidden units):', 14, 'b')
hdr = ['unit k', 'sum', 'relu', 'clamp 0..31', 'emitted']
for c, hd in enumerate(hdr): s += t(60 + c * 150, 318, hd, 12.5, 'b')
for r in range(4):
    row = [str(r), str(r1[r]), str(max(r1[r], 0)), str(y1[r]), f'ACT({16 + r}; {y1[r]})']
    for c, v in enumerate(row): s += t(60 + c * 150, 342 + r * 20, v, 12.5, 'mono')
s += t(40, 436 - 4, 'Tested: 128 of 128 feedback tokens exact on the real model (8×8, 5-bit, results/a100/v2_feedback.txt). One feedback path: Sequencer and nonlin.feedback_code agree token for token.', 12, 'm')
H = 436
s = s.replace('viewBox="0 0 1200 420"', 'viewBox="0 0 1200 440"').replace('height="420"', 'height="440"', 1)
save('v2-relu-feedback.svg', s)

# ---------------- v2-fixed-vs-prompt ----------------
W, H = 1200, 520
s = head(W, H)
s += t(40, 40, 'What is fixed in the weights, and what the prompt chooses', 22, 'b')
s += t(40, 62, 'The line between a compiled accelerator and a program: everything on the left needs a recompile, everything on the right is data in the prompt.', 13, 'm')
left = ['Array size (Dmax, n) and number format (5-bit)', 'The cycle READ → value → READ', 'Chain-only topology: layer b feeds layer b+1', 'Relu then clamp (no signed outputs)', 'The successor table (schedule compiled in)', 'That the chain ends after block Dmax−1, then HALT', 'Bias convention: row n−1 is constant 1']
right = ['Depth, 1 to Dmax (where x is written)', 'Per-layer width (zero padding), up to n−1', 'Every weight and bias', 'The input vector', 'Which classifier it is (same model, other prompt)', 'Nothing else: no READ, no OUT, no op token']
s += f'<rect x="30" y="90" width="560" height="360" rx="16" fill="var(--soft)" stroke="var(--mute)" stroke-width="2"/>'
s += t(310, 122, 'Compiled into the weights (fixed)', 17, 'b', 'middle', 'var(--mute)')
s += f'<rect x="610" y="90" width="560" height="360" rx="16" fill="var(--vfill)" stroke="var(--violet)" stroke-width="2"/>'
s += t(890, 122, 'Chosen by the prompt (data)', 17, 'b', 'middle', 'var(--violet)')
for i, l in enumerate(left):
    s += f'<circle cx="62" cy="{156 + i * 40}" r="5" fill="var(--mute)"/>' + t(78, 161 + i * 40, l, 14)
for i, l in enumerate(right):
    s += f'<circle cx="642" cy="{156 + i * 40}" r="5" fill="var(--violet)"/>' + t(658, 161 + i * 40, l, 14)
s += t(40, 482, 'A general instruction memory (program fetched from DeltaNet state) would move the left column to the right. It is designed in docs/V2_SEQUENCER.md section 5 and not built.', 12.5, 'm')
s += t(40, 504, 'The host builds the prompt and runs argmax; append. It does no arithmetic and never picks a chip token.', 12.5, 'm')
save('v2-fixed-vs-prompt.svg', s)

def _flat_light(svg):
    """rsvg-convert has no CSS variables: bake the light theme's colours into a copy for the PNG."""
    import re
    root = re.search(r':root\{([^}]*)\}', STYLE).group(1)
    cols = dict(re.findall(r'--(\w+):(#[0-9a-fA-F]+)', root))
    return re.sub(r'var\(--(\w+)\)', lambda m: cols[m.group(1)], re.sub(r'<style>.*?</style>', '<style>text{font-family:Helvetica,Arial,sans-serif}.b{font-weight:700}.mono{font-family:Menlo,monospace}</style>', svg, flags=re.S))
if shutil.which('rsvg-convert'):
    for n in OUTS:
        tmp = OUT + n[:-4] + '.light.tmp.svg'; open(tmp, 'w').write(_flat_light(open(OUT + n).read()))
        subprocess.run(['rsvg-convert', '-w', '1800', tmp, '-o', OUT + n[:-4] + '.png'], check=True); os.remove(tmp)
print('wrote', ', '.join(OUTS))
