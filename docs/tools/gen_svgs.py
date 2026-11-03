"""Generates docs/*.svg and docs/site/og.svg (+ og.png if rsvg-convert is installed). Pure stdlib.
Run:  python3 docs/tools/gen_svgs.py   (rewrites arch, token-timeline, gdn-state, datapath, adder-tree, packing, chip-path,
life-of-a-matvec, bug-museum, hardware [from docs/site/data/results.json "hardware"], og). Reads site/data/{results,presets}.json.
SVGs use CSS variables with a prefers-color-scheme dark theme; og.svg is fixed dark. Edit text here, not in the SVGs."""
import html, json, os, shutil, subprocess
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = DOCS + '/'
def _j(name, default):
    try: return json.load(open(DOCS + '/site/data/' + name))
    except Exception: return default
def fmt_n(v): return f'{v:,}'
RES = _j('results.json', {})
PRE = _j('presets.json', {})
STYLE='''<style>
:root{--bg:#fbfbfd;--panel:#ffffff;--soft:#eef0f9;--ink:#14172b;--mute:#5a6285;--line:#c9cde3;--teal:#0f9d8a;--violet:#6246ea;--amber:#c97a0a;--rose:#d1385c;--vfill:#ece8ff;--tfill:#dcf5f0;--afill:#fff0d6;--rfill:#ffe3ea}
@media (prefers-color-scheme:dark){:root{--bg:#0b0e1a;--panel:#12172b;--soft:#171d36;--ink:#e6e9ff;--mute:#9aa4cf;--line:#2b3252;--teal:#35d6c0;--violet:#9d86ff;--amber:#ffb454;--rose:#ff6b8b;--vfill:#1d1a3d;--tfill:#10312f;--afill:#33260f;--rfill:#3a1824}}
text{font-family:Inter,"Helvetica Neue",Helvetica,Arial,sans-serif;fill:var(--ink)}
.m{fill:var(--mute)}.b{font-weight:700}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre}
.flow{stroke-dasharray:7 6;animation:dash 1.1s linear infinite}
@keyframes dash{to{stroke-dashoffset:-13}}
.pulse{animation:pl 2.4s ease-in-out infinite}@keyframes pl{50%{opacity:.45}}
@media (prefers-reduced-motion:reduce){.flow,.pulse{animation:none}}
</style>'''
def head(w,h,defs=''):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img">{STYLE}<defs><marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--mute)"/></marker>{defs}</defs><rect width="{w}" height="{h}" rx="18" fill="var(--bg)"/>'
def t(x,y,s,size=14,cls='',anchor='start',fill=None):
    f=f' fill="{fill}"' if fill else ''
    return f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}" text-anchor="{anchor}"{f}>{html.escape(s)}</text>'
def box(x,y,w,h,col,fill,title,lines=(),size=14,flow=False):
    s=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="var(--{fill})" stroke="var(--{col})" stroke-width="2"/>'
    s+=t(x+w/2,y+24,title,size+1,'b','middle',f'var(--{col})')
    for i,l in enumerate(lines): s+=t(x+w/2,y+46+i*17,l,size-2,'m','middle')
    return s
def arrow(x1,y1,x2,y2,flow=True,col='mute'):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var(--{col})" stroke-width="2" {chr(99)+"lass=\"flow\"" if flow else ""} marker-end="url(#ar)"/>'
def path(d,flow=True,col='mute',w=2):
    return f'<path d="{d}" fill="none" stroke="var(--{col})" stroke-width="{w}" {chr(99)+"lass=\"flow\"" if flow else ""} marker-end="url(#ar)"/>'

# ---------------- arch.svg ----------------
W,H=1200,500
s=head(W,H)
s+=t(40,40,'qwenTPU architecture: the layer stack is the datapath',22,'b')
s+=t(40,62,'No program counter, no instruction stream. Weights and partial sums live in DeltaNet state. 8×8 u3·s3 is 2 chip layers; wider operands add adder-tree layers.',13,'m')
x0,bw,gap,y0,bh=30,165,24,92,92
tops=[('Chip tokens','teal','tfill',['CELL · ACT · READ','one token per dataflow step']),
 ('Embedding','violet','vfill',['original rows + new','chip rows (widened hidden)']),
 ('Qwen3.5 layers','mute','soft',['24 / 32 original layers','muted for chip tokens:','a=b=z=−1024, gate −1024']),
 ('TPU layers (appended)','violet','vfill',['compiled from our IR','details below ↓']),
 ('final norm + lm_head','amber','afill',['chip rows only read the','code dims; tau hides them','from text positions']),
 ('argmax → OUT(v)','teal','tfill',['the sampled token','is the layer output'])]
xs=[]
for i,(a,c,f,ls) in enumerate(tops):
    x=x0+i*(bw+gap); xs.append(x)
    s+=box(x,y0,bw,bh,c,f,a,ls,13)
    if i: s+=arrow(x-gap+2,y0+bh/2,x-2,y0+bh/2)
# feedback rail
fx=xs[5]+bw
s+=path(f'M{fx} {y0+bh/2} H1172 V458 H14 V{y0+bh/2} H{x0-2}',True,'teal',2.5)
s+=f'<rect x="340" y="444" width="520" height="28" rx="14" fill="var(--panel)" stroke="var(--teal)" stroke-width="1.5"/>'
s+=t(600,463,'host relabels OUT(v) as the next layer’s ACT(i,v): no arithmetic',13,'b','middle','var(--teal)')
# expansion
ex,ey,ew,eh,eg=56,236,200,132,22
s+=f'<path d="M{xs[3]+bw/2-20} {y0+bh} L{ex+20} {ey-18} M{xs[3]+bw/2+20} {y0+bh} L{ex+5*ew+4*eg-20} {ey-18}" stroke="var(--violet)" stroke-width="1.5" stroke-dasharray="3 4" fill="none"/>'
s+=f'<rect x="{ex-14}" y="{ey-18}" width="{5*ew+4*eg+28}" height="{eh+80}" rx="16" fill="none" stroke="var(--violet)" stroke-width="1.5" stroke-dasharray="3 4"/>'
st=[('W heads','violet','vfill',['one key head per column','CELL(r,c,w): key = AND(row r,','col c) free from conv+silu','value = w bits, stays put']),
 ('AND-product FFN','amber','afill',['act bit j AND weight bit b','reads row i of every W head','AB×WB SwiGLU units / column','(the multipliers)']),
 ('Accumulator heads','teal','tfill',['ACT(i,v) overwrites row i','every vector rewrites all rows','READ(c): multi-hot query =','exact sum of rows → counts']),
 ('Adder-tree FFNs','rose','rfill',['s_j = Σ_b w_b·count_jb','Horner top-down, q digits/step','clamp only if the range needs it,','in ≤256-wide passes (bf16-exact)']),
 ('Code FFN','violet','vfill',['last Horner step merged in:','thresholds on Z → code bits','→ lm_head → OUT(v)',''])]
for i,(a,c,f,ls) in enumerate(st):
    x=ex+i*(ew+eg); s+=box(x,ey,ew,eh,c,f,a,ls,13)
    if i: s+=arrow(x-eg+2,ey+eh/2,x-2,ey+eh/2)
s+=t(ex-2,ey+eh+34,'Each stage is a real Qwen3.5 sublayer: heads are Gated DeltaNet key heads (conv1d taps, gated RMSNorm), FFNs are SwiGLU units  coef · a · relu(b).',12.5,'m')
s+=t(ex-2,ey+eh+54,'Operands are bit-planes, so every stored or read value is a small integer that bf16 represents exactly. The compiler checks that bit for bit.',12.5,'m')
# legend
s+=t(40,H-8,'',1)
s+='</svg>'
open(OUT+'arch.svg','w').write(s)

# ---------------- token-timeline.svg ----------------
W,H=1200,470
s=head(W,H)
s+=t(40,40,'One matrix-vector product = KR/g + NC/h tokens',22,'b')
s+=t(40,62,'Weights load once, one CELL token per cell. Then every vector costs KR/g ACT + NC/h READ tokens (g = h = 1 shown) and the model emits NC/h OUT tokens. No flush.',13,'m')
# token chips
chips=[('CELL(r,c,w) …','violet',2,'once'),('ACT(0,v)','teal',1,''),('ACT(1,v)','teal',1,''),('…','mute',1,''),('ACT(KR−1,v)','teal',1.4,''),('READ(0)','amber',1,''),('READ(1)','amber',1,''),('…','mute',1,''),('READ(NC−1)','amber',1.4,'')]
unit=78; x=250; y=96; pos=[]
for lab,c,wd,_ in chips:
    w=unit*wd-8; s+=f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="9" fill="var(--panel)" stroke="var(--{c})" stroke-width="2"/>'+t(x+w/2,y+22,lab,12,'mono b','middle',f'var(--{c})'); pos.append((x,w)); x+=unit*wd
# outputs row
oy=y+54
s+=t(40,oy+24,'model output at each READ:',12.5,'m')
for k,i in enumerate((5,6,8)):
    ox,ow=pos[i]
    lab=['OUT(v₀)','OUT(v₁)','OUT(v_NC−1)'][k]
    s+=f'<rect x="{ox}" y="{oy+4}" width="{ow}" height="30" rx="9" fill="var(--tfill)" stroke="var(--teal)" stroke-width="2"/>'+t(ox+ow/2,oy+24,lab,12,'mono b','middle','var(--teal)')
    s+=path(f'M{ox+ow/2} {y+34} V{oy+3}',False,'teal',1.5)
# lanes
ly=oy+66; lh=50
lanes=[('W heads','violet','vfill',[(0,'write cell (r,c)'),(1,'read row i'),(2,''),(4,'')],'written at CELL, read per ACT'),
 ('AND-product FFN','amber','afill',[(1,'a_j AND w_b'),(2,''),(4,'')],'act bit AND weight bit'),
 ('Accumulator heads','teal','tfill',[(1,'write row i'),(2,''),(4,''),(5,'Σ rows'),(6,'Σ rows'),(8,'')],'row i at ACT, Σ at READ'),
 ('Adder tree + code FFN','rose','rfill',[(5,'Horner'),(6,''),(8,'')],'counts → u → code → OUT')]
for n,(name,c,f,act,desc) in enumerate(lanes):
    yy=ly+n*(lh+8)
    s+=f'<rect x="30" y="{yy}" width="{W-60}" height="{lh}" rx="10" fill="var(--soft)"/>'+t(44,yy+22,name,13,'b','',f'var(--{c})')+t(44,yy+40,desc,11.5,'m')
    for i,lab in act:
        px,pw=pos[i]; s+=f'<rect x="{px}" y="{yy+8}" width="{pw}" height="{lh-16}" rx="7" fill="var(--{f})" stroke="var(--{c})" stroke-width="1.5" class="pulse"/>'
        if lab: s+=t(px+pw/2,yy+lh/2+4,lab,11,'','middle')
    # label region left sits over first chip: shift by offset
s+='</svg>'
# lane titles overlap the LOAD column; move titles above lane instead
open(OUT+'token-timeline.svg','w').write(s)

# ---------------- gdn-state.svg ----------------
W,H=1200,470
s=head(W,H)
s+=t(40,40,'Gated DeltaNet state is already a RAM',22,'b')
s+=t(40,64,'S ← αS + k·(β(v − Sᵀk))ᵀ',16,'mono')+t(380,64,'o = Sᵀq',16,'mono')
def mat(x,y,rows,hot,col,fill,label,rw=170,rh=22,vals=None,faint=False):
    r=''
    r+=t(x+rw/2,y-10,label,13,'b','middle')
    for i in range(rows):
        on=i in hot
        r+=f'<rect x="{x}" y="{y+i*(rh+3)}" width="{rw}" height="{rh}" rx="5" fill="{"var(--"+fill+")" if on else "var(--soft)"}" stroke="{"var(--"+col+")" if on else "var(--line)"}" stroke-width="{2 if on else 1}"/>'
        r+=t(x-8,y+i*(rh+3)+15,f'r{i}',11,'m mono','end')
        if vals: r+=t(x+rw/2,y+i*(rh+3)+15,vals[i] if on else ('0' if faint else '· · ·'),11,'mono','middle')
    return r
px=[50,450,850]
# panel 1 write
s+=f'<rect x="30" y="92" width="360" height="340" rx="14" fill="var(--panel)" stroke="var(--line)"/>'
s+=t(210,120,'1 · one-hot key = write a row',15,'b','middle','var(--violet)')
s+=mat(110,152,6,{2},'violet','vfill','S (rows × 128)',210,26,['','','v','','',''])
s+=t(50,400,'k = e₂, α=β=1  ⇒  row 2 := v  exactly',13,'mono')
s+=t(50,420,'loads a weight row, or stores a partial sum',12.5,'m')
s+=path('M56 223 H78',True,'violet')
s+=t(40,227,'k',13,'mono b','',"var(--violet)")
# panel 2 read
s+=f'<rect x="420" y="92" width="360" height="340" rx="14" fill="var(--panel)" stroke="var(--line)"/>'
s+=t(600,120,'2 · multi-hot query = sum of rows',15,'b','middle','var(--teal)')
s+=mat(500,152,6,{0,2,3,5},'teal','tfill','S (rows × 128)',210,26,['r0','','r2','r3','','r5'])
s+=t(450,400,'q = e₀+e₂+e₃+e₅   ⇒   o = r₀+r₂+r₃+r₅',13,'mono')
s+=t(450,420,'a whole reduction over the inner dim in one read',12.5,'m')
s+=path('M452 229 H468',True,'teal')
s+=t(436,233,'q',13,'mono b','','var(--teal)')
# panel 3 flush
s+=f'<rect x="810" y="92" width="360" height="340" rx="14" fill="var(--panel)" stroke="var(--line)"/>'
s+=t(990,120,'3 · decay α = 0 = flush (not used)',15,'b','middle','var(--rose)')
s+=mat(890,152,6,set(),'rose','rfill','S (rows × 128)',210,26,None)
s+=t(840,400,'selector 238 → exp(g) = 0   ⇒   S := 0',13,'mono')
s+=t(840,420,'in the substrate; qwenTPU never needs it',12.5,'m')
s+='</svg>'
open(OUT+'gdn-state.svg','w').write(s)

# ---------------- og.svg ----------------
W,H=1200,630
o=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b0e1a"/><stop offset="1" stop-color="#1a1440"/></linearGradient><linearGradient id="t" x1="0" x2="1"><stop offset="0" stop-color="#35d6c0"/><stop offset="1" stop-color="#9d86ff"/></linearGradient></defs><rect width="{W}" height="{H}" fill="url(#g)"/>'
F='font-family="Inter,Helvetica,Arial,sans-serif"'
# systolic grid art on the right
for r in range(5):
    for c in range(6):
        on=(r*3+c*5)%4==0
        o+=f'<rect x="{770+c*66}" y="{70+r*66}" width="52" height="52" rx="10" fill="{"#9d86ff" if on else "#171d36"}" opacity="{0.9 if on else 1}" stroke="#2b3252"/>'
o+=f'<text x="60" y="150" {F} font-size="30" font-weight="700" fill="#9aa4cf">qwenTPU</text>'
o+=f'<text x="60" y="250" {F} font-size="76" font-weight="800" fill="#e6e9ff" letter-spacing="-2">A TPU inside</text><text x="60" y="335" {F} font-size="76" font-weight="800" fill="url(#t)" letter-spacing="-2">Qwen’s weights.</text>'
o+=f'<text x="60" y="395" {F} font-size="26" fill="#9aa4cf">Matrix-multiply hardware compiled into Qwen3.5.</text>'
def _outputs():
    try: return open(DOCS+'/../../results/outputs.txt').read().strip()
    except Exception: return fmt_n((RES.get('headline') or {}).get('outputs_replayed', 0))
for i,(n,l) in enumerate([(_outputs(),'outputs replayed, 0 mismatches'),('1,024 MACs','per matvec on a 32×32 array'),('KL = 0.0','text after a chip-only run')]):
    x=60+i*370
    o+=f'<rect x="{x}" y="450" width="340" height="110" rx="16" fill="#12172b" stroke="#2b3252"/><text x="{x+22}" y="508" {F} font-size="40" font-weight="800" fill="#e6e9ff">{n}</text><text x="{x+22}" y="540" {F} font-size="19" fill="#9aa4cf">{l}</text>'
o+=f'<text x="1140" y="600" text-anchor="end" {F} font-size="20" fill="#9aa4cf">github.com/Bobliuuu/qwentpu</text></svg>'
open(DOCS + '/site/og.svg','w').write(o)


# ---------------- datapath.svg ----------------
W,H=1200,400
s=head(W,H)
s+=t(40,40,'One matrix-vector product = KR/g + NC/h tokens',22,'b')
s+=t(40,62,'Weights stay put in DeltaNet state; activations and reads are the only tokens that flow.',13,'m')
stg=[('ACT tokens','teal','tfill',['g activations per token']),('Weight array','violet','vfill',['W heads: state row i,','head c (CELL-loaded)']),('Multipliers','amber','afill',['SwiGLU: act bit','AND weight bit']),
 ('Accumulators','teal','tfill',['row i of head c;','all rows rewritten']),('READ(c) token','amber','afill',['multi-hot query =','sum of rows → counts']),('Adder tree','rose','rfill',['Horner clamp chain,','last step in code FFN']),('OUT(v) token','teal','tfill',['lm_head argmax =','next layer’s activation'])]
bw,gp=144,20; x0=40
for i,(a,c,f,ls) in enumerate(stg):
    x=x0+i*(bw+gp); s+=box(x,100,bw,100,c,f,a,ls,12.5)
    if i: s+=arrow(x-gp+2,150,x-2,150)
fx=x0+6*(bw+gp)+bw/2
s+=path(f'M{fx} 202 V270 H{x0+bw/2} V202',True,'teal',2.5)
s+=t(600,296,'the model’s own token is fed back as the next layer’s ACT(i,v)',13.5,'b','middle','var(--teal)')
s+=t(40,350,'Weights load once (CELL(r,c,w) tokens, one per cell) and never move. Every vector rewrites all KR accumulator rows, so no flush.',13,'m')
s+=t(40,372,'No program counter, no fetch, no decode: the layer stack is the datapath.',13,'m')
s+='</svg>'
open(OUT+'datapath.svg','w').write(s)

# ---------------- adder-tree.svg ----------------
W,H=1200,680
s=head(W,H)
s+=t(40,40,'The adder tree: bit-plane counts → output code, without ever forming the wide sum',21,'b')
s+=t(40,62,'Example: 16×16 array, 4-bit activations × 4-bit weights. Ranges are the real ones from the compiler’s Horner plan (clamp_passes).',13,'m')
# counts matrix
cx,cy,cw,ch=60,120,52,30
s+=t(cx+2*cw,cy-34,'1 · counts cnt[j][b]',14,'b','middle','var(--teal)')
s+=t(cx+2*cw,cy-16,'# rows with act bit j AND weight bit b',11.5,'m','middle')
for j in range(4):
    s+=t(cx-8,cy+j*(ch+4)+20,f'j={j}',11,'m mono','end')
    for b in range(4):
        s+=f'<rect x="{cx+b*(cw+4)}" y="{cy+j*(ch+4)}" width="{cw}" height="{ch}" rx="6" fill="var(--tfill)" stroke="var(--teal)" stroke-width="1.5"/>'+t(cx+b*(cw+4)+cw/2,cy+j*(ch+4)+20,f'c{j}{b}',12,'mono','middle')
for b in range(4): s+=t(cx+b*(cw+4)+cw/2,cy+4*(ch+4)+14,f'b={b}',11,'m mono','middle')
s+=t(cx,cy+4*(ch+4)+36,'read from the accumulator heads in one multi-hot query',11.5,'m')
# s_j
sx=360
s+=arrow(cx+4*(cw+4)+6,cy+2*(ch+4),sx-12,cy+2*(ch+4))
s+=t(sx+75,cy-34,'2 · digit sums s_j',14,'b','middle','var(--amber)')
s+=t(sx+75,cy-16,'s_j = Σ_b wt_b · cnt[j][b]',11.5,'m mono','middle')
for j in range(4):
    s+=f'<rect x="{sx}" y="{cy+j*(ch+4)}" width="150" height="{ch}" rx="6" fill="var(--afill)" stroke="var(--amber)" stroke-width="1.5"/>'+t(sx+75,cy+j*(ch+4)+20,f's_{j}  ∈ [−128, 112]',12,'mono','middle')
s+=t(sx,cy+4*(ch+4)+14,'wt_b = 2^b, top bit negative (two’s complement)',11.5,'m')
s+=t(sx,cy+4*(ch+4)+36,'y = Σ 2^j s_j: too wide to compare',11.5,'m')
# Horner chain
hx=640; hy=104; hw=160; hh=74
s+=arrow(sx+156,cy+2*(ch+4),hx-10,cy+2*(ch+4))
s+=t(hx+240,cy-34,'3 · Horner, top digit first',14,'b','middle','var(--rose)')
s+=t(hx+240,cy-16,'u ← clamp(2^q·u + next q digits)',11.5,'m mono','middle')
steps=[('step 1 (q=1)','Z ∈ [−128, 112]','pass [−98, 112]'),('step 2 (q=1)','Z ∈ [−324, 336]','pass [−84, 100]'),('step 3 (q=1)','Z ∈ [−296, 312]','pass [−56, 72]'),('step 4 (q=1, last)','Z ∈ [−240, 256]','merged into code FFN')]
for i,(a,b_,c_) in enumerate(steps):
    x=hx+(i%2)*(hw+40); y=hy+(i//2)*(hh+30)+30
    last=i==3
    col,fill=('violet','vfill') if last else ('rose','rfill')
    s+=f'<rect x="{x}" y="{y}" width="{hw}" height="{hh}" rx="10" fill="var(--{fill})" stroke="var(--{col})" stroke-width="2"/>'+t(x+hw/2,y+20,a,12,'b','middle',f'var(--{col})')+t(x+hw/2,y+40,b_,11.5,'mono','middle')+t(x+hw/2,y+58,c_,11.5,'mono','middle')
    if i==0: s+=arrow(x+hw+2,y+hh/2,x+38+hw,y+hh/2)
    if i==1: s+=path(f'M{x+hw/2} {y+hh+2} V{y+hh+14} H{hx+hw/2} V{y+hh+28}',False,'mute',1.8)
    if i==2: s+=arrow(x+hw+2,y+hh/2,x+38+hw,y+hh/2)
# output
s+=t(hx+240,hy+2*(hh+30)+20,'4 · code FFN → OUT(v)',14,'b','middle','var(--violet)')
s+=t(hx+240,hy+2*(hh+30)+40,'thresholds on Z give the code bits; lm_head turns them into the token',11.5,'m','middle')
# formulas
fy=420
s+=f'<rect x="40" y="{fy}" width="{W-80}" height="110" rx="12" fill="var(--soft)"/>'
s+=t(60,fy+26,'clamp bounds at digit position p: the values beyond which the lower digits can no longer change relu+clamp(y)',12.5,'b')
s+=t(60,fy+50,'U_lo(p) = ⌊−R_max / 2^p⌋      U_hi(p) = ⌈(AMAX − R_min) / 2^p⌉      R_min = s_min·(2^p − 1)      R_max = s_max·(2^p − 1)',12.5,'mono')
s+=t(60,fy+74,'clamp(Z) = Z + relu(U_lo − Z) − relu(Z − U_hi): two SwiGLU units, and only the sides that can actually be hit are built',12.5,'mono')
s+=t(60,fy+96,'q is the largest digit count whose clamp fits one pass (else q=1 and several). The last step stops early once its thresholds fit.',12.5,'m')
s+=f'<rect x="40" y="{fy+124}" width="{W-80}" height="82" rx="12" fill="var(--rfill)" stroke="var(--rose)" stroke-width="1.5"/>'
s+=t(60,fy+150,'Multi-pass clamp: each pass is one FFN level and may reach at most 256 beyond the current range edge (the bf16-exact window), so it composes losslessly.',12.5,'b')
s+=t(60,fy+174,'32×32 u4·s4, step 2:  Z ∈ [−648, 672]  →  pass 1: [−392, 416]  →  pass 2: [−168, 196]',12,'mono')
s+=t(60,fy+196,'Chip layers with multi-pass clamps: 16×16 u5·s5 = 10, 32×32 u4·s4 = 10, 64×64 u3·s3 = 12.',12,'m')
s+='</svg>'
open(OUT+'adder-tree.svg','w').write(s)

# ---------------- shared helpers for the explainer diagrams ----------------
def chip(x,y,w,label,col,fill=None,size=11.5,h=30):
    f=f'fill="var(--{fill})"' if fill else 'fill="var(--panel)"'
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" {f} stroke="var(--{col})" stroke-width="1.8"/>'+t(x+w/2,y+h/2+4,label,size,'mono b','middle',f'var(--{col})')
def pre(key,field,default): return (PRE.get(key) or {}).get(field,default)

# ---------------- packing.svg ----------------
W,H=1200,640
s=head(W,H)
s+=t(40,40,'Token packing: carry g activations and read h columns per token',22,'b')
s+=t(40,62,'Example: 8×8 array, 3-bit activations × 3-bit weights. The same 64 multiply-adds, in 16 tokens or in 4.',13,'m')
# unpacked row
s+=t(40,100,'Unpacked (g = 1, h = 1): 8 ACT + 8 READ = 16 tokens, 4 MACs per token',14,'b')
x=40
for i in range(8): s+=chip(x,112,64,f'ACT{i}','teal','tfill',11); x+=70
x+=10
for i in range(8): s+=chip(x,112,64,f'READ{i}','amber','afill',11); x+=70
# packed row
s+=t(40,176,'Packed (g = 4, h = 4): 2 ACT + 2 READ = 4 tokens, 16 MACs per token',14,'b')
s+=chip(40,188,250,'ACT(0; v0 v1 v2 v3)','teal','tfill',12.5,36)
s+=chip(300,188,250,'ACT(1; v4 v5 v6 v7)','teal','tfill',12.5,36)
s+=chip(600,188,150,'READ(0)','amber','afill',12.5,36)
s+=chip(760,188,150,'READ(1)','amber','afill',12.5,36)
s+=t(940,206,'one token = four activations,',12.5,'m')+t(940,224,'one READ = four columns',12.5,'m')
s+=chip(600,232,150,'OUT(y0 y1 y2 y3)','teal','tfill',11.5,30)
s+=chip(760,232,150,'OUT(y4 y5 y6 y7)','teal','tfill',11.5,30)
s+=path('M675 220 V230',False,'teal',1.4)+path('M835 220 V230',False,'teal',1.4)
# head copies
s+=t(40,310,'What it costs: every column keeps g copies of its weight head and g accumulator heads',14,'b')
cx=60
for j in range(4):
    x=cx+j*270
    s+=f'<rect x="{x}" y="326" width="250" height="150" rx="12" fill="var(--panel)" stroke="var(--line)"/>'
    s+=t(x+125,348,f'copy {j} of column c',12.5,'b','middle')
    # weight head rows
    s+=t(x+14,370,'weight head',11,'m')
    for r in range(2):
        s+=f'<rect x="{x+90}" y="{358+r*18}" width="70" height="14" rx="4" fill="var(--vfill)" stroke="var(--violet)"/>'+t(x+95,369+r*18,f'row {r*4+j}',9.5,'mono')
    s+=t(x+14,420,'accumulator',11,'m')
    for r in range(2):
        s+=f'<rect x="{x+90}" y="{408+r*18}" width="140" height="14" rx="4" fill="var(--tfill)" stroke="var(--teal)"/>'+t(x+95,419+r*18,f'products of v{j}',9.5,'mono')
    s+=t(x+125,462,f'holds weight rows ≡ {j} (mod 4)',11,'m','middle')
s+=path('M185 482 V510 H1080 V482',False,'mute',1.5)
s+=t(600,530,'ACT(s; v0..v3) hits state row s of all four copies at once; copy j ANDs its own value v_j',12.5,'b','middle')
nh=pre('8,8,3,3,4,4','heads',64); nl=pre('8,8,3,3,4,4','layers',4); nv=pre('8,8,3,3,4,4','vocab',12803)
bh=pre('8,8,3,3,1,1','heads',16); bl=pre('8,8,3,3,1,1','layers',2); bv=pre('8,8,3,3,1,1','vocab',593)
s+=f'<rect x="40" y="552" width="{W-80}" height="64" rx="12" fill="var(--soft)"/>'
s+=t(60,578,f'Price for g = h = 4: {nh} key heads instead of {bh} ({nl} chip layers instead of {bl}), and a vocabulary of {fmt_n(nv)} chip tokens instead of {fmt_n(bv)}.',12.5,'b')
s+=t(60,600,'Each layer has 16 key heads, so more heads means more layers. An ACT token ranges over (KR/g) · 8^g values. The arithmetic stays exact.',12.5,'m')
s+='</svg>'
open(OUT+'packing.svg','w').write(s)

# ---------------- chip-path.svg ----------------
cp=RES.get('chip_path') or {}
W,H=1200,520
s=head(W,H)
s+=t(40,40,'The chip-only path: skip the layers that do nothing',22,'b')
s+=t(40,62,'Chip tokens are muted inside Qwen’s own layers, so each of those layers is exactly the identity on chip dimensions.',13,'m')
def stack(y,label,skip):
    r=t(40,y-14,label,14,'b')
    r+=chip(40,y,110,'embed','violet','vfill',12,44)
    x=170
    for i in range(10):
        if skip: r+=f'<rect x="{x}" y="{y}" width="46" height="44" rx="8" fill="var(--soft)" stroke="var(--line)" stroke-dasharray="4 3"/>'+t(x+23,y+27,'id',11,'m mono','middle')
        else: r+=f'<rect x="{x}" y="{y}" width="46" height="44" rx="8" fill="var(--soft)" stroke="var(--mute)"/>'+t(x+23,y+27,'Qwen',9.5,'m','middle')
        x+=52
    r+=t(170,y+62,'10 of the 24 Qwen layers shown (32 on 4B), muted for chip tokens' if not skip else '',11.5,'m')
    r+=chip(x+10,y,120,'TPU layers','rose','rfill',12,44)
    r+=chip(x+140,y,90,'final norm','amber','afill',11.5,44)
    return r,x
r,x=stack(130,'Full model: every layer runs, mostly to compute zero',False)
s+=r+chip(x+240,130,170,'lm_head: all rows','amber','afill',11.5,44)
r,x=stack(280,'ChipPath: embed, jump straight to the TPU layers',True)
s+=r+chip(x+240,280,170,'lm_head: chip rows','teal','tfill',11.5,44)
s+=path(f'M95 326 C95 372 {x+66} 372 {x+66} 330',True,'teal',2.5)
s+=t((95+x+66)/2,380,'bypass: the same bits, none of the work',12.5,'b','middle','var(--teal)')
s+=f'<rect x="40" y="406" width="{W-80}" height="86" rx="12" fill="var(--soft)"/>'
s+=t(60,432,'Bit-identical to the full model (a test checks it) and Qwen’s own cache is never touched, so no restore is needed.',12.5,'b')
s+=t(60,456,'A100, 0.8B, 8×8, batch 1: 1,997 tok/s (full) to 24,030 tok/s (TPU layers only), 12×. Xeon, same replay: 171.5 to 2,128 tok/s, 12×.',12.5,'mono')
s+=t(60,478,'The work saved is about 1 − (TPU layers ÷ all layers), which is why small arrays with few chip layers gain the most.',12.5,'m')
s+='</svg>'
open(OUT+'chip-path.svg','w').write(s)

# ---------------- life-of-a-matvec.svg ----------------
W,H=1200,740
s=head(W,H)
s+=t(40,40,'The life of one matrix-vector product',22,'b')
s+=t(40,62,'From loading the weights to feeding the answer to the next layer. Every step is an ordinary forward pass of the model.',13,'m')
cards=[
 ('1','Load weights (once)','violet','vfill',['CELL(r,c,w): ONE token per weight.','Key = AND(row r, col c), made free by','conv + silu. The weight bits land in a','state row of that column’s head.','They never move again.']),
 ('2','Stream activations','teal','tfill',['ACT(s; v0..v(g-1)) carries g inputs.','The token reads state row s of the','weight heads, ANDs the activation bits','with the weight bits (SwiGLU),','and overwrites row s of the accumulators.']),
 ('3','No flush needed','amber','afill',['Every vector rewrites all KR/g rows,','so last vector’s numbers are simply','overwritten. The accumulators hold','bit-plane COUNTS: small integers','that bf16 stores exactly.']),
 ('4','Read the columns','amber','afill',['READ(k): a multi-hot query over all','written rows returns their exact SUM.','That is the whole reduction over the','inner dimension, in one read.','h columns per token when packed.']),
 ('5','Adder tree','rose','rfill',['Counts → digit sums s_a (a linear form).','Horner, top digit first: Z = 2^q u + digits,','then clamp in ≤256-wide FFN passes.','The last step merges into the code FFN.','Out comes relu + clamp, already requantised.']),
 ('6','Say the answer','violet','vfill',['Code FFN writes the output code,','lm_head turns it into a token:','OUT(y0..y(h-1)).','The model samples it like any','other next token.']),
 ('7','Feed it back','teal','tfill',['The host relabels OUT(v) as the next','layer’s ACT tokens. No arithmetic.','Same weights stay loaded for the','next vector; a new layer loads','its own CELL tokens first.']),
]
cw,chh,gx,gy=270,238,26,30
pos=[]
for i,(n,ttl,c,f,ls) in enumerate(cards):
    row,col=(0,i) if i<4 else (1,i-4)
    x=40+col*(cw+gx); y=96+row*(chh+gy+10)
    if row==1: x=40+col*(cw+gx)+ (cw+gx)//2
    pos.append((x,y))
    s+=f'<rect x="{x}" y="{y}" width="{cw}" height="{chh}" rx="14" fill="var(--{f})" stroke="var(--{c})" stroke-width="2"/>'
    s+=f'<circle cx="{x+26}" cy="{y+28}" r="14" fill="var(--{c})"/>'+t(x+26,y+33,n,14,'b','middle','var(--bg)')
    s+=t(x+50,y+33,ttl,15,'b','',f'var(--{c})')
    for k,l in enumerate(ls): s+=t(x+16,y+66+k*20,l,11.5,'' if k else 'b')
    # mini picture
    py=y+chh-40
    if i==0:
        for r_ in range(3): s+=f'<rect x="{x+16+r_*0}" y="{py+r_*0}" width="0" height="0"/>'
        s+=chip(x+16,py,150,'CELL(2,5,-1)','violet',None,11,26)+path(f'M{x+170} {py+13} H{x+196}',False,'violet',1.5)
        s+=f'<rect x="{x+200}" y="{py}" width="54" height="26" rx="5" fill="var(--panel)" stroke="var(--violet)"/>'+t(x+227,py+17,'row',10,'mono','middle')
    if i==1:
        s+=chip(x+16,py,150,'ACT(0; 3 1)','teal',None,11,26)+path(f'M{x+170} {py+13} H{x+196}',False,'teal',1.5)
        s+=f'<rect x="{x+200}" y="{py}" width="54" height="26" rx="5" fill="var(--panel)" stroke="var(--teal)"/>'+t(x+227,py+17,'AND',10,'mono','middle')
    if i==2:
        for r_ in range(4):
            for c_ in range(9): s+=f'<rect x="{x+16+c_*12}" y="{py-4+r_*9}" width="10" height="7" rx="2" fill="var(--{"teal" if (r_*5+c_*3)%4==0 else "soft"})"/>'
    if i==3:
        for r_ in range(4): s+=f'<rect x="{x+16}" y="{py-6+r_*8}" width="120" height="6" rx="2" fill="var(--amber)" opacity="0.65"/>'
        s+=t(x+150,py+14,'Σ rows',11.5,'mono b')
    if i==4:
        s+=chip(x+16,py,70,'s_a','rose',None,11,26)+path(f'M{x+90} {py+13} H{x+110}',False,'rose',1.5)+chip(x+112,py,70,'Horner','rose',None,11,26)+path(f'M{x+186} {py+13} H{x+206}',False,'rose',1.5)+chip(x+208,py,46,'u','rose',None,11,26)
    if i==5: s+=chip(x+16,py,230,'OUT(5 0 2 7)','teal',None,11.5,26)
    if i==6: s+=chip(x+16,py,230,'OUT(v) = next ACT(i,v)','teal',None,11,26)
# arrows between cards
for i in range(6):
    (x1,y1),(x2,y2)=pos[i],pos[i+1]
    if i<3: s+=arrow(x1+cw+2,y1+chh/2,x2-2,y2+chh/2)
    elif i==3: s+=path(f'M{x1+cw/2} {y1+chh+2} V{y1+chh+16} H{x2-18} V{y2+chh/2} H{x2-2}',False,'mute',2)
    else: s+=arrow(x1+cw+2,y1+chh/2,x2-2,y2+chh/2)
# loop back note
(x7,y7)=pos[6]
s+=path(f'M{x7+cw/2} {y7+chh+2} V{y7+chh+22} H{pos[1][0]+cw/2} V{pos[1][1]+chh+4}',True,'teal',2.2)
s+=t(x7+cw/2+10,y7+chh+40,'next vector: back to step 2',12.5,'b','', 'var(--teal)')
s+=t(40,H-14,'g = h = 1 shown in the text; with packing each ACT/OUT token carries g/h values instead of one.',12,'m')
s+='</svg>'
open(OUT+'life-of-a-matvec.svg','w').write(s)

# ---------------- bug-museum.svg ----------------
def face(cx,cy,r,sad=True):
    f=f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="var(--afill)" stroke="var(--amber)" stroke-width="2"/>'
    f+=f'<circle cx="{cx-r*0.35}" cy="{cy-r*0.2}" r="{r*0.1}" fill="var(--ink)"/><circle cx="{cx+r*0.35}" cy="{cy-r*0.2}" r="{r*0.1}" fill="var(--ink)"/>'
    d=f'M{cx-r*0.4} {cy+r*0.5} Q{cx} {cy+(0.15 if sad else 0.85)*r} {cx+r*0.4} {cy+r*0.5}'
    f+=f'<path d="{d}" fill="none" stroke="var(--ink)" stroke-width="2.2" stroke-linecap="round"/>'
    return f
W,H=1200,640
s=head(W,H)
s+=t(40,40,'The bug museum: two numerics traps that bit us',22,'b')
s+=t(40,62,'Both look like “the model got the answer slightly wrong”. Both turned out to be bf16 rounding in places we did not control.',13,'m')
# exhibit A
s+=f'<rect x="40" y="86" width="545" height="510" rx="16" fill="var(--panel)" stroke="var(--rose)" stroke-width="2"/>'
s+=t(60,116,'Exhibit A: the irrational scale',16,'b','','var(--rose)')
s+=face(520,130,28,True)
s+=t(60,146,'a count came back one bf16 step off',13,'mono b')
s+=t(60,180,'Qwen l2-normalises the query, then scales by 1/√128.',12.5)
s+=t(60,200,'Our multi-hot query has n_q ones, so the norm is √n_q and the',12.5)
s+=t(60,220,'count c is multiplied by  1 / √(n_q · 128).',12.5)
s+=f'<rect x="60" y="240" width="505" height="86" rx="10" fill="var(--rfill)"/>'
s+=t(76,266,'n_q = 16:  1/√2048  is irrational, so bf16 rounds it',12.5,'mono')
s+=t(76,288,'each count c · scale rounds differently',12.5,'mono b')
s+=t(76,310,'so the stored counts drift and the sums are wrong.',12.5,'mono')
s+=f'<rect x="60" y="346" width="505" height="86" rx="10" fill="var(--tfill)"/>'
s+=t(76,372,'Fix: pad the query with unused rows so n_q = 2^odd',12.5,'mono b')
s+=t(76,394,'n_q = 32:  1/√4096 = 1/64, an exact power of two,',12.5,'mono')
s+=t(76,416,'so every count survives bf16 untouched.',12.5,'mono')
# little diagram: scale bar
s+=t(60,460,'n_q · 128 must be a power of 4:',12.5,'b')
for i,(nq,ok) in enumerate([(1,False),(2,True),(4,False),(8,True),(16,False),(32,True)]):
    x=60+i*84
    s+=chip(x,472,76,f'n_q={nq}','teal' if ok else 'rose','tfill' if ok else 'rfill',11,30)
s+=t(60,530,'exact: 2, 8, 32, 128 (2^odd).   Drift: 1, 4, 16, 64 (seen at 16).',12,'m')
s+=t(60,552,'Lesson: the library’s normalisation is part of your circuit.',12.5,'b')
# exhibit B
s+=f'<rect x="615" y="86" width="545" height="510" rx="16" fill="var(--panel)" stroke="var(--violet)" stroke-width="2"/>'
s+=t(635,116,'Exhibit B: bf16 split-K',16,'b','','var(--violet)')
s+=face(1095,130,28,True)
s+=t(635,146,'one GPU, two sequence lengths, wrong bits',13,'mono b')
s+=t(635,180,'On an L4, traces of 288 and 320 tokens came back wrong. The',12.5)
s+=t(635,200,'cause is cuBLAS split-K: the long dot product is cut into',12.5)
s+=t(635,220,'slices, each is summed, then the slices are added together.',12.5)
# pizza: 4 slices
import math
cx,cy,R=720,300,56
for k in range(4):
    a0,a1=k*math.pi/2,(k+1)*math.pi/2
    x0,y0=cx+R*math.cos(a0),cy+R*math.sin(a0); x1,y1=cx+R*math.cos(a1),cy+R*math.sin(a1)
    s+=f'<path d="M{cx} {cy} L{x0:.1f} {y0:.1f} A{R} {R} 0 0 1 {x1:.1f} {y1:.1f} Z" fill="var(--{["vfill","tfill","afill","rfill"][k]})" stroke="var(--ink)" stroke-width="1.2"/>'
s+=t(cx,cy+R+20,'K split in 4 slices',11.5,'m','middle')
s+=path(f'M{cx+R+14} {cy} H{cx+R+60}',False,'mute',1.8)
s+=chip(cx+R+66,cy-18,150,'add in bf16','rose','rfill',11.5,36)
s+=t(cx+R+66,cy+36,'the partial sums are rounded to 8 bits',11,'m')
s+=t(cx+R+66,cy+52,'before they meet, so the counts go wrong.',11,'m')
s+=f'<rect x="635" y="404" width="505" height="86" rx="10" fill="var(--tfill)"/>'
s+=t(651,430,'Fix: tell the backend to keep the reduction in fp32',12.5,'mono b')
s+=t(651,452,'torch.backends.cuda.matmul.',12,'mono')
s+=t(651,472,'  allow_bf16_reduced_precision_reduction = False',12,'mono b')
s+=t(635,520,'Same weights, same tokens: only the summation order differs.',12.5)
s+=t(635,542,'Lesson: “exact on one GPU” is a claim about a software stack.',12.5,'b')
s+=t(635,564,'Hence the hardware matrix in the README.',12.5,'m')
s+='</svg>'
open(OUT+'bug-museum.svg','w').write(s)

# ---------------- hardware.svg (from results.json "hardware") ----------------
hw=RES.get('hardware') or []
ncol=4; cw,chh,gx,gy=262,168,16,16
rows=max(1,(len(hw)+ncol-1)//ncol)
W=40*2+ncol*cw+(ncol-1)*gx; H=120+rows*(chh+gy)+40
s=head(W,H)
s+=t(40,40,'Does it work on your hardware?',22,'b')
nv=sum(1 for d in hw if d.get('mismatches')==0 and (d.get('replay_outputs') or 0)>0)
s+=t(40,64,f'{nv} of {len(hw)} devices replayed with 0 mismatches against NumPy; the rest ran the test suites only (dashed).',13,'m')
for i,d in enumerate(hw):
    r,c=divmod(i,ncol); x=40+c*(cw+gx); y=96+r*(chh+gy)
    ok=d.get('mismatches')==0 and (d.get('replay_outputs') or 0)>0
    col='teal' if ok else 'mute'
    dash='' if ok else ' stroke-dasharray="6 4"'
    s+=f'<rect x="{x}" y="{y}" width="{cw}" height="{chh}" rx="14" fill="var({"--tfill" if ok else "--panel"})" stroke="var(--{col})" stroke-width="2"{dash}/>'
    s+=t(x+16,y+30,(d.get('device','?').split(' (')[0]),17,'b')
    s+=f'<rect x="{x+cw-80}" y="{y+14}" width="64" height="22" rx="11" fill="var(--soft)" stroke="var(--line)"/>'+t(x+cw-48,y+30,(d.get('kind') or '').upper(),10.5,'b','middle')
    s+=t(x+16,y+52,d.get('arch',''),12,'m')
    if ok:
        s+=t(x+16,y+80,f"{fmt_n(d['replay_outputs'])} outputs",13,'b','', 'var(--teal)')
        s+=t(x+16,y+100,'0 mismatches',13,'b','', 'var(--teal)')
    else:
        s+=t(x+16,y+84,'suites only, no replay',13,'b','','var(--mute)')
        s+=t(x+16,y+102,(d.get('suite_note') or '')[:40],11.5,'m')
    mt=', '.join(d.get('models_tested') or []) or 'models: pending'
    at=(f"{len(d.get('arrays_tested') or [])} arrays replayed" if d.get('arrays_tested') else 'no replay')
    s+=t(x+16,y+126,mt[:38],11.5,'m')+t(x+16,y+144,at[:38],11.5,'m')
s+='</svg>'
open(OUT+'hardware.svg','w').write(s)

# ---------------- og.png ----------------
png=DOCS+'/site/og.png'
if shutil.which('rsvg-convert'):
    subprocess.run(['rsvg-convert','-w','1200','-h','630','-o',png,DOCS+'/site/og.svg'],check=True)
else:
    print('rsvg-convert not found: og.png not regenerated')
