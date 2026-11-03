"""Generates docs/*.svg and docs/site/og.svg (+ og.png if rsvg-convert is installed). Pure stdlib.
Run:  python3 docs/tools/gen_svgs.py     (rewrites arch, token-timeline, gdn-state, datapath, adder-tree, og)
SVGs use CSS variables with a prefers-color-scheme dark theme; og.svg is fixed dark. Edit text here, not in the SVGs."""
import html, os, shutil, subprocess
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = DOCS + '/'
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
 ('Adder-tree FFNs','rose','rfill',['s_j = Σ_b w_b·count_jb','Horner top-down, q digits/step','clamp only if the range needs it','windows ≤ 256 (bf16-exact)']),
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
s+=t(40,40,'One matrix-vector product = KR + NC tokens',22,'b')
s+=t(40,62,'Weights load once, one CELL token per cell. Then every vector costs KR × ACT + NC × READ and the model emits NC OUT tokens. No flush.',13,'m')
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
s+=path('M58 223 H88',True,'violet')
s+=t(40,227,'k',13,'mono b','',"var(--violet)")
# panel 2 read
s+=f'<rect x="420" y="92" width="360" height="340" rx="14" fill="var(--panel)" stroke="var(--line)"/>'
s+=t(600,120,'2 · multi-hot query = sum of rows',15,'b','middle','var(--teal)')
s+=mat(500,152,6,{0,2,3,5},'teal','tfill','S (rows × 128)',210,26,['r0','','r2','r3','','r5'])
s+=t(450,400,'q = e₀+e₂+e₃+e₅   ⇒   o = r₀+r₂+r₃+r₅',13,'mono')
s+=t(450,420,'a whole reduction over the inner dim in one read',12.5,'m')
s+=path('M452 229 H478',True,'teal')
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
for i,(n,l) in enumerate([('393,984','outputs replayed, 0 mismatches'),('1,024 MACs','per matvec on a 32×32 array'),('KL ≤ 0.005','text after a TPU run, unchanged')]):
    x=60+i*370
    o+=f'<rect x="{x}" y="450" width="340" height="110" rx="16" fill="#12172b" stroke="#2b3252"/><text x="{x+22}" y="508" {F} font-size="40" font-weight="800" fill="#e6e9ff">{n}</text><text x="{x+22}" y="540" {F} font-size="19" fill="#9aa4cf">{l}</text>'
o+=f'<text x="1140" y="600" text-anchor="end" {F} font-size="20" fill="#9aa4cf">github.com/Bobliuuu/qwentpu</text></svg>'
open(DOCS + '/site/og.svg','w').write(o)


# ---------------- datapath.svg ----------------
W,H=1200,400
s=head(W,H)
s+=t(40,40,'One matrix-vector product = KR + NC tokens',22,'b')
s+=t(40,62,'Weights stay put in DeltaNet state; activations and reads are the only tokens that flow.',13,'m')
stg=[('ACT(i,v) tokens','teal','tfill',['one activation per token']),('Weight array','violet','vfill',['W heads: state row i,','head c (CELL-loaded)']),('Multipliers','amber','afill',['SwiGLU: act bit','AND weight bit']),
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
W,H=1200,560
s=head(W,H)
s+=t(40,40,'The adder tree: bit-plane counts → output code, without ever forming the wide sum',21,'b')
s+=t(40,62,'Example: 16×16 array, 4-bit activations × 4-bit weights. Ranges are the real ones from the compiler’s Horner plan.',13,'m')
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
steps=[('step 1 (q=1)','Z ∈ [−128, 112]','clamp [−98, 114]'),('step 2 (q=1)','Z ∈ [−324, 336]','clamp [−84, 100]'),('step 3 (q=1)','Z ∈ [−296, 312]','clamp [−56, 72]'),('step 4 (q=1, last)','Z ∈ [−240, 256]','merged into code FFN')]
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
s+=t(60,fy+96,'q is the largest digit count whose windows still fit 256 (bf16-exact). The final step is merged into the code FFN when its thresholds fit too.',12.5,'m')
s+='</svg>'
open(OUT+'adder-tree.svg','w').write(s)

# ---------------- og.png ----------------
png=DOCS+'/site/og.png'
if shutil.which('rsvg-convert'):
    subprocess.run(['rsvg-convert','-w','1200','-h','630','-o',png,DOCS+'/site/og.svg'],check=True)
else:
    print('rsvg-convert not found: og.png not regenerated')
