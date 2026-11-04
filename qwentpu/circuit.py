"""The qwenTPU circuit, expressed in qwentpu's own IR (qwentpu/compiler) so that it compiles to *real* Qwen3.5
Gated DeltaNet + SwiGLU layers (conv1d, gated RMSNorm, input RMSNorm, bf16 and all).

Machine: KR x NC weight-stationary array (configure(kr, nc, ab, wb, g, h); default 8x8, u3 activations, s3 weights),
exact i32 accumulate, ReLU + clamp-to-max requantisation.  Output tokens are activations, so layers chain.

  CELL(r,c,w)         weight load:   ONE token writes weight w into row r of column head c
  ACT(s; v_0..v_{g-1})  stream:      ONE token carries g activations (rows s*g .. s*g+g-1).  g copies of every
                                     weight head (copy j holds rows = j mod g) are read at once -> AB*WB AND-products
                                     per column and copy (FFN) -> written into row s of g accumulator heads
  READ(k)             ones-query on the accumulators of h columns (k*h .. k*h+h-1) at once -> bit-plane counts ->
                      h carry-save Horner chains -> OUT(v_0..v_{h-1}): ONE token carries h outputs
Packing (g, h) cuts the tokens of a matvec from KR + NC to KR/g + NC/h; the price is g*NC weight/accumulator
heads (16 key heads per layer, so more layers) and a bigger vocabulary.
Weight bits and accumulator counts are stored as bits in DeltaNet state rows, exactly like Qwenputer's RAM.
Every vector rewrites all KR accumulator rows, so no flush is ever needed.

Adder tree.  cnt[j][b] = #rows with activation bit j AND weight bit b.  s_j = sum_b wt_b cnt[j][b] (wt_b = 2^b,
top bit negative) and y = sum_j 2^j s_j, but y is far too wide for a bf16-exact comparator.  So it is folded
top-down (Horner):  u <- clamp(2^q u + <q digits>), where the clamp bounds [Ulo(p), Uhi(p)] are exactly the values
beyond which the lower digits can no longer change relu+clamp(y).  The clamp costs two SwiGLU units
(Z + relu(lo-Z) - relu(Z-hi)); the final u is already the requantised output.
"""
import os

import numpy as np

from .compiler.gadgets import Circ
from .compiler.ir import A_BIT, A_REF, Chan, K

# --- configuration: rows (inner dim), columns, activation bits (unsigned), weight bits (two's complement) -----
KR = NC = AB = WB = 0
WINDOW = 256
SEQ = None          # optional v2 sequencer (sequencer.install) that decides the code (next-token) dims


def configure(kr=8, nc=8, ab=3, wb=3, g=1, h=1):
    """Pick the array.  kr/g: power of two (ref-column normalisation) <= 128.  nc: any multiple of h (heads spill
    to more layers).  g: activations per ACT token, h: outputs per READ/OUT token."""
    global KR, NC, AB, WB, G, H, KS, QROWS, NOUT, NW, AMAX, NOP, OUT0, CELL0, ACT0, READ0, NACT, NREAD, VOCAB, CODE
    assert 1 <= wb <= 6 and 1 <= ab <= 6 and kr % g == 0 and nc % h == 0
    KR, NC, AB, WB, G, H, KS = kr, nc, ab, wb, g, h, kr // g
    assert KS & (KS - 1) == 0 and KS <= 128, "KR/g must be a power of two"
    # Qwen l2-normalises q and scales it by 1/sqrt(128): the factor is exact iff n_q*128 is a power of 4, i.e.
    # n_q = 2^odd.  Otherwise bf16 rounds c*scale differently for each count c and the counts drift.
    QROWS = KS if (KS.bit_length() - 1) % 2 else 2 * KS
    assert QROWS <= 128, "KR/g = 64 needs 128 query rows; 128 is unsupported"
    NOUT, NW, AMAX = 1 << AB, 1 << WB, (1 << AB) - 1
    OUT0 = 0; NOP = NOUT ** H; CELL0 = NOP + 1; ACT0 = CELL0 + KR * NC * NW
    NACT = KS * NOUT ** G; READ0 = ACT0 + NACT; NREAD = NC // H
    VOCAB = list(range(READ0 + NREAD)); CODE = max(8, (len(VOCAB) - 1).bit_length())
    return KR, NC, AB, WB, G, H


def tok_out(*vs):
    """OUT token carrying h outputs (slot t in base-NOUT digit t); a single int works for h = 1."""
    return OUT0 + sum(int(v) * NOUT ** t for t, v in enumerate(vs))
def tok_cell(r, c, w): return CELL0 + (r * NC + c) * NW + (w & (NW - 1))
def tok_act(s, *vs): return ACT0 + s * NOUT ** G + sum(int(v) * NOUT ** j for j, v in enumerate(vs))
def tok_read(k): return READ0 + k
def out_values(t): return [(t // NOUT ** i) % NOUT for i in range(H)]
def wlo(): return -(1 << (WB - 1))
def whi(): return (1 << (WB - 1)) - 1


def rand_case(rng, gentle, nvec=4):
    """Random weights and activation vectors.  `gentle` keeps outputs off the 0 / max rails."""
    if gentle:
        W = rng.integers(max(wlo(), -1), min(whi(), 2) + 1, (KR, NC)); X = rng.integers(0, min(AMAX, 3) + 1, (nvec, KR))
        if KR > 16:                                   # keep wide rows from saturating: sparse activations
            X = X * (rng.random(X.shape) < 16 / KR)
    else:
        W = rng.integers(wlo(), whi() + 1, (KR, NC)); X = rng.integers(0, AMAX + 1, (nvec, KR))
    return W, X


def clamp_passes(zlo, zhi, L, H, stop=None):
    """Saturate Z in [zlo, zhi] to [L, H] with as many one-FFN passes as the 256-wide windows need.  Each pass
    clamps to a *partial* bound (<= WINDOW beyond the current range edge) and composes losslessly.  `stop(lo, hi)`
    ends early when the remaining range is already good enough.  Returns [(lo', hi')] and the final range."""
    out = []
    while (zlo < L or zhi > H) and not (stop and stop(zlo, zhi)):
        lo = L if L - zlo <= WINDOW else zlo + WINDOW
        hi = H if zhi - H <= WINDOW else zhi - WINDOW
        lo, hi = (lo if zlo < L else zlo), (hi if zhi > H else zhi)
        out.append((lo, hi)); zlo, zhi = max(zlo, lo), min(zhi, hi)
    return out, (zlo, zhi)


def horner_plan():
    """Static schedule of the adder tree, top-down: [(q digits absorbed, zlo, zhi, L, H, passes)].  p = weight of
    u's lsb.  u may be saturated to [Ulo(p), Uhi(p)] without changing relu+clamp(y).  A step whose clamp does not
    fit one 256-wide pass takes several (each is one FFN level); q is the largest that needs at most one."""
    smin, smax = -(1 << (WB - 1)) * KR, ((1 << (WB - 1)) - 1) * KR
    ulo = lambda p: (-smax * ((1 << p) - 1)) // (1 << p)
    uhi = lambda p: -(-(AMAX - smin * ((1 << p) - 1)) // (1 << p))
    plan, p, lo, hi = [], AB, 0, 0
    while p > 0:
        for q in range(p, 0, -1):
            zlo, zhi = (lo << q) + smin * ((1 << q) - 1), (hi << q) + smax * ((1 << q) - 1)
            L, H = ulo(p - q), uhi(p - q)
            if (p - q == 0 and fits_code(zlo, zhi)) or len(clamp_passes(zlo, zhi, L, H)[0]) <= 1:
                break
        else:                                   # nothing fits in one pass: one digit, several passes
            q = 1; zlo, zhi = (lo << 1) + smin, (hi << 1) + smax; L, H = ulo(p - 1), uhi(p - 1)
        L, H = ulo(p - q), uhi(p - q)
        merged = p - q == 0 and fits_code(zlo, zhi)
        passes, (lo, hi) = ([], (zlo, zhi)) if merged else clamp_passes(zlo, zhi, L, H, fits_code if p - q == 0 else None)
        plan.append((q, zlo, zhi, L, H, len(passes))); p -= q
    return plan


def fits_code(zlo, zhi):
    """Can the output code be read straight off an unclamped Z in [zlo, zhi]?"""
    return all((zhi - t + 1 <= WINDOW) or (t - zlo <= WINDOW) for t in range(1, AMAX + 1))


def build():
    g = Circ()
    one, f = g.one, g.token_feature
    cell = lambda t: CELL0 <= t < ACT0
    act = lambda t: ACT0 <= t < READ0
    rf = [f(f"row{r}", lambda t, r=r: int(cell(t) and (t - CELL0) // NW // NC == r)) for r in range(KR)]
    cf = [f(f"col{c}", lambda t, c=c: int(cell(t) and (t - CELL0) // NW % NC == c)) for c in range(NC)]
    wb = [f(f"wb{b}", lambda t, b=b: ((t - CELL0) % NW >> b) & 1 if cell(t) else 0) for b in range(WB)]
    arow = [f(f"arow{s}", lambda t, s=s: int(act(t) and (t - ACT0) // NOUT ** G == s)) for s in range(KS)]
    ab = [[f(f"ab{j}.{a}", lambda t, j=j, a=a: (((t - ACT0) % NOUT ** G // NOUT ** j % NOUT) >> a) & 1 if act(t) else 0)
           for a in range(AB)] for j in range(G)]
    rd = [f(f"rd{k}", lambda t, k=k: int(t == tok_read(k))) for k in range(NREAD)]
    is_read = f("is_read", lambda t: int(READ0 <= t < READ0 + NREAD))
    code = [f(f"code{i}", lambda t, i=i: 128 * (2 * ((t >> i) & 1) - 1)) for i in range(CODE)]
    ref = {0: A_REF * one, 1: A_REF * one}

    # weight array: g copies of one key head per column.  CELL(r,c,w) writes row r // g of copy r % g of head c:
    # key = AND(row r, col c) is silu(K (row + col - 1)) -- one-hot for free in the conv/silu -- value = w's bits.
    wbits = [[None] * G for _ in range(NC)]
    for c in range(NC):
        for j in range(G):
            key = {s: K * (rf[s * G + j] + cf[c] - one) for s in range(KS)}
            qry = {s: K * arow[s] for s in range(KS)}
            val = {**ref, **{2 + b: A_BIT * wb[b] for b in range(WB)}}
            outs = [(2 + b, 2.0**7) for b in range(WB)]
            wbits[c][j] = g.head(f"W{c}.{j}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs)

    # AB*WB AND-products per column and copy:  a_a & w_b   (the PEs' multipliers)
    prod = [[[[g.ffn([(1, one, ab[j][a] + wbits[c][j][b] - one)], f"p{c}.{j}.{a}{b}") for b in range(WB)]
              for a in range(AB)] for j in range(G)] for c in range(NC)]

    # accumulators: row s of copy j of head c <- products of ACT(s; ..);  READ(k) sums the rows of h columns' copies
    cnt = [[None] * G for _ in range(NC)]
    scale = 2.0 ** (7 + KS.bit_length() - 1)      # KS rows share the ref column: counts come back / KS
    chans = [(a, b) for a in range(AB) for b in range(WB)]
    for c in range(NC):
        for j in range(G):
            key = {s: K * arow[s] for s in range(KS)}
            qry = {s: K * rd[c // H] for s in range(QROWS)}   # rows >= KS are never written: they only fix the l2-norm
            val = {**ref, **{2 + n: A_BIT * prod[c][j][a][b] for n, (a, b) in enumerate(chans)}}
            outs = [(2 + n, scale) for n in range(len(chans))]
            cnt[c][j] = g.head(f"A{c}.{j}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs)

    # carry-save adder tree, once per output slot: Horner over the activation-bit digits with exact saturation
    neg = 1 << (WB - 1)
    plan = horner_plan()
    slots = []
    for t in range(H):
        cols = [c for c in range(NC) if c % H == t]
        sd = [sum((cnt[c][j][WB * a + b] * (-neg if b == WB - 1 else (1 << b))
                   for c in cols for j in range(G) for b in range(WB)), 0 * one) for a in range(AB)]
        u, p = 0 * one, AB
        for n, (q, zlo, zhi, L, Hh, npass) in enumerate(plan):
            Z = u * (1 << q) + sum((sd[p - q + i] * (1 << i) for i in range(q)), 0 * one)
            p -= q
            lo, hi = zlo, zhi
            if p == 0 and fits_code(zlo, zhi):
                u = Z
                break
            passes, (lo, hi) = clamp_passes(zlo, zhi, L, Hh, fits_code if p == 0 else None)
            assert len(passes) == npass
            u = Z
            for k, (cl, ch) in enumerate(passes):
                u = u + g.ffn(([(1, one, cl - u)] if cl > zlo else []) + ([(-1, one, u - ch)] if ch < zhi else []),
                              f"clamp{t}.{n}.{k}")
                zlo, zhi = max(zlo, cl), min(zhi, ch)
        slots.append((u, lo, hi))

    # output code:  next token = OUT(clamp(u_0), .., clamp(u_{h-1})) at a READ, NOP elsewhere.
    # One FFN per code bit rewrites the code dim; bit t*AB + a is bit a of slot t's value.
    S = 2**12
    for i in range(CODE):
        if SEQ is not None:                       # v2 sequencer: the model also emits the next chip token (sequencer.py)
            nxt = SEQ.nxt_terms(g, i, dict(one=one, f=f, rd=rd, is_read=is_read, slots=slots))
        else:
            nxt = [(1, one, one - is_read)] if (NOP >> i) & 1 else []
        if SEQ is None and i < H * AB:
            t, a = divmod(i, AB); u, lo, hi = slots[t]
            if (AMAX >> a) & 1: nxt += g.terms_ge(u, AMAX, lo, hi, 1, (is_read,))
            for v in range(1, AMAX):
                if (v >> a) & 1: nxt += g.terms_eq(u, v, lo, hi, (is_read,))
        g.ffn([(2 * S * c, aa, bb) for c, aa, bb in nxt] + [(-S, one, one), (-1, one, code[i]), (1, one, -1 * code[i])],
              f"code{i}", into=code[i])
    return g, {"code": code, "slots": slots, "cnt": cnt, "wbits": wbits}


# --- host-side reference & trace generation -------------------------------------------------------
def load_trace(W):
    return [tok_cell(r, c, int(W[r, c])) for r in range(KR) for c in range(NC)]


def matvec_trace(x):
    """Tokens of one vector: KR/g ACT tokens (g activations each), then NC/h READ tokens."""
    x = [int(v) for v in x]
    return [tok_act(s, *x[s * G:(s + 1) * G]) for s in range(KS)] + [tok_read(k) for k in range(NREAD)]


def expected_outs(x, W):
    """The OUT token the model must emit at each READ token of matvec_trace(x)."""
    y = ref_matvec(x, W)
    return [tok_out(*y[k * H:(k + 1) * H]) for k in range(NREAD)]


def ref_matvec(x, W):
    return np.clip(np.maximum(np.asarray(x) @ np.asarray(W), 0), 0, AMAX)


configure(*(int(v) for v in os.environ.get("QWENTPU_CFG", "8,8,3,3").split(",")))
