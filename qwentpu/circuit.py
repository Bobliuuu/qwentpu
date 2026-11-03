"""The qwenTPU circuit, expressed in qwentpu's own IR (qwentpu/compiler) so that it compiles to *real* Qwen3.5
Gated DeltaNet + SwiGLU layers (conv1d, gated RMSNorm, input RMSNorm, bf16 and all).

Machine: KR x NC weight-stationary array (configure(kr, nc, ab, wb); default 8x8, u3 activations, s3 weights),
exact i32 accumulate, ReLU + clamp-to-max requantisation.  Output tokens are activations, so layers chain.

  CELL(r,c,w)         weight load:   ONE token writes weight w into row r of column head c
  ACT(i,v)            stream:        read row i of all heads -> AB*WB AND-products per column (FFN)
                                     -> write them into row i of that column's *accumulator* head
  READ(c)             ones-query on accumulator c -> bit-plane counts -> carry-save Horner chain -> OUT(v)
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


def configure(kr=8, nc=8, ab=3, wb=3):
    """Pick the array.  kr: power of two (ref-column normalisation) <= 128.  nc: any (heads spill to more layers)."""
    global KR, NC, AB, WB, QROWS, NOUT, NW, AMAX, OUT0, NOP, CELL0, ACT0, READ0, VOCAB, CODE
    assert kr & (kr - 1) == 0 and kr <= 128 and 1 <= wb <= 6 and 1 <= ab <= 6
    KR, NC, AB, WB = kr, nc, ab, wb
    # Qwen l2-normalises q and scales it by 1/sqrt(128): the factor is exact iff n_q*128 is a power of 4, i.e.
    # n_q = 2^odd.  Otherwise bf16 rounds c*scale differently for each count c and the counts drift.
    QROWS = kr if (kr.bit_length() - 1) % 2 else 2 * kr
    assert QROWS <= 128, "KR=64 needs 128 query rows; KR=128 is unsupported"
    NOUT, NW, AMAX = 1 << AB, 1 << WB, (1 << AB) - 1
    OUT0 = 0; NOP = NOUT; CELL0 = NOP + 1; ACT0 = CELL0 + KR * NC * NW; READ0 = ACT0 + KR * NOUT
    VOCAB = list(range(READ0 + NC)); CODE = max(8, (len(VOCAB) - 1).bit_length())
    return KR, NC, AB, WB


def tok_out(v): return OUT0 + v
def tok_cell(r, c, w): return CELL0 + (r * NC + c) * NW + (w & (NW - 1))
def tok_act(i, v): return ACT0 + i * NOUT + v
def tok_read(c): return READ0 + c
def wlo(): return -(1 << (WB - 1))
def whi(): return (1 << (WB - 1)) - 1


def rand_case(rng, gentle, nvec=4):
    """Random weights and activation vectors.  `gentle` keeps outputs off the 0 / max rails."""
    if gentle:
        W = rng.integers(max(wlo(), -1), min(whi(), 2) + 1, (KR, NC)); X = rng.integers(0, min(AMAX, 3) + 1, (nvec, KR))
    else:
        W = rng.integers(wlo(), whi() + 1, (KR, NC)); X = rng.integers(0, AMAX + 1, (nvec, KR))
    return W, X


def horner_plan():
    """Static schedule of the adder tree: list of (q digits absorbed, clamp lo, clamp hi) top-down, with the
    range of the running value u checked against the bf16 window.  p = weight of u's lsb (u = y >> p, roughly)."""
    smin, smax = -(1 << (WB - 1)) * KR, ((1 << (WB - 1)) - 1) * KR
    ulo = lambda p: (-smax * ((1 << p) - 1)) // (1 << p)
    uhi = lambda p: -(-(AMAX - smin * ((1 << p) - 1)) // (1 << p))
    plan, p, lo, hi = [], AB, 0, 0
    while p > 0:
        for q in range(p, 0, -1):
            zlo, zhi = (lo << q) + smin * ((1 << q) - 1), (hi << q) + smax * ((1 << q) - 1)
            L, H = ulo(p - q), uhi(p - q)
            if (p - q == 0 and fits_code(zlo, zhi)) or (max(0, L - zlo) <= WINDOW and max(0, zhi - H) <= WINDOW):
                break
        else:
            raise AssertionError(f"adder tree does not fit the bf16 windows at p={p}")
        L, H = ulo(p - q), uhi(p - q)
        plan.append((q, zlo, zhi, L, H)); p -= q; lo, hi = max(zlo, L), min(zhi, H)
    return plan


def fits_code(zlo, zhi):
    """Can the output code be read straight off an unclamped Z in [zlo, zhi]?"""
    return all((zhi - t + 1 <= WINDOW) or (t - zlo <= WINDOW) for t in range(1, AMAX + 1))


def build():
    g = Circ()
    one, f = g.one, g.token_feature
    cell = lambda t: CELL0 <= t < ACT0
    rf = [f(f"row{r}", lambda t, r=r: int(cell(t) and (t - CELL0) // NW // NC == r)) for r in range(KR)]
    cf = [f(f"col{c}", lambda t, c=c: int(cell(t) and (t - CELL0) // NW % NC == c)) for c in range(NC)]
    wb = [f(f"wb{b}", lambda t, b=b: ((t - CELL0) % NW >> b) & 1 if cell(t) else 0) for b in range(WB)]
    arow = [f(f"arow{s}", lambda t, s=s: int(ACT0 <= t < READ0 and (t - ACT0) // NOUT == s)) for s in range(KR)]
    ab = [f(f"ab{j}", lambda t, j=j: ((t - ACT0) % NOUT >> j) & 1 if ACT0 <= t < READ0 else 0) for j in range(AB)]
    rd = [f(f"rd{c}", lambda t, c=c: int(t == tok_read(c))) for c in range(NC)]
    is_read = f("is_read", lambda t: int(t >= READ0))
    code = [f(f"code{i}", lambda t, i=i: 128 * (2 * ((t >> i) & 1) - 1)) for i in range(CODE)]
    ref = {0: A_REF * one, 1: A_REF * one}

    # weight array: one key head per column.  CELL(r,c,w) writes row r of head c: key = AND(row r, col c) is
    # silu(K (row + col - 1)) -- one-hot for free in the conv/silu -- and the value is w's bits.
    wbits = []
    for c in range(NC):
        key = {s: K * (rf[s] + cf[c] - one) for s in range(KR)}
        qry = {s: K * arow[s] for s in range(KR)}
        val = {**ref, **{2 + b: A_BIT * wb[b] for b in range(WB)}}
        outs = [(2 + b, 2.0**7) for b in range(WB)]
        wbits.append(g.head(f"W{c}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs))

    # AB*WB AND-products per column:  a_j & w_b   (the PEs' multipliers)
    prod = [[[g.ffn([(1, one, ab[j] + wbits[c][b] - one)], f"p{c}.{j}{b}") for b in range(WB)]
             for j in range(AB)] for c in range(NC)]

    # accumulators: row i of head c <- products of ACT(i, .);  READ(c) sums the rows
    cnt = []
    scale = 2.0 ** (7 + KR.bit_length() - 1)      # KR rows share the ref column: counts come back / KR
    for c in range(NC):
        key = {s: K * arow[s] for s in range(KR)}
        qry = {s: K * rd[c] for s in range(QROWS)}   # rows >= KR are never written: they only fix the l2-norm
        chans = [(j, b) for j in range(AB) for b in range(WB)]
        val = {**ref, **{2 + n: A_BIT * prod[c][j][b] for n, (j, b) in enumerate(chans)}}
        outs = [(2 + n, scale) for n in range(len(chans))]
        cnt.append(g.head(f"A{c}", qry, key, val, 0, {col: K * one for col, _ in outs}, outs))

    # carry-save adder tree: Horner over the activation-bit digits with exact saturation
    neg = 1 << (WB - 1)
    s = [sum((cnt[c][WB * j + b] * (-neg if b == WB - 1 else (1 << b)) for c in range(NC) for b in range(WB)), 0 * one)
         for j in range(AB)]
    u, p = 0 * one, AB
    for n, (q, zlo, zhi, L, H) in enumerate(horner_plan()):
        Z = u * (1 << q) + sum((s[p - q + i] * (1 << i) for i in range(q)), 0 * one)
        p -= q
        if p == 0 and fits_code(zlo, zhi):
            u, lo, hi = Z, zlo, zhi
            break
        terms = ([(1, one, L - Z)] if zlo < L else []) + ([(-1, one, Z - H)] if zhi > H else [])
        u = Z + g.ffn(terms, f"clamp{n}") if terms else Z
        lo, hi = max(zlo, L), min(zhi, H)

    # output code:  next token = OUT(clamp(u)) at a READ, NOP elsewhere.  One FFN per code bit rewrites the code dim.
    S = 2**12
    for i in range(CODE):
        nxt = [(1, one, one - is_read)] if (NOP >> i) & 1 else []
        if i < AB:
            if (AMAX >> i) & 1: nxt += g.terms_ge(u, AMAX, lo, hi, 1, (is_read,))
            for v in range(1, AMAX):
                if (v >> i) & 1: nxt += g.terms_eq(u, v, lo, hi, (is_read,))
        g.ffn([(2 * S * c, a, b) for c, a, b in nxt] + [(-S, one, one), (-1, one, code[i]), (1, one, -1 * code[i])],
              f"code{i}", into=code[i])
    return g, {"code": code, "u": u, "cnt": cnt, "wbits": wbits}


# --- host-side reference & trace generation -------------------------------------------------------
def load_trace(W):
    return [tok_cell(r, c, int(W[r, c])) for r in range(KR) for c in range(NC)]


def matvec_trace(x):
    """Tokens of one vector: KR activations, then one READ per column.  Returns (tokens, number of outputs)."""
    return [tok_act(i, int(x[i])) for i in range(KR)] + [tok_read(c) for c in range(NC)]


def ref_matvec(x, W):
    return np.clip(np.maximum(np.asarray(x) @ np.asarray(W), 0), 0, AMAX)


configure(*(int(v) for v in os.environ.get("QWENTPU_CFG", "8,8,3,3").split(",")))
