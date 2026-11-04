"""v2 nonlinearity + rescaling gadgets, built from the same SwiGLU unit terms as the v1 adder tree.

Input: the signed integer accumulator y, as the *carry-save digit list* the bit-plane counting already
produces:   y = sum_j 2^j * d_j,   d_j an Expr with a known integer range [lo_j, hi_j]
(`column_digits` builds it from the A-head counts of one output column).

    requant(g, digits, shift=k, amax=M)  ->  Requant with  bits / value = clamp(floor(y / 2^k), 0, M)

ReLU, arithmetic right shift (floor) by k and the saturating clamp to the activation range are ONE operation,
because  clamp(floor(y/2^k), 0, M) = clamp(relu(y) >> k, 0, M).  Two phases, both exact integer arithmetic:

  1. low digits (floor shift).  y = 2^k * (sum_{j>=k} 2^(j-k) d_j  +  c_k)  +  low,  and
     c_{j+1} = floor((d_j + c_j) / 2)  is the carry out of bit j.  The carry stays within the digit range, so
     each step is a short staircase  floor(t/2^q) = floor(tlo/2^q) + sum_m [t >= m 2^q]  (2 SwiGLU units per
     step, groups of q bits per FFN level while the 256-wide gate window allows).  This is the "bit-plane
     re-count": the dropped planes are re-counted into the carry that rounds toward -inf.
  2. high digits.  v1's top-down Horner fold  u <- clamp(2^q u + <q digits>)  with the exact saturation bounds
     [Ulo(p), Uhi(p)] for the target clamp(., 0, M) (relu = lower bound 0), then bits of u by alternating
     thresholds  bit_a = sum_m (-1)^(m+1) [u >= m 2^a].

Output bits are 0/1 residual dims, i.e. exactly the form of the ACT operand bit features (ab[j][a]) and, via
`code_ffn`, of the sign-bit output code.  Pure IR: nothing here touches the host.
"""
from dataclasses import dataclass, field

from .gadgets import Circ
from .ir import Expr

WINDOW = 256


def clamp_passes(zlo, zhi, L, H, stop=None):
    """Saturate Z in [zlo, zhi] to [L, H] with as many one-FFN passes as the 256-wide windows need."""
    out = []
    while (zlo < L or zhi > H) and not (stop and stop(zlo, zhi)):
        lo = L if L - zlo <= WINDOW else zlo + WINDOW
        hi = H if zhi - H <= WINDOW else zhi - WINDOW
        lo, hi = (lo if zlo < L else zlo), (hi if zhi > H else zhi)
        out.append((lo, hi)); zlo, zhi = max(zlo, lo), min(zhi, hi)
    return out, (zlo, zhi)


def _apply_passes(g, u, zlo, zhi, passes, name):
    for k, (cl, ch) in enumerate(passes):
        u = u + g.ffn(([(1, g.one, cl - u)] if cl > zlo else []) + ([(-1, g.one, u - ch)] if ch < zhi else []),
                      f"{name}.{k}")
        zlo, zhi = max(zlo, cl), min(zhi, ch)
    return u, (zlo, zhi)


def _ge_ok(theta, lo, hi):
    return theta > hi or theta <= lo or hi - theta + 1 <= WINDOW or theta - lo <= WINDOW


def _stair_ok(tlo, thi, D):
    return all(_ge_ok(m * D, tlo, thi) for m in range(tlo // D + 1, thi // D + 1))


def relu(g, y, lo, hi, cap=None, name="relu"):
    """clamp(y, 0, cap) for an integer Expr y in [lo, hi] (cap=None: plain ReLU).  Returns (Expr, (lo', hi'))."""
    cap = hi if cap is None else cap
    passes, rng = clamp_passes(lo, hi, 0, cap)
    return _apply_passes(g, y, lo, hi, passes, name)


@dataclass
class Requant:
    bits: list            # amax.bit_length() Exprs, LSB first, each 0/1 -- the next ACT operand's bit features
    value: Expr           # sum 2^a bits[a] = clamp(floor(relu(y)/2^k), 0, amax)  (free linear form)
    amax: int
    shift: int
    levels: tuple         # (first FFN level used, last FFN level used)
    carry_levels: int = 0
    fold_levels: int = 0
    info: dict = field(default_factory=dict)


def requant(g: Circ, digits, shift=0, amax=31, name="rq", bits=True):
    """Append ReLU + (>> shift) + clamp(0, amax) to an accumulator given as digits=[(Expr, lo, hi), ...]
    (y = sum 2^j digit_j).  amax must be 2^n - 1.  Returns Requant; costs
    ceil-ish(shift / q) carry levels + fold levels + 1 bit level (see Requant.info)."""
    M = amax
    assert M >= 1 and M & (M + 1) == 0, "amax must be 2^n - 1"
    one = g.one
    n = len(digits)
    dig = lambda j: digits[j] if j < n else (0 * one, 0, 0)
    start_levels = len(g.used) and None
    first = None
    nodes0 = len(g.nodes)

    # ---- phase 1: carry out of the low `shift` bits ---------------------------------------------------------
    c, clo, chi = 0 * one, 0, 0
    j = 0; carry_groups = 0
    while j < shift:
        for q in range(min(shift - j, 16), 0, -1):
            parts = [dig(j + i) for i in range(q)]
            tlo = clo + sum(p[1] << i for i, p in enumerate(parts))
            thi = chi + sum(p[2] << i for i, p in enumerate(parts))
            if _stair_ok(tlo, thi, 1 << q):
                break
        else:
            raise ValueError(f"carry staircase for bit {j} spans [{tlo},{thi}]: wider than the 256 gate window; "
                             "split the accumulator (fewer rows per read) or use a smaller per-digit range")
        t = c + sum((p[0] * (1 << i) for i, p in enumerate(parts)), 0 * one)
        D = 1 << q
        base = tlo // D
        units = []
        for m in range(base + 1, thi // D + 1):
            units += g.terms_ge(t, m * D, tlo, thi)
        c = g.node(units, f"{name}.carry{j}") + base
        clo, chi = base, thi // D
        j += q; carry_groups += 1

    # ---- phase 2: top-down Horner over the remaining digits, target clamp(., 0, M) -----------------------
    e = [dig(shift + i) for i in range(max(n - shift, 1))]
    e[0] = (e[0][0] + c, e[0][1] + clo, e[0][2] + chi)
    rlo = lambda p: sum(e[i][1] << i for i in range(p))
    rhi = lambda p: sum(e[i][2] << i for i in range(p))
    ulo = lambda p: (-rhi(p)) // (1 << p)
    uhi = lambda p: -(-(M - rlo(p)) // (1 << p))
    fits = lambda zlo, zhi: all(_ge_ok(t, zlo, zhi) for t in range(1, M + 1))
    u, lo, hi = 0 * one, 0, 0
    p = len(e); fold_groups = 0
    while p > 0:
        for q in range(p, 0, -1):
            zlo = (lo << q) + sum(e[p - q + i][1] << i for i in range(q))
            zhi = (hi << q) + sum(e[p - q + i][2] << i for i in range(q))
            L, H = ulo(p - q), uhi(p - q)
            if (p - q == 0 and fits(zlo, zhi)) or len(clamp_passes(zlo, zhi, L, H)[0]) <= 1:
                break
        else:
            q = 1
            zlo = (lo << 1) + e[p - 1][1]; zhi = (hi << 1) + e[p - 1][2]; L, H = ulo(p - 1), uhi(p - 1)
        Z = u * (1 << q) + sum((e[p - q + i][0] * (1 << i) for i in range(q)), 0 * one)
        p -= q
        passes, _ = clamp_passes(zlo, zhi, L, H, fits if p == 0 else None)
        u, (lo, hi) = _apply_passes(g, Z, zlo, zhi, passes, f"{name}.fold{fold_groups}")
        fold_groups += len(passes)
    info = {"carry_groups": carry_groups, "fold_passes": fold_groups, "u_range": (lo, hi)}

    # ---- bits of min(u, M) by alternating thresholds (u < 0 gives 0) -----------------------------------------
    nb = M.bit_length()
    outs = []
    for a in range(nb):
        units = []
        for m in range(1, (M >> a) + 1):
            units += g.terms_ge(u, m << a, lo, hi, 1 if m % 2 else -1)
        outs.append(g.node(units, f"{name}.bit{a}"))
    value = sum((b * (1 << a) for a, b in enumerate(outs)), 0 * one)
    lv = [n_.level for n_ in g.nodes[nodes0:]]
    return Requant(outs, value, M, shift, (min(lv), max(lv)) if lv else (0, 0), carry_groups, fold_groups, info)


def requant_expr(g, y, lo, hi, shift=0, amax=31, name="rq"):
    """ReLU + >> shift + clamp for a single wide integer Expr y in [lo, hi]: pre-clamp to [0, amax*2^shift]
    (relu and saturation; floor-shift of the clamped value is unchanged), then the digit path with one digit."""
    top = amax << shift
    u, (l2, h2) = relu(g, y, lo, hi, top, f"{name}.pre")
    return requant(g, [(u, l2, h2)], shift, amax, name)


def column_digits(cnt_col, g_copies, WB, AB, KR, cols=None):
    """Digit list of one output column from the A-head counts.  cnt_col[j][a*WB+b] is the count Expr of
    activation-bit a, weight-bit b in copy j (two's complement weight, top bit negative).  y = sum_a 2^a d_a."""
    neg = 1 << (WB - 1)
    smin, smax = -neg * KR, (neg - 1) * KR
    out = []
    for a in range(AB):
        d = Expr()
        for j in range(g_copies):
            for b in range(WB):
                d = d + cnt_col[j][WB * a + b] * (-neg if b == WB - 1 else (1 << b))
        out.append((d, smin, smax))
    return out


def slot_bits(rqs, h):
    """Bits of the OUT token of a READ that serves h columns (column c sits in slot c % h; only the READ's own
    columns are non-zero, the rest requantise y = 0 to 0, so the per-slot sum is the active column's bits).
    Returns [bit Expr] with bit index t*nbits + a = bit a of slot t -- circuit.tok_out's base-2^nbits layout."""
    nb = len(rqs[0].bits)
    out = []
    for t in range(h):
        for a in range(nb):
            out.append(sum((r.bits[a] for c, r in enumerate(rqs) if c % h == t), Expr()))
    return out


def code_ffn(g, bits, code_dims, S=2 ** 12, name="nlcode"):
    """Rewrite the sign-bit output code dims (positive = 1, negative = 0, magnitude S) from 0/1 bit Exprs:
    code_i <- S*(2 bit_i - 1).  code_dims[i] are Exprs on token-feature code dims (see circuit.build)."""
    one = g.one
    for i, cd in enumerate(code_dims):
        nxt = [(2 * S, one, bits[i])] if i < len(bits) else []
        g.ffn(nxt + [(-S, one, one), (-1, one, cd), (1, one, -1 * cd)], f"{name}{i}", into=cd)


def act_code(g, value, lo, hi, rd, is_read, act0, nout, nop, ncode, S=2 ** 12, name="actcode"):
    """Sign-bit output code that spells the *next ACT token*  ACT(k; y) = act0 + k*nout + y  at READ(k), for g == h == 1
    (square array, row s = column k), and NOP elsewhere -- so the model's own output is the next layer's input.
    `value` in [lo, hi] is the (possibly unclamped) relu+clamp Horner result u (circuit.build()["slots"][0]) or
    Requant.value; values <= 0 mean 0, >= nout-1 mean nout-1; inactive columns contribute 0.  rd[k] / is_read are the
    READ token features.  The sum act0 + k*nout + y is split at nout: low = (c0 + y) mod nout by equality units on y,
    high = act0//nout + k (+1 if c0 + y >= nout) from the one-hot rd[k] and one carry threshold.  Creates and returns
    fresh code token features (one FFN stage, `ncode` dims); pass them as the model's code dims."""
    one = g.one
    code = [g.token_feature(f"{name}_c{i}", lambda t, i=i: 128 * (2 * ((t >> i) & 1) - 1)) for i in range(ncode)]
    assert nout & (nout - 1) == 0
    ab = nout.bit_length() - 1
    c0, base = act0 % nout, act0 // nout
    M = nout - 1
    for i in range(ncode):
        nxt = [(1, one, one - is_read)] if (nop >> i) & 1 else []
        if i < ab:
            for v in range(nout):
                if (((c0 + v) % nout) >> i) & 1:
                    if v == 0: nxt += g.terms_lt(value, 1, lo, hi, 1, (is_read,))
                    elif v == M: nxt += g.terms_ge(value, M, lo, hi, 1, (is_read,))
                    else: nxt += g.terms_eq(value, v, lo, hi, (is_read,))
        else:
            j = i - ab
            for k, r in enumerate(rd):
                b0 = ((base + k) >> j) & 1; b1 = ((base + k + 1) >> j) & 1
                if b0: nxt.append((1, one, r))
                if c0 and b1 != b0:
                    nxt += g.terms_ge(value, nout - c0, lo, hi, b1 - b0, (r,))
        g.ffn([(2 * S * c, a, b) for c, a, b in nxt] + [(-S, one, one), (-1, one, code[i]), (1, one, -1 * code[i])],
              f"{name}{i}", into=code[i])
    return code


def feedback_code(g, d, C):
    """v2.0 fused op, zero extra layers: replace v1's OUT code by the code of the next ACT token.  Needs g == h == 1
    and KR == NC (call after circuit.build(); `C` is the configured circuit module).  Returns the new code dims
    (token-feature Exprs) -- hand THESE to the model builder instead of d['code']."""
    assert C.G == 1 and C.H == 1 and C.KR == C.NC
    rd, is_read = read_features(g, C.NREAD)
    u, lo, hi = d["slots"][0]
    return act_code(g, u, lo, hi, rd, is_read, C.ACT0, C.NOUT, C.NOP, C.CODE)


def read_features(g, nread):
    """(rd list, is_read) token-feature Exprs of a circuit.build() graph, found by name."""
    f = lambda n: Expr({g.names.index(n): 1})
    return [f(f"rd{k}") for k in range(nread)], f("is_read")


def reference(y, shift, amax):
    """Host reference for tests: clamp(floor(y / 2^shift), 0, amax) (python/numpy floor shift)."""
    import numpy as np
    return np.clip(np.asarray(y) >> shift, 0, amax)
