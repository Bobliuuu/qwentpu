"""Exact float64 evaluation of a circuit on a token stream, asserting that real bf16 hardware
would reproduce every intermediate bit for bit.

Text tokens are passed as -1: every chip dim is 0 there (ballast and ONE included).
"""
import numpy as np

from .ir import A_BIT, A_REF, DECAY, FFN, K, ONE, REF_COLS, HD, Head

RESIDUAL_MAX = 2.0 ** 28    # x^2 must vanish next to the ballast's m * 2^80 in the norms' fp32 sum


class ExactnessError(AssertionError):
    pass


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)   # round to nearest even
    return u.view(np.float32).astype(np.float64)


def is_bf16(x): return np.asarray(bf16(x) == np.asarray(x, dtype=np.float64))


def silu_ok(x):   # silu(x) == relu(x) exactly in fp32:  0, >= 18 (sigmoid rounds to 1), <= -104 (underflows)
    return (x == 0) | (x >= 18) | (x <= -104)


def require(ok, what, x=None):
    ok = np.asarray(ok)
    if not ok.all():
        bad = np.flatnonzero(~ok.reshape(-1))[0]
        val = None if x is None else np.asarray(x).reshape(-1)[min(bad, np.asarray(x).size - 1)]
        raise ExactnessError(f"{what}: bad value {val!r} at {bad}")


def ev(e, m):
    if not e.t:
        return np.zeros(m.shape[0])
    idx = np.fromiter(e.t.keys(), dtype=np.int64)
    return m[:, idx] @ np.array([float(c) for c in e.t.values()])


def conv_silu(c, m, check):
    x = ev(c.expr, m)
    if check:
        require(is_bf16(x) | (c.taps[0] * bf16(x) <= -104) & (not any(c.taps[1:])), "projection", x)
    out = np.zeros(len(x))
    for lag, w in enumerate(c.taps):
        if w:
            out[lag:] += w * x[: len(x) - lag]
    if check:
        require(is_bf16(out) & silu_ok(out), "conv output (silu input)", out)
    return np.maximum(out, 0.0)


def eval_ffn(n, m, check):
    ideal = np.zeros(m.shape[0]); hw = np.zeros(m.shape[0])
    for c, a, b in n.terms:
        bv, av = ev(b, m), ev(a, m)
        ideal += float(c) * av * np.maximum(bv, 0.0)
        if check:
            gate = bf16(K * bv)
            require(silu_ok(gate), "gate outside silu's exact range", gate)
            hw += float(c) * bf16(np.maximum(gate, 0.0) * bf16(av / K))
    if check:
        require(bf16(hw) == ideal, "bf16 output differs from exact", bf16(hw))
    if n.into:
        ideal = m[:, n.out] + ideal
        if check:
            require(is_bf16(ideal), "residual add", ideal)
    m[:, n.out] = ideal


def eval_head(n, m, chip, check):
    T = m.shape[0]
    q = np.stack([conv_silu(c, m, check) for c in n.q.values()], 1)
    k = np.stack([conv_silu(c, m, check) for c in n.k.values()], 1)
    qi, ki = list(n.q), list(n.k)
    if check:
        nz = (k > 0).sum(1)
        require(nz <= 1, "key must be one-hot", nz)
        for t in np.flatnonzero((q > 0).sum(1) > 1):
            vals = q[t][q[t] > 0]; require(vals == vals[0], "multi-hot query must have equal entries", vals)
            r = np.sqrt(len(vals) * HD)     # Qwen's q scale 1/(sqrt(n) sqrt(HD)) must be a power of two, or bf16
            require(r == 2.0 ** np.round(np.log2(r)), "multi-hot query size: n*HD must be a power of 4", len(vals))
    has_key = k.any(1); has_q = q.any(1)
    for vh in n.vx:
        v = np.zeros((T, HD))
        for col, c in vh.v.items():
            v[:, col] = conv_silu(c, m, check)
        a = ev(vh.a, m); z = {col: ev(e, m) for col, e in vh.z.items()}
        beta = np.ones(T) if vh.beta is None else ev(vh.beta, m)
        writes = has_key & (beta > 0)
        if check:
            require(is_bf16(a) & np.isin(a, list(DECAY)), "decay selector a", a)
            require(np.isin(beta, [0.0, 1.0]), "write gate beta", beta)
            for col, zz in z.items():
                require((is_bf16(zz) & silu_ok(zz)) | (zz <= -104), f"gate z[{col}]", zz)
            w = v[writes]
            if len(w):
                require(np.isin(np.delete(w, REF_COLS, 1), [0.0, A_BIT]), "state data column", w)
                require(np.isin(w[:, list(REF_COLS)], [0.0, A_REF]), "state ballast column", w)
        state = {}; out = np.zeros((T, HD)); decay = np.array([DECAY[int(x)] for x in a])
        for t in np.flatnonzero((decay != 1.0) | writes | has_q):
            if decay[t] == 0.0:
                state.clear()
            if writes[t]:
                state[ki[int(np.argmax(k[t]))]] = v[t].copy()
            if has_q[t]:
                o = sum((state[s] for s, w in zip(qi, q[t]) if w > 0 and s in state), np.zeros(HD))
                ref = (o[list(REF_COLS)] ** 2).sum()
                if ref == 0:
                    if check: require(o == 0, "read without ballast", o)
                    continue
                if check:
                    require((np.delete(o, REF_COLS) ** 2).sum() <= ref * 2.0 ** -26, "data not absorbed by ballast")
                out[t] = o / np.sqrt(ref / HD)
        for col, d, w in vh.out:
            val = out[:, col] * np.maximum(z.get(col, np.zeros(T)), 0.0)
            if check:
                require(is_bf16(out[:, col]) | (np.abs(out[:, col]) < 1e-30), f"normed column {col}", out[:, col])
                require(is_bf16(val * w), f"head output column {col}", val * w)
            m[:, d] = val * w


def evaluate(g, tokens, check=True):
    """Residual stream after every node: (n_tokens, n_dims)."""
    toks = np.asarray(tokens, dtype=np.int64); chip = toks >= 0
    m = np.zeros((len(toks), len(g.names))); m[:, ONE] = chip
    for d, fn in g.token_fns.items():
        col = np.array([float(fn(int(t))) if t >= 0 else 0.0 for t in toks])
        if check: require(is_bf16(col), f"token feature {g.names[d]}", col)
        m[:, d] = col
    for n in g.nodes:
        try:
            eval_ffn(n, m, check) if isinstance(n, FFN) else eval_head(n, m, chip, check)
        except ExactnessError as e:
            raise ExactnessError(f"{n.name}: {e}") from None
    if check:
        require(np.abs(m[:, 1:]).max(initial=0) <= RESIDUAL_MAX, "residual value would move the norms")
    return m
