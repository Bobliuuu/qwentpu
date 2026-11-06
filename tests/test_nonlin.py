"""v2 ReLU / requant gadgets: exhaustive IR evaluation (bf16/silu exactness asserted) over the full operand range."""
import itertools

import numpy as np
import pytest

from qwentpu import circuit as C
from qwentpu.compiler import evaluate as ir
from qwentpu.compiler import nonlin as N
from qwentpu.compiler.gadgets import Circ


def digit_graph(ranges):
    g = Circ(); f = g.token_feature
    combos = list(itertools.product(*[range(lo, hi + 1) for lo, hi in ranges]))
    arr = np.array(combos).reshape(len(combos), len(ranges))
    feats = [f(f"d{j}", lambda t, j=j: int(arr[t, j])) for j in range(len(ranges))]
    return g, feats, arr


def check(ranges, shift, amax, sample=None, seed=0):
    g, feats, arr = digit_graph(ranges)
    rq = N.requant(g, [(e, lo, hi) for e, (lo, hi) in zip(feats, ranges)], shift, amax)
    toks = np.arange(len(arr)) if sample is None else np.random.default_rng(seed).choice(len(arr), sample, replace=False)
    m = ir.evaluate(g, toks)
    got = sum((1 << a) * ir.ev(b, m) for a, b in enumerate(rq.bits)).astype(int)
    y = sum((arr[toks, j].astype(np.int64) << j) for j in range(len(ranges)))
    assert (got == N.reference(y, shift, amax)).all(), (ranges, shift, amax)
    for b in rq.bits:
        assert set(np.unique(ir.ev(b, m))) <= {0.0, 1.0}
    return g, rq


@pytest.mark.parametrize("shift", range(0, 7))
@pytest.mark.parametrize("amax", [7, 31])
def test_requant_exhaustive_digits(shift, amax):
    # 4 digits, each in [-20, 20]: y spans +-300; every digit combination (1.9M) is too many, so 41^3 x ... use 3 digits
    g, rq = check([(-20, 20)] * 3, shift, amax)
    print(shift, amax, g.n_layers(), rq.levels, rq.info)


@pytest.mark.parametrize("shift", [0, 2, 4, 5, 8])
def test_requant_wide_digits_sampled(shift):
    # v1-like digits: 5 digits in [-128, 120] (8 rows, 5-bit weights) -- extremes always included via sampling+corners
    ranges = [(-128, 120)] * 5
    rng = np.random.default_rng(shift)
    X = rng.integers(-128, 121, (6000, 5)); X[:2 ** 5] = np.array(list(itertools.product([-128, 120], repeat=5)))
    X[2 ** 5:2 ** 5 + 5] = 0
    g = Circ(); f = g.token_feature
    feats = [f(f"d{j}", lambda t, j=j: int(X[t, j])) for j in range(5)]
    rq = N.requant(g, [(e, -128, 120) for e in feats], shift, 31)
    m = ir.evaluate(g, np.arange(len(X)))
    got = sum((1 << a) * ir.ev(b, m) for a, b in enumerate(rq.bits)).astype(int)
    y = (X.astype(np.int64) << np.arange(5)).sum(1)
    assert (got == N.reference(y, shift, 31)).all()
    print("wide", shift, g.n_layers(), rq.info)


@pytest.mark.parametrize("shift,amax", [(0, 15), (1, 31), (2, 31), (3, 7)])
def test_requant_expr_every_value(shift, amax):
    # y = 16 h + l with bf16-exact token features h in [-45, 45], l in [0, 15]: every integer in [-720, 735]
    g = Circ(); hf = g.token_feature("h", lambda t: t // 16 - 45); lf = g.token_feature("l", lambda t: t % 16)
    lo, hi = -720, 735
    rq = N.requant_expr(g, hf * 16 + lf, lo, hi, shift, amax)
    m = ir.evaluate(g, np.arange(91 * 16))
    got = sum((1 << a) * ir.ev(b, m) for a, b in enumerate(rq.bits)).astype(int)
    assert (got == N.reference(np.arange(-720, -720 + 91 * 16), shift, amax)).all()


@pytest.mark.parametrize("cap", [None, 31, 100])
def test_relu_every_value(cap):
    g = Circ(); hf = g.token_feature("h", lambda t: t // 16 - 100); lf = g.token_feature("l", lambda t: t % 16)
    lo, hi = -1600, 1615
    r, _ = N.relu(g, hf * 16 + lf, lo, hi, cap)
    m = ir.evaluate(g, np.arange(201 * 16))
    v = np.arange(-1600, -1600 + 201 * 16)
    assert (ir.ev(r, m) == np.clip(v, 0, hi if cap is None else cap)).all()


@pytest.mark.parametrize("cfg,shift", [((8, 8, 3, 3), 3), ((8, 8, 5, 3), 4), ((8, 8, 5, 3), 6), ((16, 8, 5, 3), 5)])
def test_after_matmul_read(cfg, shift):
    """Append requant to the real v1 MATMUL (A-head counts) and check against numpy for random weights/inputs."""
    C.configure(*cfg)
    try:
        g, d = C.build()
        amax = (1 << 5) - 1
        v1_layers = g.n_layers()
        outs = []
        for c in range(C.NC):
            dg = N.column_digits(d["cnt"][c], C.G, C.WB, C.AB, C.KR)
            outs.append(N.requant(g, dg, shift, amax, f"rq{c}"))
        rng = np.random.default_rng(5)
        W = rng.integers(C.wlo(), C.whi() + 1, (C.KR, C.NC)); X = rng.integers(0, C.AMAX + 1, (5, C.KR))
        X[0] = C.AMAX; W[:, 0] = C.whi(); W[:, 1] = C.wlo()
        toks = C.load_trace(W); n0 = len(toks); stride = C.KS + C.NREAD
        for x in X:
            toks += C.matvec_trace(x)
        m = ir.evaluate(g, toks)
        for i, x in enumerate(X):
            y = x @ W
            for k in range(C.NREAD):
                t = n0 + i * stride + C.KS + k          # the READ(k) token
                for cc in range(k * C.H, (k + 1) * C.H):
                    got = sum((1 << a) * int(ir.ev(b, m[t:t + 1])[0]) for a, b in enumerate(outs[cc].bits))
                    assert got == N.reference(y[cc], shift, amax), (cfg, shift, t, cc, y[cc], got)
        print(cfg, shift, "layers", g.n_layers(), "v1 layers", v1_layers, outs[0].info, outs[0].levels)
    finally:
        C.configure(8, 8, 3, 3)


# ---- compiled into a stock HF Qwen3_5ForCausalLM: next-token == requantised matmul output ----------------------
HF_CFGS = [((8, 8, 3, 3), 3), ((8, 8, 5, 3), 4), ((8, 8, 5, 3), 6), ((8, 8, 3, 3, 2, 2), 2)]


def build_nonlin_model(cfg, shift, dtype):
    from qwentpu.compiler import tiny
    C.configure(*cfg)
    g, d = C.build()
    amax = C.NOUT - 1
    rqs = [N.requant(g, N.column_digits(d["cnt"][c], C.G, C.WB, C.AB, C.KR), shift, amax, f"rq{c}") for c in range(C.NC)]
    bits = N.slot_bits(rqs, C.H)
    code = [g.token_feature(f"nc{i}", lambda t, i=i: 128 * (2 * ((t >> i) & 1) - 1)) for i in range(C.CODE)]
    N.code_ffn(g, bits, code)
    model = tiny.to_device(tiny.build(g, C.VOCAB, [next(iter(c.t)) for c in code], dtype=dtype))
    return g, model


@pytest.mark.parametrize("cfg,shift", HF_CFGS, ids=lambda x: str(x))
def test_hf_relu_requant_after_matmul(cfg, shift):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        for dtype in (torch.bfloat16, torch.float32):
            g, model = build_nonlin_model(cfg, shift, dtype)
            rng = np.random.default_rng(7)
            W = rng.integers(C.wlo(), C.whi() + 1, (C.KR, C.NC)); X = rng.integers(0, C.AMAX + 1, (6, C.KR))
            X[0] = C.AMAX; W[:, 0] = C.whi(); W[:, 1] = C.wlo(); X[1] = 0
            toks = C.load_trace(W); want = {}
            for x in X:
                want.update({len(toks) + C.KS + k: v for k, v in enumerate(
                    C.tok_out(*N.reference(x @ W, shift, C.NOUT - 1)[k * C.H:(k + 1) * C.H]) for k in range(C.NREAD))})
                toks += C.matvec_trace(x)
            with torch.no_grad():
                logits = model(torch.tensor([toks], device=next(model.parameters()).device)).logits[0].float().cpu()
            pred = logits[:, : len(C.VOCAB)].argmax(-1).numpy()
            bad = [(t, int(pred[t]), w) for t, w in want.items() if pred[t] != w]
            assert not bad, (dtype, bad[:5])
            print(cfg, shift, dtype, "layers", g.n_layers(), "all", len(want), "readouts exact")
    finally:
        C.configure(8, 8, 3, 3)


# ---- inside the real Qwen3.5 (set QWEN_DIR): MATMUL -> ReLU -> >> shift -> clamp, bit-exact next-token ----------
def test_merged_real_qwen_requant(monkeypatch):
    import os
    src = os.environ.get("QWEN_DIR")
    torch = pytest.importorskip("torch")
    if not src:
        pytest.skip("set QWEN_DIR to a Qwen3.5 checkpoint")
    from transformers import AutoTokenizer
    from qwentpu import merge
    from qwentpu.mute import fit_anti_sink
    dev = os.environ.get("QWEN_DEVICE", "cpu")
    shift = int(os.environ.get("NL_SHIFT", "4"))
    C.configure(8, 8, 5, 3)
    orig = C.build
    try:
        def build():
            g, d = orig()
            rqs = [N.requant(g, N.column_digits(d["cnt"][c], C.G, C.WB, C.AB, C.KR), shift, C.NOUT - 1, f"rq{c}")
                   for c in range(C.NC)]
            code = [g.token_feature(f"nc{i}", lambda t, i=i: 128 * (2 * ((t >> i) & 1) - 1)) for i in range(C.CODE)]
            N.code_ffn(g, N.slot_bits(rqs, C.H), code)
            d = dict(d); d["code"] = code
            print("chip layers (v1 + relu/requant):", g.n_layers(), g.stats())
            return g, d
        monkeypatch.setattr(C, "build", build)
        tok = AutoTokenizer.from_pretrained(src)
        prompts = ["The capital of France is Paris, and the capital of Germany is"]
        cfg, sd = merge.load(src); tc0 = dict(cfg["text_config"]); tc0["tie_word_embeddings"] = False
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
        base = merge.instantiate(tc0, sd, dev); sinks = fit_anti_sink(base, tok, prompts)
        del base, sd
        tc, sd, v0, lay, g = merge.build_merged(src, sinks=sinks)
        model = merge.instantiate(tc, sd, dev)
        rng = np.random.default_rng(11)
        toks, want = [], {}
        for _ in range(3):
            W = rng.integers(C.wlo(), C.whi() + 1, (C.KR, C.NC)); X = rng.integers(0, C.AMAX + 1, (5, C.KR))
            X[0] = C.AMAX; W[:, 0] = C.whi(); W[:, 1] = C.wlo(); X[1] = 0
            toks += C.load_trace(W)
            for x in X:
                for k in range(C.NREAD):
                    want[len(toks) + C.KS + k] = C.tok_out(*N.reference(x @ W, shift, C.NOUT - 1)[k * C.H:(k + 1) * C.H])
                toks += C.matvec_trace(x)
        for prefix in ([], tok(prompts[0]).input_ids):
            ids = torch.tensor([list(prefix) + [v0 + t for t in toks]]).to(dev)
            with torch.no_grad():
                pred = model(input_ids=ids).logits[0, len(prefix):].float().argmax(-1).cpu().numpy() - v0
            bad = [(t, int(pred[t]), w) for t, w in want.items() if pred[t] != w]
            assert not bad, bad[:5]
            vals = [v for w in want.values() for v in C.out_values(w)]
            print(f"real Qwen: shift {shift}, {len(want)} readouts exact; distinct values {sorted(set(vals))}")
            assert len(set(vals)) >= 6
    finally:
        C.configure(8, 8, 3, 3)


# ---- v2.0 fused op: relu+clamp readout that spells the next ACT token ------------------------------------------
def act_graph(cfg):
    C.configure(*cfg)
    g, d = C.build()
    code = N.feedback_code(g, d, C)
    return g, d, code


def decode_tok(code, m, t):
    return sum((1 << i) * int(ir.ev(c, m[t:t + 1])[0] > 0) for i, c in enumerate(code))


@pytest.mark.parametrize("cfg", [(8, 8, 3, 3), (8, 8, 5, 5), (16, 16, 5, 5), (16, 16, 4, 4)])
def test_ir_feedback_is_next_act_token(cfg):
    try:
        g, d, code = act_graph(cfg)
        rng = np.random.default_rng(2)
        seen = set(); n_extra = g.n_layers()
        for gentle in (True, False):
            W, X = C.rand_case(rng, gentle, 5)
            if not gentle:
                X[0] = C.AMAX; W[:, 0] = C.whi(); W[:, 1] = C.wlo(); X[1] = 0
            toks = C.load_trace(W); n0 = len(toks)
            for x in X:
                toks += C.matvec_trace(x)
            m = ir.evaluate(g, toks)
            for i, x in enumerate(X):
                y = C.ref_matvec(x, W)
                for k in range(C.NREAD):
                    t = n0 + i * (C.KS + C.NREAD) + C.KS + k
                    want = C.tok_act(k, int(y[k])); seen.add(int(y[k]))
                    assert decode_tok(code, m, t) == want, (cfg, t, k, int(y[k]))
                    if k == 0: assert decode_tok(code, m, t - 1) == C.NOP     # ACT tokens still emit NOP
        assert len(seen) >= 3
        print(cfg, "chip layers", n_extra, "(v1 had", None, ")")
    finally:
        C.configure(8, 8, 3, 3)


def test_feedback_code_exhaustive_ky():
    """Every (k, y): READ(k) with a clamped result y in 0..AMAX spells ACT(k; y); also saturated/negative u."""
    C.configure(8, 8, 5, 5)
    try:
        g = Circ(); f = g.token_feature
        nout, nread = C.NOUT, C.NREAD
        rd = [f(f"rd{k}", lambda t, k=k: int(t // 4096 == k + 1)) for k in range(nread)]
        is_read = f("is_read", lambda t: int(t >= 4096))
        u = f("u", lambda t: (t % 4096) - 40)                    # u in [-40, 100): below 0 and above amax included
        code = N.act_code(g, u, -40, 59, rd, is_read, C.ACT0, nout, C.NOP, C.CODE)
        toks = [k * 4096 + 4096 + j for k in range(nread) for j in range(100)] + [j for j in range(100)]
        m = ir.evaluate(g, toks)
        for t, tk in enumerate(toks):
            if tk < 4096:
                assert decode_tok(code, m, t) == C.NOP
            else:
                k, j = tk // 4096 - 1, tk % 4096
                assert decode_tok(code, m, t) == C.tok_act(k, min(max(j - 40, 0), C.AMAX)), (k, j)
    finally:
        C.configure(8, 8, 3, 3)


def test_merged_real_qwen_feedback(monkeypatch):
    """v2.0 fused op in the real model: at READ(k) the next token is ACT(k; clip(relu(x@W),0,AMAX)) -- bit-exact."""
    import os
    src = os.environ.get("QWEN_DIR")
    torch = pytest.importorskip("torch")
    if not src:
        pytest.skip("set QWEN_DIR to a Qwen3.5 checkpoint")
    from transformers import AutoTokenizer
    from qwentpu import merge
    from qwentpu.mute import fit_anti_sink
    dev = os.environ.get("QWEN_DEVICE", "cpu")
    cfg = tuple(int(v) for v in os.environ.get("NL_CFG", "16,16,5,5").split(","))
    C.configure(*cfg)
    orig = C.build
    try:
        def build():
            g, d = orig()
            d = dict(d); d["code"] = N.feedback_code(g, d, C)
            print("chip layers:", g.n_layers(), g.stats())
            return g, d
        monkeypatch.setattr(C, "build", build)
        tok = AutoTokenizer.from_pretrained(src)
        prompts = ["The capital of France is Paris, and the capital of Germany is"]
        cfg_, sd = merge.load(src); tc0 = dict(cfg_["text_config"]); tc0["tie_word_embeddings"] = False
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
        base = merge.instantiate(tc0, sd, dev); sinks = fit_anti_sink(base, tok, prompts)
        del base, sd
        tc, sd, v0, lay, g = merge.build_merged(src, sinks=sinks)
        model = merge.instantiate(tc, sd, dev)
        rng = np.random.default_rng(13)
        toks, want = [], {}
        for _ in range(2):
            W, X = C.rand_case(rng, True, 4)
            W2, X2 = C.rand_case(rng, False, 4)
            W = np.where(rng.random(W.shape) < .5, W, W2); X = np.vstack([X, X2])
            toks += C.load_trace(W)
            for x in X:
                y = C.ref_matvec(x, W)
                for k in range(C.NREAD):
                    want[len(toks) + C.KS + k] = C.tok_act(k, int(y[k]))
                toks += C.matvec_trace(x)
        for prefix in ([], tok(prompts[0]).input_ids):
            ids = torch.tensor([list(prefix) + [v0 + t for t in toks]]).to(dev)
            with torch.no_grad():
                pred = model(input_ids=ids).logits[0, len(prefix):].float().argmax(-1).cpu().numpy() - v0
            bad = [(t, int(pred[t]), w) for t, w in want.items() if pred[t] != w]
            assert not bad, bad[:5]
            ys = sorted({(w - C.ACT0) % C.NOUT for w in want.values()})
            print(f"real Qwen {cfg}: {len(want)} ACT feedback tokens exact; distinct y {ys}")
            assert len(ys) >= 6
    finally:
        C.configure(8, 8, 3, 3)
