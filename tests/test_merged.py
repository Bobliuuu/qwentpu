"""TPU inside a real Qwen3.5 (set QWEN_DIR: 0.8B, 2B or 4B): exact TPU, untouched text, muted Qwen."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SRC = os.environ.get("QWEN_DIR")
pytestmark = pytest.mark.skipif(not SRC, reason="set QWEN_DIR to a Qwen3.5 checkpoint")
from qwentpu import circuit as C
from qwentpu.merge import build_merged, load
from qwentpu.merge import instantiate as _inst

DEV = os.environ.get("QWEN_DEVICE", "cpu")
if DEV != "cpu":
    torch.set_default_device(DEV)


def instantiate(tc, sd):
    return _inst(tc, sd, DEV)

rng = np.random.default_rng(5)
PROMPTS = ["The capital of France is Paris, and the capital of Germany is",
           "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n\nprint(fib(10))",
           "In 1969, Apollo 11 landed on the Moon. Neil Armstrong was the first to"]


def _build_base():
    cfg, sd = load(SRC); tc = dict(cfg["text_config"]); tc["tie_word_embeddings"] = False
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    return instantiate(tc, sd)


class OnDemand:
    """The unmodified base model is rebuilt for each forward pass and freed right after, so base + merged + the
    mutation copy never sit in VRAM together (24 GB cards) and the base never sits in host RAM next to the merged
    state dict (16 GB hosts).  On the CPU it is simply cached."""
    _cache = None

    def build(self):
        if DEV == "cpu":
            OnDemand._cache = OnDemand._cache or _build_base()
            return OnDemand._cache
        return _build_base()

    def __call__(self, **kw):
        import gc
        m = self.build()
        try: return m(**kw)
        finally:
            if DEV != "cpu":
                del m; gc.collect(); torch.cuda.empty_cache()


def base_model(): return OnDemand()


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SRC)


@pytest.fixture(scope="module")
def merged(tok):
    from qwentpu.mute import fit_anti_sink
    import gc
    base = base_model(); m = base.build()
    sinks = fit_anti_sink(m, tok, PROMPTS)
    if DEV != "cpu":
        del m; gc.collect(); torch.cuda.empty_cache()
    tc, sd, v0, lay, g = build_merged(SRC, sinks=sinks)
    return instantiate(tc, sd), base, v0, (tc, sd, g)


def trace(W, X):
    toks = C.load_trace(W); want = [C.NOP] * len(toks)
    for x in X:
        toks += C.matvec_trace(x); want += [C.NOP] * C.KS + C.expected_outs(x, W)
    return toks, want


def run(model, v0, toks, prefix=()):
    ids = torch.tensor([list(prefix) + [v0 + t for t in toks]])
    with torch.no_grad():
        return model(input_ids=ids).logits[0, len(prefix):].float().argmax(-1).cpu().numpy() - v0


def test_tpu_exact_inside_qwen(merged):
    model, _, v0, _ = merged
    toks, want = [], []
    for _ in range(int(os.environ.get("N_MATS", 3))):
        W, X = C.rand_case(rng, True, 3); t, w = trace(W, X)
        toks += t; want += w
    assert sum(1 <= v <= C.AMAX - 1 for w in want if w != C.NOP for v in C.out_values(w - C.OUT0)) >= 12     # not just saturated 0/7
    pred = run(model, v0, toks)
    bad = [(t, int(pred[t]), w) for t, w in enumerate(want) if pred[t] != w]
    assert not bad, bad[:5]


def test_tpu_exact_after_text(merged, tok):
    """Text in Qwen's own state before the trace must not leak into the TPU."""
    model, _, v0, _ = merged
    W, X = C.rand_case(rng, True, 3); toks, want = trace(W, X)
    pred = run(model, v0, toks, prefix=tok(PROMPTS[0]).input_ids)
    assert all(pred[t] == w for t, w in enumerate(want))


def kl(a, b):
    return float(torch.nn.functional.kl_div(b.log_softmax(-1), a.log_softmax(-1), log_target=True, reduction="batchmean"))


def test_text_unchanged(merged, tok):
    model, base, v0, _ = merged
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            a = base(input_ids=ids).logits[0].float(); lg = model(input_ids=ids).logits[0].float()
        b = lg[:, : a.shape[-1]]
        agree = float((a.argmax(-1) == b.argmax(-1)).float().mean())
        print(f"text KL {kl(a, b):.2e}  argmax agree {agree:.2f}  max chip logit {float(lg[:, v0:].max()):.0f}")
        assert kl(a, b) < 5e-3 and agree >= 0.9
        assert lg[:, v0:].max() < -1e3                 # tau: chip tokens are never sampled in prose


def test_text_after_chip_run(merged, tok):
    """text -> TPU trace -> text.  Qwen's 3-token conv window sees zeros for the first 3 tokens after a
    chip run (a boundary effect, independent of trace length); everything later should be back to noise."""
    model, base, v0, _ = merged
    t1 = tok(PROMPTS[2]).input_ids
    t2 = tok(" Berlin. The capital of Italy is Rome, and the capital of Spain is Madrid, while the capital of Japan is Tokyo").input_ids
    W, X = C.rand_case(rng, True, 2); toks, _ = trace(W, X)
    with torch.no_grad():
        ref = base(input_ids=torch.tensor([t1 + t2])).logits[0, -len(t2):].float()
        got = model(input_ids=torch.tensor([t1 + [v0 + t for t in toks] + t2])).logits[0, -len(t2):, : ref.shape[-1]].float()
    pos = [float(torch.nn.functional.kl_div(got[i].log_softmax(-1), ref[i].log_softmax(-1), log_target=True, reduction="sum")) for i in range(len(t2))]
    print("per-position KL after a chip run:", [round(x, 3) for x in pos[:6]], "| mean from token 4 on:", round(float(np.mean(pos[4:])), 4))
    assert np.mean(pos[4:]) < 0.1 and (ref[4:].argmax(-1) == got[4:].argmax(-1)).float().mean() >= 0.8


def test_exact_for_every_trace_length(merged):
    """GEMM split-K with bf16 partial sums (cuBLAS picks it by shape and GPU) broke 288- and 320-token traces on an
    L4: the result must not depend on the trace length."""
    model, _, v0, _ = merged
    for nvec in (1, 2, 3, 14, 16):                       # 8x8 traces are 64 + 16n tokens: 288 and 320 are the L4 failures
        W, X = C.rand_case(rng, True, nvec); toks, want = trace(W, X)
        pred = run(model, v0, toks)
        assert [int(p) for p in pred] == want, (nvec, len(toks))


def test_chip_only_path_is_bit_identical(merged):
    """Qwen's layers are the identity on chip tokens, so running only the TPU layers returns the same tokens
    (single prefill and fed in the chunks the driver uses, with a cache)."""
    from qwentpu.driver import TpuSession
    model, _, v0, _ = merged
    W, X = C.rand_case(rng, True, 3); toks, want = trace(W, X)
    full = TpuSession(model, v0, fast=False).feed(toks)
    heads = TpuSession(model, v0, readout="lm_head").feed(toks)
    code = TpuSession(model, v0, readout="code").feed(toks)
    assert full == heads == code and [int(p) for p in code] == want
    a, b = TpuSession(model, v0, fast=False), TpuSession(model, v0, fast=True)
    for sess in (a, b): sess.load_weights(W)
    for x in X: assert (a.matvec(x) == b.matvec(x)).all() and (b.matvec(x) == C.ref_matvec(x, W)).all()


@pytest.mark.skipif(not DEV.startswith("cuda"), reason="CUDA graphs")
def test_cuda_graph_executor_is_bit_identical(merged):
    """The graph-replayed chip path equals the eager one, chunk by chunk, with state carried between graphs."""
    from qwentpu.driver import TpuSession
    model, _, v0, _ = merged
    W, X = C.rand_case(rng, True, 3)
    eager, graph = TpuSession(model, v0, fast=True), TpuSession(model, v0, fast="graph")
    for sess in (eager, graph): sess.load_weights(W)
    for _ in range(2):                          # twice: the second round replays existing graphs
        for x in X:
            a, b = eager.matvec(x), graph.matvec(x)
            assert (a == b).all() and (b == C.ref_matvec(x, W)).all(), (a, b)


def run_once(tc, sd, v0, toks):
    """Build a model, run one trace, and free it again (24 GB cards cannot hold four 2B models)."""
    import gc
    model = instantiate(tc, sd)
    out = run(model, v0, toks)
    del model; gc.collect()
    if DEV != "cpu": torch.cuda.empty_cache()
    return out


def test_mutation_breaks_it(merged):
    _, _, v0, (tc, sd, g) = merged
    n0 = tc["num_hidden_layers"] - g.n_layers()
    W, X = C.rand_case(rng, True, 2); toks, want = trace(W, X)
    ok = run_once(tc, sd, v0, toks)
    sd2 = dict(sd)
    for L in range(n0, n0 + g.n_layers()):
        k = f"model.layers.{L}.linear_attn.in_proj_qkv.weight"; sd2[k] = torch.zeros_like(sd[k])
    bad = run_once(tc, sd2, v0, toks)
    wrong = [(t, ok[t], w) for t, w in enumerate(want) if ok[t] != w]
    assert not wrong, wrong[:6]
    assert any(bad[t] != w for t, w in enumerate(want))


def test_driver_xor_and_random_mlp(merged):
    """Whole networks through the real model with a cache; layers chain via the model's own tokens."""
    from qwentpu.driver import TpuSession, ref_mlp
    model, _, v0, _ = merged
    s = TpuSession(model, v0)
    W1 = np.array([[1, 1], [1, 1], [0, -1]]); W2 = np.array([[1], [-2], [0]])     # XOR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert s.mlp([W1, W2], X).ravel().tolist() == [0, 1, 1, 0]
    lo, hi = max(C.wlo(), -1), min(C.whi(), 2)          # stay inside the configured weight range
    Ws = [rng.integers(lo, hi + 1, (5, 4)), rng.integers(lo, hi + 1, (5, 3))]
    X = rng.integers(0, min(C.AMAX, 3) + 1, (3, 4))
    assert (s.mlp(Ws, X) == ref_mlp(Ws, X)).all()


def test_hybrid_restore_makes_chip_runs_invisible(merged, tok):
    """text -> TPU (inside chip()) -> text matches the no-trace logits at EVERY position, first token included;
    and the TPU keeps its weights across the text."""
    from qwentpu.driver import HybridSession
    model, base, v0, _ = merged
    t1 = tok(PROMPTS[2]).input_ids
    t2 = tok(" Berlin. The capital of Italy is Rome, and the capital of Spain is Madrid").input_ids
    W, X = C.rand_case(rng, True, 2); x1, x2 = X
    ids = torch.tensor([t1 + t2])
    with torch.no_grad():
        ref = base(input_ids=ids).logits[0, -len(t2):].float()
    ctl = HybridSession(model, v0); ctl.text(t1); control = ctl.text(t2)[:, : ref.shape[-1]]   # same chunking, no chip run
    s = HybridSession(model, v0)
    s.text(t1)
    with s.chip():
        s.load_weights(W); y1 = s.matvec(x1)
    got = s.text(t2)[:, : ref.shape[-1]]
    kl = lambda a, b: [float(torch.nn.functional.kl_div(a[i].log_softmax(-1), b[i].log_softmax(-1), log_target=True, reduction="sum")) for i in range(len(t2))]
    vs_base, vs_ctl = kl(got, ref), kl(got, control)
    print("per-position KL with restore vs base:", [round(p, 4) for p in vs_base[:5]], "max", round(max(vs_base), 4))
    print("per-position KL with restore vs no chip run at all:", [round(p, 6) for p in vs_ctl[:5]], "max", round(max(vs_ctl), 6))
    s2 = HybridSession(model, v0); s2.text(t1); s2.load_weights(W); y_fast = s2.matvec(x1)   # fast path: no restore needed
    assert max(kl(s2.text(t2)[:, : ref.shape[-1]], control)) == 0.0 and (y_fast == C.ref_matvec(x1, W)).all()
    assert max(vs_ctl) < 1e-4          # the restore itself: a chip run leaves Qwen's context exactly as it was
    assert max(vs_base) < 1e-2         # and that is within the widening noise floor of the unmodified model
    with s.chip():                                      # weights are still resident: no reload
        y2 = s.matvec(x2)
    ok = lambda x: np.clip(np.maximum(x @ W, 0), 0, C.AMAX)
    assert (y1 == ok(x1)).all() and (y2 == ok(x2)).all()


def test_hybrid_restore_full_network(merged, tok):
    """The restore only matters when Qwen's own layers run (fast=False): a chip run in the middle of a text must leave
    the text logits identical to a session that never ran the chip."""
    from qwentpu.driver import HybridSession
    model, _, v0, _ = merged
    t1 = tok(PROMPTS[2]).input_ids
    t2 = tok(" Berlin. The capital of Italy is Rome").input_ids
    W, X = C.rand_case(rng, True, 1)
    ctl = HybridSession(model, v0, fast=False); ctl.text(t1); control = ctl.text(t2)
    s = HybridSession(model, v0, fast=False); s.text(t1)
    with s.chip():
        s.load_weights(W); y = s.matvec(X[0])
    got = s.text(t2)
    assert (y == C.ref_matvec(X[0], W)).all()
    assert torch.equal(got, control), float((got - control).abs().max())
    with pytest.raises(RuntimeError):
        with HybridSession(model, v0, fast=False).chip(): pass
