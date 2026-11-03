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


def base_model():
    cfg, sd = load(SRC); tc = dict(cfg["text_config"]); tc["tie_word_embeddings"] = False
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    return instantiate(tc, sd)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SRC)


@pytest.fixture(scope="module")
def merged(tok):
    from qwentpu.mute import fit_anti_sink
    base = base_model()
    sinks = fit_anti_sink(base, tok, PROMPTS)
    tc, sd, v0, lay, g = build_merged(SRC, sinks=sinks)
    return instantiate(tc, sd), base, v0, (tc, sd, g)


def trace(W, X):
    toks = C.load_trace(W); want = [C.NOP] * len(toks)
    for x in X:
        toks += C.matvec_trace(x); want += [C.NOP] * C.KR
        want += [C.tok_out(int(np.clip(max(0, (x @ W)[c]), 0, C.AMAX))) for c in range(C.NC)]
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
    assert sum(1 <= w - C.OUT0 <= C.AMAX - 1 for w in want if w != C.NOP) >= 12     # not just saturated 0/7
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


def test_mutation_breaks_it(merged):
    _, _, v0, (tc, sd, g) = merged
    n0 = tc["num_hidden_layers"] - g.n_layers()
    W, X = C.rand_case(rng, True, 2); toks, want = trace(W, X)
    ok = run(instantiate(tc, sd), v0, toks)
    sd2 = dict(sd)
    for L in range(n0, n0 + g.n_layers()):
        k = f"model.layers.{L}.linear_attn.in_proj_qkv.weight"; sd2[k] = torch.zeros_like(sd[k])
    bad = run(instantiate(tc, sd2), v0, toks)
    assert all(ok[t] == w for t, w in enumerate(want)) and any(bad[t] != w for t, w in enumerate(want))


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
    with torch.no_grad():
        ref = base(input_ids=torch.tensor([t1 + t2])).logits[0, -len(t2):].float()
    s = HybridSession(model, v0)
    s.text(t1)
    with s.chip():
        s.load_weights(W); y1 = s.matvec(x1)
    got = s.text(t2)[:, : ref.shape[-1]]
    pos = [float(torch.nn.functional.kl_div(got[i].log_softmax(-1), ref[i].log_softmax(-1), log_target=True, reduction="sum")) for i in range(len(t2))]
    print("per-position KL with restore:", [round(p, 4) for p in pos[:5]], "max", round(max(pos), 4))
    assert max(pos) < 5e-3
    with s.chip():                                      # weights are still resident: no reload
        y2 = s.matvec(x2)
    ok = lambda x: np.clip(np.maximum(x @ W, 0), 0, C.AMAX)
    assert (y1 == ok(x1)).all() and (y2 == ok(x2)).all()
