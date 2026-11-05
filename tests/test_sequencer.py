"""v2 sequencer: the compiled network emits every chip token after the prompt.

CPU, no checkpoint: the circuit IR in float64 with all exactness conditions asserted (as test_circuit_ir.py).
With a checkpoint (QWEN_DIR): the real merged Qwen3.5, autoregressive, host loop = argmax + append only."""
import os

import numpy as np
import pytest

from qwentpu import circuit as C
from qwentpu.compiler import evaluate as ir
from qwentpu.sequencer import Sequencer, generate, ref_chain


@pytest.fixture(autouse=True)
def restore():
    yield
    Sequencer.uninstall()


def rand_net(rng, depth, width, scale=2):
    """Gentle random weights (outputs mostly off the 0 / AMAX rails), biases included."""
    Ws, n_in = [], width - 1
    for b in range(depth):
        Ws.append(rng.integers(-1, scale + 1, (n_in + 1, width - 1)))
        Ws[-1][-1] = rng.integers(0, 3, width - 1)
    x = rng.integers(0, min(C.AMAX, 3) + 1, width - 1)
    return Ws, x


def run_ir(seq, prompt, emitted):
    g, d = C.build()
    toks = list(prompt) + list(emitted)
    m = ir.evaluate(g, toks)
    nxt = lambda t: sum((1 << i) * int(ir.ev(d["code"][i], m[t:t + 1])[0] > 0) for i in range(C.CODE))
    return g, [nxt(t) for t in range(len(prompt) - 1, len(toks))]


@pytest.mark.parametrize("depth,width,ab,wb", [(2, 4, 3, 3), (4, 4, 3, 3), (2, 8, 3, 3), (2, 4, 4, 4), (2, 8, 5, 5)])
def test_ir_model_emits_the_whole_chain(depth, width, ab, wb):
    rng = np.random.default_rng(depth * 100 + width)
    seq = Sequencer(depth, width, ab, wb).install()
    for trial in range(2):
        Ws, x = rand_net(rng, depth, width)
        prompt = seq.prompt(Ws, x)
        emitted = seq.expected(prompt)
        assert len(emitted) == 2 * depth * width
        g, pred = run_ir(seq, prompt, emitted)                    # teacher-forced == autoregressive for a deterministic argmax
        assert pred == emitted + [seq.halt]
        want = ref_chain(Ws, x)
        got = seq.logits(emitted)
        assert list(got[: Ws[-1].shape[1]]) == list(want), (got, want)
        assert len(set(want)) > 1 or trial == 0


@pytest.mark.parametrize("depth,width,ab,wb,vec", [(2, 4, 3, 3, 2), (2, 4, 3, 3, 4), (4, 4, 3, 3, 2), (2, 8, 3, 3, 2), (2, 8, 3, 3, 4)])
def test_ir_model_emits_the_vector_chain(depth, width, ab, wb, vec):
    """vec neurons per token: the same chain in `vec`-row slots.  The circuit IR predicts every token of the chain and the
    decoded logits equal the plain numpy network; the chain is `vec` times shorter than the scalar one."""
    rng = np.random.default_rng(depth * 100 + width + vec)
    seq = Sequencer(depth, width, ab, wb, vec=vec).install()
    assert C.G == C.H == vec and C.ACT0 % C.NOUT ** vec == 0 and seq.outk0 % C.NOUT ** vec == 0
    for trial in range(2):
        Ws, x = rand_net(rng, depth, width)
        prompt = seq.prompt(Ws, x)
        emitted = seq.expected(prompt)
        assert len(emitted) == 2 * depth * width // vec
        g, pred = run_ir(seq, prompt, emitted)
        assert pred == emitted + [seq.halt]
        assert list(seq.logits(emitted)[: Ws[-1].shape[1]]) == list(ref_chain(Ws, x))


def test_vector_and_scalar_chains_compute_the_same_network():
    """Same weights and input: the scalar and the vec=2 golden emit different tokens but decode to identical logits."""
    rng = np.random.default_rng(5)
    Ws, x = rand_net(rng, 2, 8)
    logits = []
    for vec in (1, 2, 4):
        seq = Sequencer(2, 8, 3, 3, vec=vec).install()
        logits.append(list(seq.logits(seq.expected(seq.prompt(Ws, x)))[: Ws[-1].shape[1]]))
        Sequencer.uninstall()
    assert logits[0] == logits[1] == logits[2] == list(ref_chain(Ws, x))


def test_vec_prompt_is_weights_plus_slot_tokens():
    seq = Sequencer(2, 4, 3, 3, vec=2).install()
    Ws, x = rand_net(np.random.default_rng(0), 2, 4)
    p = seq.prompt(Ws, x)
    assert len(p) == C.KR * C.NC + C.KR // 2                                  # one ACT token per 2 rows
    assert all(C.ACT0 <= t < C.READ0 for t in p[C.KR * C.NC:])


def test_prompt_has_no_schedule():
    """The prompt is weights + input only: CELL tokens then width ACT tokens; no READ, no OUT, no op tokens."""
    seq = Sequencer(2, 4).install()
    rng = np.random.default_rng(0)
    Ws, x = rand_net(rng, 2, 4)
    p = seq.prompt(Ws, x)
    assert all(C.CELL0 <= t < C.ACT0 for t in p[: C.KR * C.NC]) and all(C.ACT0 <= t < C.READ0 for t in p[C.KR * C.NC:])
    assert len(p) == C.KR * C.NC + C.KR


def test_v1_circuit_unchanged_after_uninstall():
    Sequencer(2, 4, vec=2).install(); Sequencer.uninstall()
    g, d = C.build()
    assert g.n_layers() == 2 and C.SEQ is None and len(C.VOCAB) == C.READ0 + C.NREAD


# ---------------------------------------------------------------------------------------------- real model
SRC = os.environ.get("QWEN_DIR")


@pytest.mark.skipif(not SRC, reason="set QWEN_DIR to a Qwen3.5 checkpoint")
@pytest.mark.parametrize("depth,width", [(2, 4)])
def test_merged_qwen_host_only_argmax(depth, width):
    torch = pytest.importorskip("torch")
    from qwentpu.driver import TpuSession
    from qwentpu.merge import build_merged, instantiate
    dev = os.environ.get("QWEN_DEVICE", "cpu")
    seq = Sequencer(depth, width).install()
    tc, sd, v0, lay, g = build_merged(SRC, mute_qwen=False)
    model = instantiate(tc, sd, dev)
    rng = np.random.default_rng(7)
    for _ in range(3):
        Ws, x = rand_net(rng, depth, width)
        prompt = seq.prompt(Ws, x)
        sess = TpuSession(model, v0, fast=True, readout="lm_head")
        emitted = generate(sess.feed, prompt, seq.halt)             # the host: argmax; append
        assert emitted == seq.expected(prompt)
        assert list(seq.logits(emitted)) == list(ref_chain(Ws, x))


GOLD = os.path.join(os.path.dirname(__file__), "..", "examples", "mlp", "golden.npz")


@pytest.mark.skipif(not os.path.exists(GOLD), reason="needs examples/mlp/golden.npz (ISA agent)")
def test_ir_isa_golden_mlp_logits():
    """The ISA agent's quantised MLP (15 -> 15 -> 10, u5s5) run through the sequenced block-diagonal chain (2 x 16 on 32x32)
    gives the golden logits and predictions.  (Token ids differ from golden: no banks/OP tokens, see docs/V2_SEQUENCER.md.)"""
    z = np.load(GOLD)
    seq = Sequencer(2, 16, 5, 5).install()
    for i in (0, 1, 2):
        p = seq.prompt([z["W1"], z["W2"]], z["x_test"][i]); em = seq.expected(p)
        if i == 0:
            _, pred = run_ir(seq, p, em); assert pred == em + [seq.halt]
        lg = seq.logits(em)[:10]
        assert list(lg) == list(z["logits"][i][:10]) and lg.argmax() == z["pred"][i]
