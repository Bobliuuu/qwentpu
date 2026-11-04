"""v2 program-in-prompt: one compiled model, many programs (depth 1..Dmax, widths) chosen only by the prompt."""
import os

import numpy as np
import pytest

from qwentpu import circuit as C
from qwentpu.compiler import evaluate as ir
from qwentpu.compiler import nonlin as N
from qwentpu.sequencer import Schedule, ref_chain
from qwentpu.sequencer2 import ProgramMachine, run_program
from test_sequencer import run_ir


@pytest.fixture(autouse=True)
def restore():
    yield
    ProgramMachine.uninstall()


def rand_prog(rng, depth, widths, n):
    """depth layers, widths = hidden/out widths (each <= n-1); input width widths[0]."""
    Ws, n_in = [], widths[0]
    for w in widths[1:depth + 1]:
        W = rng.integers(-1, 3, (n_in + 1, w)); W[-1] = rng.integers(0, 3, w)
        Ws.append(W); n_in = w
    return Ws, rng.integers(0, min(C.AMAX, 3) + 1, widths[0])


@pytest.mark.parametrize("dmax,n,ab,wb", [(4, 4, 3, 3), (2, 8, 3, 3)])
def test_one_compiled_model_many_programs(dmax, n, ab, wb):
    seq = ProgramMachine(dmax, n, ab, wb).install()
    g, d = C.build(); layers = g.n_layers()
    rng = np.random.default_rng(dmax * 10 + n)
    lens, shapes = set(), set()
    for depth in range(1, dmax + 1):
        for trial in range(2):
            widths = [int(rng.integers(1, n)) for _ in range(depth + 1)]
            widths[-1] = int(rng.integers(1, n + 1)) if depth == 1 else widths[-1]
            Ws, x = rand_prog(rng, depth, widths, n)
            p = seq.program_prompt(Ws, x)
            em = seq.expected(p)
            assert len(em) == 2 * depth * n                                  # cost scales with the program, not with Dmax
            _, pred = run_ir(seq, p, em)                                      # the SAME compiled graph `g` (layers) every time
            assert pred == em + [seq.halt]
            assert seq.program_logits(em, Ws[-1].shape[1]) == list(ref_chain(Ws, x))
            lens.add(len(em)); shapes.add(tuple(w.shape for w in Ws))
    assert len(lens) == dmax and len(shapes) >= dmax


def test_program_prompt_is_data_only():
    seq = ProgramMachine(4, 4).install()
    rng = np.random.default_rng(1)
    Ws, x = rand_prog(rng, 2, [3, 3, 2], 4)
    p = seq.program_prompt(Ws, x)
    assert all(C.CELL0 <= t < C.READ0 for t in p)                            # no READ / OUT / HALT / op token
    assert p[-1] == C.tok_act(2 * 4 + 3, 1)                                   # entry block 2, bias row = 1


GOLD = os.path.join(os.path.dirname(__file__), "..", "examples", "mlp", "golden.npz")


@pytest.mark.skipif(not os.path.exists(GOLD), reason="needs golden.npz")
def test_ir_golden_mlp_as_prompt_program_and_depth1_program():
    z = np.load(GOLD)
    seq = ProgramMachine(2, 16, 5, 5).install()
    for i in (0, 1):
        p = seq.program_prompt([z["W1"], z["W2"]], z["x_test"][i]); em = seq.expected(p)
        if i == 0: _, pred = run_ir(seq, p, em); assert pred == em + [seq.halt]
        lg = seq.program_logits(em, 10)
        assert lg == list(z["logits"][i][:10]) and int(np.argmax(lg)) == z["pred"][i]
    p = seq.program_prompt([z["W1"]], z["x_test"][0]); em = seq.expected(p)     # a different program: the hidden layer alone
    _, pred = run_ir(seq, p, em); assert pred == em + [seq.halt] and len(em) == 32
    assert seq.program_logits(em, 15) == list(ref_chain([z["W1"]], z["x_test"][0]))


def test_one_feedback_path_sequencer_equals_nonlin_feedback_code():
    """Reconcile: `Sequencer` with the one-layer feedback schedule (READ(k) -> ACT(k; y), nothing else) emits the same
    next token as nonlin.feedback_code at every position, with the same number of chip layers."""
    n = 8
    fb = lambda: Schedule([("act", k) for k in range(n)], {}, {j: None for j in range(n)}, n)
    n_layers = []
    rng = np.random.default_rng(0); W, X = C.rand_case(rng, False, 4); X[0] = C.AMAX
    toks = C.load_trace(W) + sum((C.matvec_trace(x) for x in X), [])
    seq = ProgramMachine(1, n, 3, 3).install(); seq.sched = fb()
    g1, d1 = C.build(); m1 = ir.evaluate(g1, toks)
    ProgramMachine.uninstall(); C.configure(n, n, 3, 3)
    g2, d2 = C.build(); code2 = N.feedback_code(g2, d2, C); m2 = ir.evaluate(g2, toks)
    assert g1.n_layers() == g2.n_layers()
    for t in range(len(toks)):
        if t < n * n: continue
        a = sum((1 << i) * int(ir.ev(d1["code"][i], m1[t:t + 1])[0] > 0) for i in range(len(d1["code"])))
        b = sum((1 << i) * int(ir.ev(c, m2[t:t + 1])[0] > 0) for i, c in enumerate(code2))
        assert a == b, (t, a, b)
