"""The compiled circuit in float64 with every bf16 / silu / state-row exactness condition asserted."""
import numpy as np
import pytest

from qwentpu import circuit as C
from qwentpu.compiler import evaluate as ir

CFGS = [(8, 8, 3, 3), (8, 8, 4, 4), (8, 8, 5, 5), (16, 16, 4, 4), (16, 16, 3, 2), (32, 8, 3, 3), (8, 32, 3, 3)]


def run_ir(g, W, X):
    toks, marks = C.load_trace(W), []
    for x in X:
        t = C.matvec_trace(x)
        marks += [(len(toks) + C.KR + c, c, x) for c in range(C.NC)]
        toks += t
    return toks, ir.evaluate(g, toks), marks


def decode(d, m, t):
    return sum((1 << i) * int(ir.ev(d["code"][i], m[t:t + 1])[0] > 0) for i in range(C.CODE))


@pytest.mark.parametrize("cfg", CFGS)
def test_ir_matvec_exact(cfg):
    C.configure(*cfg)
    try:
        g, d = C.build()
        print(cfg, g.stats(), C.horner_plan())
        rng = np.random.default_rng(1)
        for gentle in (False, True):
            W, X = C.rand_case(rng, gentle, 5)
            if not gentle:      # worst cases for the adder tree: all-max and all-min rows
                X[0] = C.AMAX; X[1] = C.AMAX; X[2] = 0
                W[:, 0] = C.whi(); W[:, 1] = C.wlo()
            toks, m, marks = run_ir(g, W, X)
            seen = set()
            for t, c, x in marks:
                want = int(np.clip(max(0, (x @ W)[c]), 0, C.AMAX)); seen.add(want)
                assert decode(d, m, t) == C.tok_out(want), (cfg, t, c, want)
            assert len(seen) >= 2
    finally:
        C.configure(8, 8, 3, 3)


def test_default_is_two_layers():
    C.configure(8, 8, 3, 3)
    g, _ = C.build()
    assert g.n_layers() == 2, g.stats()
