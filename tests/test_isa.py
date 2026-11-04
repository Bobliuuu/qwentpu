import numpy as np
import pytest

from qwentpu import isa, program

CFGS = [isa.Arr(8, 8, 3, 3, 1, 1, 2), isa.Arr(16, 16, 5, 5, 1, 1, 2), isa.Arr(8, 8, 3, 3, 2, 2, 3), isa.Arr(16, 16, 4, 4, 2, 2, 4)]


def rand_prog(a, rng, L):
    prog = []
    Ws = [rng.integers(a.wlo, a.whi + 1, (a.kr, a.nc)) for _ in range(L)]
    for b, W in enumerate(Ws): prog.append(isa.LOAD_WEIGHTS(b % a.nb, W))
    if L > a.nb: Ws = Ws[-a.nb:] if False else Ws
    x = rng.integers(0, a.amax + 1, a.kr)
    prog.append(isa.LOAD_X(x))
    banks = {}
    for b, W in enumerate(Ws): banks[b % a.nb] = W
    order = [int(rng.integers(0, min(L, a.nb))) for _ in range(L)]
    for b in order: prog += isa.MATMUL_R(b)
    return prog + [isa.OUT(), isa.HALT()], banks, x, order


@pytest.mark.parametrize("a", CFGS)
def test_interpreter_matches_numpy(a):
    rng = np.random.default_rng(0)
    for _ in range(40):
        L = int(rng.integers(1, a.nb + 1))
        prog, banks, x, order = rand_prog(a, rng, L)
        v = x.copy()
        for b in order:
            v = np.clip(np.maximum(v @ banks[b], 0), 0, a.amax)         # independent reference
        assert (isa.run(a, prog) == v).all()


@pytest.mark.parametrize("a", CFGS)
def test_encode_decode_trace(a):
    rng = np.random.default_rng(1)
    for _ in range(10):
        prog, banks, x, order = rand_prog(a, rng, int(rng.integers(1, a.nb + 1)))
        p = isa.encode(a, prog)
        assert p[-1] == a.op_halt and p[-2] == a.op_out and max(p) < a.vocab_size
        assert (isa.run(a, isa.program_from_prompt(a, p)) == isa.run(a, prog)).all()
        e = isa.emitted_trace(a, p)
        assert len(e) == 2 * len(order) * a.nread + 1
        assert (isa.decode_result(a, e) == isa.run(a, prog)).all()


def test_unsupported_forms_rejected():
    a = isa.Arr()
    base = [isa.LOAD_WEIGHTS(0, np.zeros((16, 16), int)), isa.LOAD_X(np.zeros(16, int))]
    for bad in ([isa.MATMUL(0)], [isa.MATMUL(0), isa.REQUANT(0)], [isa.MATMUL(0), isa.RELU(), isa.REQUANT(2)]):
        with pytest.raises(ValueError):
            isa.encode(a, base + bad + [isa.OUT(), isa.HALT()])
    # but the interpreter still defines them
    isa.run(a, base + [isa.MATMUL(0), isa.RELU(), isa.REQUANT(2), isa.OUT(), isa.HALT()])


def test_requant_shift_semantics():
    a = isa.Arr(8, 8, 5, 5, 1, 1, 1)
    W = np.zeros((8, 8), int); W[0, 0] = 15; W[1, 1] = -3
    p = [isa.LOAD_WEIGHTS(0, W), isa.LOAD_X([31, 5, 0, 0, 0, 0, 0, 1]), isa.MATMUL(0), isa.RELU(), isa.REQUANT(3), isa.OUT()]
    assert list(isa.run(a, p)[:2]) == [min(15 * 31 >> 3, 31), 0]


def test_token_ids_match_circuit():
    from qwentpu import circuit as C
    for a in CFGS:
        C.configure(a.kr, a.nc, a.ab, a.wb, a.g, a.h)
        assert a.vbase == len(C.VOCAB) and a.act0 == C.ACT0 and a.read0 == C.READ0
        assert a.tok_cell(3, 2, -1) == C.tok_cell(3, 2, -1) and a.tok_read(1) == C.tok_read(1)
        vs = [1, 2][: a.g]
        assert a.tok_act(1, *vs) == C.tok_act(1, *vs) and a.tok_out(*vs) == C.tok_out(*vs)


def test_mlp_builder():
    a = isa.Arr(); rng = np.random.default_rng(2)
    Ws = [rng.integers(-3, 4, (11, 12)), rng.integers(-3, 4, (13, 10))]
    for _ in range(20):
        x = rng.integers(0, 6, 10)
        assert (isa.run(a, program.mlp_program(a, Ws, x)) == program.run_mlp_np(a, Ws, x)).all()


def test_golden_file():
    import os
    f = os.path.join(os.path.dirname(__file__), "..", "examples", "mlp", "golden.npz")
    if not os.path.exists(f): pytest.skip("no golden.npz")
    g = np.load(f); a = isa.Arr(*[int(v) for v in g["cfg"]])
    for p, e, lg in zip(g["prompt_ids"], g["expected_emitted_ids"], g["logits"]):
        assert list(isa.emitted_trace(a, list(p))) == list(e)
        assert (isa.run(a, isa.program_from_prompt(a, list(p))) == lg).all()
