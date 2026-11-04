"""Program container + MLP builder for the v2 ISA (docs/V2_ISA.md)."""
import numpy as np

from . import isa


def mlp_program(a, Ws, x):
    """Ws: list of integer weight tiles, each (n_in+1) x n_out with LAST ROW = bias (see bias convention),
    x: the n_in input activations (0..amax).  Builds LOAD_WEIGHTS per layer (bank i), LOAD_X, fused MATMULs, OUT, HALT.
    Tile layout in the array: inputs rows 0..n_in-1, bias at row kr-1; a constant-1 generator column kr-1 is
    added to every non-final layer so the next layer's bias row sees 1."""
    prog = []
    for i, W in enumerate(Ws):
        prog.append(isa.LOAD_WEIGHTS(i, tile(a, W, gen_one=(i < len(Ws) - 1))))
    xv = np.zeros(a.kr, dtype=np.int64); xv[: len(x)] = x; xv[a.kr - 1] = 1
    prog.append(isa.LOAD_X(xv))
    for i in range(len(Ws)):
        prog += isa.MATMUL_R(i)
    return prog + [isa.OUT(), isa.HALT()]


def tile(a, W, gen_one=False):
    W = np.asarray(W, dtype=np.int64); n_in, n_out = W.shape[0] - 1, W.shape[1]
    assert n_in <= a.kr - 1 and n_out <= a.nc - (1 if gen_one else 0)
    T = np.zeros((a.kr, a.nc), dtype=np.int64)
    T[:n_in, :n_out] = W[:-1]; T[a.kr - 1, :n_out] = W[-1]
    if gen_one: T[a.kr - 1, a.nc - 1] = 1
    return T


def run_mlp_np(a, Ws, x):
    """Independent numpy reference (does not use isa.run)."""
    v = np.zeros(a.kr, dtype=np.int64); v[: len(x)] = x; v[a.kr - 1] = 1
    for i, W in enumerate(Ws):
        v = np.clip(np.maximum(v @ tile(a, W, i < len(Ws) - 1), 0), 0, a.amax)
    return v
