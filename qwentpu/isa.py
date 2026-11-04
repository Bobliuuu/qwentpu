"""qwenTPU v2 ISA: assembler, token codec and bit-exact numpy reference interpreter.  See docs/V2_ISA.md.

Pure numpy; token ids are computed from an `Arr` config with the same formulas as qwentpu.circuit (a test checks
they agree with circuit.tok_* after circuit.configure)."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arr:
    kr: int = 16
    nc: int = 16
    ab: int = 5
    wb: int = 5
    g: int = 1
    h: int = 1
    nb: int = 2          # weight banks

    def __post_init__(self):
        assert self.g == self.h, "chaining needs g == h"
        assert self.kr == self.nc, "chaining needs a square array"
        assert self.kr % self.g == 0 and self.nc % self.h == 0 and 1 <= self.nb <= 4

    @property
    def amax(self): return (1 << self.ab) - 1
    @property
    def wlo(self): return -(1 << (self.wb - 1))
    @property
    def whi(self): return (1 << (self.wb - 1)) - 1
    @property
    def nout(self): return 1 << self.ab
    @property
    def nw(self): return 1 << self.wb
    @property
    def ks(self): return self.kr // self.g
    @property
    def nop(self): return self.nout ** self.h
    @property
    def cell0(self): return self.nop + 1
    @property
    def act0(self): return self.cell0 + self.kr * self.nc * self.nw
    @property
    def read0(self): return self.act0 + self.ks * self.nout ** self.g
    @property
    def nread(self): return self.nc // self.h
    @property
    def vbase(self): return self.read0 + self.nread       # == len(circuit.VOCAB)
    # chip-token constructors (same formulas as circuit.tok_*)
    def tok_cell(self, r, c, w): return self.cell0 + (r * self.nc + c) * self.nw + (int(w) & (self.nw - 1))
    def tok_act(self, s, *vs): return self.act0 + s * self.nout ** self.g + sum(int(v) * self.nout ** j for j, v in enumerate(vs))
    def tok_read(self, k): return self.read0 + k
    def tok_out(self, *vs): return sum(int(v) * self.nout ** t for t, v in enumerate(vs))
    def out_values(self, t): return [(t // self.nout ** i) % self.nout for i in range(self.h)]
    # ISA tokens
    def op_loadw(self, b): return self.vbase + b
    def op_matmul(self, b): return self.vbase + self.nb + b
    @property
    def op_loadx(self): return self.vbase + 2 * self.nb
    @property
    def op_out(self): return self.op_loadx + 1
    @property
    def op_halt(self): return self.op_loadx + 2
    @property
    def vocab_size(self): return self.op_halt + 1


# ---- instructions -------------------------------------------------------------------------------------------
@dataclass
class Ins:
    op: str                 # LOAD_WEIGHTS LOAD_X MATMUL RELU REQUANT OUT HALT
    bank: int = 0
    arg: object = None      # W for LOAD_WEIGHTS, x for LOAD_X, shift for REQUANT


def LOAD_WEIGHTS(b, W): return Ins("LOAD_WEIGHTS", b, np.asarray(W, dtype=np.int64))
def LOAD_X(x): return Ins("LOAD_X", 0, np.asarray(x, dtype=np.int64))
def MATMUL(b): return Ins("MATMUL", b)
def RELU(): return Ins("RELU")
def REQUANT(shift=0): return Ins("REQUANT", 0, int(shift))
def OUT(): return Ins("OUT")
def HALT(): return Ins("HALT")
def MATMUL_R(b): return [MATMUL(b), RELU(), REQUANT(0)]       # the fused v2.0 instruction


def _pad_w(a, W):
    Wp = np.zeros((a.kr, a.nc), dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    assert W.shape[0] <= a.kr and W.shape[1] <= a.nc
    Wp[: W.shape[0], : W.shape[1]] = W
    assert Wp.min() >= a.wlo and Wp.max() <= a.whi, "weight out of range"
    return Wp


def run(a, prog):
    """Reference interpreter.  Returns the final output vector (np int64, length nc) at OUT.
    `acc` = raw accumulator (exact int), `v` = current activation vector."""
    banks = [np.zeros((a.kr, a.nc), dtype=np.int64) for _ in range(a.nb)]
    v = np.zeros(a.kr, dtype=np.int64); acc = None; out = None
    for ins in prog:
        if ins.op == "LOAD_WEIGHTS":
            banks[ins.bank] = _pad_w(a, ins.arg)
        elif ins.op == "LOAD_X":
            x = np.zeros(a.kr, dtype=np.int64); x[: len(ins.arg)] = ins.arg
            assert x.min() >= 0 and x.max() <= a.amax, "activation out of range"
            v = x
        elif ins.op == "MATMUL":
            n = a.nc
            vin = np.zeros(a.kr, dtype=np.int64); vin[: len(v)] = v
            acc = vin @ banks[ins.bank]
        elif ins.op == "RELU":
            acc = np.maximum(acc, 0)
        elif ins.op == "REQUANT":
            v = np.clip(acc >> ins.arg, 0, a.amax)
        elif ins.op == "OUT":
            out = v.copy()
        elif ins.op == "HALT":
            break
        else:
            raise ValueError(ins.op)
    return out


# ---- token lowering -----------------------------------------------------------------------------------------
def _fuse(prog):
    """Collapse MATMUL;RELU;REQUANT 0 triples into ('MM', bank); reject anything the datapath cannot do."""
    out, i = [], 0
    while i < len(prog):
        ins = prog[i]
        if ins.op == "MATMUL":
            ok = (i + 2 < len(prog) and prog[i + 1].op == "RELU" and prog[i + 2].op == "REQUANT" and prog[i + 2].arg == 0)
            if not ok:
                raise ValueError("not supported by v2.0 datapath: MATMUL must be followed by RELU; REQUANT 0")
            out.append(("MM", ins.bank)); i += 3
        elif ins.op in ("RELU", "REQUANT"):
            raise ValueError("not supported by v2.0 datapath: stray " + ins.op)
        else:
            out.append((ins.op, ins)); i += 1
    return out


def encode(a, prog):
    """Program -> prompt token ids (the only host-fed tokens)."""
    toks, mms, seen_x, loaded = [], [], False, set()
    f = _fuse(prog)
    for kind, ins in f:
        if kind == "LOAD_WEIGHTS":
            assert not mms, "LOAD_WEIGHTS must precede MATMUL"
            W = _pad_w(a, ins.arg); toks.append(a.op_loadw(ins.bank)); loaded.add(ins.bank)
            toks += [a.tok_cell(r, c, W[r, c]) for r in range(a.kr) for c in range(a.nc)]
        elif kind == "LOAD_X":
            x = np.zeros(a.kr, dtype=np.int64); x[: len(ins.arg)] = ins.arg
            assert x.min() >= 0 and x.max() <= a.amax
            toks.append(a.op_loadx)
            toks += [a.tok_act(s, *x[s * a.g:(s + 1) * a.g]) for s in range(a.ks)]; seen_x = True
        elif kind == "MM":
            assert seen_x and ins in loaded; toks.append(a.op_matmul(ins)); mms.append(ins)
        elif kind == "OUT":
            toks.append(a.op_out)
        elif kind == "HALT":
            toks.append(a.op_halt)
    assert toks[-1] == a.op_halt and toks[-2] == a.op_out and mms, "program must end MATMUL..., OUT, HALT"
    return toks


def decode(a, toks):
    """Prompt token ids -> (banks dict, x, [matmul banks]).  Inverse of encode (for tests / tooling)."""
    i, banks, x, mms = 0, {}, None, []
    while i < len(toks):
        t = toks[i]
        if a.vbase <= t < a.vbase + a.nb:
            b = t - a.vbase; W = np.zeros((a.kr, a.nc), dtype=np.int64)
            for j in range(a.kr * a.nc):
                u = toks[i + 1 + j] - a.cell0; cell, w = divmod(u, a.nw)
                W[cell // a.nc, cell % a.nc] = w - a.nw if w >= a.nw // 2 else w
            banks[b] = W; i += 1 + a.kr * a.nc
        elif t == a.op_loadx:
            x = []
            for s in range(a.ks):
                u = toks[i + 1 + s] - a.act0; assert u // a.nout ** a.g == s
                x += [(u // a.nout ** j) % a.nout for j in range(a.g)]
            x = np.array(x, dtype=np.int64); i += 1 + a.ks
        elif a.vbase + a.nb <= t < a.vbase + 2 * a.nb:
            mms.append(t - a.vbase - a.nb); i += 1
        elif t in (a.op_out, a.op_halt):
            i += 1
        else:
            raise ValueError(f"bad prompt token {t} at {i}")
    return banks, x, mms


def program_from_prompt(a, toks):
    banks, x, mms = decode(a, toks)
    p = [LOAD_WEIGHTS(b, W) for b, W in sorted(banks.items())] + [LOAD_X(x)]
    for b in mms: p += MATMUL_R(b)
    return p + [OUT(), HALT()]


def emitted_trace(a, prompt):
    """The exact token ids the model must emit after `prompt` (see docs/V2_ISA.md section 3), computed with the
    reference interpreter on the decoded prompt: 2*L*n + 1 tokens."""
    banks, x, mms = decode(a, prompt)
    n = a.nread; v = x.copy(); out = []
    for l, b in enumerate(mms):
        y = np.clip(np.maximum(v @ banks[b], 0), 0, a.amax)
        for k in range(n):
            out.append(a.tok_read(k))
            ys = y[k * a.h:(k + 1) * a.h]
            out.append(a.tok_out(*ys) if l == len(mms) - 1 else a.tok_act(k, *ys))
        v = y
    return out + [a.op_halt]


def decode_result(a, emitted):
    """Final output vector from an emitted token stream (the OUT tokens of the last matmul)."""
    assert emitted[-1] == a.op_halt
    n = a.nread; outs = emitted[-1 - n:-1] if len(emitted) > 1 else []
    # last matmul = last n (READ, Y) pairs
    ys = emitted[-1 - 2 * n:-1][1::2]
    assert all(t < a.nop for t in ys), "final tokens are not OUT tokens"
    return np.array([v for t in ys for v in a.out_values(t)], dtype=np.int64)
