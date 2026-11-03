"""Circuit IR.  Every value is one scalar dim of the residual stream; linear forms (Expr) are free.

Nodes (the two sublayers of a Qwen3.5 linear-attention decoder layer):
  FFN   sum of unit terms  coef * a * relu(b)      -> SwiGLU units, silu(K b) * (a / K)
  Head  one Gated DeltaNet key head (+ its value heads): query/key/value channels with conv taps,
        decay selector `a`, write gate `beta`, per-column output gate `z`, output taps `outs`.
Sublayer levels: heads sit on odd levels (2L+1), FFNs on even levels (2L+2); layer L = both.
"""
from dataclasses import dataclass, field
from fractions import Fraction

HD = 128                    # key/value head dim of every Qwen3.5 size
K = 128                     # silu gate scale
A_REF = 2 ** 24             # ballast value stored in the two reference columns of a state row
A_BIT = 2 ** 7              # value of a set data bit in a state row
REF_COLS = (0, 1)
DECAY = {0: 1.0, 238: 0.0}  # decay selector a -> exp(g)   (dt_bias = -110, A_log = 0)
ONE = 0


class Expr:
    """Linear form  sum coef[d] * dim[d];  dim 0 (ONE) is the constant 1 on chip tokens."""
    __slots__ = ("t",)

    def __init__(self, t=None):
        self.t = {d: c for d, c in (t or {}).items() if c != 0}

    def __add__(self, o):
        o = as_expr(o); r = dict(self.t)
        for d, c in o.t.items():
            r[d] = r.get(d, 0) + c
        return Expr(r)
    __radd__ = __add__

    def __neg__(self): return Expr({d: -c for d, c in self.t.items()})
    def __sub__(self, o): return self + (-as_expr(o))
    def __rsub__(self, o): return as_expr(o) + (-self)

    def __mul__(self, k):
        assert isinstance(k, (int, float, Fraction)), "Expr * Expr is not linear"
        return Expr({d: c * k for d, c in self.t.items()})
    __rmul__ = __mul__


def as_expr(x):
    if isinstance(x, Expr):
        return x
    return Expr({ONE: x}) if x else Expr()


@dataclass
class Chan:
    expr: Expr
    taps: tuple = (1.0, 0.0, 0.0, 0.0)   # weights on x_t, x_{t-1}, x_{t-2}, x_{t-3}  (conv1d, kernel 4)


@dataclass
class VHead:
    v: dict; a: Expr; z: dict; out: list; beta: Expr | None = None


@dataclass
class Head:
    name: str; q: dict; k: dict; level: int
    vx: list = field(default_factory=list)    # one VHead per value head of this key head


@dataclass
class FFN:
    name: str; terms: list; out: int; level: int; into: bool = False


@dataclass
class Target:
    """Per-layer capacity of the host model."""
    key_heads: int = 16
    value_heads: int = 16
    intermediate: int = 3584


class Graph:
    def __init__(self, target: Target | None = None):
        self.target = target or Target()
        self.names = ["one"]; self.kind = ["one"]; self.level_of = [0]
        self.token_fns = {}
        self.nodes = []
        self.used = {}              # level -> units / key heads used
        self.one = Expr({ONE: 1})

    # -- dims ---------------------------------------------------------------------------------
    def _dim(self, name, kind, level):
        self.names.append(name); self.kind.append(kind); self.level_of.append(level)
        return len(self.names) - 1

    def token_feature(self, name, fn):
        d = self._dim(name, "token", 0)
        self.token_fns[d] = fn
        return Expr({d: 1})

    def need(self, *exprs):
        return max((self.level_of[d] for e in exprs for d in e.t if d != ONE), default=0)

    def _place(self, need, odd, amount, cap):
        lvl = need + 1
        while lvl % 2 != (1 if odd else 0) or self.used.get(lvl, 0) + amount > cap:
            lvl += 1
        self.used[lvl] = self.used.get(lvl, 0) + amount
        return lvl

    # -- nodes --------------------------------------------------------------------------------
    def ffn(self, terms, name="ffn", into=None):
        terms = [(c, as_expr(a), as_expr(b)) for c, a, b in terms if c != 0]
        lvl = self.need(*(e for _, a, b in terms for e in (a, b)))
        if into is not None:
            (d,) = into.t
            lvl = max(lvl, self.level_of[d])
        sub = self._place(lvl, False, len(terms), self.target.intermediate)
        if into is not None:
            self.level_of[d] = max(self.level_of[d], sub)
        else:
            d = self._dim(name, "ffn", sub)
        self.nodes.append(FFN(name, terms, d, sub, into is not None))
        return Expr({d: 1})

    def head(self, name, q, k, v, a, z, outs, beta=None):
        """One key head with one value head.  q/k: {key channel: Expr|Chan}; v: {column: Expr|Chan}
        (columns 0,1 are the reference ballast); z: {column: Expr}; outs: [(column, weight)]
        -> one new dim per out."""
        ch = lambda d: {i: c if isinstance(c, Chan) else Chan(as_expr(c)) for i, c in d.items()}
        q, k, v = ch(q), ch(k), ch(v)
        a = as_expr(a); z = {i: as_expr(e) for i, e in z.items()}
        beta = None if beta is None else as_expr(beta)
        exprs = [c.expr for d in (q, k, v) for c in d.values()] + [a, *z.values()] + ([beta] if beta else [])
        sub = self._place(self.need(*exprs), True, 1, self.target.key_heads)
        dims = [self._dim(f"{name}[{col}]", "head", sub) for col, _ in outs]
        self.nodes.append(Head(name, q, k, sub, [VHead(v, a, z, [(c, d, w) for (c, w), d in zip(outs, dims)], beta)]))
        return [Expr({d: 1}) for d in dims]

    def n_layers(self):
        return (max((n.level for n in self.nodes), default=0) + 1) // 2

    def stats(self):
        return {"layers": self.n_layers(), "dims": len(self.names),
                "heads": sum(isinstance(n, Head) for n in self.nodes),
                "ffn_units": sum(len(n.terms) for n in self.nodes if isinstance(n, FFN))}

    def feature_value(self, d, t):
        return self.token_fns[d](t)
