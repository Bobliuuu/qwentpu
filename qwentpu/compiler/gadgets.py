"""Exact integer gadgets as unit-term lists (coef, a, b) meaning coef * a * relu(b).

Every b that feeds a unit stays an integer with |b| <= 256 when positive, and either 0 or <= -1
when not, so that silu(128*b) is exactly relu(128*b) (see evaluate.silu_ok).
"""
from .ir import Expr, Graph


class Circ(Graph):
    def terms_const(self, c): return [(c, self.one, self.one)]

    def terms_and(self, lits, coef=1, a=None):
        """coef * a * AND(lits)   (0/1 literals):  gate 4*(sum - n + 1/2) is +2 or <= -2."""
        a = self.one if a is None else a
        return [(coef / 2, a, 4 * sum(lits, Expr()) - (4 * len(lits) - 2))]

    def terms_ge(self, z, theta, lo, hi, coef=1, lits=()):
        """coef * [z >= theta] * AND(lits) for an integer z in [lo, hi]."""
        n = len(lits)
        if theta > hi: return []
        if theta <= lo and not n: return self.terms_const(coef)
        if hi - theta + 1 <= 256:
            m = 1 << (hi - lo).bit_length()
            zz = z + m * (sum(lits, Expr()) - n) if n else z
            return [(coef, self.one, zz - theta + 1), (-coef, self.one, zz - theta)]
        assert theta - lo <= 256, (theta, lo, hi)
        base = self.terms_and(lits, coef) if n else self.terms_const(coef)
        return base + self.terms_lt(z, theta, lo, hi, -coef, lits)

    def terms_lt(self, z, theta, lo, hi, coef=1, lits=()):
        n = len(lits)
        if theta <= lo: return []
        if theta > hi and not n: return self.terms_const(coef)
        if theta - lo <= 256:
            m = 1 << (hi - lo).bit_length()
            zz = z - m * (sum(lits, Expr()) - n) if n else z
            return [(coef, self.one, theta - zz), (-coef, self.one, theta - zz - 1)]
        assert hi - theta + 1 <= 256, (theta, lo, hi)
        base = self.terms_and(lits, coef) if n else self.terms_const(coef)
        return base + self.terms_ge(z, theta, lo, hi, -coef, lits)

    def terms_eq(self, z, v, lo, hi, lits=()):
        a, b = max(v, lo), min(v + 1, hi + 1)
        if a >= b: return []
        return self.terms_ge(z, a, lo, hi, 1, lits) + self.terms_ge(z, b, lo, hi, -1, lits)

    def node(self, terms, name): return self.ffn(terms, name) if terms else Expr()
