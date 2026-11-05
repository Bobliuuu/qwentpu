"""v2 sequencer: the model itself emits every chip token after the prompt.

The v1 host chose each ACT/READ token and relabelled each OUT token as the next layer's ACT.  Here the compiled
network does both.  Its next-token readout (the `code` dims, sign bits of the token id) is, in general, a
function of (current token, array state).  We use exactly that:

  * the token-only part is a successor table baked into the embedding (free token features: no extra layer);
  * the state-dependent part is the requantised output value u of a READ, which the existing code stage already
    reads out as bits; we add it to a per-READ base id:   next = base(READ k) + u.

So  READ(k) -> ACT(row; y_k)  (the next layer's input row, as a model-emitted token; no host relabel)
    ACT(row;.) -> READ(k')     (fixed successor)        OUTK(j; y) -> READ(k'') / HALT.
The schedule is a table (`Schedule`): it is the chip's *control ROM*, compiled into the weights, not host logic.

Multi-layer chains need the second layer's weights while the first is running, and a host can no longer inject a
LOAD between layers.  Instead of weight banks we pack the layers block-diagonally into the one array:
layer b owns rows [b n, (b+1) n) and columns [b n, (b+1) n); layer b's output column j feeds layer b+1's row j.
Cross-block weights are 0, so reading column k of block b sums only block b's rows (other rows contribute 0
whatever they hold).  All weights are loaded once, in the prompt.  Array = (depth * width)^2.

Host after the prompt: `generate()` below = `argmax; append` until the model emits HALT (the end-of-sequence
token) -- nothing matrix-specific.  See docs/V2_SEQUENCER.md.
"""
import numpy as np

from . import circuit as C


class Schedule:
    """Control table.  routes[k]: where READ(k)'s value goes: ("act", row) = ACT token of that row, ("out", j) =
    OUTK(j) result token.  act_next[row] = READ column that follows ACT(row;.); out_next[j] = READ column or None
    (None = HALT).  Tokens not listed emit NOP (as in v1)."""

    def __init__(self, routes, act_next, out_next, width=None):
        self.routes, self.act_next, self.out_next, self.width = routes, act_next, out_next, width

    @classmethod
    def chain(cls, depth, width):
        """y = f_{d-1}(... f_0(x)): layer b = block b.  The prompt supplies x in rows 0..width-1; the model then emits
        READ(0) ACT(w;y0) READ(1) ACT(w+1;y1) ... and, for the last layer, OUTK(0;y) READ OUTK(1;y) ... HALT."""
        n, L = width, depth
        routes = [("act", k + n) if k < (L - 1) * n else ("out", k - (L - 1) * n) for k in range(L * n)]
        act_next = {s: s + 1 - n for s in range(n - 1, L * n)}          # ACT(n-1) is the last prompt token
        out_next = {j: ((L - 1) * n + j + 1 if j + 1 < n else None) for j in range(n)}
        return cls(routes, act_next, out_next, n)


class Sequencer:
    """Compile-time object: extends the circuit's vocabulary with OUTK/HALT tokens and supplies the next-token bits."""

    def __init__(self, depth=2, width=4, ab=3, wb=3, vec=1):
        """vec = neurons per token (g = h = vec of the circuit): an ACT token carries `vec` activations and a READ/OUTK
        token `vec` outputs, so the chain emits 2*depth*width/vec tokens.  vec=1 is the scalar chain (docs/V2_VECTORIZED.md)."""
        assert width % vec == 0, "width must be a multiple of vec"
        self.depth, self.width, self.ab, self.wb, self.vec = depth, width, ab, wb, vec
        self.sched = Schedule.chain(depth, width // vec)          # the chain in token slots: each slot = vec rows / columns

    # -- install ----------------------------------------------------------------------------------------------
    def install(self):
        """Configure the circuit for the packed array and add the sequencer's tokens.  Call before C.build() /
        merge.build_merged(); call `uninstall()` to get the plain v1 circuit back."""
        d, n, v = self.depth, self.width, self.vec
        C.configure(d * n, d * n, self.ab, self.wb, v, v, align_act=v > 1)
        self.low = C.ACT0 % C.NOUT ** v                           # alignment shared by ACT and OUTK ids (0 when vec > 1)
        base = C.READ0 + C.NREAD
        self.outk0 = base + (self.low - base) % C.NOUT ** v
        self.halt = self.outk0 + (n // v) * C.NOUT ** v
        C.VOCAB = list(range(self.halt + 1)); C.CODE = max(8, (len(C.VOCAB) - 1).bit_length())
        C.SEQ = self
        return self

    @staticmethod
    def uninstall():
        C.SEQ = None; C.configure(8, 8, 3, 3)

    # -- tokens -----------------------------------------------------------------------------------------------
    def tok_outk(self, j, *vs): return self.outk0 + j * C.NOUT ** C.H + sum(int(v) * C.NOUT ** t for t, v in enumerate(vs))

    def base(self, k):
        kind, i = self.sched.routes[k]
        return C.tok_act(i, *[0] * C.G) if kind == "act" else self.tok_outk(i, *[0] * C.H)

    def read_index(self, t): return t - C.READ0 if C.READ0 <= t < C.READ0 + C.NREAD else None

    def succ(self, t):
        """Fixed successor of a non-READ token (NOP if none)."""
        S = self.sched
        if C.ACT0 <= t < C.READ0:
            k = S.act_next.get((t - C.ACT0) // C.NOUT ** C.G)
            return C.NOP if k is None else C.tok_read(k)
        if self.outk0 <= t < self.halt:
            k = S.out_next[(t - self.outk0) // C.NOUT ** C.H]
            return self.halt if k is None else C.tok_read(k)
        return self.halt if t == self.halt else C.NOP

    # -- circuit hook -----------------------------------------------------------------------------------------
    def nxt_terms(self, g, i, ctx):
        """Unit terms (coef, a, b) whose sum is bit i of the next token id (0/1), added by circuit.build().
        next = base(READ k) + u, with base = NOUT*Hk + low (low = ACT0 mod NOUT, same for every family):
          bit i < AB : (u + low) mod NOUT, bit i       -> per-value indicator terms (gated by is_read)
          bit i >= AB: bit (i-AB) of Hk + [u >= NOUT-low]   -> token feature  +  carry feature * [u >= NOUT-low]
        For tokens that are not READ the table `succ` supplies the bits."""
        if self.vec > 1: return self._nxt_terms_vec(g, i, ctx)
        one, f, is_read, (u, lo, hi) = ctx["one"], ctx["f"], ctx["is_read"], ctx["slots"][0]
        AB, N, low = C.AB, C.NOUT, self.low

        def s_fn(t):
            k = self.read_index(t)
            if k is None: return (self.succ(t) >> i) & 1
            return ((self.base(k) // N * N) >> i) & 1

        terms = [(1, one, f(f"seq_s{i}", s_fn))]
        if i < AB:
            for v in range(C.AMAX + 1):
                if not (((v + low) % N) >> i) & 1: continue
                # u is the unclamped Z where the adder tree ends: value 0 means u <= 0, value AMAX means u >= AMAX
                if v == 0: terms += g.terms_lt(u, 1, lo, hi, 1, (is_read,))
                elif v == C.AMAX: terms += g.terms_ge(u, C.AMAX, lo, hi, 1, (is_read,))
                else: terms += g.terms_eq(u, v, lo, hi, (is_read,))
        elif low:
            def d_fn(t):
                k = self.read_index(t)
                if k is None: return 0
                Hk = self.base(k) // N
                return (((Hk + 1) * N >> i) & 1) - ((Hk * N >> i) & 1)
            d = f(f"seq_d{i}", d_fn)
            terms += [(c, d, b) for c, _, b in g.terms_ge(u, N - low, lo, hi, 1, (is_read,))]
        return terms

    def _nxt_terms_vec(self, g, i, ctx):
        """vec > 1: ACT0 and OUTK are multiples of NOUT**vec, so next = base(READ k) + sum_t clamp(u_t) NOUT**t has no carry.
        Bits >= vec*AB are the token feature (base, or the fixed successor of a non-READ token); bit t*AB + a below that is
        bit a of slot t's requantised value, read off the adder tree exactly as in the v1 code stage."""
        one, f, is_read = ctx["one"], ctx["f"], ctx["is_read"]

        def s_fn(t):
            k = self.read_index(t)
            return ((self.succ(t) if k is None else self.base(k)) >> i) & 1

        terms = [(1, one, f(f"seq_s{i}", s_fn))]
        if i < self.vec * C.AB:
            t, a = divmod(i, C.AB); u, lo, hi = ctx["slots"][t]
            if (C.AMAX >> a) & 1: terms += g.terms_ge(u, C.AMAX, lo, hi, 1, (is_read,))
            for v in range(1, C.AMAX):
                if (v >> a) & 1: terms += g.terms_eq(u, v, lo, hi, (is_read,))
        return terms

    # -- prompt / reference -----------------------------------------------------------------------------------
    def prompt(self, Ws, x):
        """Chip-token prompt for y = f(...f(x)): all weights (block-diagonal CELLs) then the zero-filled rows of layers 1.. and the input rows of layer 0.
        Ws[b]: (n_in_b + 1) x n_out_b, last row = bias (as TpuSession.mlp); n_in, n_out <= width - 1 (the last row/column
        of every block carries the constant 1).  Entries of each block are 0..AMAX-quantised by the caller."""
        n, L, KR = self.width, self.depth, C.KR
        assert len(Ws) == L
        Wp = np.zeros((KR, KR), dtype=int)
        for b, W in enumerate(Ws):
            W = np.asarray(W); n_in = W.shape[0] - 1
            assert n_in <= n - 1 and W.shape[1] <= n - (b < L - 1), (b, W.shape)
            Wp[b * n: b * n + n_in, b * n: b * n + W.shape[1]] = W[:-1]
            Wp[b * n + n - 1, b * n: b * n + W.shape[1]] = W[-1]                 # bias row
            if b < L - 1: Wp[b * n + n - 1, b * n + n - 1] = 1                    # column n-1 emits the 1 for the next layer
        x = list(np.asarray(x, dtype=int)) + [0] * (n - 1 - len(x)) + [1]
        assert len(x) == n and 0 <= min(x) and max(x) <= C.AMAX
        # every accumulator row must hold a write before the first READ (the head normalises by the number of written
        # rows), so the other layers' rows are written with 0 first; the input rows come last (ACT(n-1) -> READ(0)).
        gg = C.G                                                   # activations per ACT token (1 unless vec > 1)
        return (C.load_trace(Wp) + [C.tok_act(s, *[0] * gg) for s in range(n // gg, KR // gg)]
                + [C.tok_act(s, *x[s * gg:(s + 1) * gg]) for s in range(n // gg)])

    def expected(self, prompt):
        """Numpy emulation of what the model must emit (golden).  Independent of the circuit: it parses the prompt
        into weights/activations and applies the schedule with exact integer arithmetic."""
        Wp = np.zeros((C.KR, C.NC), dtype=int); a = np.zeros(C.KR, dtype=int)
        for t in prompt:
            if C.CELL0 <= t < C.ACT0:
                c, w = divmod(t - C.CELL0, C.NW); r, c = divmod(c, C.NC)
                Wp[r, c] = w - C.NW if w >= C.NW // 2 else w
            elif C.ACT0 <= t < C.READ0:
                self._write(a, t)
        out, cur, H = [], prompt[-1], C.H
        while len(out) < 8 * C.KR:
            k = self.read_index(cur)
            if k is None: nxt = self.succ(cur)
            else:
                y = np.clip(np.maximum(a @ Wp[:, k * H:(k + 1) * H], 0), 0, C.AMAX)
                nxt = self.base(k) + sum(int(v) * C.NOUT ** j for j, v in enumerate(y))
            if nxt == self.halt: break
            out.append(nxt); cur = nxt
            if C.ACT0 <= nxt < C.READ0: self._write(a, nxt)
        return out

    @staticmethod
    def _write(a, t):
        """ACT token t writes its G activations into rows s*G .. s*G+G-1."""
        s, vs = divmod(t - C.ACT0, C.NOUT ** C.G)
        for j in range(C.G): a[s * C.G + j] = (vs // C.NOUT ** j) % C.NOUT

    def logits(self, emitted):
        """The network output: the values carried by the OUTK tokens, in order."""
        return np.array([(((t - self.outk0) % C.NOUT ** C.H) // C.NOUT ** j) % C.NOUT
                         for t in emitted if self.outk0 <= t < self.halt for j in range(C.H)])


def ref_chain(Ws, x):
    """Plain numpy network (no packing, no tokens): relu + clamp after every layer, bias = trailing row."""
    v = np.asarray(x, dtype=int)
    for W in Ws:
        v = np.clip(np.maximum(np.concatenate([v, [1]]) @ np.asarray(W), 0), 0, C.AMAX)
    return v


def generate(feed, prompt, halt, max_new=4096):
    """THE host loop.  feed(chip_tokens) -> the model's argmax next token at each fed position (TpuSession.feed).
    Feed the prompt, then append the model's own token and repeat until it emits `halt`.  Knows nothing about
    matrices, layers, rows or columns; `halt` is the end-of-sequence id, like an LM's EOS."""
    nxt = feed(list(prompt))[-1]
    out = []
    while nxt != halt and len(out) < max_new:
        out.append(nxt)
        nxt = feed([nxt])[-1]
    return out
