"""v2 program-in-prompt: ONE compiled model runs different networks (depth 1..Dmax, any width <= n-1 per layer),
selected only by the prompt.

Mechanism (no new circuit, no extra layer; reuses `Sequencer` unchanged):

  * the program memory is the DeltaNet *weight state*: the packed (Dmax*n)^2 array.  The prompt's CELL tokens write
    layer weights into block-diagonal blocks b = Dmax-d .. Dmax-1; the other blocks stay 0.  A layer's width is a
    zero-padding choice (zero columns requantise to 0 and are emitted like any other).
  * the program counter is the *current token* (compiled successor table: ACT(row) -> READ(row+1-n), READ(k) ->
    ACT(k+n) | OUTK, OUTK -> READ | HALT).  The prompt sets the initial program counter by choosing WHERE the input
    activations are written: input in block s makes the last prompt token ACT(s n + n-1), whose compiled successor is
    READ(s n); the model then walks blocks s .. Dmax-1 and halts after the last.  Entry block s = Dmax-d  <=>  depth d.
  * pieces that stay compiled into the weights: array size (Dmax, n, ab, wb), the cycle READ->ACT->READ, that the
    chain ends (OUTK, HALT) after block Dmax-1, that layer b feeds rows of block b+1, relu+clamp requantisation.
    Anything else (depth, per-layer widths, all weights/biases, input) is prompt data.
"""
import numpy as np

from . import circuit as C
from .sequencer import Sequencer, generate, ref_chain  # noqa: F401  (re-exported)


class ProgramMachine(Sequencer):
    """A Sequencer whose chain is entered at a prompt-chosen block.  `Sequencer(depth, width)` = this with entry 0."""

    @property
    def dmax(self): return self.depth

    def program_prompt(self, Ws, x):
        """Prompt for the network y = f_{d-1}(... f_0(x)), d = len(Ws) in 1..Dmax.  Ws[j]: (n_in+1) x n_out, last row = bias
        (as `Sequencer.prompt`), n_in <= n-1, n_out <= n-1 for j < d-1 and <= n for the last layer.  Tokens:
        CELL*(KR*NC)  ACT(r;0) for every row outside the entry block  ACT(s n + i; x_i) for the entry block (last)."""
        n, D, KR = self.width, self.depth, C.KR
        d = len(Ws); assert 1 <= d <= D, f"depth {d} outside 1..{D}"
        s = D - d                                                       # entry block = initial program counter
        Wp = np.zeros((KR, KR), dtype=int)
        for j, W in enumerate(Ws):
            W = np.asarray(W); b = s + j; n_in = W.shape[0] - 1
            assert n_in <= n - 1 and W.shape[1] <= n - (b < D - 1), (j, W.shape)
            Wp[b * n: b * n + n_in, b * n: b * n + W.shape[1]] = W[:-1]
            Wp[b * n + n - 1, b * n: b * n + W.shape[1]] = W[-1]
            if b < D - 1: Wp[b * n + n - 1, b * n + n - 1] = 1
        x = list(np.asarray(x, dtype=int)) + [0] * (n - 1 - len(x)) + [1]
        assert len(x) == n and 0 <= min(x) and max(x) <= C.AMAX
        gg = C.G; m = n // gg                                          # token slots per block (gg = 1: one row per token)
        entry = range(s * m, s * m + m)
        return (C.load_trace(Wp) + [C.tok_act(r, *[0] * gg) for r in range(KR // gg) if r not in entry]
                + [C.tok_act(s * m + i, *x[i * gg:(i + 1) * gg]) for i in range(m)])

    def program_logits(self, emitted, n_out):
        return list(self.logits(emitted)[:n_out])


def run_program(feed, seq, Ws, x):
    """Host side, complete: build the prompt, `generate` (argmax; append until HALT), detokenise the OUTK ids."""
    p = seq.program_prompt(Ws, x)
    em = generate(feed, p, seq.halt)
    return em, seq.program_logits(em, np.asarray(Ws[-1]).shape[1])
