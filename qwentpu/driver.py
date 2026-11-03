"""Host driver (qwenputer's RvChat analogue).  It streams TPU micro-op tokens into a merged Qwen3.5 with a
real cache and feeds the model's *own* output tokens forward as the next layer's activations.  The host
does no arithmetic: every product, sum, ReLU and clamp happens in the compiled layers."""
import numpy as np
import torch

from . import circuit as C


class TpuSession:
    def __init__(self, model, v0):
        self.m, self.v0, self.past, self.n_tokens = model, v0, None, 0

    @torch.no_grad()
    def feed(self, toks):
        """Feed chip tokens; returns the model's argmax next-token (as a chip token) at each position."""
        ids = torch.tensor([[self.v0 + t for t in toks]])
        out = self.m(input_ids=ids, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values; self.n_tokens += len(toks)
        return (out.logits[0].float().argmax(-1) - self.v0).tolist()

    def load_weights(self, W):
        """Weight-stationary load: W[r][c] (KR x NC, two's complement WB bits) into the DeltaNet weight array."""
        Wp = np.zeros((C.KR, C.NC), dtype=int); Wp[: W.shape[0], : W.shape[1]] = W
        self.feed(C.load_trace(Wp))

    def matvec(self, x, ncols=C.NC):
        """One vector (KR activations 0..AMAX) through the loaded array: ReLU + clamp to 0..AMAX."""
        self.feed([C.tok_act(i, int(v)) for i, v in enumerate(x)])
        out = self.feed([C.tok_read(c) for c in range(ncols)])
        assert all(C.OUT0 <= t < C.OUT0 + C.NOUT for t in out), out
        return np.array([t - C.OUT0 for t in out])

    def mlp(self, Ws, X):
        """Layers chain through the model's own output tokens.  W: (n_in + 1) x n_out, last row = bias,
        n_in <= KR - 1 (the bias is a constant-1 activation in row KR-1)."""
        X = np.asarray(X)
        for W in Ws:
            n_in = W.shape[0] - 1
            Wp = np.zeros((C.KR, C.NC), dtype=int); Wp[:n_in, : W.shape[1]] = W[:-1]; Wp[C.KR - 1, : W.shape[1]] = W[-1]
            self.load_weights(Wp)
            X = np.stack([self.matvec(np.concatenate([x, np.zeros(C.KR - 1 - len(x), int), [1]]), W.shape[1]) for x in X])
        return X


def ref_mlp(Ws, X):
    X = np.asarray(X)
    for W in Ws:
        X = np.clip(np.maximum(np.concatenate([X, np.ones((len(X), 1), int)], 1) @ W, 0), 0, C.AMAX)
    return X


import copy
from contextlib import contextmanager


class HybridSession(TpuSession):
    """Text and TPU in one model, with the chip run invisible to Qwen.

    `chip()` snapshots the cache of Qwen's own layers, lets the TPU run, and then puts the snapshot back, so
    Qwen's conv windows, recurrent states and KV (and its position counter) are exactly as if the chip tokens
    had never been fed.  The TPU layers keep their state (the loaded weights).  Results come back to Qwen the
    way qwenputer's do: the host relabels an output token as an ordinary text token (`text()`), no arithmetic.
    """

    def __init__(self, model, v0):
        super().__init__(model, v0)
        self.n_qwen = sum(1 for _ in range(model.config.num_hidden_layers)) - self.n_chip_layers(model)

    @staticmethod
    def n_chip_layers(model):
        t = model.config.layer_types
        return len(t) - next(i for i in range(len(t) - 1, -1, -1) if t[i] == "full_attention") - 1

    @torch.no_grad()
    def text(self, ids):
        out = self.m(input_ids=torch.tensor([list(ids)]), past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        return out.logits[0, :, : self.v0].float()

    @contextmanager
    def chip(self):
        from transformers import DynamicCache
        snap = copy.deepcopy(self.past) if self.past is not None else DynamicCache(config=self.m.config)
        try:
            yield self
        finally:
            for i in range(self.n_qwen):
                self.past.layers[i] = snap.layers[i]
