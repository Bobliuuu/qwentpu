"""Host driver (qwenputer's RvChat analogue).  It streams TPU micro-op tokens into a merged Qwen3.5 with a
real cache and feeds the model's *own* output tokens forward as the next layer's activations.  The host
does no arithmetic: every product, sum, ReLU and clamp happens in the compiled layers."""
import numpy as np
import torch

from . import circuit as C


class ChipPath:
    """Chip tokens through ONLY the appended TPU layers: embed -> TPU layers -> final norm -> next token.

    Qwen's own layers are muted on chip tokens (their mixer output is exactly 0 and their MLP sees a zero
    vector), so on the chip dims every one of them is the identity; skipping them returns the same bits
    (tests/test_merged.py::test_chip_only_path_is_bit_identical) for ~(1 - n_chip/n_layers) less work.
    Qwen's cache entries are never touched, so a chip run leaves the text context exactly as it was.

    The next token is read straight off the code dims, which spell its id in sign bits (bit i = [code_i > 0]);
    the lm_head over the chip vocabulary computes the same argmax (readout="lm_head", used by the tests) but
    costs hidden x |vocab| MACs per token, more than the whole TPU stack once the vocabulary is large."""

    def __init__(self, model, v0, code_dims=None, readout="code"):
        t = model.config.layer_types
        code_dims = code_dims or getattr(model.config, "qwentpu_code_dims", None)
        if code_dims is None: readout = "lm_head"
        self.m, self.v0, self.readout = model, v0, readout
        self.first = (next(i for i in range(len(t) - 1, -1, -1) if t[i] == "full_attention")) + 1
        from . import circuit as C
        self.head = model.lm_head.weight[v0: v0 + len(C.VOCAB)]
        self.code_dims = None if code_dims is None else torch.tensor(code_dims, device=model.device)

    @torch.no_grad()
    def hidden(self, ids, past):
        m = self.m.model
        h = m.embed_tokens(ids)
        for layer in m.layers[self.first:]:
            h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=past)
        return m.norm(h)

    @torch.no_grad()
    def tokens(self, ids, past):
        """argmax next chip token at every position, shape (B, T)."""
        h = self.hidden(ids, past)
        if self.readout == "code":
            bits = (h[..., self.code_dims] > 0).to(torch.int64)
            return (bits * (1 << torch.arange(bits.shape[-1], device=bits.device))).sum(-1)
        return (h @ self.head.T).argmax(-1)

    __call__ = tokens


class GraphedChip(ChipPath):
    """ChipPath replayed from CUDA graphs: a whole chunk (all TPU layers, readout included) is ONE graph launch.

    HF's linear-attention cache copies into fixed buffers, so once the states exist (and `has_previous_state` is
    set; zero states are exactly a fresh start) every chunk of a given length has static shapes and addresses.
    One graph per distinct chunk length (KR/g ACT tokens, NC/h READ tokens, ...) is captured on first use; the
    cache is shared, so state flows between graphs exactly as between eager calls (tested bit-identical).
    Free-running chains are bound by per-call launch overhead, not arithmetic: this removes it."""

    def __init__(self, model, v0, code_dims=None, readout="code"):
        super().__init__(model, v0, code_dims, readout)
        self.graphs, self.past = {}, None

    def attach(self, past):
        """Initialise the chip layers' state tensors in `past` and mark them live (zero state == fresh start)."""
        self.past = past
        ids = torch.full((1, 1), self.v0, dtype=torch.long, device=self.m.device)
        super().tokens(ids, past)                                   # lazy init of the state tensors
        for L in range(self.first, self.m.config.num_hidden_layers):
            lyr = past.layers[L]
            for i in range(lyr.number_of_states):
                lyr.has_previous_state[i] = True
        self.zero()

    def _states(self):
        for L in range(self.first, self.m.config.num_hidden_layers):
            lyr = self.past.layers[L]
            for i in range(lyr.number_of_states):
                yield lyr.conv_states[i]; yield lyr.recurrent_states[i]

    def zero(self):
        for t in self._states(): t.zero_()

    @torch.no_grad()
    def tokens(self, ids, past):
        T = ids.shape[1]
        if T not in self.graphs:
            static = torch.full((1, T), self.v0, dtype=torch.long, device=ids.device)
            saved = [t.clone() for t in self._states()]             # warm-up runs must not eat the live state
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):                              # warm up (autotune, allocator) off the graph
                for _ in range(2): ChipPath.tokens(self, static, past)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): out = ChipPath.tokens(self, static, past)
            self.graphs[T] = (g, static, out)
            for t, v in zip(self._states(), saved): t.copy_(v)
        g, static, out = self.graphs[T]
        static.copy_(ids); g.replay()
        return out.clone()


class TpuSession:
    def __init__(self, model, v0, fast=True, readout="code", code_dims=None):
        """fast: False = whole network (reference); True = TPU layers only; "graph" = TPU layers from CUDA graphs."""
        self.m, self.v0, self.past, self.n_tokens = model, v0, None, 0
        self.path = None if not fast else (GraphedChip if fast == "graph" else ChipPath)(model, v0, code_dims, readout)

    @torch.no_grad()
    def feed(self, toks):
        """Feed chip tokens; returns the model's argmax next-token (as a chip token) at each position."""
        if self.path is not None:
            from transformers import DynamicCache
            if self.past is None:
                self.past = DynamicCache(config=self.m.config)
                if isinstance(self.path, GraphedChip): self.path.attach(self.past)
            ids = torch.tensor([[self.v0 + t for t in toks]], device=self.m.device)
            self.n_tokens += len(toks)
            return self.path.tokens(ids, self.past)[0].tolist()
        ids = torch.tensor([[self.v0 + t for t in toks]], device=self.m.device)
        out = self.m(input_ids=ids, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values; self.n_tokens += len(toks)
        return (out.logits[0].float().argmax(-1) - self.v0).tolist()

    def load_weights(self, W):
        """Weight-stationary load: W[r][c] (KR x NC, two's complement WB bits) into the DeltaNet weight array."""
        Wp = np.zeros((C.KR, C.NC), dtype=int); Wp[: W.shape[0], : W.shape[1]] = W
        self.feed(C.load_trace(Wp))

    def matvec(self, x, ncols=None):
        """One vector (KR activations 0..AMAX) through the loaded array: ReLU + clamp to 0..AMAX.  The tokens are
        KR/g ACT tokens (g activations each) and NC/h READ tokens (h outputs each); the host only relabels."""
        ncols = C.NC if ncols is None else ncols
        x = [int(v) for v in x]
        self.feed([C.tok_act(s, *x[s * C.G:(s + 1) * C.G]) for s in range(C.KS)])
        out = self.feed([C.tok_read(k) for k in range(-(-ncols // C.H))])
        assert all(C.OUT0 <= t < C.NOP for t in out), out
        return np.array([v for t in out for v in C.out_values(t - C.OUT0)][:ncols])

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

    def __init__(self, model, v0, fast=True, readout="code", code_dims=None):
        super().__init__(model, v0, fast, readout, code_dims)
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
