"""IR -> Qwen3.5 layer tensors.

Residual layout of the chip region (starting at Layout.off; Qwen's own dims come first in a merged model):
    [ballast x m] [IR dim 1] [IR dim 2] ...
A chip token carries B = 2^40 in every ballast dim.  If m*B^2/hidden = 4^k then every RMSNorm on a
chip token divides by exactly 2^k, so a residual value x reaches a projection as x * 2^-k, and the
constant 1 is the normed ballast B * 2^-k.  Weights carry the inverse factor.  (Other residual values
vanish next to the ballast in the fp32 sum of squares because |x| <= 2^28.)
"""
import math
from dataclasses import dataclass

import torch

from .ir import FFN, HD, K, ONE, Head

LOG2_B = 40


@dataclass
class Layout:
    hidden: int
    off: int = 0
    m: int = 3

    @property
    def B(self): return 2.0 ** LOG2_B

    @property
    def k(self):
        """rms of a chip token's residual is 2^k."""
        mean = self.m * self.B ** 2 / self.hidden
        k4 = math.log(mean, 4)
        assert abs(k4 - round(k4)) < 1e-9, "m * B^2 / hidden must be a power of 4"
        return round(k4)

    def dim(self, d): return self.off + self.m + d - 1    # IR dim d >= 1


def pick_layout(base: int, n_dims: int) -> Layout:
    """Smallest hidden = m * 4^j (multiple of 128) that holds `base` Qwen dims, m ballast dims and n_dims."""
    best = None
    for m in range(3, 65):
        j = 0
        while m * 4 ** j < base + m + n_dims or (m * 4 ** j) % 128:
            j += 1
        if best is None or m * 4 ** j < best.hidden:
            best = Layout(m * 4 ** j, base, m)
    return best


def bf16_parts(x, n):
    """x as a sum of at most n bf16 values."""
    parts = []
    for _ in range(n):
        if x == 0: break
        p = torch.tensor(x, dtype=torch.float64).to(torch.bfloat16).double().item()
        parts.append(p); x -= p
    if x != 0: raise ValueError(f"constant needs more than {n} bf16 parts")
    return parts


def row(e, lay, out, w=1.0):
    """out += w * e as a projection row over the normed residual."""
    for d, c in e.t.items():
        if d == ONE:
            for i, p in enumerate(bf16_parts(w * float(c) * 2.0 ** lay.k / lay.B, lay.m)):
                out[lay.off + i] += p
        else:
            out[lay.dim(d)] += w * float(c) * 2.0 ** lay.k


def check_bf16(name, t):
    b = t.to(torch.bfloat16).to(t.dtype)
    bad = ((b != t) | ~torch.isfinite(b)).nonzero()
    if len(bad):
        raise ValueError(f"{name}: weight {t[tuple(bad[0].tolist())].item()!r} is not a bf16 value")


def emit_layers(g, lay, tc, first, prefix):
    """Tensors of the chip layers, named f'{prefix}layers.{first+i}.*'."""
    h, nk, nv, inter = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["intermediate_size"]
    r = nv // nk; assert r * nk == nv
    f32 = torch.float32; out = {}
    for L in range(g.n_layers()):
        p = f"{prefix}layers.{first + L}."
        qkv = torch.zeros(2 * nk * HD + nv * HD, h); conv = torch.zeros(qkv.shape[0], 1, tc["linear_conv_kernel_dim"])
        zw, bw, aw = torch.zeros(nv * HD, h), torch.zeros(nv, h), torch.zeros(nv, h)
        ow = torch.zeros(h, nv * HD)
        gate, up, down = torch.zeros(inter, h), torch.zeros(inter, h), torch.zeros(h, inter)
        heads = [n for n in g.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
        for j, n in enumerate(heads):
            assert len(n.vx) <= r
            for base, chans in ((j * HD, n.q), (nk * HD + j * HD, n.k)):
                for i, c in chans.items():
                    row(c.expr, lay, qkv[base + i])
                    for lag, w in enumerate(c.taps): conv[base + i, 0, -1 - lag] = w
            for x, vh in enumerate(n.vx):
                v = j * r + x
                for col, c in vh.v.items():
                    row(c.expr, lay, qkv[2 * nk * HD + v * HD + col])
                    for lag, w in enumerate(c.taps): conv[2 * nk * HD + v * HD + col, 0, -1 - lag] = w
                # beta = sigmoid(128) = 1, or sigmoid(256 g - 128) = 1 / exactly 0
                row(g.one * 128 if vh.beta is None else vh.beta * 256 - g.one * 128, lay, bw[v])
                row(vh.a, lay, aw[v])
                for col, e in vh.z.items(): row(e, lay, zw[v * HD + col])
                for col, d, w in vh.out: ow[lay.dim(d), v * HD + col] += w
        units = [(c, a, b, n.out) for n in g.nodes if isinstance(n, FFN) and n.level == 2 * L + 2 for c, a, b in n.terms]
        assert len(units) <= inter, (L, len(units), inter)
        for u, (c, a, b, d) in enumerate(units):
            row(b, lay, gate[u], K); row(a, lay, up[u], 1.0 / K); down[lay.dim(d), u] = float(c)
        t = {"input_layernorm.weight": torch.zeros(h), "post_attention_layernorm.weight": torch.zeros(h),
             "linear_attn.in_proj_qkv.weight": qkv, "linear_attn.in_proj_z.weight": zw,
             "linear_attn.in_proj_a.weight": aw, "linear_attn.in_proj_b.weight": bw,
             "linear_attn.conv1d.weight": conv, "linear_attn.dt_bias": torch.full((nv,), -110.0),
             "linear_attn.A_log": torch.zeros(nv), "linear_attn.norm.weight": torch.ones(HD),
             "linear_attn.out_proj.weight": ow, "mlp.gate_proj.weight": gate, "mlp.up_proj.weight": up,
             "mlp.down_proj.weight": down}
        for k_, v_ in t.items():
            check_bf16(p + k_, v_); out[p + k_] = v_
    return out


def embedding_rows(g, lay, vocab, hidden):
    e = torch.zeros(len(vocab), hidden)
    e[:, lay.off: lay.off + lay.m] = lay.B
    for d, fn in g.token_fns.items():
        e[:, lay.dim(d)] = torch.tensor([float(fn(t)) for t in vocab])
    check_bf16("embedding", e)
    return e
