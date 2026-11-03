"""Make Qwen's own layers ignore chip tokens, bit for bit on text.

On a chip token every RMSNorm output on a ballast dim is the same constant nb = B * 2^-k, and on a text
token it is exactly 0.  So a weight on one ballast dim acts on chip tokens only:

  Gated DeltaNet   in_proj_a = -1024:  softplus(-1024 + dt_bias) = 0  ->  g = 0, the state does not decay
                   in_proj_b = -1024:  beta = 0, nothing is written
                   in_proj_z = -1024:  silu(z) = -0, the layer outputs exactly 0
  full attention   output gate (second half of q_proj) = -1024:  sigmoid = 0, the layer outputs exactly 0
                   k_proj = anti-sink key: chip positions score very negative against text queries, so
                   they take no softmax mass from the text around them
So Qwen's recurrent state passes through a run of chip tokens untouched and nothing is added to the
residual.  What remains is the conv window: the 3 text tokens after a chip run see zero inputs in it.
"""
import torch

from .compiler.emit import check_bf16

MUTE = -1024.0
ROT = 64          # rotary dims of a 256-wide head (partial_rotary_factor 0.25); the sink key avoids them


def mute(sd, tc, lay, n0, sinks=None):
    nb = lay.B * 2.0 ** -lay.k                   # normed ballast value on a chip token
    w = MUTE / nb; b0 = lay.off                  # one ballast dim carries it
    for i in range(n0):
        p = f"model.layers.{i}."
        if tc["layer_types"][i] == "linear_attention":
            for name in ("in_proj_a", "in_proj_b", "in_proj_z"):
                sd[p + f"linear_attn.{name}.weight"][:, b0] = w
        else:
            hd = tc["head_dim"]; q = sd[p + "self_attn.q_proj.weight"]
            q.view(-1, 2, hd, q.shape[-1])[:, 1, :, b0] = w          # [head, (query|gate), hd, hidden]
            if sinks is not None:
                kw = sd[p + "self_attn.k_proj.weight"]; gain = 1.0 + sd[p + "self_attn.k_norm.weight"].float()
                for g_, u in enumerate(sinks[i]):                    # u: (head_dim,) desired post-norm direction
                    kv = u / gain; kv[:ROT] = 0.0
                    kw.view(-1, hd, kw.shape[-1])[g_, :, b0] = (kv / nb).to(kw.dtype)


@torch.no_grad()
def fit_anti_sink(model, tok, prompts):
    """Per full-attention layer: (n_kv_heads, head_dim) keys opposite to the mean text query."""
    tc = model.config
    layers = {i: l.self_attn for i, l in enumerate(model.model.layers) if tc.layer_types[i] == "full_attention"}
    sums = {i: 0 for i in layers}; n = 0
    hooks = [a.q_norm.register_forward_hook(lambda m, inp, out, i=i: sums.__setitem__(i, sums[i] + out.float().flatten(0, -3).sum(0)))
             for i, a in layers.items()]
    dev = next(model.parameters()).device
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(dev); model(input_ids=ids); n += ids.shape[1]
    for h in hooks: h.remove()
    hd, nkv = tc.head_dim, tc.num_key_value_heads; grp = tc.num_attention_heads // nkv; out = {}
    for i in layers:
        mean = sums[i] / n                                            # (heads, head_dim)
        u = -mean.view(nkv, grp, hd).mean(1)
        u[:, :ROT] = 0.0
        out[i] = (16.0 * u / u.norm(dim=-1, keepdim=True)).cpu()      # unit-ish direction
    return out
