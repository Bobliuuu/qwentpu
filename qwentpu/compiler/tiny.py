"""A standalone stock Qwen3_5ForCausalLM that holds only the circuit (same head shapes as Qwen3.5-0.8B)."""
import torch

from .emit import Layout, check_bf16, embedding_rows, emit_layers
from .ir import Target

BASE = {"hidden_act": "silu", "intermediate_size": 3584, "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 16, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
        "rms_norm_eps": 1e-6, "attn_output_gate": True, "max_position_embeddings": 262144,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        "tie_word_embeddings": True}
CODE_GAIN = 2.0 ** 20      # final-norm gain on the output code dims


def to_device(model):
    """Move a tiny model to the GPU when there is one. Where flash-linear-attention is installed its kernels need
    GPU tensors, and a model left on the CPU fails with a Triton "cpu tensor" error; the split-K guard goes on
    with it, since the circuit's integer arithmetic must not be re-associated by cuBLAS."""
    if torch.cuda.is_available():
        from ..merge import exact_matmul
        exact_matmul()
        return model.to("cuda")
    return model


def build(g, vocab, code_dims, dtype=torch.bfloat16):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from .emit import pick_layout
    lay = pick_layout(0, len(g.names))
    n = g.n_layers(); V = -(-len(vocab) // 64) * 64
    tc = dict(BASE, hidden_size=lay.hidden, vocab_size=V, num_hidden_layers=n + 1,
              layer_types=["linear_attention"] * n + ["full_attention"])
    sd = emit_layers(g, lay, tc, 0, "model.")
    emb = torch.zeros(V, lay.hidden); emb[: len(vocab)] = embedding_rows(g, lay, vocab, lay.hidden)
    sd["model.embed_tokens.weight"] = emb
    norm = torch.full((lay.hidden,), -1.0)          # (1 + w) = 0: the lm_head sees only the code dims
    for d in code_dims: norm[lay.dim(d)] = CODE_GAIN
    check_bf16("final norm", norm); sd["model.norm.weight"] = norm
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tc))
    for p in model.parameters(): p.data.zero_()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith(f"model.layers.{n}.") or k == "lm_head.weight" for k in missing), missing[:4]
    model.tie_weights(); model.layout = lay
    return model.to(dtype).eval()
