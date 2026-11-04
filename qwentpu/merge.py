"""Splice the compiled TPU into a real Qwen3.5 checkpoint (0.8B / 2B / 4B), text-only, as a stock
Qwen3_5ForCausalLM.

  1. widen   zero-pad every matrix that touches the residual to the new hidden size; rescale the RMSNorm
             gains that average over the wider residual by sqrt(d0 / hidden)
  2. append  the TPU's layers after Qwen's own
  3. tokens  new embedding rows for the TPU vocabulary; an *untied* lm_head so text tokens can carry
             tau (below) without chip tokens doing so
  4. norm    final-norm gain 0 on every chip dim except the output code
  5. mute    (mute.py) Qwen's own layers ignore chip tokens

tau: a residual dim holding 2^-40 on every text token.  The final norm amplifies it by 2^60 and the
chip rows of the lm_head read it with weight -1, so chip tokens score about -2^20 at text positions and
can never be sampled in prose; at chip positions it is 0 and the code dims decide.
"""
import json
import re
from pathlib import Path

import torch

from . import circuit as C
from .compiler.emit import check_bf16, embedding_rows, emit_layers, pick_layout
from .compiler.tiny import CODE_GAIN

TEXT_IN = "model.language_model."
TEXT_OUT = "model."
IN_COLS = re.compile(r"(in_proj_(qkv|z|a|b)|q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$|embed_tokens\.weight$")
OUT_ROWS = re.compile(r"(out_proj|o_proj|down_proj)\.weight$")
WIDE_NORMS = re.compile(r"(input_layernorm|post_attention_layernorm)\.weight$|^model\.norm\.weight$")
TAU_TEXT, TAU_GAIN = 2.0 ** -40, 2.0 ** 60
FIT_PROMPTS = ["The capital of France is Paris, and the capital of Germany is",
               "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n\nprint(fib(10))",
               "In 1969, Apollo 11 landed on the Moon. Neil Armstrong was the first to"]   # same prompts the tests fit with


def load(path):
    from safetensors import safe_open
    path = Path(path); cfg = json.loads((path / "config.json").read_text()); sd = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                if k.startswith(TEXT_IN):
                    sd[TEXT_OUT + k[len(TEXT_IN):]] = fh.get_tensor(k)       # text tower only; vision/mtp dropped
    return cfg, sd


def widen(sd, d0, hidden):
    scale = (d0 / hidden) ** 0.5
    for k in list(sd):
        w = sd[k]
        if IN_COLS.search(k):
            sd[k] = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
        elif OUT_ROWS.search(k):
            sd[k] = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
        elif WIDE_NORMS.search(k):
            g = (1.0 + w.float()) * scale - 1.0                 # zero-centred gain (1 + w)
            sd[k] = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])
    return sd


def build_merged(src, mute_qwen=True, sinks=None):
    """Returns (text config dict, state dict, chip token offset, layout, graph).  Always built on the CPU."""
    with torch.device("cpu"):
        return _build_merged(src, mute_qwen, sinks)


def _build_merged(src, mute_qwen, sinks):
    g, d = C.build()
    cfg, sd = load(src)
    tc = dict(cfg["text_config"]); d0, v0, n0 = tc["hidden_size"], tc["vocab_size"], tc["num_hidden_layers"]
    lay = pick_layout(d0, len(g.names) + 1)          # +1: tau
    hidden = lay.hidden; n_chip = g.n_layers(); tau = hidden - 1
    assert lay.off + lay.m + len(g.names) - 1 <= tau
    lay.k
    sd = widen(sd, d0, hidden)
    tc.update(hidden_size=hidden, num_hidden_layers=n0 + n_chip, tie_word_embeddings=False,
              layer_types=list(tc["layer_types"]) + ["linear_attention"] * n_chip)
    tc.pop("full_attention_interval", None); tc["mtp_num_hidden_layers"] = 0
    sd.update({k: v.to(torch.bfloat16) for k, v in emit_layers(g, lay, tc, n0, TEXT_OUT).items()})
    # tokens
    nt = len(C.VOCAB); V = -(-(v0 + nt) // 64) * 64; tc["vocab_size"] = V
    E = torch.zeros(V, hidden, dtype=torch.bfloat16)
    E[:v0] = sd[TEXT_OUT + "embed_tokens.weight"]; E[:v0, tau] = TAU_TEXT
    chip = embedding_rows(g, lay, C.VOCAB, hidden).to(torch.bfloat16)
    E[v0: v0 + nt] = chip
    head = E.clone()
    head[:v0, tau] = 0.0                              # text rows do not read tau
    head[v0: v0 + nt, : lay.off + lay.m] = 0.0        # chip rows read only the code dims (ballast gain is 0 anyway)
    head[v0:, tau] = -1.0                             # every non-text row (incl. padding)
    sd[TEXT_OUT + "embed_tokens.weight"] = E; sd["lm_head.weight"] = head
    # final norm: chip dims invisible to the lm_head except the code (and tau)
    norm = sd["model.norm.weight"].float(); norm[d0:] = -1.0
    lay.code_dims = [lay.dim(next(iter(e.t))) for e in d["code"]]      # hidden-state dims that spell the next chip token
    for dd in lay.code_dims: norm[dd] = CODE_GAIN
    tc["qwentpu_code_dims"] = lay.code_dims        # lets the driver read the next token off the sign bits
    norm[tau] = TAU_GAIN
    check_bf16("final norm", norm[d0:]); sd["model.norm.weight"] = norm.to(torch.bfloat16)
    if mute_qwen:
        from .mute import mute
        mute(sd, tc, lay, n0, sinks)
    return tc, sd, v0, lay, g


def exact_matmul():
    """cuBLAS may split a bf16 GEMM's K dimension and combine the partials *in bf16* (shape- and GPU-dependent:
    seen on an L4 for traces of 288 and 320 tokens, never on an A100).  That breaks the integer-exact arithmetic
    this circuit relies on, so every model built here turns it off."""
    import torch
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def instantiate(tc, sd, device="cpu"):
    exact_matmul()
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    config = Qwen3_5TextConfig(**{k: v for k, v in tc.items() if k not in ("model_type",)})
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not missing and not unexpected, (missing[:4], unexpected[:4])
    model.model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config)   # non-persistent buffers were on meta
    return model.to(torch.bfloat16).to(device).eval()


def save(tc, sd, v0, lay, dst):
    from safetensors.torch import save_file
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, dst / "model.safetensors", metadata={"format": "pt"})
    (dst / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", **tc}, indent=2))
    (dst / "qwentpu.json").write_text(json.dumps({"chip_token_offset": v0, "hidden": lay.hidden, "m": lay.m}))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--no-mute", action="store_true")
    a = ap.parse_args()
    sinks = None
    if not a.no_mute:                                   # the anti-sink keys need the base model's own text queries
        from transformers import AutoTokenizer
        from .mute import fit_anti_sink
        cfg, base_sd = load(a.src); btc = dict(cfg["text_config"]); btc["tie_word_embeddings"] = False
        base_sd["lm_head.weight"] = base_sd["model.embed_tokens.weight"].clone()
        base = instantiate(btc, base_sd, "cuda" if torch.cuda.is_available() else "cpu"); del base_sd
        sinks = fit_anti_sink(base, AutoTokenizer.from_pretrained(a.src), FIT_PROMPTS); del base
    tc, sd, v0, lay, g = build_merged(a.src, mute_qwen=not a.no_mute, sinks=sinks)
    save(tc, sd, v0, lay, a.dst); print("saved", a.dst, "hidden", lay.hidden)
