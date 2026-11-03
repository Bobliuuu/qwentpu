"""Compiled circuit == IR == a stock HF Qwen3_5ForCausalLM (bf16 and fp32)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from qwentpu.compiler import tiny
from qwentpu import circuit as C

rng = np.random.default_rng(3)


def make_tokens(n_vec=8):
    W, X = C.rand_case(rng, True, n_vec)
    toks = C.load_trace(W); want = [C.NOP] * len(toks)
    for x in X:
        toks += C.matvec_trace(x); want += [C.NOP] * C.KS + C.expected_outs(x, W)
    return toks, want


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_hf_matches_ir_and_numpy(dtype):
    g, d = C.build()
    model = tiny.build(g, C.VOCAB, [next(iter(c.t)) for c in d["code"]], dtype=dtype)
    toks, want = make_tokens()
    ids = torch.tensor([toks])
    with torch.no_grad():
        logits = model(ids).logits[0].float()
    pred = logits[:, : len(C.VOCAB)].argmax(-1).numpy()
    bad = [(t, int(pred[t]), w) for t, w in enumerate(want) if pred[t] != w]
    assert not bad, bad[:5]
    print(f"{dtype}: {len(toks)} tokens, hidden {model.layout.hidden}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, all next-tokens exact")
