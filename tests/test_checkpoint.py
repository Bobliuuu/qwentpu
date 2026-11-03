"""The merged model survives a save / from_pretrained round trip as an ordinary Hugging Face checkpoint."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SRC = os.environ.get("QWEN_DIR")
pytestmark = pytest.mark.skipif(not SRC, reason="set QWEN_DIR to a Qwen3.5 checkpoint")
from qwentpu import circuit as C
from qwentpu.merge import build_merged, save

DEV = os.environ.get("QWEN_DEVICE", "cpu")


def test_saved_checkpoint_loads_with_plain_hf_and_computes(tmp_path):
    from transformers import AutoModelForCausalLM
    tc, sd, v0, lay, g = build_merged(SRC)
    save(tc, sd, v0, lay, tmp_path)
    assert (tmp_path / "model.safetensors").stat().st_size > 1e8
    model = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.bfloat16).to(DEV).eval()
    rng = np.random.default_rng(11)
    W, X = C.rand_case(rng, True, 3)
    toks = C.load_trace(W); want = [C.NOP] * len(toks)
    for x in X:
        toks += C.matvec_trace(x); want += [C.NOP] * C.KR
        want += [C.tok_out(int(np.clip(max(0, (x @ W)[c]), 0, C.AMAX))) for c in range(C.NC)]
    with torch.no_grad():
        h = model.model(input_ids=torch.tensor([[v0 + t for t in toks]], device=DEV)).last_hidden_state
        pred = (h @ model.lm_head.weight[v0: v0 + len(C.VOCAB)].T).argmax(-1)[0].cpu().tolist()
    assert pred == want
