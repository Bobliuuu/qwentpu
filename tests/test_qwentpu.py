import numpy as np
import torch
from qwentpu import QwenTPU
from qwentpu.gdn import DT, GDNLayer, l2norm
from qwentpu.reference import ref_mlp

rng = np.random.default_rng(0)


def test_delta_rule_matches_matrix_form():
    # S_t = S_{t-1} (alpha (I - beta k k^T)) + beta v k^T   (state stored transposed)
    L = GDNLayer(8, 1, 4, 3)
    L.Wq = torch.randn(4, 8, dtype=DT); L.Wk = torch.randn(4, 8, dtype=DT); L.Wv = torch.randn(3, 8, dtype=DT)
    L.Wb = torch.randn(1, 8, dtype=DT); L.Wa = torch.randn(1, 8, dtype=DT); L.Wo = torch.randn(8, 3, dtype=DT)
    S = L.init_state(); St = torch.zeros(3, 4, dtype=DT)
    for _ in range(20):
        h = torch.randn(8, dtype=DT)
        _, S = L(h, S)
        k = l2norm(L.Wk @ h); v = L.Wv @ h; b = torch.sigmoid(L.Wb @ h)[0]
        a = torch.exp(-torch.exp(L.A_log) * torch.nn.functional.softplus(L.Wa @ h))[0]
        St = St @ (a * (torch.eye(4, dtype=DT) - b * torch.outer(k, k))) + b * torch.outer(v, k)
        assert torch.allclose(S[0].T, St, atol=1e-12)


def test_matvec_exact():
    t = QwenTPU()
    W = rng.integers(-8, 9, (16, 16)); t.load_weights(W)
    for _ in range(3):
        x = rng.integers(-8, 9, 16)
        assert (t.matvec(x, 16) == np.clip(x @ W, -128, 127)).all()
        assert (t.matvec(x, 16, relu=True) == np.maximum(np.clip(x @ W, -128, 127), 0)).all()


def test_saturation_and_flush():
    t = QwenTPU()
    W = np.full((16, 16), 100); t.load_weights(W)
    assert (t.matvec(np.ones(16, int) * 100, 4) == 127).all()
    assert (t.matvec(-np.ones(16, int) * 100, 4) == -128).all()
    assert (t.matvec(np.zeros(16, int), 4) == 0).all()       # FLUSH cleared the accumulators


def test_xor_mlp():
    t = QwenTPU()
    W1 = np.array([[1, 0], [1, 0], [0, -1]]).T * 0  # placeholder replaced below
    W1 = np.array([[1, 1], [1, 1], [0, -1]])          # rows: x1, x2, bias
    W2 = np.array([[1], [-2], [0]])
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert t.mlp([W1, W2], X).ravel().tolist() == [0, 1, 1, 0]


def test_random_mlp_vs_reference():
    t = QwenTPU()
    Ws = [rng.integers(-3, 4, (6, 5)), rng.integers(-3, 4, (6, 3))]
    X = rng.integers(0, 4, (3, 5))
    assert (t.mlp(Ws, X) == ref_mlp(Ws, X)).all()
