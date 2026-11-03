"""Host driver (qwenputer's RvChat analogue): turns matmuls/MLPs into trace tokens and feeds the
model's own output tokens back.  The host does no arithmetic -- every product and sum happens in
the compiled layers."""
import numpy as np

from .model import N, QwenTPUModel


class QwenTPU:
    def __init__(self):
        self.m = QwenTPUModel()

    def load_weights(self, W):
        """Weight-stationary load: W[r][c] -> DeltaNet head c, row r (shape <= N x N, zero padded)."""
        Wp = np.zeros((N, N), dtype=int); Wp[: W.shape[0], : W.shape[1]] = W
        for r in range(N):
            for c in range(N):
                self.m.step(("ADDR", r * N + c)); self.m.step(("WVAL", int(Wp[r, c])))

    def matvec(self, x, ncols, relu=False):
        self.m.step(("FLUSH",))
        for i, v in enumerate(x):
            if v:
                self.m.step(("ACT", i, int(v)))
        out = []
        for c in range(ncols):
            t = self.m.step(("READR" if relu else "READ", c))
            out.append(t[1])           # ("WVAL", y) emitted by the model itself
        return np.array(out)

    def mlp(self, Ws, X, relu_last=False):
        """Run X [B, in] through layers Ws (each [in+1, out], last row = bias).  Bias = constant-1 input."""
        X = np.asarray(X)
        for li, W in enumerate(Ws):
            self.load_weights(W)
            last = li == len(Ws) - 1
            X = np.stack([self.matvec(np.append(x, 1), W.shape[1], relu=(not last) or relu_last) for x in X])
        return X
