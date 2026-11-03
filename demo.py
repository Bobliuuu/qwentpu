"""XOR on the compiled TPU (same task as TinyTPU's xor_demo)."""
import numpy as np
from qwentpu import QwenTPU

t = QwenTPU()
W1 = np.array([[1, 1], [1, 1], [0, -1]])   # rows: x1, x2, bias   -> h = relu(x1+x2), relu(x1+x2-1)
W2 = np.array([[1], [-2], [0]])            # y = h1 - 2 h2
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = t.mlp([W1, W2], X).ravel()
print("inputs ", X); print("outputs", y.tolist(), "(expect [0,1,1,0])")
print(f"model: {t.m.n_params()/1e6:.2f}M params, hidden={t.m.E.shape[1]}, {t.m.n_tokens} trace tokens")
