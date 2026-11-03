import numpy as np


def ref_mlp(Ws, X, relu_last=False):
    X = np.asarray(X)
    for li, W in enumerate(Ws):
        X = np.clip(np.concatenate([X, np.ones((len(X), 1), int)], 1) @ W, -128, 127)
        if li < len(Ws) - 1 or relu_last:
            X = np.maximum(X, 0)
    return X
