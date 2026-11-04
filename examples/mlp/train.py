"""Train + quantize a digits classifier for the qwenTPU v2 array and emit program/golden files.

    python examples/mlp/train.py [--kr 16] [--xmax 12] [--epochs 250] [--seeds 4] [--ntest 20]

Pipeline: sklearn load_digits (8x8) -> average-pool to 15 non-negative features (kr=16) -> quantize to 0..xmax ->
integer-aware training (STE through round/clamp, leaky gradient outside the linear range) of a 15->15->10 net on the
EXACT datapath arithmetic  a = clip(relu(x @ W), 0, amax)  with the bias as a weight on the constant-1 row and NO shift
(docs/V2_ISA.md) -> golden files.  A float MLP of the same shape on the same features is the reference accuracy.
Why xmax is small: with no REQUANT shift the scale is fixed, so products must stay in the 0..31 window; 2-3 bit
inputs and small integer weights are what fit.  (xmax swept 3..31 at 250 epochs: train acc peaks at 12; test 0.835-0.889 across the sweep.)
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from qwentpu import isa, program  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def data(F, seed=0):
    """Non-negative features in 0..1: kr=16: 2x2 average pool -> 4x4=16 blocks, drop the always-dark corner -> 15;
    kr=32: 1x2 pool -> 8x4=32, drop one -> 31.  Other F: PCA (offsets make biases big; not recommended)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits(); I = d.images / 16.0
    if F == 15: Z = I.reshape(-1, 4, 2, 4, 2).mean((2, 4)).reshape(len(I), 16)[:, 1:]
    elif F == 31: Z = I.reshape(-1, 8, 4, 2).mean(3).reshape(len(I), 32)[:, 1:]
    else:
        X = d.data / 16.0; mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False); Z = (X - mu) @ Vt[:F].T
    return train_test_split(Z, d.target, test_size=0.3, random_state=seed, stratify=d.target)


def quant_in(Z, xmax):
    return np.clip(np.rint(Z * xmax), 0, xmax).astype(np.int64)      # pooled pixel in 0..1 -> 0..xmax


def adam(params, lr):
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; t = [0]
    def step(grads, lr_=None):
        t[0] += 1
        for p, g, mi, vi in zip(params, grads, m, v):
            mi *= .9; mi += .1 * g; vi *= .999; vi += .001 * g * g
            p -= (lr_ or lr) * (mi / (1 - .9 ** t[0])) / (np.sqrt(vi / (1 - .999 ** t[0])) + 1e-8)
    return step


def int_forward(a, w1, w2, x):
    a1 = np.clip(np.maximum(x @ w1[:-1] + w1[-1], 0), 0, a.amax)
    return a1, np.clip(np.maximum(a1 @ w2[:-1] + w2[-1], 0), 0, a.amax)


def acc_of(a, W, X, y): return (int_forward(a, *W, X)[1].argmax(1) == y).mean()


def qat(a, Xq, y, H, epochs, seed, lr=0.02, leak=0.15, T=4.0):
    """Latent float weights in integer units; forward uses round+clip to [wlo, whi].  Returns best-train-acc int weights."""
    rng = np.random.default_rng(seed); F = Xq.shape[1]; X = Xq.astype(np.float64); N = len(X); amax = a.amax
    ps = [rng.normal(0, .5, (F, H)), np.ones((1, H)), rng.normal(0, .5, (H, 10)), np.zeros((1, 10))]; opt = adam(ps, lr)
    Rr = lambda p: np.clip(np.rint(p), a.wlo, a.whi)
    lk = lambda z: np.where((z > 0) & (z < amax), 1.0, leak)
    best = (-1, None)
    for ep in range(epochs):
        idx = rng.permutation(N); cur = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)) * 0.9 + 0.1)
        for s in range(0, N, 64):
            i = idx[s:s + 64]; x = X[i]; t = np.eye(10)[y[i]]
            q1, b1, q2, b2 = [Rr(p) for p in ps]
            z1 = x @ q1 + b1; a1 = np.clip(np.maximum(z1, 0), 0, amax); z2 = a1 @ q2 + b2
            a2 = np.clip(np.maximum(z2, 0), 0, amax)
            e = np.exp((a2 - a2.max(1, keepdims=True)) / T); p = e / e.sum(1, keepdims=True)
            d2 = (p - t) / T / len(i) * lk(z2); d1 = (d2 @ q2.T) * lk(z1)
            opt([x.T @ d1, d1.sum(0, keepdims=True), a1.T @ d2, d2.sum(0, keepdims=True)], cur)
            for p_ in ps: np.clip(p_, a.wlo - .49, a.whi + .49, out=p_)
        if ep % 5 == 4 or ep == epochs - 1:
            W = [np.vstack([Rr(ps[0]), Rr(ps[1])]).astype(np.int64), np.vstack([Rr(ps[2]), Rr(ps[3])]).astype(np.int64)]
            ac = acc_of(a, W, Xq, y)                                  # model selection on TRAIN accuracy only
            if ac > best[0]: best = (ac, W)
    return best


def float_mlp(A, y, H, seed=1, epochs=300):
    rng = np.random.default_rng(seed); F = A.shape[1]
    W1 = rng.normal(0, .3, (F, H)); b1 = np.zeros(H); W2 = rng.normal(0, .3, (H, 10)); b2 = np.zeros(10)
    ps = [W1, b1, W2, b2]; opt = adam(ps, 0.01)
    for ep in range(epochs):
        idx = rng.permutation(len(A))
        for st in range(0, len(A), 64):
            i = idx[st:st + 64]; z1 = A[i] @ W1 + b1; a1 = np.maximum(z1, 0); z2 = a1 @ W2 + b2
            p = np.exp(z2 - z2.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); d2 = (p - np.eye(10)[y[i]]) / len(i)
            d1 = (d2 @ W2.T) * (z1 > 0); opt([A[i].T @ d1, d1.sum(0), a1.T @ d2, d2.sum(0)])
    return W1, b1, W2, b2


def predict(a, Ws, Xq):
    return np.stack([program.run_mlp_np(a, Ws, x)[:10] for x in Xq])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kr", type=int, default=16); ap.add_argument("--ab", type=int, default=5)
    ap.add_argument("--xmax", type=int, default=12); ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--ntest", type=int, default=20); ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=4); ap.add_argument("--out", default=HERE)
    o = ap.parse_args()
    a = isa.Arr(o.kr, o.kr, o.ab, o.ab, 1, 1, 2); F = o.kr - 1; H = o.hidden or o.kr - 1
    Ztr, Zte, ytr, yte = data(F)
    fl = float_mlp(Ztr, ytr, H)
    facc = ((np.maximum(Zte @ fl[0] + fl[1], 0) @ fl[2] + fl[3]).argmax(1) == yte).mean()
    Xtr, Xte = quant_in(Ztr, o.xmax), quant_in(Zte, o.xmax)
    runs = [qat(a, Xtr, ytr, H, o.epochs, s) for s in range(o.seeds)]
    for s, (ac, _) in enumerate(runs): print(f"  seed {s}: train acc {ac:.4f}")
    tr_acc, Ws = max(runs, key=lambda r: r[0])                         # pick by train accuracy
    iacc = acc_of(a, Ws, Xte, yte)
    print(f"array {o.kr}x{o.kr} u{a.ab}s{a.wb} features={F} hidden={H} xmax={o.xmax} | float(same shape, same features) test acc "
          f"{facc:.4f} | integer train {tr_acc:.4f} test {iacc:.4f} (n_test={len(yte)})")
    n = o.ntest; prompts, emitted, logits = [], [], []
    for i in range(n):
        prog = program.mlp_program(a, Ws, Xte[i])
        ref = isa.run(a, prog); assert (ref == program.run_mlp_np(a, Ws, Xte[i])).all()
        p = isa.encode(a, prog); e = isa.emitted_trace(a, p)
        assert (isa.decode_result(a, e) == ref).all()
        prompts.append(np.array(p)); emitted.append(np.array(e)); logits.append(ref)
    lg = np.stack(logits)
    np.savez(os.path.join(o.out, "golden.npz"), prompt_ids=np.stack(prompts), expected_emitted_ids=np.stack(emitted),
             logits=lg, label=yte[:n], pred=lg[:, :10].argmax(1), W1=Ws[0], W2=Ws[1], x_test=Xte[:n],
             cfg=np.array([a.kr, a.nc, a.ab, a.wb, a.g, a.h, a.nb]), float_acc=facc, int_acc=iacc, int_train_acc=tr_acc)
    print(f"golden: {n} programs, prompt {len(prompts[0])} tokens, {len(emitted[0])} emitted tokens each, "
          f"golden argmax matches label on {(lg[:, :10].argmax(1) == yte[:n]).mean():.2f} of them")


if __name__ == "__main__":
    main()
