"""Compiler: TPU circuit -> weights of a hybrid DeltaNet/MLP transformer.  Nothing is learned.

Layer stack (one token = one TPU micro-op, exactly like qwenputer's trace tokens):
  0 GDN  ADDR latch     one head stores the last ADDR token's (row, col) one-hot       [address reg]
  1 GDN  WEIGHT ARRAY   N heads (one per output column), dk=N rows: S[c][r] = W[r][c]  [weights, stationary]
  2 MLP  MAC            prod_c = act * W[i][c]   (SwiGLU gate*up multiplier)           [multipliers]
  3 GDN  ACCUMULATORS   N heads; ACT(i,v) writes row i = v*W[i][c]; READ query = ones  [partial sums + adder tree]
  4 MLP  ReLU           y += isR * relu(-y)                                             [VPU activation]
  5 MLP  saturate       clamp to int8                                                   [VPU requant]
  6 MLP  one-hot        oh[v] = tent(y - v);  lm_head reads oh                          [output word]
"""
import torch

from .gdn import DT, GDNLayer, SwiGLU

N = 16               # array is N x N
BIG, ON = 1000.0, 60.0   # l2norm makes BIG irrelevant; ON saturates sigmoid/softplus to exactly 0/1
G = 128.0            # silu(G*z) == G*relu(z) exactly for integer z (silu(-128)~1e-54)


class Dims:
    def __init__(self):
        self.n, self.names = 0, {}

    def add(self, name, k=1):
        self.names[name] = self.n
        self.n += k
        return self.names[name]

    def __getitem__(self, k):
        return self.names[k]


def build():
    D = Dims()
    for nm, k in [("one", 1), ("val", 1), ("wval", 1), ("idx", N), ("colsel", N), ("isADDR", 1), ("isWVAL", 1),
                  ("isACT", 1), ("isREAD", 1), ("isR", 1), ("isFLUSH", 1), ("eaddr", 2 * N), ("addr", 2 * N),
                  ("rowvec", N), ("prod", N), ("y", 1), ("oh", 256)]:
        D.add(nm, k)
    H = D.n
    layers = []

    # 0: address latch (dk=1, dv=2N)
    L = GDNLayer(H, 1, 1, 2 * N)
    L.Wk[0, D["one"]] = BIG; L.Wq[0, D["one"]] = BIG
    for j in range(2 * N):
        L.Wv[j, D["eaddr"] + j] = 1.0
        L.Wo[D["addr"] + j, j] = 1.0
    L.Wb[0, D["isADDR"]] = 2 * ON; L.Wb[0, D["one"]] = -ON
    L.Wa[0, D["one"]] = -ON
    layers.append(("gdn", L))

    # 1: weight array. head c, row r holds W[r][c].  write: key=onehot(r) from addr, v=wval, beta=isWVAL&&col==c
    L = GDNLayer(H, N, N, 1)
    for c in range(N):
        for r in range(N):
            L.Wk[c * N + r, D["addr"] + r] = BIG
            L.Wq[c * N + r, D["idx"] + r] = BIG
        L.Wv[c, D["wval"]] = 1.0
        L.Wb[c, D["isWVAL"]] = 2 * ON; L.Wb[c, D["addr"] + N + c] = 2 * ON; L.Wb[c, D["one"]] = -3 * ON
        L.Wa[c, D["one"]] = -ON
        L.Wo[D["rowvec"] + c, c] = 4.0           # undo q scale dk**-0.5 = 1/4
    layers.append(("gdn", L))

    # 2: MAC: prod_c = val * rowvec_c  =  val*relu(r) - val*relu(-r)
    M = SwiGLU(H, 2 * N)
    for c in range(N):
        for s, sign in ((0, 1), (1, -1)):
            n = 2 * c + s
            M.Wg[n, D["rowvec"] + c] = sign * G
            M.Wu[n, D["val"]] = 1.0
            M.Wd[D["prod"] + c, n] = sign / G
    layers.append(("mlp", M))

    # 3: accumulators. ACT(i,v): head c row i <- prod_c.  FLUSH: alpha=0.  READ(c): q = ones on head c only.
    L = GDNLayer(H, N, N, 1)
    for c in range(N):
        for r in range(N):
            L.Wk[c * N + r, D["idx"] + r] = BIG
            L.Wq[c * N + r, D["colsel"] + c] = BIG
        L.Wv[c, D["prod"] + c] = 1.0
        L.Wb[c, D["isACT"]] = 2 * ON; L.Wb[c, D["one"]] = -ON
        L.Wa[c, D["isFLUSH"]] = 2000.0; L.Wa[c, D["one"]] = -ON
        L.Wo[D["y"], c] = float(N)               # q = ones/sqrt(N) * dk**-.5 = ones/16 -> times 16
    layers.append(("gdn", L))

    # 4: ReLU on READR:  y += isR * relu(-y)
    M = SwiGLU(H, 1)
    M.Wg[0, D["y"]] = -G; M.Wu[0, D["isR"]] = 1.0; M.Wd[D["y"], 0] = 1 / G
    layers.append(("mlp", M))

    # 5: saturate to [-128,127]:  y -= relu(y-127);  y += relu(-128-y)
    M = SwiGLU(H, 2)
    M.Wg[0, D["y"]] = G; M.Wg[0, D["one"]] = -G * 127; M.Wu[0, D["one"]] = 1.0; M.Wd[D["y"], 0] = -1 / G
    M.Wg[1, D["y"]] = -G; M.Wg[1, D["one"]] = -G * 128; M.Wu[1, D["one"]] = 1.0; M.Wd[D["y"], 1] = 1 / G
    layers.append(("mlp", M))

    # 6: one-hot of integer y:  oh[v] = relu(y-v+1) - 2 relu(y-v) + relu(y-v-1)
    M = SwiGLU(H, 3 * 256)
    for i in range(256):
        v = i - 128
        for j, (off, co) in enumerate(((1, 1), (0, -2), (-1, 1))):
            n = 3 * i + j
            M.Wg[n, D["y"]] = G; M.Wg[n, D["one"]] = G * (off - v); M.Wu[n, D["one"]] = 1.0
            M.Wd[D["oh"] + i, n] = co / G
    layers.append(("mlp", M))

    # vocab -> embedding rows
    vocab, emb = {}, []

    def tok(name, **dims):
        e = torch.zeros(H, dtype=DT)
        for k, val in dims.items():
            if isinstance(val, tuple):
                e[D[k] + val[0]] = val[1]
            else:
                e[D[k]] = val
        vocab[name] = len(emb); emb.append(e)

    for a in range(N * N):
        e = {"isADDR": 1.0, "one": 1.0}
        tok(("ADDR", a), eaddr=(a // N, 1.0), **{"one": 1.0, "isADDR": 1.0})
        emb[-1][D["eaddr"] + N + a % N] = 1.0
    for w in range(-128, 128):
        tok(("WVAL", w), wval=float(w), isWVAL=1.0, one=1.0)
    for i in range(N):
        for v in range(-128, 128):
            tok(("ACT", i, v), idx=(i, 1.0), val=float(v), isACT=1.0, one=1.0)
    for c in range(N):
        tok(("READ", c), colsel=(c, 1.0), isREAD=1.0, one=1.0)
        tok(("READR", c), colsel=(c, 1.0), isREAD=1.0, isR=1.0, one=1.0)
    tok(("FLUSH",), isFLUSH=1.0, one=1.0)
    E = torch.stack(emb)
    lm = torch.zeros(len(emb), H, dtype=DT)
    for v in range(-128, 128):
        lm[vocab[("WVAL", v)], D["oh"] + v + 128] = 1.0
    return layers, E, lm, vocab, D


class QwenTPUModel:
    """The compiled model. step(token) -> argmax token, carrying DeltaNet states between calls."""

    def __init__(self):
        self.layers, self.E, self.lm, self.vocab, self.D = build()
        self.inv = {v: k for k, v in self.vocab.items()}
        self.reset()

    def reset(self):
        self.states = [l.init_state() if k == "gdn" else None for k, l in self.layers]
        self.n_tokens = 0

    def step(self, name):
        h = self.E[self.vocab[name]].clone()
        for i, (kind, l) in enumerate(self.layers):
            if kind == "gdn":
                d, self.states[i] = l(h, self.states[i]); h = h + d
            else:
                h = h + l(h)
        logits = self.lm @ h
        self.n_tokens += 1
        self.last_logits = logits
        return self.inv[int(logits.argmax())]

    def n_params(self):
        n = self.E.numel() + self.lm.numel()
        for k, l in self.layers:
            n += sum(t.numel() for t in vars(l).values() if isinstance(t, torch.Tensor))
        return n
