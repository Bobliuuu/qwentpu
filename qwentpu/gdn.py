"""Gated DeltaNet layer with Qwen3.5 semantics (single-token recurrent form), plus SwiGLU MLP.

Per head, state S is [dk, dv].  For each token:
    q, k = l2norm(Wq h), l2norm(Wk h);  q *= dk**-0.5
    beta = sigmoid(Wb h);  alpha = exp(-exp(A_log) * softplus(Wa h + dt_bias))
    S   <- alpha * S
    S   <- S + k (beta * (v - S^T k))^T          # delta rule
    o    = S^T q
With alpha=1, beta=1 and one-hot k this overwrites row(k) with v (RAM write); a one-hot q reads it
back; a *dense* q returns q @ S, a matrix-vector product (the TPU's systolic array).
Omitted vs the real HF layer: causal conv1d, output RMSNorm gate, input RMSNorm (see WRITEUP.md).
"""
import torch
import torch.nn.functional as F

DT = torch.float64


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class GDNLayer:
    def __init__(self, H, nh, dk, dv):
        self.H, self.nh, self.dk, self.dv = H, nh, dk, dv
        z = lambda *s: torch.zeros(*s, dtype=DT)
        self.Wq, self.Wk, self.Wv = z(nh * dk, H), z(nh * dk, H), z(nh * dv, H)
        self.Wb, self.Wa = z(nh, H), z(nh, H)
        self.A_log, self.dt_bias = z(nh), z(nh)
        self.Wo = z(H, nh * dv)

    def init_state(self):
        return torch.zeros(self.nh, self.dk, self.dv, dtype=DT)

    def __call__(self, h, S):
        nh, dk, dv = self.nh, self.dk, self.dv
        q = l2norm((self.Wq @ h).view(nh, dk)) * dk ** -0.5
        k = l2norm((self.Wk @ h).view(nh, dk))
        v = (self.Wv @ h).view(nh, dv)
        beta = torch.sigmoid(self.Wb @ h)
        g = -torch.exp(self.A_log) * F.softplus(self.Wa @ h + self.dt_bias)
        S = S * torch.exp(g)[:, None, None]
        pred = torch.einsum("hkv,hk->hv", S, k)
        S = S + torch.einsum("hk,hv->hkv", k, beta[:, None] * (v - pred))
        o = torch.einsum("hkv,hk->hv", S, q)
        return self.Wo @ o.reshape(-1), S


class SwiGLU:
    """down(silu(gate h) * up h) -- the FFN. Neurons are compiled as unit terms coef*a*relu(b)."""

    def __init__(self, H, n):
        self.Wg = torch.zeros(n, H, dtype=DT)
        self.Wu = torch.zeros(n, H, dtype=DT)
        self.Wd = torch.zeros(H, n, dtype=DT)

    def __call__(self, h):
        return self.Wd @ (F.silu(self.Wg @ h) * (self.Wu @ h))
