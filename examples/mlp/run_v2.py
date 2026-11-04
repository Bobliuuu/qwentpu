#!/usr/bin/env python3
"""Run the golden digit MLP (15 -> 15 -> 10, 5-bit integer weights) as a v2 *program in the prompt*.

    python examples/mlp/run_v2.py                    # CPU simulator, 5 digits (seconds, numpy only)
    python examples/mlp/run_v2.py -n 20 -v           # all 20 golden digits, print every emitted token
    python examples/mlp/run_v2.py --ir 1             # also run the compiled circuit IR (float64, CPU, ~1 min per digit)
    QWEN_DIR=q08 python examples/mlp/run_v2.py --cuda   # optional: the real Qwen3.5-0.8B with the chip layers merged in (needs CUDA)

What happens: the host builds ONE prompt (the weights as CELL tokens, the input as ACT tokens, the entry point chosen by where the input
is written) and then runs `argmax; append` until the model emits HALT. Every READ, ACT and OUTK token in between is the model's own.
The logits are the OUTK values; the prediction is their argmax. The result is checked against the integer golden model (examples/mlp/golden.npz).
"""
import argparse, os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from qwentpu import circuit as C
from qwentpu.sequencer2 import ProgramMachine, run_program


class SimFeed:
    """CPU simulator of the compiled model's *behaviour*: for every fed token it returns the next token the model emits.
    It keeps the accumulator rows and the weight array like the DeltaNet state does, and applies the compiled successor table plus
    relu + clamp at READ. Same interface as TpuSession.feed, so the host loop (`generate`) is identical for every backend."""
    def __init__(self, seq):
        self.seq = seq; self.W = np.zeros((C.KR, C.NC), dtype=int); self.a = np.zeros(C.KR, dtype=int)

    def feed(self, toks):
        s, out = self.seq, []
        for t in toks:
            if C.CELL0 <= t < C.ACT0:                                    # CELL(r, c, w): weight write
                rc, w = divmod(t - C.CELL0, C.NW); r, c = divmod(rc, C.NC)
                self.W[r, c] = w - C.NW if w >= C.NW // 2 else w
            elif C.ACT0 <= t < C.READ0:                                  # ACT(row; v): activation write
                self.a[(t - C.ACT0) // C.NOUT] = (t - C.ACT0) % C.NOUT
            k = s.read_index(t)
            out.append(s.base(k) + int(np.clip(max(self.a @ self.W[:, k], 0), 0, C.AMAX)) if k is not None else s.succ(t))
        return out


def ir_feed_factory(seq):
    """The compiled circuit IR (float64, every bf16 exactness condition checked) as the model. Teacher-forced over the whole stream,
    which equals autoregressive decoding for a deterministic argmax. Returns a feed that replays the stream it is asked to predict."""
    from qwentpu.compiler import evaluate as ir
    g, d = C.build()
    def predict(stream):
        m = ir.evaluate(g, list(stream))
        return [sum((1 << i) * int(ir.ev(d["code"][i], m[t:t + 1])[0] > 0) for i in range(C.CODE)) for t in range(len(stream))]
    return predict


def cuda_feed(seq):
    """Optional: the real Qwen3.5 checkpoint with the chip layers appended (needs a GPU and QWEN_DIR)."""
    import torch
    from qwentpu.driver import TpuSession
    from qwentpu.merge import build_merged, instantiate
    src = os.environ.get("QWEN_DIR") or sys.exit("set QWEN_DIR to a Qwen3.5 checkpoint directory (hf download Qwen/Qwen3.5-0.8B --local-dir q08)")
    tc, sd, v0, lay, g = build_merged(src, mute_qwen=False)
    model = instantiate(tc, sd, "cuda")
    return lambda: TpuSession(model, v0, fast=True, readout="lm_head").feed        # a fresh session (empty state) per digit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=5, help="how many golden digits (max 20)")
    ap.add_argument("-v", action="store_true", help="print every emitted token")
    ap.add_argument("--ir", type=int, default=0, metavar="N", help="also check the first N digits against the circuit IR (slow)")
    ap.add_argument("--cuda", action="store_true", help="run on the real model (needs CUDA and QWEN_DIR)")
    args = ap.parse_args()

    z = np.load(os.path.join(ROOT, "examples", "mlp", "golden.npz"))
    W1, W2 = z["W1"], z["W2"]                                              # 16 x 15 and 16 x 10 (last row = bias)
    seq = ProgramMachine(2, 16, 5, 5).install()                            # Dmax = 2 blocks of width 16, 5-bit: a 32 x 32 array
    n_chip = C.build()[0].n_layers()
    print(f"compiled model: Dmax=2, width 16, u5s5 -> {n_chip} chip layers; float acc {float(z['float_acc']):.3f}, integer acc {float(z['int_acc']):.3f} (540 held-out digits)")
    mk_feed = cuda_feed(seq) if args.cuda else (lambda: SimFeed(seq).feed)
    ok = hit = 0
    for i in range(min(args.n, len(z["label"]))):
        x = z["x_test"][i]
        t0 = time.time()
        emitted, logits = run_program(mk_feed(), seq, [W1, W2], x)         # host: build prompt; argmax; append until HALT
        prompt_len = len(seq.program_prompt([W1, W2], x))
        pred = int(np.argmax(logits)); label = int(z["label"][i])
        exact = list(logits) == [int(t) for t in z["logits"][i][:10]] and pred == int(z["pred"][i])
        ok += exact; hit += pred == label
        print(f"digit {i:2d}: prompt {prompt_len} tokens, model emitted {len(emitted)} tokens, logits {list(map(int, logits))} -> predicted {pred}, label {label}"
              f"  {'match' if exact else 'MISMATCH'}  ({time.time() - t0:.2f}s)")
        if args.v:
            names = []
            for t in emitted:
                if C.READ0 <= t < C.READ0 + C.NREAD: names.append(f"READ({t - C.READ0})")
                elif C.ACT0 <= t < C.READ0: names.append(f"ACT({(t - C.ACT0) // C.NOUT};{(t - C.ACT0) % C.NOUT})")
                else: names.append(f"OUTK({(t - seq.outk0) // C.NOUT};{(t - seq.outk0) % C.NOUT})")
            print("   emitted:", " ".join(names), "HALT")
        if i < args.ir:
            pred_ir = ir_feed_factory(seq)(list(seq.program_prompt([W1, W2], x)) + list(emitted))
            n_p = prompt_len
            assert pred_ir[n_p - 1:] == emitted + [seq.halt], "circuit IR disagrees with the simulator"
            print(f"   circuit IR (float64) next-token at every emitted position: equal, ends in HALT")
    n = min(args.n, len(z["label"]))
    print(f"{ok}/{n} equal the integer golden logits; {hit}/{n} predicted digits equal the label")
    ProgramMachine.uninstall()
    sys.exit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
