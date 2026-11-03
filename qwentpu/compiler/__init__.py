"""qwentpu's own circuit compiler: IR -> Qwen3.5 Gated DeltaNet + SwiGLU weights.

Design credit: Qwenputer (Agrawal, 2026) showed that a Qwen3.5 residual stream with a huge
"ballast" makes RMSNorm a known constant, that DeltaNet state rows are exact RAM when keys are
one-hot and beta/alpha are saturated, and that silu(128*n) = 128*relu(n) on integers.  This package
re-derives those facts from the HF modeling code and implements them independently.
"""
