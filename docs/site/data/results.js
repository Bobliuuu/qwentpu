window.RESULTS = {
 "_note": "Data-driven: every chart/number reads this file. Run docs/site/sync.sh after editing.",
 "headline": {
  "outputs_replayed": 1139712,
  "mismatches": 0,
  "peak_macs": 1024,
  "peak_macs_per_token": 16.0,
  "text_kl_after_chip": 0.0022,
  "a100_peak_tok_s": 45371.0
 },
 "arrays": [
  {
   "id": "8x8",
   "label": "8×8 u3·s3",
   "kr": 8,
   "nc": 8,
   "ab": 3,
   "wb": 3,
   "macs": 64,
   "tokens": 16,
   "macs_per_token": 4.0,
   "chip_layers": 2,
   "verified": "0.8B · 2B · 4B"
  },
  {
   "id": "8x8-u4s4",
   "label": "8×8 u4·s4",
   "kr": 8,
   "nc": 8,
   "ab": 4,
   "wb": 4,
   "macs": 64,
   "tokens": 16,
   "macs_per_token": 4.0,
   "chip_layers": 4,
   "verified": "0.8B · 2B · 4B"
  },
  {
   "id": "8x8-u5s5",
   "label": "8×8 u5·s5",
   "kr": 8,
   "nc": 8,
   "ab": 5,
   "wb": 5,
   "macs": 64,
   "tokens": 16,
   "macs_per_token": 4.0,
   "chip_layers": 7,
   "verified": "0.8B · 2B · 4B"
  },
  {
   "id": "16x16-u3s2",
   "label": "16×16 u3·s2",
   "kr": 16,
   "nc": 16,
   "ab": 3,
   "wb": 2,
   "macs": 256,
   "tokens": 32,
   "macs_per_token": 8.0,
   "chip_layers": 2,
   "verified": "0.8B · 2B · 4B"
  },
  {
   "id": "16x16-u4s4",
   "label": "16×16 u4·s4",
   "kr": 16,
   "nc": 16,
   "ab": 4,
   "wb": 4,
   "macs": 256,
   "tokens": 32,
   "macs_per_token": 8.0,
   "chip_layers": 5,
   "verified": "0.8B · 2B · 4B"
  },
  {
   "id": "32x32",
   "label": "32×32 u3·s3",
   "kr": 32,
   "nc": 32,
   "ab": 3,
   "wb": 3,
   "macs": 1024,
   "tokens": 64,
   "macs_per_token": 16.0,
   "chip_layers": 6,
   "verified": "0.8B · 2B · 4B"
  }
 ],
 "speed": [
  {
   "model": "0.8B",
   "array": "8×8",
   "cpu": null,
   "a100_b1": 1996.7,
   "a100_b16": 33074.0
  },
  {
   "model": "0.8B",
   "array": "16×16 u4·s4",
   "cpu": null,
   "a100_b1": 5620.0,
   "a100_b16": 36362.7
  },
  {
   "model": "0.8B",
   "array": "32×32",
   "cpu": null,
   "a100_b1": 15481.7,
   "a100_b16": 36478.0
  },
  {
   "model": "2B",
   "array": "8×8",
   "cpu": null,
   "a100_b1": 2097.5,
   "a100_b16": 28400.9
  },
  {
   "model": "2B",
   "array": "16×16 u4·s4",
   "cpu": null,
   "a100_b1": 5372.9,
   "a100_b16": 26059.0
  },
  {
   "model": "2B",
   "array": "32×32",
   "cpu": null,
   "a100_b1": 16821.0,
   "a100_b16": 25267.2
  },
  {
   "model": "4B",
   "array": "8×8",
   "cpu": null,
   "a100_b1": 1483.9,
   "a100_b16": 13772.7
  },
  {
   "model": "4B",
   "array": "16×16 u4·s4",
   "cpu": null,
   "a100_b1": 4347.1,
   "a100_b16": 12728.4
  },
  {
   "model": "4B",
   "array": "32×32",
   "cpu": null,
   "a100_b1": 10455.5,
   "a100_b16": 12361.3
  }
 ],
 "restore": [
  {
   "model": "0.8B",
   "muting_first": 1.316,
   "muting_worst": null,
   "restore_first": 0.0019,
   "restore_worst": 0.0022
  },
  {
   "model": "2B",
   "muting_first": 1.219,
   "muting_worst": null,
   "restore_first": 0.0004,
   "restore_worst": 0.0018
  },
  {
   "model": "4B",
   "muting_first": 1.098,
   "muting_worst": null,
   "restore_first": 0.0005,
   "restore_worst": 0.0019
  }
 ],
 "models": [
  {
   "model": "0.8B",
   "layers": "24 + 2",
   "hidden": "1024 → 1280",
   "heads": "16 / 16",
   "text_kl": "4.6e-4 – 6.9e-4",
   "mismatches": "0 / 131328"
  },
  {
   "model": "2B",
   "layers": "24 + 2",
   "hidden": "2048 → 2304",
   "heads": "16 / 16",
   "text_kl": "2.5e-4 – 6.0e-4",
   "mismatches": "0 / 131328"
  },
  {
   "model": "4B",
   "layers": "32 + 2",
   "hidden": "2560 → 2816",
   "heads": "32 / 16",
   "text_kl": "3.1e-4 – 5.4e-4",
   "mismatches": "0 / 131328"
  }
 ],
 "hardware": [
  {
   "device": "A100 40GB",
   "kind": "gpu",
   "arch": "Ampere sm80",
   "models_tested": [
    "0.8B",
    "2B",
    "4B"
   ],
   "arrays_tested": [
    "16x16 u3s2",
    "16x16 u4s4",
    "16x16 u5s5",
    "32x32 u3s3",
    "32x32 u4s4",
    "64x64 u3s3",
    "8x8 u3s3",
    "8x8 u4s4",
    "8x8 u5s5"
   ],
   "replay_outputs": 705024,
   "mismatches": 0,
   "suite_note": "18 of 18 test-suite runs fully passed"
  },
  {
   "device": "A10G 24GB",
   "kind": "gpu",
   "arch": "Ampere sm86",
   "models_tested": [
    "0.8B",
    "2B"
   ],
   "arrays_tested": [
    "16x16 u3s2",
    "16x16 u4s4",
    "16x16 u5s5",
    "32x32 u3s3",
    "32x32 u4s4",
    "8x8 u3s3",
    "8x8 u4s4",
    "8x8 u5s5"
   ],
   "replay_outputs": 221184,
   "mismatches": 0,
   "suite_note": "12 of 12 test-suite runs fully passed"
  },
  {
   "device": "L4 24GB",
   "kind": "gpu",
   "arch": "Ada sm89",
   "models_tested": [
    "0.8B",
    "2B"
   ],
   "arrays_tested": [
    "16x16 u3s2",
    "16x16 u4s4",
    "16x16 u5s5",
    "32x32 u3s3",
    "32x32 u4s4",
    "8x8 u3s3",
    "8x8 u4s4",
    "8x8 u5s5"
   ],
   "replay_outputs": 190464,
   "mismatches": 0,
   "suite_note": "12 of 12 test-suite runs fully passed"
  },
  {
   "device": "T4 16GB",
   "kind": "gpu",
   "arch": "Turing sm75",
   "models_tested": [
    "0.8B",
    "2B"
   ],
   "arrays_tested": [],
   "replay_outputs": null,
   "mismatches": null,
   "suite_note": "9 of 12 test-suite runs fully passed"
  },
  {
   "device": "x86 Xeon (c6i, Ice Lake)",
   "kind": "cpu",
   "arch": "x86-64",
   "models_tested": [
    "0.8B",
    "2B",
    "4B"
   ],
   "arrays_tested": [
    "16x16 u3s2",
    "16x16 u4s4",
    "16x16 u5s5",
    "32x32 u3s3",
    "32x32 u4s4",
    "8x8 u3s3",
    "8x8 u4s4",
    "8x8 u5s5"
   ],
   "replay_outputs": 23040,
   "mismatches": 0,
   "suite_note": "14 of 14 test-suite runs fully passed"
  }
 ],
 "packing": [
  {
   "array": "8×8 u3·s3",
   "g": 1,
   "h": 1,
   "layers": 2,
   "tokens_per_matvec": 16,
   "macs_per_token": 4.0,
   "tok_s_b16": null,
   "macs_s_b16": null,
   "pending": true
  },
  {
   "array": "8×8 u3·s3",
   "g": 2,
   "h": 2,
   "layers": 2,
   "tokens_per_matvec": 8,
   "macs_per_token": 8.0,
   "tok_s_b16": null,
   "macs_s_b16": null,
   "pending": true
  },
  {
   "array": "8×8 u3·s3",
   "g": 4,
   "h": 4,
   "layers": 4,
   "tokens_per_matvec": 4,
   "macs_per_token": 16.0,
   "tok_s_b16": null,
   "macs_s_b16": null,
   "pending": true
  },
  {
   "array": "16×16 u3·s2",
   "g": 1,
   "h": 1,
   "layers": 2,
   "tokens_per_matvec": 32,
   "macs_per_token": 8.0,
   "tok_s_b16": null,
   "macs_s_b16": null,
   "pending": true
  },
  {
   "array": "16×16 u3·s2",
   "g": 4,
   "h": 4,
   "layers": 8,
   "tokens_per_matvec": 8,
   "macs_per_token": 32.0,
   "tok_s_b16": null,
   "macs_s_b16": null,
   "pending": true
  },
  {
   "array": "32×32 u3·s3",
   "g": 1,
   "h": 1,
   "layers": 6,
   "tokens_per_matvec": 64,
   "macs_per_token": 16.0,
   "tok_s_b16": null,
   "macs_s_b16": null,
   "pending": true
  },
  {
   "array": "32×32 u3·s3",
   "g": 4,
   "h": 4,
   "layers": 18,
   "tokens_per_matvec": 16,
   "macs_per_token": 64.0,
   "tok_s_b16": null,
   "macs_s_b16": null,
   "pending": true
  }
 ],
 "chip_path": {
  "model": "0.8B",
  "array": "8×8",
  "b1_full": 1997,
  "b1_chip": 24030,
  "b16_full": 33074,
  "b16_chip": 125208,
  "source": "A100: results/a100/final.jsonl (full) and sweep.jsonl (tpu-layers)"
 }
};
