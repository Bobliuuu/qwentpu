window.RESULTS = {
 "_note": "Data-driven: every chart/number reads this file. Run docs/site/sync.sh after editing.",
 "headline": {
  "outputs_replayed": 393984,
  "mismatches": 0,
  "peak_macs": 1024,
  "peak_macs_per_token": 16.0,
  "text_kl_after_chip": 0.0047,
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
   "muting_first": 1.31,
   "muting_worst": null,
   "restore_first": 0.002,
   "restore_worst": 0.0031
  },
  {
   "model": "2B",
   "muting_first": 1.19,
   "muting_worst": null,
   "restore_first": 0.0008,
   "restore_worst": 0.0017
  },
  {
   "model": "4B",
   "muting_first": 1.14,
   "muting_worst": 0.61,
   "restore_first": 0.0014,
   "restore_worst": 0.0047
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
 ]
};
