#!/bin/bash
# Verify every array on every model: the model suite and a mass replay against NumPy.
# usage: ./run_matrix.sh <dir with q08 q2 q4 checkpoints> [device=cuda]   -> results.jsonl
root=${1:?checkpoint dir}; dev=${2:-cuda}; export PYTHONPATH=.; : > results.jsonl
for d in q08 q2 q4; do
  for cfg in 8,8,3,3 8,8,4,4 8,8,5,5 16,16,3,2 16,16,4,4 32,32,3,3; do
    QWENTPU_CFG=$cfg QWEN_DIR=$root/$d QWEN_DEVICE=$dev python -m pytest -q tests/test_merged.py || echo "SUITE FAILED $d $cfg"
    for b in 1 16; do
      QWENTPU_CFG=$cfg QWEN_DIR=$root/$d python bench.py $((b == 1 ? 6 : 48)) --device $dev --batch $b | grep '^{' >> results.jsonl
    done
  done
done
