#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv-cu128/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

DATA_DIR="${DATA_DIR:-data}"
RUN_DIR="${RUN_DIR:-runs/tinystories_baseline}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

BATCH_SIZE="${BATCH_SIZE:-128}"
STEPS="${STEPS:-10000}"
EVAL_ITERS="${EVAL_ITERS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
MAX_LR="${MAX_LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SEED="${SEED:-42}"

"$PYTHON" scripts/train_lm.py \
  --train-data "$DATA_DIR/tinystories_train.npy" \
  --val-data "$DATA_DIR/tinystories_val.npy" \
  --out-dir "$RUN_DIR" \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --d-ff 1344 \
  --num-layers 4 \
  --num-heads 16 \
  --rope-theta 10000 \
  --batch-size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --eval-iters "$EVAL_ITERS" \
  --eval-interval "$EVAL_INTERVAL" \
  --log-interval "$LOG_INTERVAL" \
  --checkpoint-interval "$CHECKPOINT_INTERVAL" \
  --max-lr "$MAX_LR" \
  --min-lr "$MIN_LR" \
  --warmup-iters "$WARMUP_ITERS" \
  --weight-decay "$WEIGHT_DECAY" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED"
