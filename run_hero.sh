#!/usr/bin/env bash
# Reproduce the LEDGAR smoke run: base eval -> fine-tune -> fine-tuned eval.
# Usage: bash run_hero.sh qlora  # 24GB+ CUDA GPU
#        bash run_hero.sh bf16   # DGX Spark or 80GB+ accelerator
set -euo pipefail
trap 'echo "=== FAILED at line $LINENO ==="' ERR

MODE="${1:-qlora}"
PYTHON="${PYTHON:-python3}"
TRAIN_ARGS=()

case "$MODE" in
  qlora)
    BATCH=4
    GRAD_ACCUM=2
    ;;
  bf16)
    BATCH=16
    GRAD_ACCUM=1
    TRAIN_ARGS+=(--no-4bit)
    ;;
  *)
    echo "usage: bash run_hero.sh [qlora|bf16]" >&2
    exit 2
    ;;
esac

echo "=== base eval (no training, with the 100-label list) ==="
"$PYTHON" eval_local.py --dataset ledgar --limit 300 --name "base SmolLM3-3B"

echo "=== fine-tune (4000 examples, 2 epochs, no label list in the prompt) ==="
TRAIN_ARGS+=(
  --dataset ledgar
  --train-size 4000
  --epochs 2
  --batch "$BATCH"
  --grad-accum "$GRAD_ACCUM"
  --max-length 1536
  --out adapters/ledgar-smollm3-3b
)
"$PYTHON" train_lora.py "${TRAIN_ARGS[@]}" 2>&1 | tee hero.log

echo "=== fine-tuned eval ==="
"$PYTHON" eval_local.py --dataset ledgar --adapter adapters/ledgar-smollm3-3b \
  --limit 300 --name "fine-tuned SmolLM3-3B (ours)"

echo "=== done. add the frontier baselines with eval_frontier.py, then run report.py ==="
