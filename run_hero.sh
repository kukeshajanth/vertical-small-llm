#!/usr/bin/env bash
# Reproduce the hero result on LEDGAR: base eval -> fine-tune -> fine-tuned eval.
# On a 24GB consumer GPU, remove --no-4bit to train in 4-bit (QLoRA).
set -eo pipefail
trap 'echo "=== FAILED at line $LINENO ==="' ERR

echo "=== base eval (no training, with the 100-label list) ==="
python eval_local.py --dataset ledgar --limit 300 --name "base SmolLM3-3B"

echo "=== fine-tune (4000 examples, 2 epochs, no label list in the prompt) ==="
python train_lora.py --dataset ledgar --no-4bit --train-size 4000 --epochs 2 \
  --batch 16 --grad-accum 1 --max-length 1536 --out adapters/ledgar-smollm3-3b

echo "=== fine-tuned eval ==="
python eval_local.py --dataset ledgar --adapter adapters/ledgar-smollm3-3b \
  --limit 300 --name "fine-tuned SmolLM3-3B (ours)"

echo "=== done. add the frontier baselines with eval_frontier.py, then run report.py ==="
