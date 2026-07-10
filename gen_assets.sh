#!/usr/bin/env bash
# Optional: regenerate the figures in assets/ from a completed run.
# Expects a fine-tuned adapter at adapters/ledgar-smollm3-3b (run run_hero.sh first).
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "=== base @300 (dumps predictions to runs/) ==="
"$PYTHON" eval_local.py --dataset ledgar --limit 300 --name "base-300p"

echo "=== fine-tuned @300 (dumps predictions to runs/) ==="
"$PYTHON" eval_local.py --dataset ledgar --adapter adapters/ledgar-smollm3-3b --limit 300 --name "ft-300p"

echo "=== charts ==="
"$PYTHON" gen_charts.py

echo "=== assets done ==="
