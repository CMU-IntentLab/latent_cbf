#!/usr/bin/env bash
# Grid search over zs_weight × relu_weight for dreamer_offline WM training.
# Usage: from repo root:  bash src/latent_cbf/scripts/hp_ablation.sh
#    or from src/latent_cbf:  bash scripts/hp_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATENT_CBF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LATENT_CBF_ROOT}"

if [ ! -f "scripts/dreamer_offline.py" ]; then
  echo "Error: scripts/dreamer_offline.py not found (expected under ${LATENT_CBF_ROOT})."
  exit 1
fi

zs_weights=(0.05 0.1 0.2)
relu_weights=(0.5 1 2)

for zs_weight in "${zs_weights[@]}"; do
  for relu_weight in "${relu_weights[@]}"; do
    echo "============================================================"
    echo "Running zs_weight=${zs_weight} relu_weight=${relu_weight}"
    echo "============================================================"
    python scripts/dreamer_offline.py \
      --zs_weight "${zs_weight}" \
      --relu_weight "${relu_weight}" \
      --gp_weight 20.0
  done
done

echo "All ${#zs_weights[@]}×${#relu_weights[@]} runs finished."
