#!/usr/bin/env bash
set -euo pipefail
port="${1:-6006}"
: "${OPENPI_ROOT:?Set OPENPI_ROOT to the OpenPI 15a9616 checkout}"
: "${SFT_CHECKPOINT_DIR:?Set SFT_CHECKPOINT_DIR to the 20k checkpoint directory}"
export PYTHONPATH="${PWD}:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
python -m sft.serve_policy_fixed_noise \
  --config pi05_so101_pickplace_v3 \
  --checkpoint-dir "${SFT_CHECKPOINT_DIR}" --port "${port}" --noise-seed 0
