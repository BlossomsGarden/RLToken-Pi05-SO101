#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
port="${1:-6006}"
: "${OPENPI_ROOT:?Set OPENPI_ROOT to the OpenPI checkout}"
: "${SFT_CHECKPOINT_DIR:?Set SFT_CHECKPOINT_DIR to the SFT checkpoint}"
export PYTHONPATH="${PWD}:${OPENPI_ROOT}/src:${PYTHONPATH:-}"
args=(--config "${SFT_CONFIG:-pi05_so101_pickplace_v3}" --checkpoint-dir "${SFT_CHECKPOINT_DIR}" --port "$port" --noise-seed "${NOISE_SEED:-0}")
if [[ "${NEW_SO101:-0}" == 1 ]]; then
    : "${MAPPING_FILE:?Set MAPPING_FILE to the verified joint mapping JSON}"
    args+=(--new-so101 --mapping-file "$MAPPING_FILE")
fi
python -m sft.serve_policy_fixed_noise "${args[@]}"
