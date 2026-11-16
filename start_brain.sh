#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs/rlt_online
if [[ -f outputs/rlt_online/brain.pid ]] && kill -0 "$(cat outputs/rlt_online/brain.pid)" 2>/dev/null; then
  echo "Brain is already running" >&2
  exit 1
fi
config=configs/rlt_online.json
if [[ $# -gt 0 ]]; then config="$1"; fi
nohup python -m rlt_online.serve_rlt_brain --config "$config" >outputs/rlt_online/brain.log 2>&1 &
echo "$!" > outputs/rlt_online/brain.pid
echo "Brain PID: $!; inspect outputs/rlt_online/brain.log and run the health client"
