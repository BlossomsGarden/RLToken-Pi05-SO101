#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pidfile=outputs/rlt_online/brain.pid
if [[ ! -f "$pidfile" ]]; then
  echo "No brain PID file" >&2
  exit 1
fi
pid=$(cat "$pidfile")
kill -TERM "$pid"
for ((attempt=0; attempt<60; attempt++)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pidfile"
    echo "Brain stopped cleanly; checkpoint saved"
    exit 0
  fi
  sleep 1
done
echo "Brain did not exit within 60 seconds; inspect logs before resuming" >&2
exit 1
