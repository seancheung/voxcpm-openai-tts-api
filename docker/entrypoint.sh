#!/usr/bin/env bash
set -euo pipefail

: "${VOXCPM_MODEL:=openbmb/VoxCPM2}"
: "${VOXCPM_VOICES_DIR:=/voices}"
: "${VOXCPM_DEVICE:=auto}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

export VOXCPM_MODEL VOXCPM_VOICES_DIR VOXCPM_DEVICE HOST PORT LOG_LEVEL

if [ "$#" -eq 0 ]; then
  exec uvicorn app.server:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"
fi
exec "$@"
