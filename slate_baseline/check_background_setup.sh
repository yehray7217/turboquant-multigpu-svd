#!/usr/bin/env bash
set -eo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="${BASE_DIR}/background_setup.pid"
LOG_FILE="${BASE_DIR}/background_setup.log"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if ps -p "${pid}" -o pid,ppid,stat,etime,cmd >/dev/null 2>&1; then
    ps -p "${pid}" -o pid,ppid,stat,etime,cmd
  else
    echo "PID file exists but process is not running: ${pid}"
  fi
else
  echo "No PID file. Background setup is not running."
fi

echo "--- log tail ---"
tail -n 40 "${LOG_FILE}" 2>/dev/null || echo "No log yet."
