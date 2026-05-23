#!/usr/bin/env bash
set -eo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="${BASE_DIR}/background_setup.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PID file. Nothing to stop."
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if ps -p "${pid}" >/dev/null 2>&1; then
  kill "${pid}"
  echo "Stopped PID=${pid}"
else
  echo "Process already not running: PID=${pid}"
fi

rm -f "${PID_FILE}"
