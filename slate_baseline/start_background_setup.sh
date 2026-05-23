#!/usr/bin/env bash
set -eo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="${BASE_DIR}/background_setup.pid"
LOG_FILE="${BASE_DIR}/background_setup.log"

cd "${BASE_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if ps -p "${old_pid}" >/dev/null 2>&1; then
    echo "Background setup already running. PID=${old_pid}"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
fi

nohup ./run_background_setup.sh >/dev/null 2>&1 < /dev/null &
new_pid=$!
echo "${new_pid}" > "${PID_FILE}"

echo "Started background setup. PID=${new_pid}"
echo "Log: ${LOG_FILE}"
