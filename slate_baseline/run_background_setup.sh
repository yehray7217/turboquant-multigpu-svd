#!/usr/bin/env bash
set -eo pipefail

BASE_DIR="/home/rax10101010/hpc_final/turboquant-multigpu-svd/slate_baseline"
LOG_FILE="${BASE_DIR}/background_setup.log"
PID_FILE="${BASE_DIR}/background_setup.pid"

cd "${BASE_DIR}"

echo "[$(date '+%F %T')] start background setup" >> "${LOG_FILE}"

# Clean potentially corrupted previous SLATE build objects.
rm -rf ../third_party/slate-build

./install_slate_local.sh >> "${LOG_FILE}" 2>&1
source ./env_taiwania2.sh >> "${LOG_FILE}" 2>&1

make clean >> "${LOG_FILE}" 2>&1 || true
make -j >> "${LOG_FILE}" 2>&1
make doctor >> "${LOG_FILE}" 2>&1

echo "[$(date '+%F %T')] background setup done" >> "${LOG_FILE}"

rm -f "${PID_FILE}"
