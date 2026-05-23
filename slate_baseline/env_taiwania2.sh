#!/usr/bin/env bash
# Source this file to prepare Taiwania2 build/run environment for SLATE baseline.
# Usage:
#   source ./env_taiwania2.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script instead of executing it:" >&2
  echo "  source ./env_taiwania2.sh" >&2
  exit 1
fi

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module purge >/dev/null 2>&1 || true
module load cuda/12.8
module load ucx/1.14.1
module load cmake/3.23.2
module load git/2.46.2
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

# Some OpenMPI modulefiles force CUDA-aware MPI flags that may not match
# the actual OpenMPI build on this node; disable to avoid startup failure.
unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

# Use MKL runtime directly (without loading full intel module, which can
# override MPI/UCX paths and break OpenMPI linking).
DEFAULT_MKLROOT="/opt/ohpc/twcc/intel/2020/update1/compilers_and_libraries_2020.1.217/linux/mkl"
if [[ -z "${MKLROOT:-}" && -d "${DEFAULT_MKLROOT}" ]]; then
  export MKLROOT="${DEFAULT_MKLROOT}"
fi
if [[ -n "${MKLROOT:-}" && -d "${MKLROOT}/lib/intel64" ]]; then
  export LIBRARY_PATH="${MKLROOT}/lib/intel64:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${MKLROOT:-}" && -d "${MKLROOT}/bin/pkgconfig" ]]; then
  export PKG_CONFIG_PATH="${MKLROOT}/bin/pkgconfig:${PKG_CONFIG_PATH:-}"
fi

SLATE_BASELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLATE_INSTALL_PREFIX="${SLATE_INSTALL_PREFIX:-${SLATE_BASELINE_DIR}/../third_party/slate-install}"
export SLATE_INSTALL_PREFIX

if [[ -d "${SLATE_INSTALL_PREFIX}/lib/pkgconfig" ]]; then
  export PKG_CONFIG_PATH="${SLATE_INSTALL_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
if [[ -d "${SLATE_INSTALL_PREFIX}/lib64/pkgconfig" ]]; then
  export PKG_CONFIG_PATH="${SLATE_INSTALL_PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
if [[ -d "${SLATE_INSTALL_PREFIX}/lib64" ]]; then
  export LD_LIBRARY_PATH="${SLATE_INSTALL_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${SLATE_INSTALL_PREFIX}/lib64:${LIBRARY_PATH:-}"
fi
if [[ -d "${SLATE_INSTALL_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${SLATE_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="${SLATE_INSTALL_PREFIX}/lib:${LIBRARY_PATH:-}"
fi
if [[ -d "${SLATE_INSTALL_PREFIX}/bin" ]]; then
  export PATH="${SLATE_INSTALL_PREFIX}/bin:${PATH}"
fi

echo "[env] CUDA_HOME=${CUDA_HOME:-unset}"
echo "[env] MKLROOT=${MKLROOT:-unset}"
echo "[env] mpirun=$(command -v mpirun || echo missing)"
echo "[env] mpicxx=$(command -v mpicxx || echo missing)"
echo "[env] pkg-config slate=$(pkg-config --exists slate && echo yes || echo no)"
