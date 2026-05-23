#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_taiwania2.sh"

THIRD_PARTY_DIR="${SCRIPT_DIR}/../third_party"
SLATE_SRC_DIR="${THIRD_PARTY_DIR}/slate-src"
SLATE_BUILD_DIR="${THIRD_PARTY_DIR}/slate-build"
SLATE_INSTALL_PREFIX="${SLATE_INSTALL_PREFIX:-${THIRD_PARTY_DIR}/slate-install}"
SLATE_GIT_REF="${SLATE_GIT_REF:-master}"

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${SLATE_SRC_DIR}/.git" ]]; then
  git clone --recursive https://github.com/icl-utk-edu/slate.git "${SLATE_SRC_DIR}"
else
  git -C "${SLATE_SRC_DIR}" fetch --all --tags
fi

git -C "${SLATE_SRC_DIR}" checkout "${SLATE_GIT_REF}"
git -C "${SLATE_SRC_DIR}" submodule update --init --recursive

mkdir -p "${SLATE_BUILD_DIR}" "${SLATE_INSTALL_PREFIX}"

export CC=mpicc
export CXX=mpicxx
export FC=mpifort

MKL_RT="${MKLROOT:-}/lib/intel64/libmkl_rt.so"
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${SLATE_INSTALL_PREFIX}"
  -DBUILD_SHARED_LIBS=ON
  -Dbuild_tests=OFF
  -Dgpu_backend=cuda
  -DCMAKE_CUDA_ARCHITECTURES=70
)

if [[ -f "${MKL_RT}" ]]; then
  CMAKE_ARGS+=(
    -DBLAS_LIBRARIES="${MKL_RT};-lpthread;-lm;-ldl"
    -DLAPACK_LIBRARIES="${MKL_RT};-lpthread;-lm;-ldl"
  )
else
  echo "[warn] MKL runtime not found at ${MKL_RT}, fallback to automatic BLAS/LAPACK detection."
fi

cmake -S "${SLATE_SRC_DIR}" -B "${SLATE_BUILD_DIR}" "${CMAKE_ARGS[@]}"

cmake --build "${SLATE_BUILD_DIR}" -j
cmake --install "${SLATE_BUILD_DIR}"

echo ""
echo "[done] SLATE installed to: ${SLATE_INSTALL_PREFIX}"
echo "[next] source ${SCRIPT_DIR}/env_taiwania2.sh"
echo "[next] cd ${SCRIPT_DIR} && make doctor && make -j"
