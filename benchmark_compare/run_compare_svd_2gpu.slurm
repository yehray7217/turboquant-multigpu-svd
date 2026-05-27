#!/bin/bash
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --job-name=compare-svd-2gpu
#SBATCH --output=benchmark_compare/compare_svd_2gpu_%j.out
#SBATCH --error=benchmark_compare/compare_svd_2gpu_%j.err

set -eo pipefail

REPO=$HOME/turboquant-multigpu-svd
cd "${REPO}"

echo "SVD method comparison"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "====================================="

echo
echo "[1/3] cuSOLVER exact/full SVD, single-GPU square cases"
echo "-------------------------------------"
module purge >/dev/null 2>&1 || true
module load cuda/12.8
make -C cusolver

for size in 4096 8192; do
    echo "Running cuSOLVER exact SVD: M=${size} N=${size}"
    (
        cd cusolver
        ./cusolver -M "${size}" -N "${size}"
    )
    echo "-------------------------------------"
done

echo
echo "[2/3] SLATE randomized SVD, 2 MPI ranks / 2 GPUs"
echo "-------------------------------------"
(
    cd slate_baseline
    source ./env_taiwania2.sh
    if ! pkg-config --exists slate && [[ ! -f ../third_party/slate-install/lib64/libslate.so ]] && [[ ! -f ../third_party/slate-install/lib/libslate.so ]]; then
        echo "SLATE is not installed yet."
        echo "Run this once before the comparison job:"
        echo "  cd ${REPO}/slate_baseline"
        echo "  ./start_background_setup.sh"
        echo "  ./check_background_setup.sh"
        exit 2
    fi
    make -j
    srun --mpi=pmix ./slate_rsvd_naive_baseline \
        --m 32768 \
        --n 8192 \
        --k 256 \
        --oversample 64 \
        --seed 12345 \
        --expected-mpi-size 2 \
        --grid-p 1 \
        --grid-q 2
)
echo "-------------------------------------"

echo
echo "[3/3] v2 randomized SVD, 2 GPUs, warm repeat timing"
echo "-------------------------------------"
module purge >/dev/null 2>&1 || true
module load cuda/12.8
make -C randomized_svd_baseline_v2

V2_ARGS=(
    --m 32768
    --n 8192
    --k 256
    --oversample 64
    --ngpus 2
    --seed 1234
    --device-random-input
    --skip-form-u
    --repeat 100
    --repeat-print-every 25
    --no-check-error
    --no-check-b-error
)

echo "Running v2 RSVD warm timing: none"
randomized_svd_baseline_v2/randomized_svd_multigpu_v2 \
    "${V2_ARGS[@]}" \
    --compress-b-mode none \
    --compress-b-bits 0
echo "-------------------------------------"

echo "Running v2 RSVD warm timing: TQ 4-bit"
randomized_svd_baseline_v2/randomized_svd_multigpu_v2 \
    "${V2_ARGS[@]}" \
    --compress-b-mode tq \
    --compress-b-bits 4 \
    --qjl-dim 0 \
    --qjl-alpha 0.0
echo "-------------------------------------"

echo "Running v2 RSVD warm timing: TQ+QJL 4-bit"
randomized_svd_baseline_v2/randomized_svd_multigpu_v2 \
    "${V2_ARGS[@]}" \
    --compress-b-mode tq-qjl \
    --compress-b-bits 4 \
    --qjl-dim 64 \
    --qjl-alpha 0.01
echo "-------------------------------------"

echo "Comparison complete!"
