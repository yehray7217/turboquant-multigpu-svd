# SLATE RSVD Naive Baseline

This baseline is intentionally configured as an **unoptimized naive baseline** to expose communication and scheduling overhead:

- `Lookahead = 0` (no communication hiding)
- `nb = 256` (small tile size)
- process grid is configurable via CLI (`--grid-p`, `--grid-q`)
- expected MPI size is configurable via CLI (`--expected-mpi-size`)
- `Target = Devices` (GPU execution)
- single precision (`float`)

## Files

- `slate_rsvd_naive_baseline.cc`: complete C++17 MPI + SLATE RSVD program.
- `Makefile`: build via `pkg-config slate` with doctor/run helpers.
- `env_taiwania2.sh`: source this to load CUDA + MPI + local SLATE paths.
- `install_slate_local.sh`: build and install SLATE locally under `../third_party/slate-install`.
- `run_slate_rsvd_naive_baseline.slurm`: current launch template (currently 1 node, 2 GPUs, 2 MPI ranks).
- `run_background_setup.sh`: internal background workflow (install + build).
- `start_background_setup.sh`: start background setup (SSH disconnect safe).
- `check_background_setup.sh`: check background setup status/log.
- `stop_background_setup.sh`: stop background setup.

## Quick Start (for your current errors)

```bash
cd /home/rax10101010/hpc_final/turboquant-multigpu-svd/slate_baseline
source ./env_taiwania2.sh
make doctor
```

If `slate.pc visible: no`, install local SLATE once:

```bash
./install_slate_local.sh
source ./env_taiwania2.sh
make doctor
```

Then build:

```bash
make -j
```

## Run on compute nodes (recommended)

```bash
sbatch run_slate_rsvd_naive_baseline.slurm
```

Current default in this repository:

- `#SBATCH --nodes=1`
- `#SBATCH --ntasks-per-node=2`
- `#SBATCH --gres=gpu:2`
- runtime args: `--expected-mpi-size 2 --grid-p 1 --grid-q 2`

## Background Setup (SSH disconnect safe)

```bash
cd /home/rax10101010/hpc_final/turboquant-multigpu-svd/slate_baseline
./start_background_setup.sh
./check_background_setup.sh
```

When done, submit job:

```bash
sbatch run_slate_rsvd_naive_baseline.slurm
```

## Notes

- Current SLATE API uses `slate::svd(...)` (equivalent role to LAPACK-style `gesvd`).
- For explicit `Q`, the program uses `geqrf` + `qr_multiply_by_q` on an identity matrix, which is the `ungqr`-equivalent workflow in current SLATE.
- Your `mpirun: command not found` on login node means MPI module is not loaded yet; `source ./env_taiwania2.sh` fixes that.
- If `pkg-config` cannot find `slate`, the `Makefile` will fall back to local SLATE libraries at `../third_party/slate-install/lib64`.
