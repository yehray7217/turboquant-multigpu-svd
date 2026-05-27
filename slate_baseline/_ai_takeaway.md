# AI Takeaway — slate_baseline/

## Purpose

**Library-level distributed randomized SVD reference.** Uses the SLATE library
(Scalable Linear Algebra Tiled Execution), which implements distributed tiled
linear algebra on top of MPI + CUDA.

This baseline is intentionally **unoptimized** (naive) to show communication
overhead without tuning — making it a fair comparison point to demonstrate that
the custom v2/v3 pipeline with TurboQuant compression is competitive with
(or better than) a standard library baseline.

---

## Files

| File | Description |
|------|-------------|
| `slate_rsvd_naive_baseline.cc` | Main C++17 MPI + SLATE program. Intentionally naive: `Lookahead=0`, small tile `nb=256`. |
| `Makefile` | Uses `pkg-config slate`. Falls back to local `../third_party/slate-install`. |
| `env_taiwania2.sh` | Source to load CUDA + MPI + local SLATE paths. |
| `install_slate_local.sh` | Builds and installs SLATE locally under `../third_party/slate-install/`. |
| `run_slate_rsvd_naive_baseline.slurm` | Launch: 1 node, 2 GPUs, 2 MPI ranks. |
| `start_background_setup.sh` | Start SLATE build in background (SSH-disconnect safe). |
| `check_background_setup.sh` | Check if background build is still running. |
| `stop_background_setup.sh` | Kill background build process. |
| `run_background_setup.sh` | Internal script used by `start_background_setup.sh`. |

---

## Key Concepts

### SLATE Configuration (Intentionally Naive)

| Parameter | Value | Reason for being naive |
|-----------|-------|----------------------|
| `Lookahead` | 0 | No communication hiding — maximizes visible latency |
| `nb` (tile size) | 256 | Small tiles → more communication events |
| `Target` | `Devices` | GPU execution |
| `Precision` | `float` | Matches v2/v3 |

This configuration surfaces communication overhead that would be hidden by
a production-tuned SLATE run. That makes it the right baseline for showing
that our custom compressed pipeline is meaningful.

### SLATE API Used

```cpp
// QR-based randomized projection
slate::geqrf(Y, T, opts);        // QR of Y
slate::qr_multiply_by_q(slate::Side::Left, slate::Op::ConjTrans, Y, T, identity, opts);  // form Q
slate::gemm(1.0, Qt, A, 0.0, B, opts);  // B = Q^T A

// SVD on small B
slate::svd(B, Sigma, U_tilde, Vt, opts);
```

### Process Grid

Configurable via `--grid-p` and `--grid-q` flags. For 2 GPUs: `--grid-p 1 --grid-q 2`.

---

## How to Build & Run

### First-time setup (install SLATE locally — takes ~20 min)

```bash
cd slate_baseline
source ./env_taiwania2.sh

# Check if system SLATE is available
make doctor

# If not found, install locally in background:
SLATE_BUILD_JOBS=2 ./start_background_setup.sh
./check_background_setup.sh   # poll until done
```

### Build and run

```bash
source ./env_taiwania2.sh
make -j
sbatch run_slate_rsvd_naive_baseline.slurm
```

### Troubleshooting

- `mpirun: command not found` → forgot to `source ./env_taiwania2.sh`
- `pkg-config cannot find slate` → run `install_slate_local.sh` first
- Background setup check: `cat run_background_setup.log`

---

## Current Configuration (Slurm Script)

```
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
runtime args: --expected-mpi-size 2 --grid-p 1 --grid-q 2
```

Matrix size used for comparison: `M=32768, N=8192, K=256, oversample=64`.

---

## What To Compare (for presentation)

```
SLATE fields:
  total RSVD core time
  Y = A*Omega
  geqrf(Y)
  B = Q^T*A
  svd(B)

v2 equivalent fields:
  warm_compute_avg_ms
  local_projection_Yi
  local_qr_Yi
  build_reduce_Bi
  svd_B_on_gpu0
```

**Story**: SLATE is a library baseline showing raw distributed RSVD cost.
v2 none-mode is our custom equivalent. v2 TQ/TQ-QJL is the compressed variant.

---

## Next Steps / What To Try

1. **Run the comparison script** `benchmark_compare/run_compare_svd_2gpu.slurm`
   which runs SLATE + cuSOLVER + v2 in one job and summarizes all timings.
2. **Try a larger tile size** (`nb=512` or `nb=1024`) to see how much SLATE's
   performance improves with better tiling — this quantifies the "naive" penalty.
3. **Do not optimize SLATE here** — it's a fixed reference. Real optimization
   happens in v2/v3.
4. **Note**: SLATE's RSVD uses a different power iteration strategy than v2,
   so the reconstruction errors may differ even at the same `k`.
