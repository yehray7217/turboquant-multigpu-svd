# AI Takeaway — randomized_svd_baseline/ (v1)

## Purpose

**Baseline B v1**: the first, simpler multi-GPU randomized SVD implementation.
This version is primarily a **readable research baseline** — the pipeline is easy
to follow but is not the main optimization target.

The key difference from v2: intermediate results (`Y_i`) are gathered to **host
memory** between GPUs, not kept on device. This limits its scalability and makes
its compression target less interesting for the project's final story.

---

## Files

| File | Description |
|------|-------------|
| `randomized_svd_multigpu.cu` | ~580 lines. Full pipeline: host-side A → GPU row blocks → Y_i = A_i Ω on each GPU → host gather Y → GPU 0 QR → B = QᵀA → SVD(B) → form U_k. |
| `Makefile` | `nvcc -O3 -std=c++17 -arch=sm_70 -lcublas -lcusolver -lcudart` |
| `run_randomized_svd_multigpu.slurm` | Slurm script: 8 GPUs, small/medium test configs. |
| `test1.out` | A sample output file from a previous run (useful to see expected output format). |

---

## Key Concepts

### Pipeline Flow

```
Host: generate A  →  GPU i: A_i = row block i  →  GPU i: Y_i = A_i * Omega
   →  Host: gather all Y_i  →  [TurboQuant/QJL insertion point here]
   →  GPU 0: QR on Y  →  GPU 0: B = Q^T A  →  GPU 0: SVD(B)  →  GPU 0: U = Q U_tilde
```

### Compression Point in v1

The compressible tensor is `Y_i` before the host gather. Payload formula:
```
m * l * sizeof(float)   where l = k + oversample
```

This payload does **not** grow with GPU count — more GPUs just split the same `Y`
into smaller row blocks. This is why v1's compression story is weaker than v2.

For the small benchmark (m=4096, n=2048, k=64, l=80):
- Total Y payload: ~1.25 MiB (constant regardless of GPU count)

### Why v2 Is Better

v2 targets `B_i` reduction, where payload = `ngpus * l * n * sizeof(float)`.
This grows with GPU count, so compression has larger absolute benefit at 8+ GPUs.

---

## How to Build & Run

```bash
module load cuda/12.8
cd randomized_svd_baseline
make
sbatch run_randomized_svd_multigpu.slurm
```

Command-line options:
```
--m <rows>       Matrix rows (default varies)
--n <cols>       Matrix cols
--k <rank>       Target rank
--oversample <p> Oversampling parameter (l = k + p)
--ngpus <g>      Number of GPUs to use
--seed <s>       Random seed
--no-check-error Skip reconstruction error check
```

---

## Current Status

Working for correctness validation. Used to verify that the randomized SVD
algorithm produces the correct singular values.

**Not the main benchmark target** — the project's primary results come from v2
and v3.

---

## Next Steps / What To Try

- **Use for correctness cross-check**: run with the same matrix parameters as v2
  and compare singular values to validate the pipeline.
- **Read `test1.out`**: this sample output shows what the report format looks like —
  useful to understand timing breakdowns before reading v2 output.
- **Do not add optimization here**: any new compression or performance work should
  go in v2 or v3, not v1.
