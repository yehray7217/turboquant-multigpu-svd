# AI Takeaway — randomized_svd_baseline_v4/

## Purpose

**Experiment framework on top of the v2/v3 pipeline** (collaborator: pbr03617 / Gino).
v4 keeps the same TSQR-style distributed randomized SVD + TurboQuant `B_i` compression,
but adds the pieces needed to study *accuracy under realistic inputs*:

- **Subspace iteration** to improve singular-vector capture.
- **Synthetic spectrum-decay test matrices** so the input `A` has fast-decaying singular
  values like real scientific / engineering / NLP matrices, instead of the near-flat
  spectrum of pure Gaussian input.
- A **theoretical-error / error-ratio** accuracy metric.
- **RHT-rotation validation** for TurboQuant.

It is an MPI build (like v3): `mpicxx` + `nvcc`, linking `../turboquant/turboquant.cu`.

> The canonical, most up-to-date status doc is `skills/PROJECT_CONTEXT.md` — Gino feeds it
> directly to an AI to continue the work. Read that first.

---

## Files

| File | Description |
|------|-------------|
| `randomized_svd_multigpu_v4.cu` | Main source. v3 MPI pipeline + subspace iteration + synthetic spectra + new accuracy metric. |
| `Makefile` | `mpicxx`/`nvcc`, `-arch=sm_70`, links `../turboquant`. Builds into `.build/`. |
| `README.md` | v4 development history (Chinese): subspace iteration, spectrum generation, RHT validation, error-ratio metric. |
| `TODO.md` | Next experiments: TQ sweep on polynomial spectra, QJL (low priority), SLATE comparison. |
| `skills/PROJECT_CONTEXT.md` | Full project context for teammate/AI handoff — read first. |
| `skills/EXPERIMENT_GUIDELINE.md` | How to run/record v4 experiments consistently. |
| `exp_subspace_iteration/` | Experiment: does subspace iteration pay for itself? (positive — q=1 best) |
| `exp_subspace_stabilization/` | Experiment: QR-stabilize the `Z = AᵀQ` reduce? (negative) |
| `.build/` | Compiled objects/binary (currently checked in; normally a build artifact). |

---

## Key Concepts

### What's new vs v3

| Feature | CLI | Notes |
|---------|-----|-------|
| Subspace iteration | `--subspace-iter <q>` | Adds `q` rounds of `Z = Σ Aᵢᵀ Qᵢ` → `Y = A Z` → re-QR. Each round adds one `n×l` reduce. |
| Synthetic spectra | `--spectrum-decay-mode random\|polynomial\|exponential` | `polynomial`: σᵢ ∼ i⁻ᵖ; `exponential`: σᵢ ∼ exp(−i). Default `random`. |
| Spectrum params | `--spectrum-decay-param <float>`, `--spectrum-rank <int>` | Decay rate `p`; number of nonzero singular values (default `min(m,n)`). |
| Z stabilization (exp) | `--stabilize-subspace-z` | QR-orthogonalize `Z` before `Y = A Z`. Negative result (see below). |
| Optional B-error | `--check-b-error` | B relative error is now opt-in (off by default) to keep timing clean. |

### New Accuracy Metric

For `polynomial`/`exponential` spectra the *theoretical* randomized-SVD error is known
in closed form:

```math
\sqrt{ \frac{\sum_{i=k+1}^{\text{rank}} \sigma_i^2}{\sum_{i=1}^{\text{rank}} \sigma_i^2} }
```

v4 reports `error ratio = our reconstruction error / theoretical error`, which makes it
easy to see how much TurboQuant inflates error above the unavoidable theoretical floor —
clearer than a raw error percentage.

### Subspace Iteration Result (exp_subspace_iteration, 32k×8k polynomial, no TQ/QJL)

```text
Theoretical:               4.85%
No subspace iteration:     9.06%
q=1 subspace iteration:    5.46%   ← best
q=2 subspace iteration:    6.11%
```

`q=1` is the sweet spot. Caveat: the measured overhead is large (≈48 ms → ≈203 ms) because
TQ currently compresses only `B`, not the new `Z = Σ Aᵢᵀ Qᵢ` reduce — so subspace iteration
adds an uncompressed `n×l` communication each round. Compressing `Z` is the natural next step.

### Z Stabilization Result (exp_subspace_stabilization — negative)

QR-orthogonalizing `Z` before `Y = A Z` **worsened** final reconstruction error
(43.68% → 45.35%) and added ~15.35 ms. Abandoned for now.

### RHT Validation

TurboQuant requires that, after the rotation, vector components are ~Gaussian. v4 confirmed
that the practical RHT rotation (random ±1 signs + Fast Walsh–Hadamard Transform) achieves
this just as well as a dense O(d²) random rotation — so the fast rotation is safe to use.
Evidence: `docs/notes/rht-distribution-test/rht_distribution.png`.

---

## How to Build & Run

```bash
module load cuda/12.8 ucx/1.14.1 openmpi/5.0.2_ucx1.14.1_cuda12.3
cd randomized_svd_baseline_v4
make

# Example: 1-step subspace iteration on a polynomial-decay matrix
sbatch exp_subspace_iteration/exp.slurm
```

Recommended baseline config (from `TODO.md`):

```text
m=32768, n=8192, k=250, oversample=6
spectrum_decay_mode=polynomial, spectrum_rank=8192
subspace_iter=1, repeat=50
sweep polynomial p ∈ {0.4, 0.6, 0.8, 1.0}   (use p=0.6 as first-pass stress test)
compare: none / TQ 4-bit / TQ 2-bit
```

---

## Relationship to v2/v3

- v2/v3 (and `OPTIMIZATION_HISTORY.md`) own the **compression** story: TQ/QJL/FP16 on `B`,
  the pre-alloc fix, and the multi-node speedup claim (TQ 4-bit −20.3% at 64k×16k).
- v4 owns the **accuracy-under-realistic-inputs** story: subspace iteration and synthetic
  spectra, currently measured without TQ to establish a clean accuracy baseline.

---

## Next Steps / What To Try

1. **Run the TODO.md TQ sweep** on polynomial spectra (p = 0.4/0.6/0.8/1.0), comparing
   none / TQ 4-bit / TQ 2-bit with the error-ratio metric.
2. **Compress the subspace-iteration `Z` reduce** with TQ — the main overhead source once
   `--subspace-iter` is enabled.
3. **Decide whether QJL is worth revisiting** only after TQ baselines on realistic spectra
   are stable (currently low priority / negative — see `OPTIMIZATION_HISTORY.md` §13.2).
4. **SLATE comparison** as a standard distributed-SVD baseline (see `TODO.md` §3).
5. **Stop committing `.build/`** — add it to `.gitignore` so binaries stay out of the repo.
