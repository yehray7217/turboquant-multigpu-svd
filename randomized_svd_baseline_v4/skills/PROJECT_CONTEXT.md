# TurboQuant-QJL Accelerated Multi-GPU Randomized SVD — Project Context

This document summarizes the current v4 baseline status for teammates. The main working directory is:

```text
/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4
```

Main implementation:

```text
randomized_svd_multigpu_v4.cu
```

## 1. Project Goal

We are studying distributed randomized SVD on multi-node / multi-GPU systems. The original bottleneck is communication of intermediate matrices across GPUs and MPI ranks. We apply TurboQuant-style random rotation plus low-bit quantization to reduce communication payload while controlling numerical error.

The project currently focuses on:

1. Building a realistic randomized SVD baseline with controllable singular spectra.
2. Measuring whether subspace iteration improves accuracy enough to justify its extra communication.
3. Applying TurboQuant to communication-heavy stages, starting with `B_i = Q_i^T A_i` reduce and likely extending to subspace iteration communication.
4. Validating that our practical RHT rotation is a reasonable replacement for dense random rotation.

Platform:

- Taiwania 2
- NVIDIA Tesla V100 SXM2 32GB
- Typical allocation: 2 nodes x 8 GPUs = 16 GPUs
- CUDA 12.8, cuBLAS, cuSOLVER, OpenMPI
- Intra-node: NVLink
- Inter-node: InfiniBand

## 2. Distributed Randomized SVD

We row-partition the input matrix:

```text
A = [A_0; A_1; ...; A_{G-1}]
```

Each GPU owns one row block `A_i`.

Basic randomized SVD flow:

```text
Input: A in R^{m x n}, target rank k, oversampling p
l = k + p

1. Generate Omega in R^{n x l}
2. Local projection:
     Y_i = A_i Omega
3. Local QR:
     Y_i = Qbar_i R_i
4. TSQR:
     stack(R_i) = T R
     Q_i = Qbar_i T_i
5. Build local projected matrix:
     B_i = Q_i^T A_i
6. Reduce:
     B = sum_i B_i
7. SVD on GPU0:
     B = Utilde Sigma V^T
8. Optional:
     U_i = Q_i Utilde
```

Important communication costs:

| Stage | Payload | Importance |
|---|---:|---|
| TSQR `R_i` gather | `O(l^2)` | small |
| `B_i` reduce | `O(nl)` | large; original TurboQuant target |
| broadcast `Utilde` | `O(lk)` | small |

## 3. Subspace Iteration

Subspace iteration is implemented in v4 via:

```bash
--subspace-iter <q>
```

Current non-stabilized implementation:

```text
Q = qr(A Omega)

repeat q times:
    Z_i = A_i^T Q_i
    Z = sum_i Z_i              # n x l, communication O(nl)
    Y_i = A_i Z
    Q = qr(Y) via local QR + TSQR
```

This means each subspace iteration adds another communication-heavy `n x l` reduce, comparable in size to the `B_i` reduce. Therefore, once subspace iteration is enabled, compressing only `B_i` is no longer enough: the subspace iteration `Z` communication can become an equally important compression target.

There is also an experimental stabilized variant:

```bash
--stabilize-subspace-z
```

which changes the loop to:

```text
Z_i = A_i^T Q_i
Z = sum_i Z_i
Qz = qr(Z)
Y_i = A_i Qz
Q = qr(Y)
```

This was tested but did not improve accuracy in the current setup.

## 4. Synthetic Spectrum Test Matrices

Gaussian random matrices have nearly flat spectra, making final reconstruction error very high and not representative of many scientific, engineering, and NLP matrices. v4 therefore supports synthetic matrices with controllable singular values:

```bash
--spectrum-decay-mode random|polynomial|exponential
--spectrum-decay-param <float>
--spectrum-rank <int>
```

Polynomial:

```text
sigma_i = i^{-p}
```

Exponential:

```text
sigma_i = exp(-alpha * (i - 1))
```

Implementation detail:

```text
A_i = U_i Sigma V^T = U_i (V Sigma)^T
```

where `U_i` and `V` are deterministic DCT-like orthonormal bases generated on GPU. The code forms each row block with cuBLAS GEMM. This avoids CPU-side `O(m n rank)` generation and allows `spectrum_rank = min(m,n)` for current 32k x 8k experiments.

Current default:

```text
spectrum_rank = min(m, n)
```

## 5. Accuracy Metric

The reported final reconstruction error is a fast Frobenius relative error estimate:

```text
sqrt((||A||_F^2 - sum_{i=1}^{k} S_i^2) / ||A||_F^2)
```

For synthetic spectrum tests, the code also reports the theoretical best rank-k error:

```text
sqrt(
  sum_{i=k+1}^{spectrum_rank} sigma_i^2
  /
  sum_{i=1}^{spectrum_rank} sigma_i^2
)
```

The output now includes:

```text
mean, min, stddev, theoretical, err ratio
```

where:

```text
err ratio = mean final reconstruction error / theoretical error
```

Because the theoretical error is constant for a fixed experiment configuration, this is equivalent to averaging per-repeat error ratios.

Timing-only runs print only `Total Time`. Final-error runs print only accuracy metrics.

### Current Experiment Settings

The current v4 experiments intentionally changed several settings from the earlier v3/v4 trials:

1. `spectrum_rank` is no longer capped at `1024`.
   - Earlier experiments used `spectrum_rank = 1024`, which made `k=250` capture about 24% of the nonzero singular directions.
   - This was too small and not representative of realistic low-rank approximation settings.
   - Current default and recommended setting:
     ```text
     spectrum_rank = min(m, n)
     ```
   - For the standard `32768 x 8192` experiment, this means:
     ```text
     spectrum_rank = 8192
     k / spectrum_rank = 250 / 8192 ~= 3.05%
     ```

2. Repeated measurements use `repeat = 50`.
   - Timing summaries skip the first cold run and report warm statistics over the remaining repeats.
   - Accuracy summaries use all 50 randomized trials with different seeds.

3. The most important accuracy metric is now `err ratio`.
   - Absolute final reconstruction error is still reported.
   - However, the main comparison should be:
     ```text
     err ratio = mean final reconstruction error / theoretical best rank-k error
     ```
   - This makes experiments comparable across different spectrum decay parameters.

## 6. TurboQuant / RHT Status

TurboQuant compresses each column of communication payloads by:

```text
1. Random sign flip D
2. Fast Walsh-Hadamard Transform H
3. Low-bit quantization
4. Transmit codes + scale
5. Decode with inverse transform
```

Original TurboQuant uses a dense random orthogonal rotation. Our implementation uses Randomized Hadamard Transform:

```text
Pi = (1 / sqrt(d)) H D
```

This reduces rotation cost from dense `O(d^2)` to FWHT `O(d log d)`.

RHT validation has been completed:

```text
/work/pbr03617/turboquant-multigpu-svd/docs/notes/rht-distribution-test/rht-distribution.py
```

Result:

- RHT output coordinates match the expected normal-like distribution.
- This holds not only for random unit vectors but also for clustered / structured vectors.
- This supports the key TurboQuant assumption that random rotation spreads coordinate mass and makes quantization more uniform.

Generated figure:

```text
/work/pbr03617/turboquant-multigpu-svd/docs/notes/rht-distribution-test/rht_distribution.png
```

## 7. Current Experimental Results

### 7.1 Subspace Iteration

Experiment folder:

```text
exp_subspace_iteration/
```

Representative setup:

```text
m = 32768
n = 8192
k = 250
oversample = 6
l = 256
spectrum = polynomial, sigma_i = i^-1
spectrum_rank = 8192
compression = none
repeat = 50
```

Results:

| Method | Final Error | Theoretical | Error Ratio | Interpretation |
|---|---:|---:|---:|---|
| q = 0 | 9.05529% | 4.85% | 1.87x | baseline randomized sketch |
| q = 1 | 5.46298% | 4.85% | 1.13x | best current setting |
| q = 2 | 6.10679% | 4.85% | 1.26x | worse than q=1, likely FP32 / small oversample instability |

Interpretation:

- One subspace iteration clearly improves accuracy.
- Two iterations are worse than one in the current FP32 implementation with small oversampling.
- The likely reason is that q=2 effectively amplifies singular values as roughly `sigma^(2q+1) = sigma^5`, which can over-emphasize the leading directions and numerically suppress rank-boundary directions near `k`.

### 7.2 Z-Side Stabilization

Experiment folder:

```text
exp_subspace_stabilization/
```

Goal:

```text
Check whether QR-orthogonalizing Z = A^T Q improves q=2 stability.
```

Result:

```text
Negative.
```

Observed result:

```text
Without Z stabilization: 43.68%
With Z stabilization:    45.35%
```

Runtime also increased by about `15 ms`. Current decision: do not prioritize this path.

## 8. Important Implementation Notes

Current useful CLI options:

```text
--m, --n, --k, --oversample
--ngpus
--gpus-per-rank
--compress-b-mode none|lowbit|tq|tq-qjl
--compress-b-bits 0|2|4|8
--compress-subspace-mode none|lowbit|tq|tq-qjl
--compress-subspace-bits 0|2|4|8
--qjl-dim
--qjl-alpha
--device-random-input
--skip-form-u
--subspace-iter <q>
--stabilize-subspace-z
--spectrum-decay-mode random|polynomial|exponential
--spectrum-decay-param <p or alpha>
--spectrum-rank <rank>
--repeat <N>
--summary-only
--no-check-error
--check-b-error
```

Removed/deprecated options:

```text
--allow-host-tq-prototype
--repeat-print-every
--host-reduce-b
```

Current folder structure:

```text
randomized_svd_baseline_v4/
├── randomized_svd_multigpu_v4.cu
├── Makefile
├── README.md
├── TODO.md
├── skills/
│   ├── PROJECT_CONTEXT.md
│   └── EXPERIMENT_GUIDELINE.md
├── .build/
├── exp_subspace_iteration/
│   ├── README.md
│   ├── ctrl.slurm
│   ├── exp.slurm
│   ├── exp2.slurm
│   ├── output_logs/
│   └── error_logs/
└── exp_subspace_stabilization/
    ├── README.md
    ├── ctrl.slurm
    ├── exp.slurm
    ├── output_logs/
    └── error_logs/
```

## 9. Main Next Steps

### 9.1 Validate TurboQuant on Subspace Iteration Communication

Subspace iteration TQ is now implemented behind explicit CLI flags. The communication-heavy matrix is:

```text
Z = sum_i A_i^T Q_i
```

Its size is:

```text
n x l
```

This is the same asymptotic size as:

```text
B = sum_i Q_i^T A_i
```

Therefore, if `--subspace-iter` is enabled, compressing only `B_i` leaves a major communication cost untouched. The implementation now supports compressing this `Z` path through:

```text
--compress-subspace-mode tq
--compress-subspace-bits 4
```

Implementation note:

```text
Each MPI rank compresses its local Z contribution, all ranks exchange compressed payloads, and each rank decodes/accumulates the global Z on GPU0 before computing Y_i = A_i Z.
```

Questions to answer experimentally:

- What is the effect on final reconstruction error?
- How much runtime is saved versus the added encode/decode cost?
- Does q=1 + TQ on both `Z` and `B` give a better speed/accuracy tradeoff than q=0 + TQ only on `B`?

### 9.2 Run TurboQuant Experiments on v4 Synthetic Spectra

This is likely the next task for teammates.

Recommended starting experiment:

```text
m = 32768
n = 8192
k = 250
oversample = 6
spectrum = polynomial
spectrum_decay_param = 0.6
spectrum_rank = 8192
subspace_iter = 1
repeat = 50
```

Recommended TQ sweep:

```text
spectrum_decay_param = 0.4, 0.6, 0.8, 1.0
```

Use `0.6` as the first-pass stress-test config. Keep `1.0` as an easier sanity check, not the only reported setting.

Compare:

```text
control:
  compress-b-mode = none
  compress-subspace-mode = none

B-only TQ:
  compress-b-mode = tq, compress-b-bits = 4 or 2
  compress-subspace-mode = none

subspace-only TQ:
  compress-b-mode = none
  compress-subspace-mode = tq, compress-subspace-bits = 4

B + subspace TQ:
  compress-b-mode = tq, compress-b-bits = 4 or 2
  compress-subspace-mode = tq, compress-subspace-bits = 4 or 2
```

Metrics to record:

- Total Time mean / min / stddev
- Final Reconstruction Error mean / min / stddev
- Theoretical error
- Error ratio
- Payload size / compression ratio if available in detailed logs

Suggested interpretation:

- TQ 4-bit is expected to be the useful tradeoff.
- TQ 2-bit is expected to be faster but may damage accuracy heavily.
- If q=1 is used, B-only compression may not show the full possible speedup because subspace iteration `Z` communication can be comparable to the `B` reduce.

## 10. Known Issues / Open Questions

- q=2 is worse than q=1 in current FP32 experiments.
- Z-side QR stabilization did not help.
- QJL residual correction remains a negative result in the current implementation.
- TQ is currently applied to `B` reduce, not yet to subspace iteration `Z` reduce.
- More experiment guidelines are needed so teammates can run TQ tests consistently.

## 11. References

| Reference | Used for |
|---|---|
| Halko, Martinsson & Tropp, SIAM Review 2011 | Randomized SVD and subspace iteration |
| TurboQuant paper | Random rotation + quantization compression idea |
| QJL paper | Residual correction via quantized JL sketch |
| Ailon & Chazelle, SICOMP 2009 | Fast JL Transform / RHT justification |
| Eikmeier & Gleich, KDD 2017 | Power-law singular spectra in real graph matrices |
| Udell & Townsend, SIMODS 2019 | Why big data matrices are often approximately low rank |
| Eckart-Young / Mirsky theorem | Optimal rank-k approximation baseline |
