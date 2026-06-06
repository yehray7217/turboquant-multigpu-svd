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
3. Applying TurboQuant to communication-heavy stages, including both `B_i = Q_i^T A_i` reduce and subspace iteration `Z_i = A_i^T Q_i` reduce.
4. Validating that our practical RHT rotation is a reasonable replacement for dense random rotation.
5. Moving beyond synthetic matrices by supporting raw `.f32` input files for real-world OISST / POD / LoftQ-style testcases.

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
| subspace `Z_i` reduce | `O(nl)` | large when `q > 0`; current second TurboQuant target |
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

v4 now supports compressing this subspace `Z` communication path with the same compression interface used by `B`:

```bash
--compress-subspace-mode none|lowbit|tq|tq-qjl
--compress-subspace-bits <bits>
```

Current TurboQuant experiments generally compare:

```text
No TQ:
  compress-b-mode = none
  compress-subspace-mode = none

TQ8:
  compress-b-mode = tq, compress-b-bits = 8
  compress-subspace-mode = tq, compress-subspace-bits = 8

TQ4:
  compress-b-mode = tq, compress-b-bits = 4
  compress-subspace-mode = tq, compress-subspace-bits = 4
```

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

Synthetic spectra are still the preferred way to isolate accuracy behavior because the theoretical best rank-k error is known.

## 5. External Input File Mode

v4 now supports reading a real matrix `A` from a raw FP32 file:

```bash
--input-file <path>
```

Required file format:

```text
raw float32 binary
row-major layout
no header
file size = m * n * sizeof(float)
```

The shape is still specified by the existing CLI parameters:

```bash
--m <rows>
--n <cols>
--input-file testcases_real_world/OISST/matrix.f32
```

Implementation behavior:

- `--input-file` makes the matrix source `file`.
- Each MPI rank / GPU reads only its assigned row block using file seek.
- The host row-major block is converted to the local column-major layout expected by cuBLAS before copying to `w.d_Ai`.
- `||A||_F^2` is computed from the file data with double/long-double accumulation and combined with MPI.
- File loading and row-major to column-major conversion are setup work and are not counted in `Total Time`.
- Repeat runs keep `A` fixed and only regenerate/copy randomized SVD `Omega`.
- Synthetic spectrum options are ignored in file mode.
- `--input-file` is mutually exclusive with `--device-random-input`.

Because real input files do not provide an analytic singular spectrum, theoretical error and error ratio are reported as `n/a` in file mode. Use raw `Final Reconstruction Error` and comparison against the no-compression baseline for real-world datasets.

## 6. Accuracy Metric

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

Timing-only runs print timing / communication summaries. Final-error runs print accuracy metrics.

Recommended experiment pattern:

```text
Timing phase:
  add --no-check-error
  record Total Time, GPU Compute Time, Host/Staging Time, NVLink/InfiniBand time, payloads

Final-error phase:
  do not add --no-check-error
  record Final Reconstruction Error, theoretical, err ratio
  optionally add --check-b-error for B compression diagnostics
```

For real-world `--input-file` runs, theoretical and err ratio are `n/a`; record raw `Final Reconstruction Error` instead.

`Global B Relative Error` is a compression diagnostic. It is useful when studying compression damage to the intermediate `B` matrix, but it is not a primary final accuracy metric.

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

## 7. TurboQuant / RHT Status

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

### Current TurboQuant Status

TurboQuant is now applied to:

```text
B reduce:
  B = sum_i Q_i^T A_i

Subspace iteration reduce:
  Z = sum_i A_i^T Q_i
```

The current default good configuration is pure TQ 8-bit on both paths. TQ 4-bit can be faster but has a visible accuracy cost. TQ-QJL has been tested as a residual correction path and is currently a negative result: the extra GPU compute cost has not beaten pure TQ at the same bit setting.

Reference experiment folder:

```text
exp_turboquant/
```

Representative synthetic setup:

```text
m = 65536
n = 16384
k = 250
oversample = 6
l = 256
spectrum_decay_mode = polynomial
spectrum_decay_param = 0.6
spectrum_rank = 8192
subspace_iter = 1
repeat = 50
```

Observed high-level result:

```text
No TQ  -> Total Time about 187.7 ms, Error Ratio about 1.063
TQ8    -> Total Time about 161.7 ms, Error Ratio about 1.063
TQ4    -> Total Time about 137.9 ms, Error Ratio about 1.106
QJL    -> negative result in current implementation
```

Interpretation:

- TQ8 is the safest current default.
- TQ4 is useful for speed-focused experiments but must be reported with accuracy cost.
- QJL should not be included in the main TQ experiments unless explicitly requested.

## 8. Current Experimental Results

### 8.1 Subspace Iteration

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

### 8.2 Z-Side Stabilization

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

## 9. Important Implementation Notes

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
--input-file <path>
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
├── ai-prompts/
├── testcases_real_world/
│   ├── OISST/
│   │   └── guideline.md
│   ├── POD/
│   │   └── guideline.md
│   └── LoftQ/
│       ├── guideline.md
│       └── references/
├── exp_synthetic_accuracy_scaling/
│   ├── README.md
│   ├── run_accuracy_sweep.slurm
│   ├── run_scaling.slurm
│   ├── output_logs/
│   └── error_logs/
├── exp_turboquant/
│   ├── README.md
│   ├── ctrl.slurm
│   ├── b8-exp.slurm
│   ├── b4-exp.slurm
│   ├── b8-qjl-exp.slurm
│   ├── b4-qjl-exp.slurm
│   ├── output_logs/
│   └── error_logs/
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

## 10. Real-World Testcase Guidelines

The code can now consume real matrices through `--input-file`, so three real-world testcase guidelines have been added:

```text
testcases_real_world/OISST/guideline.md
testcases_real_world/POD/guideline.md
testcases_real_world/LoftQ/guideline.md
```

All real-world testcases must produce:

```text
matrix.f32
raw float32
row-major
no header
```

Each testcase should also provide:

```text
meta.json
README.md
scripts/<prepare_script>.py
exp_turboquant/
output_logs/
error_logs/
```

Large generated matrices, raw NetCDF/HDF5/model shards, and cache directories should not be committed unless explicitly agreed.

### 10.1 OISST

Goal:

```text
Sea surface temperature anomaly matrix for EOF/PCA/SVD.
```

Matrix meaning:

```text
rows    = selected ocean grid points
columns = daily time snapshots
entry   = SST anomaly after temporal mean removal
```

Target shape:

```text
32768 x 8192
```

### 10.2 POD / JHTDB CFD

Goal:

```text
CFD/turbulence snapshot matrix for Proper Orthogonal Decomposition.
```

Current status:

```text
First perform feasibility check for JHTDB datasets:
  isotropic1024coarse
  transition_bl
```

Preferred matrix:

```text
rows    = 32 x 32 x 32 spatial points
columns = time snapshots
entry   = one velocity component or pressure fluctuation
```

Target shape:

```text
32768 x 4096
```

Fallback if time-resolved access is not feasible:

```text
Spatial patch matrix from a single turbulence snapshot.
```

### 10.3 LoftQ-Inspired LLM Residual

Goal:

```text
LLM quantization residual matrix inspired by LoftQ.
```

Matrix definition:

```text
R = W - Q(W)
```

where `W` is one dense LLM weight tensor and `Q(W)` is the dequantized low-bit quantized version of that tensor.

This is not a full LoftQ reproduction and should not be described as LoRA training. It is a real AI matrix benchmark for distributed rSVD on a quantization residual.

Preferred first target:

```text
Mistral-7B
model.layers.0.mlp.gate_proj.weight
approx shape: 14336 x 4096
```

Debug target:

```text
TinyLlama-1.1B
model.layers.0.mlp.gate_proj.weight
approx shape: 5632 x 2048
```

## 11. Current Synthetic Experiment Plan

Experiment folder:

```text
exp_synthetic_accuracy_scaling/
```

Scripts:

```text
run_accuracy_sweep.slurm
run_scaling.slurm
```

### 11.1 Synthetic Accuracy Sweep

Purpose:

```text
Measure TQ accuracy inflation across polynomial spectrum difficulty.
```

Fixed setup:

```text
m = 32768
n = 8192
k = 250
oversample = 6
spectrum = polynomial
spectrum_decay_param = swept
spectrum_rank = 8192
subspace_iter = 1
repeat = 50
```

Sweep:

```text
spectrum_decay_param = 0.4, 0.6, 0.8, 1.0
method = No TQ, TQ8, TQ4
```

Main derived metrics:

```text
Speedup = Total Time_NoTQ / Total Time_TQ
Error Inflation = Error Ratio_TQ / Error Ratio_NoTQ
```

### 11.2 Synthetic Matrix-Size Scaling

Purpose:

```text
Measure whether TQ speedup grows with matrix size.
```

Current planned sizes:

```text
16384 x 4096, p = 0.6
32768 x 8192, p = 0.6
```

The `32768 x 8192, p=0.6` point is already included in the accuracy sweep and should be reused unless rerun is necessary.

Do not include QJL in this experiment unless explicitly requested.

## 12. Known Issues / Open Questions

- q=2 is worse than q=1 in current FP32 experiments.
- Z-side QR stabilization did not help.
- QJL residual correction remains a negative result in the current implementation.
- The fast final reconstruction metric is still an estimate based on the produced singular values; for real input files use it as a practical quality metric, not an analytic optimum comparison.
- For real-world datasets, theoretical error / err ratio are unavailable unless the true singular spectrum is separately computed.
- More real-world preprocessing scripts are still needed for OISST, POD/JHTDB, and LoftQ-inspired residual matrices.

## 13. References

| Reference | Used for |
|---|---|
| Halko, Martinsson & Tropp, SIAM Review 2011 | Randomized SVD and subspace iteration |
| TurboQuant paper | Random rotation + quantization compression idea |
| QJL paper | Residual correction via quantized JL sketch |
| Ailon & Chazelle, SICOMP 2009 | Fast JL Transform / RHT justification |
| Eikmeier & Gleich, KDD 2017 | Power-law singular spectra in real graph matrices |
| Udell & Townsend, SIMODS 2019 | Why big data matrices are often approximately low rank |
| Eckart-Young / Mirsky theorem | Optimal rank-k approximation baseline |
