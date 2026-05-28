# TODO

## 1. TurboQuant Experiments

動了滿多東西所以感覺要再做一次。

- Recommended baseline config:
  ```text
  m = 32768
  n = 8192
  k = 250
  oversample = 6
  spectrum_decay_mode = polynomial
  spectrum_rank = 8192
  subspace_iter = 1
  repeat = 50
  ```
- Required sweep:
  ```text
  polynomial p = 0.4, 0.6, 0.8, 1.0
  ```
- Compare:
  ```text
  none
  TQ 4-bit
  TQ 2-bit
  ```
- Main metrics:
  ```text
  Total Time mean/min/stddev
  B Relative Error mean/min/stddev
  Final Reconstruction Error mean/min/stddev
  Theoretical Error
  Error Ratio
  ```

Notes:

- Use `p=0.6` as the first-pass stress test.
- `p=1.0` is an easy sanity check, not enough by itself.
- `B Relative Error` is optional; enable only with `--check-b-error` when diagnosing compression damage.
- After B-only TQ experiments, decide whether to apply TQ to subspace iteration `Z = sum_i A_i^T Q_i`.



## 2. QJL

Goal: revisit QJL only after the TQ baseline experiments are clear.

Current status:

```text
Negative / low priority
```

Known issues:

- Previous QJL residual correction worsened error for nonzero alpha.
- QJL added significant overhead.
- Current implementation uses hash Gaussian samples, which are expensive.

Possible next steps:

- Try hash Rademacher signs instead of hash Gaussian samples.
- Sweep `qjl_dim` and `qjl_alpha` only after TQ 4-bit/2-bit baselines are stable.
- Profile QJL kernels if QJL becomes relevant again.

## 3. SLATE Baseline

Goal: compare our randomized SVD pipeline against a more standard distributed dense linear algebra baseline.

Questions to answer:

- Can SLATE compute the relevant SVD baseline on the same Taiwania 2 allocation?
- What matrix sizes fit within time and memory limits?
- Should comparison be full SVD, truncated SVD, or a reduced benchmark that approximates the cost of deterministic SVD?

Minimum deliverables:

- Build/run instructions for SLATE on the target environment.
- One successful baseline run on a smaller matrix.
- Runtime comparison table against current rSVD v4.
- Clear note explaining whether SLATE is a fair baseline for this project.
