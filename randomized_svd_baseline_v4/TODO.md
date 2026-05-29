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
- TQ is now available for both B-side reduce and subspace iteration communication.
- Keep B-only TQ as the first experiment so we have a clean baseline before enabling subspace-side compression.

Follow-up: validate TQ on subspace iteration / power iteration communication.

- Target communication path:
  ```text
  Y = A A^T A Omega
  ```
- In the distributed implementation, this mainly means investigating the intermediate reduce/broadcast path such as:
  ```text
  Z = sum_i A_i^T Q_i
  Y_i = A_i Z
  ```
- Motivation: after enabling subspace iteration, this communication can become comparable to `B` reduce, so B-only TQ may no longer address the dominant communication cost.
- Risk: quantizing the subspace iteration intermediate may amplify basis error across power iterations, so evaluate accuracy using `Final Reconstruction Error` and `Error Ratio`, not only runtime.
- Suggested experiment:
  ```text
  none
  B-only TQ
  subspace-only TQ
  B + subspace TQ
  ```
  with `subspace_iter = 1` first, then optionally `subspace_iter = 2`.



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
