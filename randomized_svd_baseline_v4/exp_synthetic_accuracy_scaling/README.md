# Synthetic Accuracy + Scaling Experiments

Controlled synthetic-spectrum experiments for measuring TurboQuant speedup and accuracy inflation.

## Purpose

This experiment answers two questions:

1. Accuracy impact: how much extra final reconstruction error does TQ introduce relative to no compression?
2. Scaling behavior: does TQ become more useful as matrix communication payloads grow?

Synthetic polynomial spectra provide a known theoretical best rank-k error, so the main accuracy metric is:

```text
Error Ratio = Final Reconstruction Error / Theoretical Error
```

For TQ runs, report:

```text
Error Inflation = Error Ratio_TQ / Error Ratio_NoTQ
```

## Common Config

```text
k = 250
oversample = 6
l = 256
subspace_iter = 1
spectrum_decay_mode = polynomial
spectrum_rank = min(m, n)
repeat = 50
ngpus = 16
gpus_per_rank = 8
skip_form_u = true
summary_only = true
device_random_input = true
```

The scripts run two phases for each config:

1. Timing: `--no-check-error`
2. Final-error: accuracy metrics, with optional `--check-b-error`

`CHECK_B_ERROR=1` by default for the final-error phase only. Use `CHECK_B_ERROR=0 sbatch ...` to skip B compression diagnostics.

## Methods

| Method | B Compression | Subspace Compression |
|---|---|---|
| No TQ | none | none |
| TQ8 | TQ 8-bit | TQ 8-bit |
| TQ4 | TQ 4-bit | TQ 4-bit |

## Scripts

```text
run_accuracy_sweep.slurm  -> p = 0.4, 0.6, 0.8, 1.0 at 32768 x 8192
run_scaling.slurm         -> 16384 x 4096, 65536 x 16384, 131072 x 32768 at p = 0.6
```

The `32768 x 8192, p=0.6` scaling point is reused from the accuracy sweep. Set `RUN_DUPLICATE_32K=1` when submitting `run_scaling.slurm` only if that duplicate point needs to be rerun.

## Accuracy Sweep

| p | Method | Total Time | Speedup | Final Error | Theoretical | Error Ratio | Error Inflation | Global B Rel Err |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | No TQ | | 1.00x | | | | 1.000 | - |
| 0.4 | TQ8 | | | | | | | |
| 0.4 | TQ4 | | | | | | | |
| 0.6 | No TQ | | 1.00x | | | | 1.000 | - |
| 0.6 | TQ8 | | | | | | | |
| 0.6 | TQ4 | | | | | | | |
| 0.8 | No TQ | | 1.00x | | | | 1.000 | - |
| 0.8 | TQ8 | | | | | | | |
| 0.8 | TQ4 | | | | | | | |
| 1.0 | No TQ | | 1.00x | | | | 1.000 | - |
| 1.0 | TQ8 | | | | | | | |
| 1.0 | TQ4 | | | | | | | |

## Scaling

| Matrix Size | Method | Total Time | Speedup | InfiniBand Time | NVLink Time | Final Error | Error Ratio | Error Inflation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 16384 x 4096 | No TQ | | 1.00x | | | | | 1.000 |
| 16384 x 4096 | TQ8 | | | | | | | |
| 16384 x 4096 | TQ4 | | | | | | | |
| 32768 x 8192 | No TQ | | 1.00x | | | | | 1.000 |
| 32768 x 8192 | TQ8 | | | | | | | |
| 32768 x 8192 | TQ4 | | | | | | | |
| 65536 x 16384 | No TQ | | 1.00x | | | | | 1.000 |
| 65536 x 16384 | TQ8 | | | | | | | |
| 65536 x 16384 | TQ4 | | | | | | | |
| 131072 x 32768 | No TQ | | 1.00x | | | | | 1.000 |
| 131072 x 32768 | TQ8 | | | | | | | |
| 131072 x 32768 | TQ4 | | | | | | | |

## Notes

- Do not compare raw final error across different `p` without using Error Ratio.
- Do not include QJL in this experiment.
- Use summary output values, not manually estimated values from long logs.
- For timing, use warm summary means and payload summaries from the `Timing Summary` / `Payload Summary` sections.
- Keep `k=250`, `oversample=6`, and `l=256` fixed for matrix-size scaling. This keeps the scaling curve focused on matrix size only; experiments that vary `l` should be separate.
