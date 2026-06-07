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

## Results

Runs executed on Taiwania 2 (`nycugpu_queue`, 2 nodes x 8 V100 = 16 GPUs, account ACD115064),
CUDA 12.8 + OpenMPI 5.0.2, `repeat=50` (warm = 49). Job IDs: accuracy p=0.4/0.6/0.8/1.0 =
946729/946730/946731/946732, scaling (16384x4096) = 946733. Values come from the program's
`Timing Summary` / `Payload Summary` / `Accuracy Summary` (means), parsed by `parse_results.py`.

Derived metrics:
- `Speedup = Total Time_NoTQ / Total Time_TQ`
- `Error Inflation = Error Ratio_TQ / Error Ratio_NoTQ`

## Accuracy Sweep (32768 x 8192)

| p | Method | Total Time (ms) | Speedup | Final Error | Theoretical | Error Ratio | Error Inflation | Global B Rel Err |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.4 | No TQ | 99.44 | 1.00x | 79.927% | 76.700% | 1.0421 | 1.0000 | - |
| 0.4 | TQ8 | 80.17 | 1.24x | 79.934% | 76.700% | 1.0422 | 1.0001 | 1.541% |
| 0.4 | TQ4 | 62.39 | 1.59x | 80.436% | 76.700% | 1.0487 | 1.0064 | 13.273% |
| 0.6 | No TQ | 98.99 | 1.00x | 44.388% | 41.775% | 1.0626 | 1.0000 | - |
| 0.6 | TQ8 | 80.08 | 1.24x | 44.416% | 41.775% | 1.0632 | 1.0006 | 1.476% |
| 0.6 | TQ4 | 61.99 | 1.60x | 46.377% | 41.775% | 1.1102 | 1.0448 | 13.274% |
| 0.8 | No TQ | 98.40 | 1.00x | 16.655% | 15.272% | 1.0906 | 1.0000 | - |
| 0.8 | TQ8 | 80.59 | 1.22x | 16.752% | 15.272% | 1.0969 | 1.0058 | 1.390% |
| 0.8 | TQ4 | 63.55 | 1.55x | 21.700% | 15.272% | 1.4209 | 1.3029 | 13.180% |
| 1.0 | No TQ | 97.91 | 1.00x | 5.463% | 4.851% | 1.1263 | 1.0000 | - |
| 1.0 | TQ8 | 80.04 | 1.22x | 5.730% | 4.851% | 1.1812 | 1.0488 | 1.321% |
| 1.0 | TQ4 | 63.06 | 1.55x | 14.365% | 4.851% | 2.9614 | 2.6295 | 13.087% |

Takeaways:
- Speedup is essentially independent of `p` (payload sizes do not depend on the spectrum): TQ8 ~1.22-1.24x, TQ4 ~1.55-1.60x.
- TQ8 error inflation stays <=1.05 across the whole sweep; it is negligible (<1.01x) for the hard/medium spectra (p=0.4-0.8).
- TQ4 error inflation grows sharply as the spectrum steepens: 1.006 (p=0.4) -> 2.63 (p=1.0). As the theoretical error shrinks, the fixed 4-bit quantization noise floor (~13% Global B error) dominates the ratio.

## Scaling (p=0.6)

`32768 x 8192` is reused from the accuracy sweep (p=0.6); only `16384 x 4096` was run by the scaling job.

| Matrix Size | Method | Total Time (ms) | Speedup | InfiniBand Time (ms) | NVLink Time (ms) | Final Error | Error Ratio | Error Inflation | IB Payload (MiB) | NVLink Payload (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16384 x 4096 | No TQ | 68.82 | 1.00x | 29.61 | 0.445 | 41.980% | 1.0742 | 1.0000 | 32.00 | 112.00 |
| 16384 x 4096 | TQ8 | 59.43 | 1.16x | 25.31 | 0.560 | 42.015% | 1.0751 | 1.0008 | 20.06 | 84.44 |
| 16384 x 4096 | TQ4 | 50.27 | 1.37x | 25.22 | 0.524 | 44.273% | 1.1329 | 1.0546 | 18.06 | 70.44 |
| 32768 x 8192 | No TQ | 98.99 | 1.00x | 33.41 | 0.797 | 44.388% | 1.0626 | 1.0000 | 48.00 | 224.00 |
| 32768 x 8192 | TQ8 | 80.08 | 1.24x | 28.78 | 0.787 | 44.416% | 1.0632 | 1.0006 | 24.12 | 168.88 |
| 32768 x 8192 | TQ4 | 61.99 | 1.60x | 27.33 | 0.685 | 46.377% | 1.1102 | 1.0448 | 20.12 | 140.88 |

Takeaways:
- TQ speedup grows with matrix size: TQ8 1.16x -> 1.24x, TQ4 1.37x -> 1.60x going 16384x4096 -> 32768x8192. Communication compression is more beneficial as payloads grow.
- IB payload is cut ~2x by TQ8 (48 -> 24 MiB) and ~2.4x by TQ4 (48 -> 20 MiB); accuracy impact at p=0.6 stays tiny for TQ8.

## Conclusion

Positive result for TQ8. At the standard 32768 x 8192 / p=0.6 stress test, TQ8 gives a 1.24x
end-to-end speedup with negligible accuracy cost (Error Inflation 1.0006), and the speedup grows
with matrix size. TQ4 is faster (1.6x) but its accuracy cost is spectrum-dependent: acceptable on
slowly decaying spectra (p<=0.6) and unacceptable on fast-decaying spectra (p>=0.8, up to 2.6x
error inflation at p=1.0). Recommended default: TQ8.


## Notes

- Do not compare raw final error across different `p` without using Error Ratio.
- Do not include QJL in this experiment.
- Use summary output values, not manually estimated values from long logs.
- For timing, use warm summary means and payload summaries from the `Timing Summary` / `Payload Summary` sections.
- Keep `k=250`, `oversample=6`, and `l=256` fixed for matrix-size scaling. This keeps the scaling curve focused on matrix size only; experiments that vary `l` should be separate.
