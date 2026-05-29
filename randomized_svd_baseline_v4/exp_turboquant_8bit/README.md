# TurboQuant 4-bit Experiment

Experiment purpose: measure whether applying 4-bit TurboQuant to both `B = sum_i Q_i^T A_i` and subspace iteration `Z = sum_i A_i^T Q_i` reduces runtime without unacceptable reconstruction error.

| Group | B Compression | Subspace Compression | Total Time mean | Total Time min | Total Time stddev | Final Error mean | Final Error min | Final Error stddev | Theoretical | Error Ratio | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | none | none | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | No compression |
| experiment | TQ 4-bit | TQ 4-bit | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Compress both B reduce and subspace Z communication |

## Config

```text
m = 32768
n = 8192
k = 250
oversample = 6
l = 256
ngpus = 16
gpus_per_rank = 8
spectrum_decay_mode = polynomial
spectrum_decay_param = 0.6
spectrum_rank = 8192
subspace_iter = 1
repeat = 50
```

## Files

```text
ctrl.slurm  -> no compression
exp.slurm   -> B + subspace TQ 4-bit
```
