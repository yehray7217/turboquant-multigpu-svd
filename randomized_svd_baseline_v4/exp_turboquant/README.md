# TurboQuant Experiment

Experiment purpose: measure whether applying TurboQuant to both `B = sum_i Q_i^T A_i` and subspace iteration `Z = sum_i A_i^T Q_i` reduces runtime without unacceptable reconstruction error.

## Takeaway

Pure TQ 8-bit is the best default configuration in this experiment: it reduces total time from `187.711 ms` to `161.719 ms` while keeping final reconstruction error almost unchanged.

Pure TQ 4-bit is the fastest configuration, but it has a visible accuracy cost. TQ-QJL is currently a negative result: QJL adds GPU compute overhead, and neither 4-bit nor 8-bit QJL beats pure TQ at the same bit setting.

## Results

| Group | B Compression | Subspace Compression | Total Time mean | GPU Compute mean | Host/Staging mean | InfiniBand mean | B Relative Error | Final Error mean | Theoretical | Error Ratio | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | none | none | 187.711 ms | 83.2787 ms | 87.9321 ms | 49.7008 ms | skipped | 44.3929% | 41.7747% | 1.06268 | No compression |
| b4-exp | TQ 4-bit | TQ 4-bit | 137.931 ms | 82.4714 ms | 48.4773 ms | 36.8028 ms | 13.2720% | 46.1877% | 41.7747% | 1.10564 | Fastest, but accuracy degrades |
| b8-exp | TQ 8-bit | TQ 8-bit | 161.719 ms | 100.945 ms | 51.6653 ms | 40.1797 ms | 1.47439% | 44.4165% | 41.7747% | 1.06324 | Recommended setting |
| b4-qjl-exp | TQ-QJL 4-bit | TQ-QJL 4-bit | 163.923 ms | 108.007 ms | 48.8285 ms | 38.9901 ms | 33.8081% | 54.4296% | 41.7747% | 1.30293 | Negative result |
| b8-qjl-exp | TQ-QJL 8-bit | TQ-QJL 8-bit | 182.037 ms | 118.305 ms | 51.2657 ms | 41.0503 ms | 2.9940% | 44.4901% | 41.7747% | 1.06500 | Accuracy is acceptable, but slower than pure TQ8 |

## Payloads

| Group | Host-GPU Payload | NVLink Payload | InfiniBand Payload |
|---|---:|---:|---:|
| control | 592.000 MiB | 224.000 MiB | 80.000 MiB |
| b4-exp | 404.625 MiB | 57.750 MiB | 24.250 MiB |
| b8-exp | 424.625 MiB | 113.750 MiB | 32.250 MiB |
| b4-qjl-exp | 405.250 MiB | 59.500 MiB | 24.500 MiB |
| b8-qjl-exp | 425.250 MiB | 115.500 MiB | 32.500 MiB |

## Config

```text
m = 65536
n = 16384
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
ctrl.slurm        -> no compression
b4-exp.slurm      -> B + subspace TQ 4-bit
b8-exp.slurm      -> B + subspace TQ 8-bit
b4-qjl-exp.slurm  -> B + subspace TQ-QJL 4-bit
b8-qjl-exp.slurm  -> B + subspace TQ-QJL 8-bit
```

Additional QJL dimension/alpha sweep results are stored in `../exp_tq8_qjl_dim_alpha_sweep`. That sweep also supports the same conclusion: QJL tuning changes the tradeoff, but no tested QJL configuration beats pure TQ 8-bit.
