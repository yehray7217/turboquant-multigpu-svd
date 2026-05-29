# TQ8 QJL Dimension/Alpha Sweep

Experiment purpose: compare pure 8-bit TurboQuant against 8-bit TQ-QJL while sweeping QJL residual sketch dimension and residual correction strength.

| Group | Compression | qjl_dim | qjl_alpha | Total Time mean | GPU Compute mean | Host/Staging mean | InfiniBand mean | Final Error mean | Theoretical | Error Ratio | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| control | TQ 8-bit | - | - | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Pure TQ 8-bit baseline |
| qjl-d64-a025 | TQ-QJL 8-bit | 64 | 0.25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Lower QJL compute/payload, weak residual correction |
| qjl-d64-a05 | TQ-QJL 8-bit | 64 | 0.5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Lower QJL compute/payload, medium residual correction |
| qjl-d64-a1 | TQ-QJL 8-bit | 64 | 1.0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Lower QJL compute/payload, full residual correction |
| qjl-d128-a025 | TQ-QJL 8-bit | 128 | 0.25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Half-size QJL sketch |
| qjl-d128-a05 | TQ-QJL 8-bit | 128 | 0.5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Half-size QJL sketch |
| qjl-d128-a1 | TQ-QJL 8-bit | 128 | 1.0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Half-size QJL sketch |
| qjl-d256-a025 | TQ-QJL 8-bit | 256 | 0.25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Current default-dimensional QJL sketch |
| qjl-d256-a05 | TQ-QJL 8-bit | 256 | 0.5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Current default-dimensional QJL sketch |
| qjl-d256-a1 | TQ-QJL 8-bit | 256 | 1.0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Current default-dimensional QJL sketch |
| qjl-d512-a025 | TQ-QJL 8-bit | 512 | 0.25 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Larger QJL sketch |
| qjl-d512-a05 | TQ-QJL 8-bit | 512 | 0.5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Larger QJL sketch |
| qjl-d512-a1 | TQ-QJL 8-bit | 512 | 1.0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Larger QJL sketch |

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
control = TQ 8-bit on B and subspace Z
experiment = TQ-QJL 8-bit on B and subspace Z
```

## Files

```text
ctrl.slurm              -> pure TQ 8-bit control
qjl-d64-a025.slurm      -> qjl_dim=64,  qjl_alpha=0.25
qjl-d64-a05.slurm       -> qjl_dim=64,  qjl_alpha=0.5
qjl-d64-a1.slurm        -> qjl_dim=64,  qjl_alpha=1.0
qjl-d128-a025.slurm     -> qjl_dim=128, qjl_alpha=0.25
qjl-d128-a05.slurm      -> qjl_dim=128, qjl_alpha=0.5
qjl-d128-a1.slurm       -> qjl_dim=128, qjl_alpha=1.0
qjl-d256-a025.slurm     -> qjl_dim=256, qjl_alpha=0.25
qjl-d256-a05.slurm      -> qjl_dim=256, qjl_alpha=0.5
qjl-d256-a1.slurm       -> qjl_dim=256, qjl_alpha=1.0
qjl-d512-a025.slurm     -> qjl_dim=512, qjl_alpha=0.25
qjl-d512-a05.slurm      -> qjl_dim=512, qjl_alpha=0.5
qjl-d512-a1.slurm       -> qjl_dim=512, qjl_alpha=1.0
```
