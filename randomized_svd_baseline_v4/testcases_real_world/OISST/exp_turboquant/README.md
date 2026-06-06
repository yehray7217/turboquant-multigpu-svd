# OISST TurboQuant Experiment

Compare distributed rSVD on the OISST SST anomaly matrix with:

- no compression
- TQ 8-bit on `B` and subspace `Z`
- TQ 4-bit on `B` and subspace `Z`

## Config

| Parameter | Value |
|---|---:|
| m | 32768 |
| n | 8192 |
| k | 250 |
| oversample | 6 |
| ngpus | 16 |
| gpus-per-rank | 8 |
| subspace-iter | 1 |
| repeat | 50 |
| input-file | `testcases_real_world/OISST/matrix.f32` |

## Results

| Output | Group | Total Time Mean (ms) | Final Error Mean | Delta vs No TQ | Notes |
|---|---|---:|---:|---:|---|
| ctrl.out | control | TBD | TBD | 0 |  |
| b8-exp.out | TQ8 | TBD | TBD | TBD |  |
| b4-exp.out | TQ4 | TBD | TBD | TBD |  |

## Notes

- Use timing phase results for runtime and payload summaries.
- Use final-error phase results for `Final Reconstruction Error`.
- `theoretical` and `err ratio` are `n/a` in `--input-file` mode.
