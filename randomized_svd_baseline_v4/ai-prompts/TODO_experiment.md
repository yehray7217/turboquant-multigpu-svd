# Guideline：Synthetic Spectrum Accuracy + Scaling Experiments

## 目標

這組實驗用 synthetic matrix 來回答兩個問題：

1. **Accuracy impact**：加入 TQ 之後，randomized SVD 的 approximation error 會額外變差多少？
2. **Scaling behavior**：矩陣變大時，TQ 是否能帶來更明顯的 runtime / communication benefit？

Synthetic matrix 的好處是我們知道 singular spectrum，因此可以計算 theoretical best rank-(k) error，進一步比較：

```text
Final Reconstruction Error / Theoretical Error
```

也就是目前程式輸出的：

```text
Error Ratio
```

---

## Common Config

除非特別說明，所有實驗都使用：

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
```

比較三組方法：

```text
No TQ
TQ 8-bit
TQ 4-bit
```

---

# Experiment A：Accuracy / Spectrum Difficulty Sweep

## 目的

測試不同 polynomial spectrum decay difficulty 下，TQ 對 final randomized SVD error 的影響。

Polynomial spectrum 定義：

```text
sigma_i = i^{-p}
```

這裡的 `p` 是 spectrum decay parameter，不是 oversampling。

## 固定參數

```text
m = 32768
n = 8192
k = 250
oversample = 6
l = 256
spectrum_rank = 8192
subspace_iter = 1
repeat = 50
```

## 變動參數

```text
spectrum_decay_param = 0.4, 0.6, 0.8, 1.0
method = No TQ, TQ 8-bit, TQ 4-bit
```

總共：

```text
4 decay settings × 3 methods = 12 runs
```

## Run Groups

For each `p ∈ {0.4, 0.6, 0.8, 1.0}`:

### No TQ

```text
--compress-b-mode none
--compress-b-bits 0
--compress-subspace-mode none
--compress-subspace-bits 0
```

### TQ 8-bit

```text
--compress-b-mode tq
--compress-b-bits 8
--compress-subspace-mode tq
--compress-subspace-bits 8
```

### TQ 4-bit

```text
--compress-b-mode tq
--compress-b-bits 4
--compress-subspace-mode tq
--compress-subspace-bits 4
```

## Metrics to Record

For each run, record:

```text
Total Time
GPU Compute Time
Host/Staging Time
NVLink Time
InfiniBand Time
Other/Sync Time
NVLink Payload
InfiniBand Payload
Final Reconstruction Error
Theoretical Error
Error Ratio
Global B Relative Error
```

## Derived Metrics

### Speedup

For each decay parameter:

```text
Speedup_TQ8 = Total Time_NoTQ / Total Time_TQ8
Speedup_TQ4 = Total Time_NoTQ / Total Time_TQ4
```

### Error Inflation

Use division, not subtraction.

```text
Error Inflation_TQ8 = Error Ratio_TQ8 / Error Ratio_NoTQ
Error Inflation_TQ4 = Error Ratio_TQ4 / Error Ratio_NoTQ
```

Interpretation:

```text
Error Inflation ≈ 1.00  means TQ adds almost no extra approximation error.
Error Inflation > 1.00  means TQ increases final error relative to No TQ.
```

Example:

```text
No TQ Error Ratio = 1.063
TQ8 Error Ratio   = 1.064

Error Inflation = 1.064 / 1.063 ≈ 1.00094
```

This means TQ8 adds almost no extra error.

---

# Experiment B：Matrix Size Scaling

## 目的

測試矩陣變大時，TQ 對 runtime / communication 是否更有幫助。

## 固定參數

```text
spectrum_decay_param = 0.6
spectrum_decay_mode = polynomial
k = 250
oversample = 6
l = 256
subspace_iter = 1
spectrum_rank = min(m, n)
repeat = 50
```

## 變動矩陣大小

```text
16384 x 4096
32768 x 8192
```

其中：

```text
32768 x 8192, p = 0.6
```

已經會在 Experiment A 裡跑到，所以 scaling experiment 實際上只需要補：

```text
16384 x 4096, p = 0.6
```

的三組方法。

## Run Groups

For each matrix size:

### No TQ

```text
--compress-b-mode none
--compress-b-bits 0
--compress-subspace-mode none
--compress-subspace-bits 0
```

### TQ 8-bit

```text
--compress-b-mode tq
--compress-b-bits 8
--compress-subspace-mode tq
--compress-subspace-bits 8
```

### TQ 4-bit

```text
--compress-b-mode tq
--compress-b-bits 4
--compress-subspace-mode tq
--compress-subspace-bits 4
```

## Metrics to Record

Same as Experiment A:

```text
Total Time
GPU Compute Time
Host/Staging Time
NVLink Time
InfiniBand Time
Other/Sync Time
NVLink Payload
InfiniBand Payload
Final Reconstruction Error
Theoretical Error
Error Ratio
Global B Relative Error
```

## Derived Metrics

### Speedup

```text
Speedup = Total Time_NoTQ / Total Time_TQ
```

### Error Inflation

```text
Error Inflation = Error Ratio_TQ / Error Ratio_NoTQ
```

### Payload Reduction

If payload metrics are available:

```text
InfiniBand Payload Reduction = InfiniBand Payload_NoTQ / InfiniBand Payload_TQ
NVLink Payload Reduction     = NVLink Payload_NoTQ / NVLink Payload_TQ
```

---

# Suggested Experiment Folder

Create:

```text
randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/
```

Recommended structure:

```text
exp_synthetic_accuracy_scaling/
├── README.md
├── run_accuracy_sweep.slurm
├── run_scaling.slurm
├── output_logs/
└── error_logs/
```

If one script becomes too long, split by parameter:

```text
p04.slurm
p06.slurm
p08.slurm
p10.slurm
scaling_16k_4k.slurm
```

---

# Suggested README Table

## Accuracy Sweep Table

|   p | Method | Total Time | Speedup | Final Error | Theoretical | Error Ratio | Error Inflation | Global B Rel Err |
| --: | ------ | ---------: | ------: | ----------: | ----------: | ----------: | --------------: | ---------------: |
| 0.4 | No TQ  |            |   1.00x |             |             |             |           1.000 |                - |
| 0.4 | TQ8    |            |         |             |             |             |                 |                  |
| 0.4 | TQ4    |            |         |             |             |             |                 |                  |
| 0.6 | No TQ  |            |   1.00x |             |             |             |           1.000 |                - |
| 0.6 | TQ8    |            |         |             |             |             |                 |                  |
| 0.6 | TQ4    |            |         |             |             |             |                 |                  |
| 0.8 | No TQ  |            |   1.00x |             |             |             |           1.000 |                - |
| 0.8 | TQ8    |            |         |             |             |             |                 |                  |
| 0.8 | TQ4    |            |         |             |             |             |                 |                  |
| 1.0 | No TQ  |            |   1.00x |             |             |             |           1.000 |                - |
| 1.0 | TQ8    |            |         |             |             |             |                 |                  |
| 1.0 | TQ4    |            |         |             |             |             |                 |                  |

## Scaling Table

| Matrix Size  | Method | Total Time | Speedup | InfiniBand Time | NVLink Time | Final Error | Error Ratio | Error Inflation |
| ------------ | ------ | ---------: | ------: | --------------: | ----------: | ----------: | ----------: | --------------: |
| 16384 x 4096 | No TQ  |            |   1.00x |                 |             |             |             |           1.000 |
| 16384 x 4096 | TQ8    |            |         |                 |             |             |             |                 |
| 16384 x 4096 | TQ4    |            |         |                 |             |             |             |                 |
| 32768 x 8192 | No TQ  |            |   1.00x |                 |             |             |             |           1.000 |
| 32768 x 8192 | TQ8    |            |         |                 |             |             |             |                 |
| 32768 x 8192 | TQ4    |            |         |                 |             |             |             |                 |

---

# Plot Suggestions

## Plot 1：Accuracy Sweep

x-axis:

```text
polynomial decay parameter p
```

y-axis:

```text
Error Inflation vs No TQ
```

lines:

```text
TQ 8-bit
TQ 4-bit
```

Expected story:

```text
TQ8 should stay close to 1.0.
TQ4 may show larger error inflation, especially for harder spectra.
```

## Plot 2：Scaling

x-axis:

```text
matrix size
```

y-axis:

```text
Total Time speedup vs No TQ
```

bars:

```text
TQ 8-bit
TQ 4-bit
```

Expected story:

```text
Larger matrices should make communication compression more useful.
```

---

# Report Interpretation

Use this experiment to say:

> Synthetic matrices with controlled polynomial singular spectra allow us to isolate the accuracy impact of TurboQuant. Since the theoretical best rank-k error is known, we compare Error Ratio and Error Inflation to measure how much additional approximation error is introduced by communication compression.

For scaling:

> Matrix-size scaling evaluates whether TurboQuant becomes more beneficial as communication payloads grow. We compare total runtime, InfiniBand/NVLink time, and payload reduction across matrix sizes.

---

# Minimum Completion Criteria

At minimum, finish:

```text
Experiment A:
p = 0.4, 0.6, 0.8, 1.0
method = No TQ, TQ8, TQ4

Experiment B:
size = 16384 x 4096 and 32768 x 8192
p = 0.6
method = No TQ, TQ8, TQ4
```

Since `32768 x 8192, p=0.6` overlaps with Experiment A, reuse that result in the scaling table.

Do not rerun duplicated configs unless necessary.

---

# Notes

1. Do not compare raw final error across different `p` without also using `Error Ratio`.
2. Use `Error Inflation = Error Ratio_TQ / Error Ratio_NoTQ` for the main accuracy-impact conclusion.
3. Keep `l = 256` because current TQ codebooks and RHT path are stable for this dimension.
4. Do not include QJL in this experiment unless explicitly requested. This experiment is about No TQ vs TQ8 vs TQ4.
5. Use `repeat = 50` for final reported numbers.
6. Use summary output, not manually estimated values from logs.

```
```
