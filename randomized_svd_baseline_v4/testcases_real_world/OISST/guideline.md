# Guideline 01：OISST Real-World Testcase

## 目標

建立一個真實科學資料矩陣，讓 `randomized_svd_multigpu_v4` 可以用：

```bash
--input-file testcases_real_world/OISST/matrix.f32
```

讀取 OISST sea surface temperature anomaly matrix，並測試：

```text
no compression
TQ 8-bit
TQ 4-bit
```

這個 testcase 的科學意義是：

> EOF / PCA / SVD of time-space sea surface temperature anomaly matrix.

也就是把海溫資料整理成「空間點 × 時間」矩陣，然後用 SVD 找主要的海溫變化模式。

---

## 目錄結構

請在：

```text
randomized_svd_baseline_v4/testcases_real_world/OISST/
```

最後目錄應該長這樣：

```text
OISST/
├── .gitignore
├── matrix.f32
├── matrix_debug.f32
├── meta.json
├── README.md
├── exp_turboquant/
│   ├── README.md
│   ├── ctrl.slurm
│   ├── b8-exp.slurm
│   ├── b4-exp.slurm
│   ├── output_logs/
│   └── error_logs/
└── scripts/
    └── prepare_oisst_matrix.py
```

其中：

* `.gitignore`：忽略大型 raw data / cache / `.f32`，避免誤 commit
* `matrix.f32`：給 CUDA program 讀的正式測資
* `matrix_debug.f32`：小尺寸 smoke test 測資，可以沒有，但不要覆蓋正式 `matrix.f32`
* `meta.json`：給人看的 metadata，不給 `.cu` 讀
* `README.md`：記錄資料來源、處理流程、sanity check
* `exp_turboquant/`：OISST no-compression / TQ 8-bit / TQ 4-bit 實驗設定與結果
* `scripts/prepare_oisst_matrix.py`：資料下載 / 轉換 / preprocessing script

注意：`matrix.f32` 和 raw NetCDF / cache 檔案通常很大，不要直接 commit 到 git，除非組員明確決定要把資料放進 repo。

建議 `.gitignore` 內容：

```gitignore
matrix*.f32
raw_data/
cache/
*.nc
*.nc4
*.hdf5
```

---

## 最終 matrix 格式

`matrix.f32` 必須是：

```text
raw float32 binary
row-major
no header
```

不要用 `.mat`、`.npy`、`.nc`、`.hdf5` 當最終輸入格式。

這些格式可以用在 preprocessing，但最後餵給 CUDA program 的檔案一律是：

```text
matrix.f32
```

---

## 為什麼使用 row-major？

目前 distributed randomized SVD 是 row-partition。

也就是矩陣 (A) 的 rows 會被分給不同 MPI rank / GPU。

如果檔案是 row-major，每個 MPI rank 可以直接：

```text
seek 到自己的 row block
只讀自己負責的 rows
```

概念上：

```text
offset_bytes = row_start * n * sizeof(float)
read_count   = local_rows * n
```

讀進來後，CUDA program 會把 local row-major host block 轉成 cuBLAS 需要的 local column-major GPU layout。

所以最終 `.f32` 檔案規格固定為：

```text
FP32 + row-major
```

---

## CUDA program 使用方式

現在 `.cu` 已支援：

```bash
--input-file <path>
```

只要有 `--input-file`，就代表從檔案讀測資；沒有 `--input-file` 則走原本 synthetic/random generator。

不要新增或使用：

```text
--input-dtype
--input-layout
--input-rows
--input-cols
```

shape 直接使用既有參數：

```text
--m <rows>
--n <cols>
```

因此如果 OISST matrix 是 (32768 \times 8192)，執行時要給：

```bash
--m 32768
--n 8192
--input-file testcases_real_world/OISST/matrix.f32
```

---

## Matrix 定義

建立：

[
A \in \mathbb{R}^{m \times n}
]

其中：

```text
rows    = selected ocean grid points
columns = selected days / time snapshots
entry   = SST anomaly at grid point x and day t
```

建議主目標：

```text
m = 32768
n = 8192
```

也就是：

```text
32768 個海洋格點 × 8192 天
```

如果 preprocessing 一開始太慢，可以先做 debug 版本：

```text
m = 8192
n = 4096
input_file = testcases_real_world/OISST/matrix_debug.f32
```

---

## SST anomaly 定義

先讀取原始 SST：

```text
S[i, t] = SST at ocean grid point i on day t
```

然後對每個 grid point 扣掉時間平均：

[
A[i,t] = S[i,t] - \frac{1}{n}\sum_{t=1}^{n}S[i,t]
]

也就是：

```text
SST anomaly = 原始 SST - 該格點的時間平均 SST
```

這樣 SVD 找到的是海溫「變化模式」，而不是平均溫度場。

第一版只需要做 temporal mean centering。

如果時間夠，可以另外做 monthly climatology removal，但不是必要。

---

## 資料處理步驟

### Step 1：下載 / 讀取 OISST

OISST 通常是 NetCDF 格式，資料維度類似：

```text
time x lat x lon
```

主要變數通常是：

```text
sst
```

請使用 Python 讀取，例如：

```text
xarray
netCDF4
numpy
```

#### Recommended download method

Do not manually download files. Write a Python script using `xarray` and NOAA PSL OPeNDAP:

    import xarray as xr

    years = range(2000, 2023)
    urls = [
        f"https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{y}.nc"
        for y in years
    ]

    ds = xr.open_mfdataset(urls, combine="by_coords")
    sst = ds["sst"]

The preprocessing script should download/read the data, select valid ocean grid points and days, remove temporal mean per grid point, and write `matrix.f32`.

Do not download data inside Slurm benchmark jobs. Data preparation is an offline preprocessing step.

---

### Step 2：選時間範圍

需要至少：

```text
8192 days
```

如果先做 debug，可用：

```text
4096 days
```

請在 `README.md` 和 `meta.json` 裡記錄：

```text
start_date
end_date
number_of_days
```

---

### Step 3：選海洋格點

把 lat-lon grid flatten 成一維 spatial points。

移除 land / missing value / NaN 過多的 grid points。

選出：

```text
32768 valid ocean grid points
```

選取方式可以是：

1. 全球 valid ocean points 中固定 seed 隨機抽樣。
2. 選一個海域區域後取 valid ocean points。
3. 對 valid ocean points 做 stride sampling。

請務必 deterministic，也就是固定 random seed。

例如：

```text
seed = 1234
```

---

### Step 4：建立矩陣

建立：

```text
S.shape = (m, n)
```

其中：

```text
m = selected ocean grid points
n = selected days
```

接著扣 temporal mean：

```python
A = S - S.mean(axis=1, keepdims=True)
```

---

### Step 5：確認沒有 NaN / Inf

最後的 `A` 必須滿足：

```text
no NaN
no Inf
dtype = float32
shape = (m, n)
```

如果還有 NaN：

1. 優先移除該 ocean grid point。
2. 若只剩極少數缺值，可用該 grid point 的 temporal mean 補值。
3. 不要讓 NaN 進入 `matrix.f32`。

---

## 儲存 matrix.f32

請用：

```python
A = np.asarray(A, dtype=np.float32, order="C")
A.tofile("matrix.f32")
```

注意：

```text
order="C" = row-major
```

不要默默 transpose。

最終 shape 必須和 `meta.json` 以及執行時的 `--m --n` 一致。

如果同時產生 debug 與正式矩陣，請用不同檔名：

```text
matrix_debug.f32  -> m=8192,  n=4096
matrix.f32        -> m=32768, n=8192
```

---

## meta.json

`meta.json` 不給 CUDA program 讀，是給組員 / AI agent / 報告整理用。

範例：

```json
{
  "name": "OISST_SST_anomaly",
  "rows": 32768,
  "cols": 8192,
  "dtype": "float32",
  "layout": "row_major",
  "file": "matrix.f32",
  "debug_file": "matrix_debug.f32",
  "source": "NOAA OISST daily sea surface temperature",
  "matrix_meaning": "rows are selected ocean grid points, columns are daily time snapshots",
  "preprocessing": "temporal mean removed per grid point",
  "nan_policy": "invalid grid points removed",
  "selection": "global valid ocean points sampled with seed 1234",
  "notes": "SVD corresponds to EOF/PCA analysis of SST anomaly field"
}
```

---

## Sanity Check

完成 `matrix.f32` 後，請跑：

```python
import json
import numpy as np

with open("meta.json") as f:
    meta = json.load(f)

m = meta["rows"]
n = meta["cols"]

A = np.fromfile("matrix.f32", dtype=np.float32).reshape(m, n)

expected_bytes = m * n * np.dtype(np.float32).itemsize
actual_bytes = A.nbytes

print("shape:", A.shape)
print("dtype:", A.dtype)
print("mean abs:", np.mean(np.abs(A)))
print("max abs:", np.max(np.abs(A)))
print("has NaN:", np.isnan(A).any())
print("has Inf:", np.isinf(A).any())
print("mean of row means:", np.mean(A.mean(axis=1)))
print("max abs row mean:", np.max(np.abs(A.mean(axis=1))))
print("file size bytes:", actual_bytes)
print("expected bytes:", expected_bytes)
assert actual_bytes == expected_bytes
```

Expected：

```text
shape = (m, n)
dtype = float32
has NaN = False
has Inf = False
row means should be close to 0
```

---

## Smoke Test

先用小矩陣測 CUDA program 是否能讀檔。

例如先做：

```text
m = 8192
n = 4096
```

然後跑：

```bash
srun --mpi=pmix "$BIN" \
    --m 8192 \
    --n 4096 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/OISST/matrix_debug.f32 \
    --compress-b-mode none \
    --compress-b-bits 0 \
    --compress-subspace-mode none \
    --compress-subspace-bits 0 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only \
    --repeat 5
```

確認能跑後，再做正式大小。

---

## 正式實驗設定

目標大小：

```text
m = 32768
n = 8192
k = 250
oversample = 6
l = 256
subspace_iter = 1
repeat = 50
input_file = testcases_real_world/OISST/matrix.f32
```

比較三組：

### 1. No TQ

```text
--compress-b-mode none
--compress-b-bits 0
--compress-subspace-mode none
--compress-subspace-bits 0
```

### 2. TQ 8-bit

```text
--compress-b-mode tq
--compress-b-bits 8
--compress-subspace-mode tq
--compress-subspace-bits 8
```

### 3. TQ 4-bit

```text
--compress-b-mode tq
--compress-b-bits 4
--compress-subspace-mode tq
--compress-subspace-bits 4
```

不要加 `--device-random-input`。目前 `--input-file` 與 `--device-random-input` 是互斥的，file mode 下矩陣 A 會固定從檔案讀入，repeat 只會更新 randomized SVD 的 Omega。

每一組都建議分成兩個 phase：

1. Timing phase：加 `--no-check-error`，只記錄 runtime / communication metrics。
2. Final-error phase：不要加 `--no-check-error`，只記錄 accuracy metrics。

原因是目前 v4 會把 timing 和 final-error 分開輸出；file loading / row-major to column-major conversion 是 setup，不應計入 `Total Time`。

Timing phase 範例：

```bash
srun --mpi=pmix "$BIN" \
    --m 32768 \
    --n 8192 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/OISST/matrix.f32 \
    --compress-b-mode tq \
    --compress-b-bits 8 \
    --compress-subspace-mode tq \
    --compress-subspace-bits 8 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only \
    --repeat 50 \
    --no-check-error
```

Final-error phase 範例：

```bash
srun --mpi=pmix "$BIN" \
    --m 32768 \
    --n 8192 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/OISST/matrix.f32 \
    --compress-b-mode tq \
    --compress-b-bits 8 \
    --compress-subspace-mode tq \
    --compress-subspace-bits 8 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only \
    --repeat 50
```

---

## 要記錄的結果

請整理：

```text
Total Time
GPU Compute Time
Host/Staging Time
NVLink Time
InfiniBand Time
NVLink Payload
InfiniBand Payload
Global B Relative Error
Final Reconstruction Error
Top-k singular values, if printed
```

現實資料沒有 synthetic theoretical error，所以不要硬套 polynomial/exponential theoretical error。程式在 `--input-file` 模式下的 `theoretical` 和 `err ratio` 會是 `n/a`；OISST 的主要 accuracy 指標應該是 raw `Final Reconstruction Error`，以及和 no-compression baseline 的差距。

`Global B Relative Error` 是 compression diagnostic，只有需要檢查壓縮對中間矩陣 B 的影響時才加 `--check-b-error`。一般正式 timing 可以先不加，避免多餘 copy/check 影響時間。

主要比較：

```text
No TQ baseline
vs
TQ 8-bit
vs
TQ 4-bit
```

重點是：

1. TQ 是否降低 Total Time / InfiniBand Time / NVLink Payload。
2. TQ 是否保持接近 FP32 no-compression 的 final reconstruction error / top-k SVD 結果。
3. 8-bit 和 4-bit 的 speed / accuracy tradeoff。

---

## 報告用一句話

> OISST sea surface temperature data forms a real climate space-time matrix. After subtracting the temporal mean, SVD corresponds to EOF/PCA analysis, where the leading singular vectors capture dominant spatial climate patterns and their temporal coefficients. We use it to test whether TurboQuant-compressed distributed rSVD preserves the FP32 rSVD result while reducing communication time.

---

## Deliverables Checklist

請交付：

```text
matrix.f32
matrix_debug.f32 (optional)
.gitignore
meta.json
README.md
scripts/prepare_oisst_matrix.py
exp_turboquant/{ctrl,b8-exp,b4-exp}.slurm
sanity-check output
```

`README.md` 至少包含：

```text
1. OISST source
2. selected date range
3. selected region or grid-point sampling method
4. final matrix shape
5. preprocessing steps
6. sanity-check result
7. commands used to run no TQ / TQ 8-bit / TQ 4-bit
```
