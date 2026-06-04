# Guideline 02：JHTDB / CFD POD Real-World Testcase

## 目標

建立一個來自 Johns Hopkins Turbulence Database, JHTDB 的 CFD / turbulence matrix，讓 `randomized_svd_multigpu_v4` 可以用：

```bash
--input-file testcases_real_world/POD/matrix.f32
```

讀取真實流體力學資料，並測試：

```text
no compression
TQ 8-bit
TQ 4-bit
```

這個 testcase 的科學意義是：

> POD / SVD of CFD flow snapshots.

POD, Proper Orthogonal Decomposition, 在流體力學中常用來把很多個 flow snapshots 疊成矩陣，再用 SVD 找出 dominant coherent structures / flow modes。

---

## 本任務分成兩階段

這個 testcase 目前有不確定性，所以請先做 feasibility check。

### Phase 1：確認 JHTDB 資料能不能抓

請優先調查以下兩個 dataset：

```text
1. isotropic1024coarse
2. transition_bl
```

重點是確認它們是否能抓到足夠多個 time snapshots。

目標是至少能做：

```text
32768 x 4096
```

更理想是：

```text
32768 x 5028
```

其中：

```text
32768 = 32 x 32 x 32 spatial points
4096 / 5028 = time snapshots
```

### Phase 2：如果可行，產生 `matrix.f32`

如果 dataset 能抓到足夠 time snapshots，就建立 POD snapshot matrix：

[
A \in \mathbb{R}^{m \times n}
]

其中：

```text
rows    = spatial sample points
columns = time snapshots
entry   = pressure or one velocity component at a spatial point and time
```

---

## 目錄結構

請在：

```text
randomized_svd_baseline_v4/testcases_real_world/POD/
```

最後目錄應該長這樣：

```text
POD/
├── .gitignore
├── matrix.f32
├── matrix_debug.f32
├── meta.json
├── README.md
├── feasibility_report.md
├── exp_turboquant/
│   ├── README.md
│   ├── ctrl.slurm
│   ├── b8-exp.slurm
│   ├── b4-exp.slurm
│   ├── output_logs/
│   └── error_logs/
└── scripts/
    └── prepare_jhtdb_pod_matrix.py
```

其中：

* `.gitignore`：忽略大型 raw data / cache / `.f32`，避免誤 commit
* `matrix.f32`：給 CUDA program 讀的正式測資
* `matrix_debug.f32`：小尺寸 smoke test 測資，可以沒有，但不要覆蓋正式 `matrix.f32`
* `meta.json`：給人看的 metadata，不給 `.cu` 讀
* `README.md`：記錄資料來源、處理流程、sanity check
* `feasibility_report.md`：記錄 JHTDB access / dataset feasibility
* `exp_turboquant/`：POD no-compression / TQ 8-bit / TQ 4-bit 實驗設定與結果
* `scripts/prepare_jhtdb_pod_matrix.py`：資料下載 / 轉換 / preprocessing script

如果最後發現 JHTDB time-resolved data 不可行，至少要產出：

```text
feasibility_report.md
```

注意：`matrix.f32`、raw JHTDB cutout、HDF5/Zarr/NetCDF cache 通常很大，不要直接 commit 到 git。

建議 `.gitignore` 內容：

```gitignore
matrix*.f32
raw_data/
cache/
*.h5
*.hdf5
*.nc
*.nc4
*.zarr/
```

---

## 最終 matrix 格式

`matrix.f32` 必須是：

```text
raw float32 binary
row-major
no header
```

不要用 `.mat`、`.npy`、`.hdf5` 當 CUDA program 的最終輸入。

HDF5 / Zarr / NetCDF 可以用於 preprocessing，但最後交給 CUDA program 的檔案一律是：

```text
matrix.f32
```

---

## 為什麼使用 row-major？

目前 distributed randomized SVD 是 row-partition。

如果檔案是 row-major，每個 MPI rank 可以直接 seek 到自己的 row block，只讀自己負責的 rows：

```text
offset_bytes = row_start * n * sizeof(float)
read_count   = local_rows * n
```

讀進 host row-major buffer 後，CUDA program 會轉成 cuBLAS 需要的 local column-major GPU layout。

所以最終 `.f32` 檔案固定為：

```text
FP32 + row-major
```

---

## Matrix 定義：time-resolved POD 版本

如果成功取得 time-resolved JHTDB data，請建立：

[
A \in \mathbb{R}^{32768 \times T}
]

其中：

```text
32768 = 32 x 32 x 32 spatial cutout
T     = number of time snapshots
```

推薦大小：

```text
A = 32768 x 4096
```

如果 `isotropic1024coarse` 可完整取得 5028 timesteps，也可以做：

```text
A = 32768 x 5028
```

注意：

```text
5028 不需要對齊 2 的冪次。
```

只要 `k + oversample = l` 是 TurboQuant 支援的 dimension，例如：

```text
l = 256
```

即可。

---

## 欄位選擇：pressure 還是 velocity？

JHTDB 通常提供：

```text
pressure
velocity components: u, v, w
```

第一版建議選一個 scalar field：

```text
u velocity component
```

或：

```text
pressure
```

建議優先選：

```text
u velocity component
```

理由：POD of velocity snapshots 在 CFD / turbulence analysis 中比較好解釋。

如果使用 pressure，也可以，但 README 要寫清楚：

```text
entry = pressure fluctuation at spatial point x and time t
```

---

## Preprocessing：是否要扣平均？

建議扣 temporal mean：

[
A[x,t] = field(x,t) - \frac{1}{T}\sum_t field(x,t)
]

這樣 SVD / POD 找的是 fluctuation modes，而不是平均流場。

如果先做 debug，也可以不扣平均，但正式 benchmark 建議扣平均。

---

## Feasibility Check

請先確認以下事項。

### 1. Dataset 是否 time-resolved？

要確認 dataset 是否提供很多時間點。

不要只看空間解析度很大，例如：

```text
4096^3
8192^3
32768^3
```

很多這類資料只有：

```text
1, 5, 6, 11, 40 snapshots
```

這種不適合做 POD time-snapshot matrix。

請找以下資訊：

```text
available time steps
time range
time step
number of frames / snapshots
available variables
cutout API support
```

---

### 2. 優先確認 `isotropic1024coarse`

我們目前知道 `isotropic1024coarse` 可能有：

```text
5028 timesteps
time t between 0 and 10.056 s
pressure + 3 velocity components
```

請確認：

```text
1. JHTDB API / giverny 是否能 access isotropic1024coarse
2. dataset name 實際叫什麼
3. time indices 如何指定
4. 是否能取得 32 x 32 x 32 cutout
5. 是否能批次抓 4096 或 5028 time snapshots
```

---

### 3. 第二候選：`transition_bl`

請確認：

```text
1. 是否有約 4701 time snapshots
2. 是否可用 JHTDB API / cutout service 抓資料
3. spatial grid 如何選取 32768 points
4. 可用變數有哪些
```

---

### 4. 如果只能取得 single snapshot

如果只能取得像 Hugging Face 的：

```text
coarse_t420.hdf5
shape = (1024, 1024, 1024, 4)
```

這代表它是單一時間點 snapshot，不是 time series。

這種不能稱為 time-resolved POD。

但可以做 fallback：

> spatial patch matrix from one turbulence snapshot.

Fallback matrix：

```text
A = 32768 x 8192
```

其中：

```text
rows    = flattened 32 x 32 x 32 patch
columns = different spatial patches
entry   = pressure or velocity value inside patch
```

這仍然是真實 CFD 資料，但請在 README 中明確寫：

```text
This is a spatial patch matrix from one turbulence snapshot, not a time-resolved POD matrix.
```

---

## 建議資料抓取流程

### Step 1：取得 API token

使用學校帳號或 JHTDB account 取得 token。

請在 README 記錄：

```text
API used: giverny / REST / cutout service / other
dataset name
variable
time range
spatial cutout
```

不要把 token commit 到 git。

---

### Step 2：small proof-of-data

先抓小矩陣確認流程：

```text
spatial cutout = 16 x 16 x 16 = 4096 points
time snapshots = 128 or 512
matrix = 4096 x 128 / 4096 x 512
```

確認：

```text
1. 能讀資料
2. shape 正確
3. 沒有 NaN / Inf
4. 能存成 row-major FP32 matrix.f32
5. CUDA program 可用 --input-file 跑
```

---

### Step 3：main POD matrix

如果 time snapshots 足夠，建立：

```text
spatial cutout = 32 x 32 x 32 = 32768 points
time snapshots = 4096 or 5028
```

得到：

```text
A = 32768 x 4096
```

或：

```text
A = 32768 x 5028
```

如果資料抓取流程一開始太慢，請先做 debug 版本：

```text
m = 4096
n = 128 or 512
input_file = testcases_real_world/POD/matrix_debug.f32
```

---

## 儲存 matrix.f32

請用：

```python
A = np.asarray(A, dtype=np.float32, order="C")
A.tofile("matrix.f32")
```

`order="C"` 代表 row-major。

不要默默 transpose。

最終 shape 必須和 `meta.json` 以及執行時的 `--m --n` 一致。

如果同時產生 debug 與正式矩陣，請用不同檔名：

```text
matrix_debug.f32  -> small proof-of-data / smoke test
matrix.f32        -> formal benchmark matrix
```

---

## meta.json

`meta.json` 不給 CUDA program 讀，只給組員、AI agent、報告整理用。

Time-resolved POD 版本範例：

```json
{
  "name": "JHTDB_isotropic1024coarse_POD",
  "rows": 32768,
  "cols": 4096,
  "dtype": "float32",
  "layout": "row_major",
  "file": "matrix.f32",
  "debug_file": "matrix_debug.f32",
  "source": "Johns Hopkins Turbulence Database",
  "dataset": "isotropic1024coarse",
  "matrix_meaning": "rows are 32x32x32 spatial points, columns are time snapshots",
  "field": "u velocity component",
  "preprocessing": "temporal mean removed per spatial point",
  "spatial_cutout": "describe x/y/z range here",
  "time_range": "describe time indices here",
  "notes": "SVD corresponds to POD of time-resolved CFD snapshots"
}
```

Fallback spatial patch 版本範例：

```json
{
  "name": "JHTDB_single_snapshot_patch_matrix",
  "rows": 32768,
  "cols": 8192,
  "dtype": "float32",
  "layout": "row_major",
  "file": "matrix.f32",
  "debug_file": "matrix_debug.f32",
  "source": "JHTDB-derived single turbulence snapshot",
  "matrix_meaning": "rows are flattened 32x32x32 spatial patch coordinates, columns are different spatial patches",
  "field": "pressure or velocity component",
  "preprocessing": "none or per-patch mean removed",
  "notes": "This is a spatial patch matrix from one 3D turbulence snapshot, not time-resolved POD"
}
```

---

## feasibility_report.md

如果調查後發現 JHTDB time-resolved data 不可行，請完成 `feasibility_report.md`，內容至少包含：

```text
1. checked datasets
2. available time snapshots
3. available variables
4. whether API token/access succeeded
5. whether cutout service succeeded
6. why POD matrix is feasible or infeasible
7. fallback recommendation
```

建議表格：

| Dataset                 | Time-resolved? | Available snapshots | Can build 32768 x 4096? | Notes                 |
| ----------------------- | -------------: | ------------------: | ----------------------: | --------------------- |
| isotropic1024coarse     |            TBD |                 TBD |                     TBD |                       |
| transition_bl           |            TBD |                 TBD |                     TBD |                       |
| single snapshot HF file |             no |                   1 |                      no | possible patch matrix |

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
```

如果做了 temporal mean removal，row means 應該接近 0。

---

## CUDA Smoke Test

如果 debug matrix 是：

```text
4096 x 512
```

可以先跑：

```bash
srun --mpi=pmix "$BIN" \
    --m 4096 \
    --n 512 \
    --k 128 \
    --oversample 128 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/POD/matrix_debug.f32 \
    --compress-b-mode none \
    --compress-b-bits 0 \
    --compress-subspace-mode none \
    --compress-subspace-bits 0 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only \
    --repeat 5
```

如果成功，再跑正式三組。

---

## 正式實驗設定

若 matrix 是：

```text
32768 x 4096
```

建議：

```text
m = 32768
n = 4096
k = 250
oversample = 6
l = 256
subspace_iter = 1
repeat = 50
input_file = testcases_real_world/POD/matrix.f32
```

若覺得 (k=250) 對 (n=4096) 太高，也可以改：

```text
k = 128
oversample = 128
l = 256
```

但若要和 synthetic / OISST 結果對齊，優先使用：

```text
k = 250
oversample = 6
l = 256
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
    --n 4096 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/POD/matrix.f32 \
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
    --n 4096 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/POD/matrix.f32 \
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

## 比較三組

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

真實資料沒有 synthetic theoretical error，所以不要硬套 polynomial / exponential theoretical error。程式在 `--input-file` 模式下的 `theoretical` 和 `err ratio` 會是 `n/a`；POD 的主要 accuracy 指標應該是 raw `Final Reconstruction Error`，以及和 no-compression baseline 的差距。

`Global B Relative Error` 是 compression diagnostic，只有需要檢查壓縮對中間矩陣 B 的影響時才加 `--check-b-error`。一般正式 timing 可以先不加，避免多餘 copy/check 影響時間。

主要比較：

```text
No TQ baseline
vs
TQ 8-bit
vs
TQ 4-bit
```

---

## 報告用一句話

Time-resolved POD 版本：

> We use a JHTDB turbulence dataset to build a CFD snapshot matrix. Each column is one flow snapshot over a fixed (32^3) spatial cutout. SVD of this matrix corresponds to POD, which extracts dominant coherent flow structures. This provides a real engineering benchmark for distributed randomized SVD.

Fallback patch matrix 版本：

> If time-resolved JHTDB access is not feasible, we construct a spatial patch matrix from a single 3D turbulence snapshot. This is not time-resolved POD, but it still provides a real CFD-derived dense matrix for testing distributed randomized SVD and TurboQuant communication compression.

---

## Deliverables Checklist

請交付：

```text
feasibility_report.md
matrix.f32, if feasible
matrix_debug.f32, optional
.gitignore
meta.json, if feasible
README.md
scripts/prepare_jhtdb_pod_matrix.py
exp_turboquant/{ctrl,b8-exp,b4-exp}.slurm, if feasible
sanity-check output
commands used to run no TQ / TQ 8-bit / TQ 4-bit
```
