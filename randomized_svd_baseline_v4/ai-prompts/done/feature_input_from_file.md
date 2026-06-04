# Guideline 00：讓 `randomized_svd_multigpu_v4.cu` 支援讀取外部 `.f32` 測資

## 目標

目前 `randomized_svd_multigpu_v4.cu` 主要支援兩種測資來源：

1. 隨機矩陣
2. synthetic spectrum matrix，例如 polynomial / exponential decay

現在要新增第三種來源：

> 從外部 raw FP32 binary file 讀取矩陣 (A)。

之後 OISST、POD、LLM 相關的矩陣都會被 preprocessing 成同一種格式：

```text
matrix.f32
```

然後 CUDA program 直接讀這個檔案作為 $A$。



## CLI 指令修改

只新增一個 CLI 參數：

```text
--input-file <path>
```

只要使用者有指定：

```text
--input-file testcases_from_real_world/oisst/matrix.f32
```

就代表矩陣 $A$ 從檔案讀取。如果沒有指定 `--input-file`，就維持原本行為：使用 synthetic / random generator。

另外矩陣檔案 layout 固定為 row-major，大小的參數沿用既有的 `--m` 和 `--n`，避免 CLI 變複雜。

範例：

```bash
srun --mpi=pmix "$BIN" \
    --m 32768 \
    --n 8192 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_from_real_world/oisst/matrix.f32 \
    --compress-b-mode tq \
    --compress-b-bits 8 \
    --compress-subspace-mode tq \
    --compress-subspace-bits 8 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only
```



## Input file 格式

外部矩陣檔案一律是：

```text
raw float32 binary
row-major layout
no header
```

也就是如果矩陣是 $A \in \mathbb{R}^{m \times n}$，則檔案中資料排列為：

```text
A[0,0], A[0,1], ..., A[0,n-1],
A[1,0], A[1,1], ..., A[1,n-1],
...
A[m-1,0], ..., A[m-1,n-1]
```

檔案大小必須剛好是：

```text
m * n * sizeof(float)
```

如果檔案大小不對，請直接報錯。

---

## 為什麼使用 row-major？

目前 distributed randomized SVD 是 row-partition。

也就是每個 MPI rank / GPU 只負責矩陣 (A) 的某幾段 rows。

如果測資檔案使用 row-major 格式，每個 MPI rank 可以直接用 file seek 讀取自己的 row block：

```text
offset_bytes = row_start * n * sizeof(float)
read_count   = local_rows * n
```

這樣不需要每個 rank 都讀完整矩陣。

讀進 host row-major buffer 之後，再在程式內轉成目前 cuBLAS 需要的 local column-major layout。

---

## 需要修改的地方

請修改：

```text
randomized_svd_multigpu_v4.cu
```

主要位置：

1. `struct Options`
2. `print_usage`
3. `parse_args`
4. setup / matrix initialization 區塊
5. repeat loop 中 local projection 前的 A 初始化邏輯

---

## Step 1：更新 `Options`

在 `struct Options` 加上：

```cpp
std::string input_file;
```

判斷方式：

```cpp
const bool use_input_file = !opt.input_file.empty();
```

---

## Step 2：更新 `print_usage`

新增 usage 說明：

```cpp
<< "  --input-file <path>   Read A from raw row-major FP32 .f32 file instead of generating synthetic A.\n"
<< "                        File must contain exactly m*n float32 values in row-major order.\n"
```

---

## Step 3：更新 `parse_args`

新增：

```cpp
else if (a == "--input-file") opt.input_file = need_value(a);
```

---

## Step 4：處理參數衝突

如果有指定 `--input-file`，矩陣 (A) 來自檔案。

因此：

1. 不應該產生 random (A)。
2. 不應該產生 synthetic spectrum (A)。
3. `--spectrum-decay-mode`、`--spectrum-decay-param`、`--spectrum-rank` 不應影響 (A)。
4. `--device-random-input` 不應覆蓋檔案讀進來的 (A)。

建議第一版採取保守策略：

```text
如果 --input-file 和 --device-random-input 同時存在，直接報錯。
```

例如：

```cpp
if (!opt.input_file.empty() && opt.device_random_input) {
    throw std::runtime_error("--input-file is incompatible with --device-random-input in the first file-input implementation.");
}
```

理由是目前 `--device-random-input` 會在 GPU 上生成 (A_i) 和 (\Omega)，容易不小心覆蓋從檔案讀入的 (A_i)。

之後如果要優化，可以再把 `--device-random-input` 重新定義成「只在 GPU 生成 Omega」，但第一版先不要複雜化。

對 spectrum 參數，建議：

```text
如果 --input-file 存在，強制 use_spectrum_decay = false。
```

也就是檔案模式下不使用 synthetic spectrum generator。

如果 root rank 看到使用者同時傳了 spectrum 參數，可以印 warning：

```text
Warning: --input-file is set; synthetic spectrum options are ignored.
```

---

## Step 5：驗證檔案大小

在 setup 階段檢查：

```text
file_size == m * n * sizeof(float)
```

若不相等，直接報錯。

注意：

```text
m * n * sizeof(float)
```

可能超過 `int`，請使用 `size_t` 或 `uint64_t`。

---

## Step 6：讀取 row block

目前 code 已經有：

```cpp
global_rows = split_rows(m, global_ngpus)
global_row0 = prefix_offsets(global_rows)
```

每個 local GPU 對應：

```cpp
w.row0
w.mi
```

請新增 helper function，概念如下：

```cpp
read_rowmajor_f32_block(
    input_file,
    m,
    n,
    row0,
    mi
)
```

它要做：

1. 開啟檔案。
2. seek 到：

```text
row0 * n * sizeof(float)
```

3. 讀取：

```text
mi * n
```

個 float。
4. 回傳 host row-major buffer。

---

## Step 7：轉成 local column-major layout

目前 `w.d_Ai` 的 layout 是：

```text
mi x n column-major
```

也就是 cuBLAS 會把它當作 leading dimension `mi` 的 column-major matrix。

但讀進來的檔案 block 是 row-major：

```text
h_rowmajor[r * n + c]
```

所以需要轉成 column-major：

```cpp
h_colmajor[c * mi + r] = h_rowmajor[r * n + c];
```

再 copy 到 GPU：

```cpp
cudaMemcpy(w.d_Ai, h_colmajor.data(), mi * n * sizeof(float), cudaMemcpyHostToDevice);
```

---

## Step 8：計算 (||A||_F^2)

目前 final error metric 需要 (||A||_F^2)。

在 file-input 模式下，請在讀取每個 local block 時順便計算：

```text
local_norm2 = sum of A_ij^2 over rows owned by this MPI rank / local GPUs
```

然後用 MPI allreduce / reduce 得到 global norm：

```text
h_A_norm2 = sum over all ranks
```

請用 double 或 long double accumulation，不要用 float。

---

## Step 9：repeat loop 裡不要重新產生 A

現在 repeat loop 裡會根據 seed 重新產生 random (A)，或在 host mode 下重新 copy (A)。

在 `--input-file` 模式下：

```text
A is fixed.
```

所以：

1. setup 階段讀入 file 並 copy 到 `w.d_Ai`。
2. repeat loop 中不要重新生成 A。
3. repeat loop 中不要覆蓋 `w.d_Ai`。
4. repeat loop 中仍然可以重新生成 (\Omega)，因為 randomized SVD 每次 repeat 可以用不同 random seed。

具體說：

```text
if use_input_file:
    do not call make_random_row_blocks
    do not call fill_random_colmajor_kernel for A
    do not call spectrum matrix generation
    keep w.d_Ai fixed across repeats
```

---

## Step 10：Omega 的處理

第一版可以沿用 host-generated Omega：

```text
h_Omega = make_random_matrix(n, l, seed + 1)
cudaMemcpy(w.d_Omega, h_Omega.data(), ...)
```

若 repeat 時需要不同 seed，仍然只更新 Omega，不更新 A。

---

## Step 11：理論誤差

外部真實資料沒有 synthetic singular spectrum，因此沒有 theoretical best rank-k error。

所以在 `--input-file` 模式下：

```text
theoretical error = N/A
```

或用目前 `-1.0` 的邏輯即可。

不要使用 `synthetic_spectrum_theoretical_error()`。

---

## Step 12：README / output banner

程式 banner 建議顯示：

```text
matrix_source=file
input_file=...
```

例如：

```text
m=32768 n=8192 k=250 oversample=6 l=256 matrix_source=file input_file=testcases_from_real_world/oisst/matrix.f32
```

如果沒有 `--input-file`，則顯示：

```text
matrix_source=synthetic
```

或維持原本即可。

---

## Acceptance Criteria

完成後應滿足：

1. 程式新增 `--input-file <path>`。
2. 若沒有 `--input-file`，原本 synthetic/random 行為不變。
3. 若有 `--input-file`，從 raw row-major FP32 file 讀取 (A)。
4. 使用既有 `--m` 和 `--n` 作為矩陣 shape。
5. 不新增 `--input-dtype`、`--input-layout`、`--input-rows`、`--input-cols`。
6. file size 必須驗證為 `m*n*sizeof(float)`。
7. 每個 MPI rank / GPU 只讀自己負責的 row block。
8. 讀進來的 row-major block 會轉成 local column-major layout 再放到 `w.d_Ai`。
9. `--input-file` 模式下不會產生 random A，也不會產生 synthetic spectrum A。
10. `--input-file` 模式下 repeat 只更新 Omega，不更新 A。
11. `h_A_norm2` 正確來自 input file。
12. `--input-file` 與 `--device-random-input` 同時出現時，第一版請直接報錯。
13. Data loading / row-major to column-major conversion 不應算進 Total Time。
14. 程式可用小矩陣 smoke test 驗證。

---

## Smoke Test 建議

先用 Python 產生一個小矩陣：

```python
import numpy as np

m, n = 16, 8
A = np.arange(m * n, dtype=np.float32).reshape(m, n)
A.tofile("test_matrix.f32")
```

跑：

```bash
./randomized_svd_multigpu_v4 \
    --m 16 \
    --n 8 \
    --k 4 \
    --oversample 2 \
    --ngpus 1 \
    --input-file test_matrix.f32 \
    --repeat 1
```

如果成功，再測較大的矩陣。

---

## Notes

這個 patch 的目標只是讓 `.cu` 能吃外部 `.f32` 測資。

不要在這個 patch 裡處理：

1. OISST preprocessing
2. POD preprocessing
3. LLM weight extraction
4. `.npy` / `.mat` / `.nc` / `.hdf5` runtime loading
5. FP64 input
6. file format auto-detection

這些都交給各自 testcase 的 preprocessing scripts。
