# TurboQuant-QJL Accelerated Multi-GPU SVD

## 專題題目

**以 TurboQuant 與 QJL Residual Correction 降低多 GPU SVD 中間資料傳輸成本**

---

## 專題概要

Singular Value Decomposition（SVD）是科學計算、資料壓縮、主成分分析（PCA）與低秩矩陣近似中的核心運算。然而，當矩陣規模增大時，SVD 的計算成本、記憶體需求與資料搬移成本都會快速上升，因此常需要透過 GPU 與多 GPU 平行化來提升效能。

在多 GPU 情境下，除了矩陣運算本身的計算成本外，**GPU 之間的資料交換與同步**也可能成為主要瓶頸。尤其是在矩陣分塊處理、低秩投影、正交化與中間結果彙整的過程中，若需要頻繁傳遞高維向量或中間矩陣，通訊成本可能限制整體加速效果。

本專題嘗試將 **TurboQuant** 的核心概念引入多 GPU SVD 計算流程，針對需要跨 GPU 傳輸的中間資料，使用：

1. **Random Rotation**
2. **低位元量化**
3. **QJL（Quantized Johnson-Lindenstrauss）Residual Correction**

將高維資料壓縮後再傳輸，並於接收端近似重建，藉此觀察：

- 是否能降低多 GPU 間的資料傳輸量；
- 是否能減少 communication time；
- 是否能改善端到端 SVD runtime；
- 加入 QJL residual correction 後，是否能在壓縮下維持較佳的內積結構與數值精度。

---

## 研究動機

傳統 GPU SVD 優化大多聚焦於：

- 提升矩陣乘法與分解 kernel 的吞吐量；
- 使用高效 CUDA library；
- 改善 block size、memory access pattern 與 GPU occupancy。

然而，當計算擴展至多 GPU 時，效能瓶頸不一定只存在於 FLOPS，也可能來自：

- GPU-GPU 資料交換；
- NCCL collective communication；
- 中間矩陣彙整；
- QR / projection 階段的跨裝置同步。

因此，本專題從 **data movement bottleneck** 切入，探討：

> 在多 GPU SVD 中，是否能利用 TurboQuant + QJL 將中間資料壓縮後傳輸，以通訊量下降換取整體效能提升，並透過 QJL residual correction 控制數值誤差。

---

## 平台環境

- **叢集平台**：台灣杉二號（Taiwania 2）
- **主要硬體**：NVIDIA Tesla V100 GPU
- **預計使用技術**
  - CUDA C/C++
  - cuBLAS
  - cuSOLVER
  - NCCL 或 MPI + CUDA
  - Nsight Systems
  - Nsight Compute

---

## Baseline 設計

本專題預計保留兩種 baseline。

### Baseline A：cuSOLVER SVD

使用 NVIDIA cuSOLVER 提供的 SVD routine 作為標準 GPU SVD 參考實作，用於：

- correctness verification；
- 單 GPU library-level 效能基準；
- 與自寫版本的數值結果互相比對。

### Baseline B：自寫 Multi-GPU SVD / Randomized SVD

為了能精確控制：

- 矩陣如何分割到多張 GPU；
- 哪些中間結果需要跨 GPU 傳輸；
- 通訊前後資料格式；
- TurboQuant 與 QJL 的插入位置；

本專題將另外實作一個 **自寫 multi-GPU SVD pipeline**，作為比較「優化前」與「優化後」的主要對象。

> 實作上，可優先考慮 **Randomized SVD**，因為其流程天然包含多 GPU 局部運算、投影矩陣整合與低秩重建，較適合觀察壓縮通訊對整體流程的影響。

---

# TurboQuant + QJL 方法概念

## 1. TurboQuant 的基本想法

TurboQuant 是一種針對高維向量設計的 online vector quantization 方法。其主要精神是：

1. 先對高維向量做隨機旋轉，使各座標的數值分布更集中；
2. 再對旋轉後的座標分別做低位元 scalar quantization；
3. 壓縮成低 bit representation；
4. 解碼時重建近似向量。

其核心流程如下：

```text
Input Vector x
│
├── Random Rotation
│   └── 將向量座標重新混合，使其更適合量化
│
├── Low-bit Scalar Quantization
│   └── 將旋轉後的每個座標壓縮為低位元表示
│
├── Encoded Representation
│   └── 儲存或傳輸 compressed codes
│
└── Approximate Reconstruction
    └── 解碼後得到近似向量 x_hat
```

---

## 2. 為什麼需要 QJL Residual Correction？

若只使用 MSE-oriented quantization，雖然可以讓重建誤差下降，但對於後續涉及：

- 內積；
- 投影；
- 正交化；
- 矩陣乘法；

的流程，**量化誤差可能導致 inner product estimation 出現偏差**。

TurboQuant 因此加入第二階段：

> **對第一階段量化後留下的 residual 進行 1-bit QJL encoding，作為額外修正資訊。**

令：

```math
\hat{x}_{\text{TQ}}
```

表示 TurboQuant 第一階段量化後的近似重建，則 residual 為：

```math
r = x - \hat{x}_{\text{TQ}}
```

接著使用 QJL 對 residual 做一個低成本 sketch：

```math
q = \mathrm{sign}(Gr)
```

其中：

- $G$ 為隨機投影矩陣；
- $\mathrm{sign}(\cdot)$ 將結果壓成 1-bit sign code；
- $q$ 為 residual 的 QJL correction code。

因此，最終傳輸內容不只是 TurboQuant 的主量化碼，而是：

```text
Compressed Payload
│
├── Main Quantized Codes
│   └── 低位元主表示
│
└── QJL Residual Sign Codes
    └── 1-bit residual correction
```

---

## 3. 本專題採用的壓縮策略

在本專題中，我們將採用：

> **TurboQuant main quantization + QJL residual correction**

作為多 GPU SVD 中間資料的壓縮傳輸策略。

整體壓縮流程如下：

```text
Intermediate Vector / Block x
│
├── 1. Random Rotation
│
├── 2. Low-bit Quantization
│
├── 3. Reconstruct Temporary Approximation x_hat_TQ
│
├── 4. Compute Residual
│       r = x - x_hat_TQ
│
├── 5. QJL Residual Encoding
│       q = sign(G r)
│
├── 6. Send
│       ├── Quantized main codes
│       └── QJL residual sign codes
│
└── 7. Receiver-side Approximate Recovery
        ├── Decode main quantized codes
        ├── Use QJL correction information
        └── Recover corrected approximation
```

---

# Multi-GPU SVD 計算流程

## 優化前：未壓縮通訊的 Multi-GPU SVD

以 randomized SVD 為例，基本流程如下。

### 數學流程

給定輸入矩陣：

```math
A \in \mathbb{R}^{m \times n}
```

目標是近似求得 rank-$k$ SVD：

```math
A \approx U_k \Sigma_k V_k^T
```

其中流程可包含：

1. 產生隨機投影矩陣：
   ```math
   \Omega \in \mathbb{R}^{n \times (k+p)}
   ```

2. 各 GPU 計算局部投影：
   ```math
   Y_i = A_i \Omega
   ```

3. 彙整或交換中間矩陣 $Y_i$，形成投影空間資訊；

4. 進行 QR 分解：
   ```math
   Y = QR
   ```

5. 計算低維矩陣：
   ```math
   B = Q^T A
   ```

6. 對 $B$ 做 SVD：
   ```math
   B = \tilde{U} \Sigma V^T
   ```

7. 回推出：
   ```math
   U = Q\tilde{U}
   ```

### 優化前資料流

```text
Input Matrix A
│
├── Partition A across multiple GPUs
│
├── Local randomized projection
│   └── Y_i = A_i Ω
│
├── Full-precision communication
│   └── Exchange / gather intermediate matrices
│
├── QR / Orthogonalization
│
├── Build reduced matrix B = QᵀA
│
├── SVD on reduced matrix B
│
└── Output low-rank factors U, Σ, Vᵀ
```

---

## 優化後：加入 TurboQuant + QJL 的 Multi-GPU SVD

本專題將 TurboQuant + QJL 插入於：

> **中間矩陣產生之後、跨 GPU 傳輸之前。**

### 優化後資料流

```text
Input Matrix A
│
├── Partition A across multiple GPUs
│
├── Local randomized projection
│   └── Y_i = A_i Ω
│
├── Split intermediate result into vectors / blocks
│
├── TurboQuant Main Compression
│   ├── Random rotation
│   ├── Low-bit quantization
│   └── Temporary reconstruction x_hat_TQ
│
├── QJL Residual Correction
│   ├── Compute residual r = x - x_hat_TQ
│   ├── Random projection of residual
│   └── 1-bit sign encoding
│
├── Compressed Communication
│   ├── Send main quantized codes
│   └── Send QJL residual codes
│
├── Receiver-side Reconstruction
│   ├── Decode main quantized representation
│   ├── Apply residual correction information
│   └── Recover corrected approximation
│
├── QR / Orthogonalization
│
├── Build reduced matrix B = QᵀA
│
├── SVD on reduced matrix B
│
└── Output low-rank factors U, Σ, Vᵀ
```

---

# 加速前後比較

| 面向 | 原始 Multi-GPU SVD | TurboQuant + QJL Multi-GPU SVD |
|---|---|---|
| 中間資料傳輸 | FP32 / FP64 | 低位元主量化碼 + 1-bit QJL residual code |
| 傳輸量 | 較大 | 預期下降 |
| 額外計算 | 無 | rotation、quantization、residual、QJL encoding、decode |
| 內積結構維持 | 取決於原始精度 | QJL 用於減少量化對 inner product 的偏差 |
| 整體目標 | 標準多 GPU SVD | 通訊壓縮下的效能 / 精度 trade-off |

---

# 預計實驗指標

## 1. 效能指標

- End-to-end runtime
- Communication time
- Quantization time
- QJL residual encoding time
- Dequantization / reconstruction time
- Overall speedup
- Scaling efficiency under:
  - 1 GPU
  - 2 GPUs
  - 4 GPUs
  - 8 GPUs

## 2. 壓縮指標

- 傳輸 byte 數
- 壓縮前後 payload 大小
- Main quantization bits per value
- QJL residual bits per vector / block
- Effective bit-rate

## 3. 數值指標

- Reconstruction error：
  ```math
  \frac{\|A - U_k\Sigma_kV_k^T\|_F}{\|A\|_F}
  ```

- Singular value relative error：
  ```math
  \frac{\|\sigma - \hat{\sigma}\|_2}{\|\sigma\|_2}
  ```

- Reduced matrix $B$ approximation error
- Projection quality
- 若實作允許，可額外比較 inner-product distortion

---

# 預計實驗變因

## GPU 數量

- 1 GPU
- 2 GPUs
- 4 GPUs
- 8 GPUs

## 矩陣規模

- Small
- Medium
- Large

## 目標 rank

- $k = 32$
- $k = 64$
- $k = 128$
- $k = 256$

## 傳輸格式

- FP32 communication
- FP16 communication
- TurboQuant only
- TurboQuant + QJL

---

# Workflow Tree

```text
Project Workflow
│
├── 1. Environment Setup
│   ├── Taiwania 2 job environment
│   ├── CUDA / cuBLAS / cuSOLVER
│   ├── NCCL or MPI + CUDA
│   └── Profiling tools
│       ├── Nsight Systems
│       └── Nsight Compute
│
├── 2. Baseline Construction
│   ├── cuSOLVER SVD baseline
│   │   ├── Correctness verification
│   │   └── Reference runtime measurement
│   │
│   └── Custom multi-GPU SVD baseline
│       ├── Matrix partitioning
│       ├── Local randomized projection
│       ├── Full-precision intermediate communication
│       ├── QR / Orthogonalization
│       └── Reduced SVD reconstruction
│
├── 3. Bottleneck Analysis
│   ├── Profile runtime breakdown
│   ├── Measure communication cost
│   ├── Identify transmitted intermediate tensors
│   └── Select compression insertion point
│
├── 4. TurboQuant Main Compression
│   ├── Vector / block partitioning
│   ├── Random rotation
│   ├── Low-bit scalar quantization
│   ├── Main code generation
│   └── Temporary reconstruction for residual computation
│
├── 5. QJL Residual Correction
│   ├── Residual computation
│   │   └── r = x - x_hat_TQ
│   ├── Random projection
│   ├── 1-bit sign encoding
│   └── Residual correction code packaging
│
├── 6. Compressed Communication Pipeline
│   ├── Send main quantized codes
│   ├── Send QJL residual codes
│   ├── Receive compressed payload
│   └── Reconstruct corrected approximation
│
├── 7. Experimental Evaluation
│   ├── Runtime comparison
│   ├── Communication time comparison
│   ├── Compression ratio analysis
│   ├── Reconstruction error analysis
│   ├── Singular value error analysis
│   └── Multi-GPU scaling analysis
│
└── 8. Final Analysis
    ├── Performance / accuracy trade-off
    ├── TurboQuant-only vs TurboQuant + QJL
    ├── Conditions where compression helps
    ├── Conditions where overhead dominates
    └── Applicability to other distributed linear algebra tasks
```

---

# Repo 初步架構

```text
turboquant-qjl-multigpu-svd/
│
├── README.md
│
├── docs/
│   ├── design_notes.md
│   └── experiment_plan.md
│
├── src/
│   ├── baseline/
│   │   ├── cusolver_svd.cu
│   │   └── randomized_multigpu_svd.cu
│   │
│   ├── turboquant/
│   │   ├── rotation.cu
│   │   ├── quantize.cu
│   │   └── dequantize.cu
│   │
│   ├── qjl/
│   │   ├── residual.cu
│   │   ├── qjl_encode.cu
│   │   └── qjl_decode.cu
│   │
│   ├── communication/
│   │   ├── nccl_sendrecv.cu
│   │   └── compressed_payload.cu
│   │
│   └── main.cu
│
├── scripts/
│   ├── build.sh
│   ├── run_single_gpu.sh
│   ├── run_multigpu.sh
│   └── run_twcc.slurm
│
├── experiments/
│   ├── configs/
│   ├── logs/
│   └── results/
│
└── CMakeLists.txt
```

---

# 預期成果

本專題預期產出：

1. 可在台灣杉二號執行的 GPU SVD baseline；
2. 可觀察多 GPU 通訊成本的自寫 SVD / randomized SVD pipeline；
3. 加入 TurboQuant main quantization 的壓縮傳輸版本；
4. 加入 QJL residual correction 的完整壓縮傳輸版本；
5. 優化前後在：
   - runtime；
   - communication time；
   - 傳輸量；
   - reconstruction error；
   - singular value error；
   之間的系統性比較；
6. 對「TurboQuant + QJL 是否適合用於分散式線性代數中間資料傳輸」給出初步結論。

---

# 可能應用

此方法未來可能延伸至：

- 大規模 PCA 與資料降維；
- 科學計算與模擬資料的低秩近似；
- 分散式矩陣分解；
- 大型模型權重壓縮或中間特徵矩陣分析。
