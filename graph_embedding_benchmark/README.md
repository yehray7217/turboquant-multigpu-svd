# Graph Embedding And Real MNIST QJL Benchmarks with TurboQuant-QJL Accelerated Multi-GPU SVD

本專案針對多 GPU 計算環境下的奇異值分解（Singular Value Decomposition, SVD）進行通訊瓶頸優化。藉由引入 **TurboQuant** 線上向量量化技術與 **QJL（Quantized Johnson-Lindenstrauss）殘差修正**機制，降低節點間或晶片間的中間矩陣傳輸開銷，在確保數值幾何保真度的前提下，提升大規模並行計算的端到端效能。

本專案提供兩個基準測試（Benchmarks）模組：**圖嵌入多卡通訊模擬測試**與**真實影像高維投影測試**。這兩個模組採用 Proxy 測試模式，抽離複雜框架的非核心開銷，精確量化通訊壓縮演算法的實體硬體加速比與數學結構保真度。

---

## 目錄

* [1. 核心實驗一：圖嵌入多卡通訊模擬測試 (Part 1)](https://www.google.com/search?q=%231-%E6%A0%B8%E5%BF%83%E5%AF%A6%E9%A9%97%E4%B8%80%E5%9C%96%E5%B5%8C%E5%85%A5%E5%A4%9A%E5%8D%A1%E9%80%9A%E8%A8%8A%E6%A8%A1%E6%93%AC%E6%B8%AC%E8%A9%A6-part-1)
* [2. 核心實驗二：真實 Fashion-MNIST 10萬維特徵測試 (Part 2)](https://www.google.com/search?q=%232-%E6%A0%B8%E5%BF%83%E5%AF%A6%E9%A9%97%E4%BA%8C%E7%9C%9F%E5%AF%A6-fashion-mnist-10%E8%90%AC%E7%B6%AD%E7%89%B9%E5%BE%B5%E6%B8%AC%E8%A9%A6-part-2)
* [3. 原始碼架構與元件說明](https://www.google.com/search?q=%233-%E5%8E%9F%E5%A7%8B%E7%A2%BC%E6%9E%B6%E6%A7%8B%E8%88%87%E5%85%83%E4%BB%B6%E8%AA%AA%E6%98%8E)
* [4. 系統執行與運作流程](https://www.google.com/search?q=%234-%E7%B3%BB%E7%B5%B1%E5%9F%B7%E8%A1%8C%E8%88%87%E9%81%8B%E4%BD%9C%E6%B5%81%E7%A8%8B)
* [5. 實驗建置與執行步驟](https://www.google.com/search?q=%235-%E5%AF%A6%E9%A9%97%E5%BB%BA%E7%BD%AE%E8%88%87%E5%9F%B7%E8%A1%8C%E6%AD%A5%E9%A9%9F)
* [6. 實驗輸出指標與數據解讀基準](https://www.google.com/search?q=%236-%E5%AF%A6%E9%A9%97%E8%BC%B8%E5%87%BA%E6%8C%87%E6%A8%99%E8%88%87%E6%95%B8%E6%93%9A%E8%A7%A3%E8%AE%80%E5%9F%BA%E6%BA%96)

---

## 1. 核心實驗一：圖嵌入多卡通訊模擬測試 (Part 1)

### 1.1 實驗機制與場景模擬

本實驗為純通訊效能評估測試。在分散式圖神經網路（如 Distributed GraphSAGE）或多卡隨機奇異值分解（Randomized SVD）中，當矩陣或圖結構被切分至多張 GPU 後，各計算節點在局部投影運算（$Y_i = A_i \Omega$）或邊邊節點特徵同步階段，必須頻繁傳遞高維稠密中間矩陣（Dense Embedding Matrix）。本測試透過 MPI 與 CUDA 建立多卡 All-Gather 通訊拓撲，在實體傳輸前、後嵌入壓縮與解壓核，用以模擬並評估實際並行計算中的網路頻寬塞車情境。

### 1.2 實驗目的

* 驗證高性能計算（HPC）中「以計算換頻寬」策略的硬體效益，即利用 GPU 內置的超高並列計算吞吐量執行量化編碼，降低 PCIe、NVLink 或 InfiniBand 的實體資料搬移延遲。
* 評估不同壓縮模式（包含未壓縮基準 `none`、傳統 8-bit 量化 `lowbit8`、4-bit 主量化 `tq4` 以及結合隨機投影的 `tq-qjl4`）在特定向量維度下的端到端通訊加速比與數值失真率（RMSE）。

---

## 2. 核心實驗二：真實 Fashion-MNIST 10萬維特徵測試 (Part 2)

### 2.1 實驗機制與場景模擬

本實驗整合了**遠期高維降維定理**與**專案實體量化函式庫**。測試採用真實世界的 Fashion-MNIST 影像資料集（包含 60,000 張訓練影像與 10,000 張測試影像）。

1. **隱式高維擴展**：程式在 GPU 顯存內部，利用隨機傅立葉特徵空間映射技巧，將原始 784 維的影像向量隱式展開至 100,000 維的超高維空間。
2. **QJL 空間降維**：調用 QJL 隨機正負號矩陣（Rademacher Matrix），將該 10 萬維度投影壓縮至 256、512 及 1024 維度。
3. **TurboQuant 疊加量化**：**此處為本次優化重點**。降維完成後，程式直接連結並調用專案的核心函式庫 `turboquant`，將 QJL 特徵進一步做 4-bit（或指定位元）的線上向量量化壓縮，並在 GPU 上即時解壓還原。

> **顯存優化設計（Streaming Chunking）**：70,000 筆資料擴展至 100,000 維度需耗費約 26 GB 顯存，易導致硬體記憶體溢出（OOM）。本程式設計分塊處理機制（Feature Chunking），每次由 CPU 隨機生成 2048 維度的映射權重，傳入 GPU 計算局部特徵並立刻進行 QJL 累加投影，隨後釋放分塊顯存，將記憶體開銷降至常數級別。

### 2.2 實驗目的

* **驗證雙重壓縮下的保真度**：追蹤資料在經歷「隨機投影降維（QJL）」與「低位元欄位量化（TurboQuant）」連續兩次高倍率壓縮後，幾何拓撲距離誤差（Pairwise Distance Error）的受損程度。
* **評估實際任務可用性**：比較「原始特徵」、「純 QJL 降維」與「QJL + TurboQuant 還原」三個階段下的最近中心分類器（Nearest Centroid Classifier）準確度。證明本專案實作之量化函式庫在極致壓縮資料體積的同時，依然能保留完整的核心語義資訊。

---

## 3. 原始碼架構與元件說明

專案核心組件由兩個高度並列化的 CUDA/C++ 原始碼模組構成：

### 3.1 `graph_embedding_comm_benchmark.cu`

* **`MpiScope`**：應用 RAII 機制，管理分散式環境中 MPI 進程的生命週期與異常安全。
* **`fill_embeddings_kernel`**：CUDA 核心，結合波動函數（`sinf`）與隨機雜訊，在 GPU 顯存中動態生成具有流形結構特徵的模擬 Embedding 向量。
* **`run_once_mode`**：測試流水線。負責配置臨時運算緩衝區，並在通訊點調用 `turboquant::quantize_fp32_device_column_tq_to_device_payload` 進行欄位級旋轉量化，隨後透過 `MPI_Allgather` 進行多卡傳輸，最後調用對應之解壓核進行還原。

### 3.2 `mnist_qjl_feature_benchmark.cu`

* **`load_idx` / `normalize_from_train**`：負責讀取標準 IDX 二進位檔案格式，並基於訓練集統計量進行 Z-score 正規化。
* **`add_bias_cos_kernel`**：執行非線性高維擴展：$\phi(x) = \sqrt{\frac{2}{D}} \cos(W^T x + b)$。
* **`accumulate_pair_dist_kernel`**：利用 GPU 共享記憶體（Shared Memory）進行多執行緒平行歸約（Reduction），累加計算 10 萬維空間中 4096 對隨機圖像的精確距離，作為評估基準。
* **`copy_prefix_rows_kernel`**：**新加入元件**。CUDA 核心，負責將 GPU 內計算出的最大維度投影矩陣，按當前遍歷的 $q$ 維度（256, 512, 1024）高效提取出前 $q$ 列，重組成連續（Contiguous）的特徵矩陣，以供 TurboQuant 函式庫進行欄位級別量化。
* **`centroid_accuracy` / `evaluate_representation**`：負責評估各階段特徵矩陣的最近中心分類準確度與拓撲結構幾何失真率。

---

## 4. 系統執行與運作流程

兩個基準測試相互獨立，但可配置於同一工作排程（如 Slurm）中序列化執行。整體資料流與控制邏輯如下圖所示：

```text
========================================================================================
【 Part 1: 多 GPU 圖嵌入通訊模擬資料流 】
========================================================================================
 啟動 mpirun 分散式進程 (例如 16 Ranks / GPUs 跨節點)
       │
       ▼
 進程編號物理綁定：`mpi.rank % device_count` 分配 local GPU 卡
       │
       ▼
 執行 `fill_embeddings_kernel` ➔ 各卡配置 32 MiB FP32 模擬稠密矩陣
       │
       ▼
 執行全精度傳輸對照組 ➔ 建立失真率計算所需之黃金參考基準 `h_ref_all`
       │
       ▼
【 遍歷壓縮模式：none ➔ lowbit8 ➔ tq4 ➔ tq-qjl4 】
   │
   ├──> 迴圈重複 20 次實體效能觀測 (排除 Warmup 迭代)
   │       │
   │       ├───> [編碼段] GPU 呼叫 TurboQuant 欄位旋轉量化 (32 MiB ➔ 4.125 MiB)
   │       ├───> [通訊段] MPI 執行 `allgather_bytes` ➔ 跨節點交換低位元編碼包
   │       └───> [解碼段] 接收端 GPU 呼叫 `dequantize_..._add_to_fp32` 並列還原
   │
   ▼ (全模式測試完畢)
 透過 `MPI_Reduce` 取最大耗時（`MPI_MAX`） ➔ 由 Rank 0 輸出硬體加速比 CSV 報告

========================================================================================
【 Part 2: 真實 Fashion-MNIST 10萬維 QJL+TurboQuant 雙重優化資料流 】
========================================================================================
 單進程啟動 ➔ 自行下載並解析 70,000 筆 Fashion-MNIST 影像二進位檔
       │
       ▼
 矩陣標準化處理 ➔ 載入至 GPU 顯存主緩衝區 `d_x` (維度: 784 x 70000)
       │
       ▼
【 100,000 維特徵分塊循環：每次步進 2048 維 】
   │
   ├───> CPU 隨機數引擎生成分塊矩陣 W, b 與 QJL Rademacher 號誌矩陣 S
   ├───> GPU 呼叫 cuBLAS `cublasSgemm` 計算線性變換：W^T * x
   ├───> GPU Kernel (`add_bias_cos_kernel`) 計算隨機傅立葉高維擴展特徵 phi
   ├───> GPU Kernel (`accumulate_pair_dist_kernel`) 平行累加隨機 4096 對原始距離
   └───> GPU cuBLAS 矩陣乘法計算 QJL 線性累加投影：d_y += S^T * phi (流式降維)
       │
       ▼ (分塊迭代完畢，最大投影矩陣儲存於 d_y，顯存無溢出)
【 目標維度循環：q = 256 ➔ 512 ➔ 1024 】
   │
   ├───> [對照組評估] 呼叫 `evaluate_representation` ➔ 計算純 QJL 降維後之誤差與 Accuracy
   ├───> [特徵連續化] GPU Kernel (`copy_prefix_rows_kernel`) ➔ 抽取前 q 列至專屬緩衝區
   ├───> [專案庫調用] 呼叫 `turboquant::quantize_fp32_device_column_tq_to_device_payload` ➔ 壓成 4-bit 壓縮包
   ├───> [並列解壓縮] 呼叫 `turboquant::dequantize_column_tq_payload_add_to_fp32` ➔ 還原近似特徵
   └───> [實驗組評估] 呼叫 `evaluate_representation` ➔ 計算經雙重壓縮後之最終誤差與 Accuracy
       │
       ▼ (全維度測試完畢)
 將所有精度對比與多階壓縮率指標寫入 CSV 報告 (`results/fashion_mnist_qjl_tq.csv`)

```

---

## 5. 實驗建置與執行步驟

### 5.1 環境建置與編譯

專案依賴 CUDA 工具鏈、高效能矩陣運算庫（cuBLAS）以及支援 CUDA-aware 的 MPI 實作。進入目的目錄後執行自動化編譯：

```bash
cd graph_embedding_benchmark
make clean
make
```

編譯器將於 `.build/` 目錄下生成 `graph_embedding_comm_benchmark` 與 `mnist_qjl_feature_benchmark` 執行檔。

### 5.2 執行圖嵌入通訊模擬測試 (以單節點 8 GPU 為例)

指定分配 8 個 MPI 進程，運行模式設定為通訊模式掃描（Sweep）：

```bash
mpirun -np 8 .build/graph_embedding_comm_benchmark \
  --halo-nodes 65536 \
  --embedding-dim 256 \
  --mode sweep \
  --repeats 20 \
  --warmup 3
```

### 5.3 執行真實 Fashion-MNIST 測試

利用標準庫下載資料集檔案，解壓後將路徑指定給引進了 TurboQuant 界面之基準測試程式：

```bash
# 下載與資料預處理
python3 prepare_mnist_data.py --dataset fashion-mnist --data-dir ./data

# 啟動高維幾何與專案庫整合測試 (指定 QJL 目標維度與主量化位元)
.build/mnist_qjl_feature_benchmark \
  --dataset-dir ./data/fashion-mnist \
  --feature-dim 100000 \
  --qjl-dims 256,512,1024 \
  --tq-bits 4
```

### 5.4 排程系統批次提交 (Slurm 範例)

專案提供整合型排程提交腳本。該腳本可一鍵將多卡並行通訊任務（跨節點 16 張 V100 GPU）與 Fashion-MNIST 串流幾何測試派發至叢集：

```bash
sbatch run_graph_embedding_comm_8gpu.slurm
```

---

## 6. 實驗輸出指標與數據解讀基準

系統產出之評估報告儲存於 `results/` 目錄之 CSV 檔案中，各核心指標定義如下：

### 6.1 通訊測試指標 (Part 1 Output)

* **`mean_ms`**：全卡集體同步之平均端到端時間，包含量化、網路傳輸與解壓還原開銷。
* **`speedup_vs_none`**：相較於標準 FP32 全精度未壓縮通訊之**實際系統加速比**。當數值 $> 1.0$ 時代表壓縮帶來的通訊時間縮短幅度大於量化產生的計算代價。
* **`relative_rmse`**：相對均方根誤差。量化失真指標，反映解壓後的 Embedding 矩陣與原始 FP32 矩陣的數值偏離程度。

### 6.2 幾何與分類測試指標 (Part 2 Output)

* **`total_compression_vs_expanded_fp32`**：總體體積壓縮比。即原始 10 萬維度 FP32 資料與經由 QJL 降維及 TurboQuant 4-bit 量化後最終 Payload 之間的體積倍數關係（數值通常高達數千倍）。
* **幾何失真對比 (`qjl_mean_..._error` vs `tq_mean_..._error`)**：分別記錄「純 QJL 降維」與「疊加 TurboQuant 量化解壓後」圖片點對相對距離的誤差率。若兩者差距極小，則證明 TurboQuant 引入的幾何擾動微乎其微。
* **分類準確度對比 (`qjl_centroid_accuracy` vs `tq_centroid_accuracy`)**：
* 將上述兩個階段的分類 Accuracy，與未經高維展開的原始影像基準 `raw_pixel_centroid_accuracy` (67.84%) 進行全面交叉對比。
* **解讀基準**：若 `tq_centroid_accuracy` 依然能與 `qjl_centroid_accuracy` 持平，甚至超越原始像素基準，則在數值理論上鐵證了本專案函式庫在極致壓縮下，依然完美保留了核心語義流形。


* **`tq_seconds`**：TurboQuant 函式庫在 GPU 上進行線上旋轉量化與並列解壓的實體計算開銷（通常在毫秒級別），用以論證本優化演算法的高吞吐量與極低延遲開銷。