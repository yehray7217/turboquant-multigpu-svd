# Baseline A：cuSOLVER SVD
## 1. 單節點cuSOLVER
使用 NVIDIA cuSOLVER 提供的 SVD routine 作為標準 GPU SVD 參考實作，用於：
- correctness verification
- 單 GPU library-level 效能基準
- 與自寫版本的數值結果互相比對

## Exeperment Steps
**load cuda**
```
module load cuda/12.8
```
**compile**
```
make
```
**run**
```
sbatch run_cusolver.sh
```

## 2. 分散式 Multi-GPU SVD Proxy App (MPI + CUDA)
使用 MPI (Message Passing Interface) 結合 CUDA 實作的多節點、多 GPU SVD 代理應用程式。將矩陣分配至不同 GPU 進行局部計算，並透過網路交換特徵向量，用於：
- 驗證多卡協同運算與弱擴展性 (Weak Scaling) 的正確性
- 獨立量測運算時間 (Compute Time) 與跨節點通訊時間 (Comm Time)，確立網路頻寬瓶頸
- 作為後續安插 TurboQuant (TQ) 與極座標通訊壓縮演算法的開發基底 (Pipeline Base)

## Experiment Steps
run_mpi_cusolver.sh 會編譯跟執行
```
sbatch run_mpi_cusolver.sh
```