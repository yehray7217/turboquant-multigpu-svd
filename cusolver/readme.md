# Baseline A：cuSOLVER SVD
使用 NVIDIA cuSOLVER 提供的 SVD routine 作為標準 GPU SVD 參考實作，用於：
- correctness verification
- 單 GPU library-level 效能基準
- 與自寫版本的數值結果互相比對

# Exeperment Steps
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
