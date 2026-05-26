# Randomized SVD v4 Optimization History

## 1. 新增不同的 testcase（矩陣 $A$）generation 方法

觀察發現現實中的科學 / 工程問題大部分矩陣的 singular spectrum 其實下降速度非常快，且有些矩陣 decay 的行為很接近 polynomial decay 或 exponential decay，因此新增了後兩種生成測資的方式（生出來的 $A$ 矩陣的 singular values 可以選擇服從 $\sigma_i \sim i^{-p}$ 或 $\sigma_i \sim \exp(-i)$）。




