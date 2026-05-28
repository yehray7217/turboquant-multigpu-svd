# Randomized SVD v4 Development History

> Note: 可以直接把 `turboquant-multigpu-svd/randomized_svd_baseline_v4/skills/PROJECT_CONTEXT.md` 拿去餵給 AI，讓它 follow up 最新的進度。


## Randomized SVD 增加 subspace iteration

將 rSVD 引入 subspace iteration。

實驗之後發現 iteration 做 1 次是最好的。

效果：overhead 變得很大（48ms 變成 203ms）。但這是因為 TQ 目前只 apply 到 B，沒有 apply 到 subspace iteration。




## 新增不同的 testcase（矩陣 $A$）generation 方法

觀察發現現實中的科學 / 工程問題大部分矩陣的 singular spectrum 其實下降速度非常快，且有些矩陣 decay 的行為很接近 polynomial decay 或 exponential decay，因此新增了後兩種生成測資的方式（生出來的 $A$ 矩陣的 singular values 可以選擇服從 $\sigma_i \sim i^{-p}$ 或 $\sigma_i \sim \exp(-i)$）。

Implemented CLI:

- `--spectrum-decay-mode random|polynomial|exponential` (default: `random`)
- `--spectrum-decay-param <float>` (default: `1.0`)
- `--spectrum-rank <int>` (default: `min(m, n)`)

<!-- > Implementation note: the code builds deterministic DCT orthonormal bases on GPU and forms row blocks by GEMM:
> ```text
> A_i = U_i Sigma V^T = U_i (V Sigma)^T
> ```
> This avoids explicitly forming huge `U` and `V` on host or doing CPU-side `O(m n rank)` loops. -->




## 已經驗證 TurboQuant 的旋轉變換改成 RHT 是完全可行的

TurboQuant 的演算法要能成功，有一個重要性質是，做旋轉變換之後向量的每個分量要服從常態分佈。

但 TurboQuant 原本的旋轉變換是最 naïve 的 $O(d^2)$ 矩陣乘法。因此為了不讓 quantization 成為瓶頸，需要更快的方法。其中一個方法是 RHT 變換：
1. 先將各個分量隨機乘上 $+1$ 或 $-1$。
2. 對該向量做 Fast Walsh-Hadamard Transform（可以想成是 FFT 的變體，FFT 是快速做加法的卷積，而 FWHT 是快速做位元運算的卷積）

但還是得驗證改成 RHT 之後，分量的分佈是否保證常態分佈。

實驗證實 RHT 完全沒問題，可以將各個分量均勻打散、變成漂亮的常態分佈。詳細結果在 `turboquant-multigpu-svd/docs/notes/rht-distribution-test/rht_distribution.png`



## 更改了誤差測量指標的呈現方式

誤差測量有兩個指標：
- `B Relative Error`：測量 TurboQuant 壓縮並解壓縮 `B_i` 之後，跟原本的 `B_i` 差多少。公式：
```math
\frac{\lVert B' - B \rVert_F}{\lVert B \rVert_F}
```
- `Final Reconstruction Error`：測量最終的 Randomized SVD 近似 $A$ 矩陣的誤差。
```math
\sqrt{
  \frac{
    \lVert A \rVert_F^2 - \sum_{i=1}^{k} \hat{\sigma}_i^2
  }{
    \lVert A \rVert_F^2
  }
}
```
  其中 $\hat{\sigma}_i$ 是 randomized SVD 最後從小矩陣 $B=Q^TA$ 做 SVD 得到的 singular values。

舊的呈現 `Final Reconstruction Error` 的方式是直接顯示誤差是多少 %。

### 更動

Randomized SVD 天然就存在理論誤差，而對於 `--spectrum-decay-mode=polynomial` 或是 `--spectrum-decay-mode=exponential` 的測資，理論誤差可以直接用公式算：
```math
\sqrt{
  \frac{
    \sum_{i=k+1}^{\text{spectrum\_rank}} \sigma_i^2
  }{
    \sum_{i=1}^{\text{spectrum\_rank}} \sigma_i^2
  }
}
```

對於能算理論誤差的測資，v4 會先算 theoretical error，接著算 error ratio：
```math
\frac{\text{our reconstruction error (\%)}}{\text{theoretical error (\%)}}
```

換成這個指標比較好看出 TurboQuant 讓 error ratio 改變了多少、會不會讓 ratio 變得很大。

<!-- ## TODO -->




<!-- ## ADDED BUT NEGATIVE

### 嘗試修復 subspace iteration 可能的精度問題

> 結果發現做了反而誤差變大，目前決定捨棄。

Implemented an experiment for checking whether the intermediate right-side sketch should be orthogonalized:

- New CLI: `--stabilize-subspace-z`
- Current control path:
  ```text
  Z = A^T Q
  Y = A Z
  Q = qr(Y)
  ```
- Stabilized experiment path:
  ```text
  Z = A^T Q
  Qz = qr(Z)
  Y = A Qz
  Q = qr(Y)
  ```

Experiment folder: `exp_subspace_stabilization/`.


#### A Fatal Flaw in the Power Iteration Pseudocode

The note provides this pseudocode for the power iteration:

```
text
Y = A Ω
Repeat q times:
    Y = A * (A^T * Y)
```

While mathematically elegant, **this will fail in a real implementation due to floating-point round-off errors.**

Repeatedly multiplying by $A A^T$ causes all the column vectors in $Y$ to rapidly align with the single dominant singular vector of $A$. After a few iterations, the columns of $Y$ become nearly identical (linearly dependent), causing you to lose information about the 2nd, 3rd, and $k$-th singular vectors.

**The Fix:** You must explicitly orthogonalize the vectors (using a QR or LU decomposition) during every iteration step to keep them from collapsing into each other.


Here is the mathematically stable and practically correct way to write the power iteration step.

To prevent the columns from collapsing into each other, you must insert an orthogonalization step (typically a QR decomposition) between the matrix multiplications.

#### The Corrected Pseudocode

```
// 1. Initial sketch
Y = A * Ω
Q, R = qr(Y)  // Orthogonalize immediately

// 2. Power Iteration with stabilized subspace
Repeat q times:
    // Step A: Multiply by A^T and orthogonalize
    Y_temp = A^T * Q
    Q_temp, R_temp = qr(Y_temp)
    
    // Step B: Multiply by A and orthogonalize again
    Y = A * Q_temp
    Q, R = qr(Y)

// The final 'Q' is now your stable orthonormal basis
```

#### Why this is necessary

In your original snippet `Y = A * (A^T * Y)`, floating-point arithmetic ruins the math. Because the largest singular value acts like a magnet, every time you multiply by $A$ and $A^T$, all the column vectors in $Y$ get pulled toward that single dominant direction.

If you do this without intermediate orthogonalization, the vectors quickly lose their linear independence. The qr() function solves this by constantly "resetting" the vectors so they remain perfectly perpendicular (orthogonal) to each other, capturing the full subspace instead of just the single largest feature.

**A quick note for your HPC project:** While QR decomposition makes the algorithm mathematically stable, computing qr() inside a loop requires significant data movement. If you are distributing this across multiple GPUs or nodes, those qr() steps become synchronization bottlenecks where the nodes have to talk to each other. Balancing that communication overhead against mathematical accuracy is exactly the kind of engineering challenge this algorithm presents in high-performance computing.

 -->
