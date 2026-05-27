### A Fatal Flaw in the Power Iteration Pseudocode

The note provides this pseudocode for the power iteration:

```text
Y = A Ω
Repeat q times:
    Y = A * (A^T * Y)

```

While mathematically elegant, **this will fail in a real implementation due to floating-point round-off errors.**

Repeatedly multiplying by $A A^T$ causes all the column vectors in $Y$ to rapidly align with the single dominant singular vector of $A$. After a few iterations, the columns of $Y$ become nearly identical (linearly dependent), causing you to lose information about the 2nd, 3rd, and $k$-th singular vectors.

**The Fix:** You must explicitly orthogonalize the vectors (using a QR or LU decomposition) during *every* iteration step to keep them from collapsing into each other.

---

Here is the mathematically stable and practically correct way to write the power iteration step.

To prevent the columns from collapsing into each other, you must insert an orthogonalization step (typically a QR decomposition) between the matrix multiplications.

### The Corrected Pseudocode

```text
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

### Why this is necessary

In your original snippet `Y = A * (A^T * Y)`, floating-point arithmetic ruins the math. Because the largest singular value acts like a magnet, every time you multiply by $A$ and $A^T$, all the column vectors in $Y$ get pulled toward that single dominant direction.

If you do this without intermediate orthogonalization, the vectors quickly lose their linear independence. The `qr()` function solves this by constantly "resetting" the vectors so they remain perfectly perpendicular (orthogonal) to each other, capturing the full subspace instead of just the single largest feature.

**A quick note for your HPC project:** While QR decomposition makes the algorithm mathematically stable, computing `qr()` inside a loop requires significant data movement. If you are distributing this across multiple GPUs or nodes, those `qr()` steps become synchronization bottlenecks where the nodes have to talk to each other. Balancing that communication overhead against mathematical accuracy is exactly the kind of engineering challenge this algorithm presents in high-performance computing.