# AI Takeaway — docs/notes/

## Purpose

**Theory notes and learning resources.** This directory contains markdown notes
and supporting visuals that explain the mathematical foundations behind the
project. Read these before diving into the CUDA source code.

---

## Files

| File | Description |
|------|-------------|
| `SVD.md` | Core SVD theory: orthogonal matrices, eigendecomposition, SVD definition, cuSOLVER implementation details. |
| `Randomized_SVD.md` | Randomized SVD algorithm: low-rank approximation, algorithm steps, power iteration, pseudocode, accuracy analysis. |
| `TurboQuant.md` | TurboQuant + QJL theory — currently marked "施工中" (under construction). |
| `code/` | Supporting Python scripts (singular value spectrum visualization). |
| `img/` | Figures and diagrams referenced by the markdown notes. |

---

## Reading Order

1. **`SVD.md`** — understand what SVD is, what U/Σ/V mean, how NVIDIA implements it.
2. **`Randomized_SVD.md`** — understand how randomized SVD approximates it cheaply.
3. **Root README.md** — understand how TurboQuant/QJL compresses the communication.
4. **`TurboQuant.md`** — intended to formalize TQ/QJL theory (not complete yet).

---

## SVD.md Summary

Key concepts covered:
- Orthogonal matrix: `QᵀQ = I`, columns are orthonormal
- Singular values: square roots of eigenvalues of `AᵀA`
- SVD: `A = U Σ Vᵀ` where U, V are orthogonal and Σ is diagonal
- Spectral theorem: symmetric positive semidefinite matrices are diagonalizable
- cuSOLVER implementation: bidiagonalization → QR iteration on bidiagonal form
  (or Jacobi variant for smaller matrices)

---

## Randomized_SVD.md Summary

Algorithm (7 steps):
1. Generate random matrix `Ω ∈ Rⁿˣˡ` where `l = k + p`
2. Compute `Y = AΩ` (random projection, samples column space of A)
3. QR: `Y = QR` (orthonormal basis for column space)
4. Project: `B = QᵀA` (small matrix `l × n`)
5. SVD: `B = Ũ Σ Vᵀ` (cheap since B is small)
6. Lift: `U = Q Ũ` (recover left singular vectors in original space)
7. Truncate to rank-k

Complexity: `O(mnl + ml² + nl²)` — much better than `O(mn·min(m,n))` for full SVD
when `l = k + p ≪ min(m,n)`.

Power iteration: replace `Y = AΩ` with `Y = (AAᵀ)^q AΩ` to amplify large singular
values. Common choice: `q = 1` or `q = 2`.

---

## TurboQuant.md Status

Currently empty / placeholder ("施工中"). The full TurboQuant theory is
documented in:
- Root `README.md` (concept overview with diagrams)
- Root `OPTIMIZATION_HISTORY.md` (what was tried and what worked)
- `turboquant/README.md` (API and algorithm flow)

---

## Next Steps / What To Do

1. **Complete `TurboQuant.md`**: document the TQ/QJL math formally:
   - Random Rademacher signs: `s ∈ {±1}^d`
   - Normalized FWHT: `Hx / sqrt(d)`
   - Low-bit quantization formula
   - QJL residual sketch: `q = sign(Gr)`, reconstruction `r̂ ≈ (√(π/2)/d_qjl) Gᵀ q`
   - Why the current QJL implementation doesn't improve error (crude estimator)
2. **Add power iteration note to Randomized_SVD.md**: the current implementation
   in v2/v3 does NOT use power iteration (standard `Y = AΩ` only). Adding `q=1`
   would reduce singular value approximation error.
3. **Add a note on TSQR**: the v2/v3 pipeline uses TSQR (Tall Skinny QR) for
   distributed QR across GPUs. This is not covered in the current notes.
