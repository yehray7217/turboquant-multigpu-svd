# AI Takeaway — docs/notes/code/

## Purpose

**Visualization scripts** for understanding the mathematical properties that
motivate the project. These are standalone Python scripts, not part of the
CUDA pipeline.

---

## Files

| File | Description |
|------|-------------|
| `sv-distribution-image.py` | Plots singular value spectrum of real image data (LFW faces from scikit-learn). Shows rapid decay and demonstrates that ~1/8 of singular values preserve 99.9% energy. |
| `randomized-svd/cpu-single-core.cpp` | NEW (v4): single-core C++ reference implementation of randomized SVD with synthetic decay test matrices (exponential/polynomial/step). The CPU oracle for the GPU pipeline's algorithm and accuracy metric. |
| `randomized-svd/run.sh` | NEW (v4): helper that compiles (`g++ -std=c++17 -O2`) and runs a single-file C++ source with passthrough args. |

---

## What sv-distribution-image.py Shows

The script:
1. Loads 15 random 250×250 grayscale face images from `sklearn.datasets.fetch_lfw_people`
2. Computes the full SVD of each image
3. Plots the singular value spectrum (σ₁, σ₂, ..., σₙ vs index)
4. Finds `k` such that `Σᵢ₌₁ᵏ σᵢ² / Σᵢ σᵢ² ≥ 0.999` (99.9% energy)
5. Reconstructs each image using only the top-k singular values
6. Saves `singular_value_spectrum_image.png` and `image_svd_compression.png`
   to `../img/`

**Key insight demonstrated**: even at 99.9% energy retention, you only need
roughly 1/8 of the singular values. This motivates rank-k approximation and
the entire randomized SVD approach.

---

## How to Run

```bash
cd docs/notes/code
pip install numpy matplotlib scikit-learn   # if not installed
python sv-distribution-image.py
```

The output images appear in `docs/notes/img/`.

---

## Dependencies

- `numpy`
- `matplotlib`
- `scikit-learn` (for `fetch_lfw_people`)

The script downloads the LFW dataset on first run (~200 MB). On HPC clusters
without internet access, you may need to pre-download and cache the dataset.

---

## Next Steps / What To Try

1. **Run locally** (not on HPC) to regenerate the figures if they need updating.
2. **Add a singular value decay plot for random Gaussian matrices**: shows that
   random matrices have flat singular value spectra — motivating why TurboQuant's
   FWHT rotation makes quantization more efficient.
3. **Add a quantization error visualization**: for a given B_i matrix, plot
   `||B_i - B_i_hat||_F / ||B_i||_F` vs bits for lowbit/TQ — matches the
   experimental results from the CUDA pipeline.
