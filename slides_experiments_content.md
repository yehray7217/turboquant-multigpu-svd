# Slides Content — Experiments & Results / Discussion

English content ready to paste into the Google Slides.
This fills the **Experiments and Results** TODOs (slide 13) and the empty **Discussion** (slide 14).

All numbers are from real runs on **Taiwania 2** (`nycugpu_queue`, 2 nodes × 8 NVIDIA V100 = 16 GPUs,
account ACD115064), CUDA 12.8 + OpenMPI 5.0.2, `repeat = 50` (1 cold + 49 warm). Source logs:
`randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/output_logs/` (job IDs 946729–946733),
parsed by `parse_results.py`.

---

## Slide — Experiment Setup

**What we measure**
- Two questions on synthetic matrices with a *known* singular spectrum:
  1. **Accuracy impact** — how much extra error does TurboQuant add to randomized SVD?
  2. **Scaling** — does TQ help more as the matrix (communication payload) grows?

**Fixed config**

| Param | Value | | Param | Value |
|---|---|---|---|---|
| m × n | 32768 × 8192 | | oversample p | 6 (l = 256) |
| target rank k | 250 | | subspace iter | 1 |
| spectrum | polynomial, σᵢ = i⁻ᵖ | | spectrum rank | 8192 = min(m,n) |
| repeat | 50 | | GPUs | 16 (2×8 V100) |

**Methods compared:** No TQ · TQ 8-bit · TQ 4-bit (compress both the `B = QᵀA` reduce and the
subspace-iteration `Z = AᵀQ` reduce). QJL is excluded.

---

## Slide — Why a polynomial spectrum? (fills TODO #1)

- Real scientific / engineering / ML matrices are **approximately low rank**: a few singular values
  dominate and the tail decays smoothly — often like a **power law** σᵢ ∼ i⁻ᵖ (Udell & Townsend 2019;
  Eikmeier & Gleich 2017).
- A Gaussian random matrix has an almost **flat** spectrum, so rank-k error is huge and unrepresentative.
- Polynomial spectra let us (a) match realistic decay and (b) **compute the theoretical best rank-k
  error in closed form**, so we can isolate exactly how much error TQ adds.
- The decay parameter `p` is a difficulty knob: small `p` = slow decay (hard), large `p` = fast decay (easy).

| p | Theoretical rank-250 error (rank=8192) | Regime |
|---:|---:|---|
| 0.4 | 76.70% | slow decay — hard |
| 0.6 | 41.77% | main stress test |
| 0.8 | 15.27% | medium-easy |
| 1.0 | 4.85% | easy / sanity check |

---

## Slide — The error story (fills TODO #2)

Three layers of error, measured by **Error Ratio = Final Reconstruction Error / Theoretical Error**:

1. **Theoretical floor** — even an exact rank-k SVD cannot beat the Eckart–Young optimum
   (Error Ratio = 1.00 by definition).
2. **Randomized SVD** adds a *small* sketching error on top → Error Ratio slightly above 1
   (e.g. 1.06 at p=0.6 with 1 subspace iteration).
3. **TurboQuant** compresses the communication. The question: how much further does the ratio grow?

**Headline:** TQ8 adds **almost nothing** — at p=0.6 the Error Ratio goes 1.0626 → 1.0632, an
**Error Inflation of just 1.0006** (0.06%). The communication is cut ~2× over InfiniBand for free.

---

## Slide — Accuracy Sweep results (32768 × 8192)

| p | Method | Total (ms) | Speedup | Error Ratio | **Error Inflation** | Global B Rel Err |
|---:|---|---:|---:|---:|---:|---:|
| 0.4 | No TQ | 99.44 | 1.00× | 1.0421 | 1.0000 | – |
| 0.4 | TQ8 | 80.17 | **1.24×** | 1.0422 | **1.0001** | 1.54% |
| 0.4 | TQ4 | 62.39 | **1.59×** | 1.0487 | 1.0064 | 13.27% |
| 0.6 | No TQ | 98.99 | 1.00× | 1.0626 | 1.0000 | – |
| 0.6 | TQ8 | 80.08 | **1.24×** | 1.0632 | **1.0006** | 1.48% |
| 0.6 | TQ4 | 61.99 | **1.60×** | 1.1102 | 1.0448 | 13.27% |
| 0.8 | No TQ | 98.40 | 1.00× | 1.0906 | 1.0000 | – |
| 0.8 | TQ8 | 80.59 | 1.22× | 1.0969 | 1.0058 | 1.39% |
| 0.8 | TQ4 | 63.55 | 1.55× | 1.4209 | 1.3029 | 13.18% |
| 1.0 | No TQ | 97.91 | 1.00× | 1.1263 | 1.0000 | – |
| 1.0 | TQ8 | 80.04 | 1.22× | 1.1812 | 1.0488 | 1.32% |
| 1.0 | TQ4 | 63.06 | 1.55× | 2.9614 | **2.6295** | 13.09% |

**Talking points**
- **Speedup is flat in `p`** (payload size doesn't depend on the spectrum): TQ8 ≈ 1.22–1.24×, TQ4 ≈ 1.55–1.60×.
- **TQ8 is safe everywhere**: Error Inflation ≤ 1.05 across the whole sweep, ≤ 1.006 for p ≤ 0.8.
- **TQ4 degrades as the spectrum steepens**: 1.006 (p=0.4) → **2.63 (p=1.0)**. The 4-bit
  quantizer has a fixed ~13% `B` error floor; when the theoretical error is tiny (4.85% at p=1.0),
  that fixed noise *relatively* dominates and the ratio blows up.

> Plot 1: x = p (0.4→1.0), y = Error Inflation, two lines (TQ8, TQ4). TQ8 stays flat near 1.0; TQ4 curves up steeply.

---

## Slide — Scaling results (p = 0.6)

| Matrix Size | Method | Total (ms) | Speedup | IB Payload | Error Inflation |
|---|---|---:|---:|---:|---:|
| 16384 × 4096 | No TQ | 68.82 | 1.00× | 32.0 MiB | 1.0000 |
| 16384 × 4096 | TQ8 | 59.43 | 1.16× | 20.1 MiB | 1.0008 |
| 16384 × 4096 | TQ4 | 50.27 | 1.37× | 18.1 MiB | 1.0546 |
| 32768 × 8192 | No TQ | 98.99 | 1.00× | 48.0 MiB | 1.0000 |
| 32768 × 8192 | TQ8 | 80.08 | **1.24×** | 24.1 MiB | 1.0006 |
| 32768 × 8192 | TQ4 | 61.99 | **1.60×** | 20.1 MiB | 1.0448 |

**Talking points**
- **TQ helps more as the matrix grows**: TQ8 speedup 1.16× → 1.24×; TQ4 1.37× → 1.60×.
- InfiniBand payload is cut ~2× (TQ8) to ~2.4× (TQ4); accuracy cost stays tiny at p=0.6.
- Communication compression pays off exactly where it matters — the large `O(nl)` reduces.

> Plot 2: bar chart, x = matrix size, y = speedup vs No TQ, bars for TQ8 and TQ4. Both grow with size.

---

## Slide — Discussion

- **TQ8 is the recommended default**: ~1.24× end-to-end speedup at the standard 32768×8192 / p=0.6
  setting for ~0.06% extra error, and the benefit grows with problem size.
- **TQ4 is a speed/accuracy trade**: up to 1.6× faster, fine on slowly decaying spectra (p ≤ 0.6),
  but risky on sharply decaying spectra (p ≥ 0.8) where its fixed quantization-noise floor dominates.
- **Always report Error Ratio, not raw error**: at p=1.0 TQ4's raw error (14.4%) looks "small" vs
  p=0.4's 80%, but relative to the 4.85% optimum it is a 2.6× inflation — the ratio exposes the real cost.
- **Why this works**: the RHT rotation spreads each vector's mass into a near-Gaussian distribution,
  so scalar quantization is uniform and 8-bit is enough to keep `B` error ~1.5%. The reduce traffic
  (`O(nl)`, the dominant communication) shrinks ~2× over InfiniBand with negligible accuracy loss.

---

## Appendix — full metrics (reference, not for slides)

See `randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling/README.md` for the complete tables
including GPU compute / host-staging / NVLink times and full payload breakdowns. Re-generate with:

```bash
cd randomized_svd_baseline_v4/exp_synthetic_accuracy_scaling
python3 parse_results.py
```
