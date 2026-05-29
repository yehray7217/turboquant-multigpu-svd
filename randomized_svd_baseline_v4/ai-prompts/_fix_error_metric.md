Please fix the final reconstruction error metric in `randomized_svd_multigpu_v4.cu`.

## Problem

The current final reconstruction error is computed with the fast proxy:

\[
\sqrt{
\frac{\|A\|_F^2 - \sum_{i=1}^{k} S_i^2}
{\|A\|_F^2}
}
\]

where \(S_i\) are the singular values computed from the final reduced matrix \(B\).

This proxy is valid for the uncompressed baseline when:

\[
B = Q^T A
\]

but it becomes invalid when lossy compression is applied to \(B\), because the compressed/decompressed matrix is:

\[
\tilde{B} = B + E
\]

Quantization noise can inflate the singular values of \(\tilde{B}\), causing:

\[
\sum_{i=1}^{k} \sigma_i(\tilde{B})^2
\]

to exceed the theoretical optimal captured energy. We observed this directly when TQ 4-bit produced `Captured Energy Ratio > 1`.

Therefore, do NOT use the fast proxy as the final reconstruction error for compressed runs.

## Required Fix

Add a compressed-safe reconstruction error metric.

For compressed runs, compute:

\[
B_{\text{exact}} = Q^T A
\]

without compression, and also compute:

\[
\tilde{B}
\]

which is the actual compressed/decompressed matrix that is passed into the final SVD.

Then after computing the rank-\(k\) approximation of \(\tilde{B}\):

\[
\tilde{B}_k = \tilde{U}_k \Sigma_k V_k^T
\]

compute the true projected-space reconstruction error:

\[
\|A - Q\tilde{B}_k\|_F^2
=
\|A\|_F^2
-
\|B_{\text{exact}}\|_F^2
+
\|B_{\text{exact}} - \tilde{B}_k\|_F^2
\]

Then report:

\[
\text{final\_error}
=
\sqrt{
\frac{
\|A\|_F^2
-
\|B_{\text{exact}}\|_F^2
+
\|B_{\text{exact}} - \tilde{B}_k\|_F^2
}{
\|A\|_F^2
}
}
\]

This metric should be used whenever lossy compression is enabled.

## Important Details

1. Preserve the old fast metric for the uncompressed baseline if useful.
2. For compressed runs, the final error must be computed using `B_exact` and `B_compressed`.
3. `B_exact` should mean the mathematically correct uncompressed reduced matrix:
   \[
   B_{\text{exact}} = Q^T A
   \]
4. `B_compressed` should mean the actual matrix that enters the final SVD after all compression/decompression and accumulation.
5. The metric should not rely only on the singular values of `B_compressed`.
6. Add a diagnostic warning if:
   \[
   \sum_{i=1}^{k} \sigma_i(B_{\text{compressed}})^2
   >
   \sum_{i=1}^{k} \sigma_i(A)^2
   \]
   or if `Captured Energy Ratio > 1`.
   This indicates that the old energy-based metric would be invalid.

## Additional Diagnostic

Please also add or fix:

\[
\text{Global B Relative Error}
=
\frac{
\|B_{\text{compressed}} - B_{\text{exact}}\|_F
}{
\|B_{\text{exact}}\|_F
}
\]

This is more meaningful than only measuring local \(B_i\) compression error.

## Expected Output Changes

The final summary should report:

- Theoretical best rank-\(k\) error
- Final reconstruction error using the compressed-safe metric
- Error ratio:
  \[
  \frac{\text{final reconstruction error}}{\text{theoretical best rank-}k\text{ error}}
  \]
- Captured Energy Ratio, but mark it as diagnostic only
- Global B Relative Error when compression is enabled

## Acceptance Criteria

After the fix:

1. TQ 4-bit should no longer report impossible final reconstruction error below theoretical best error.
2. `Captured Energy Ratio > 1` should trigger a warning and should not be interpreted as improved accuracy.
3. The final error should be based on \(B_{\text{exact}}\) vs. \(\tilde{B}_k\), not only on singular values of \(\tilde{B}\).
4. `--check-b-error` must remain diagnostic only and must not change the main computation path.
5. `--check-b-error on` and `--check-b-error off` should produce the same final \(B_{\text{compressed}}\), except for additional printed diagnostics.