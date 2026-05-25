# TurboQuant/QJL Utilities

This directory contains reusable compression utilities for SVD communication
experiments.

The implementation contains three families of compression methods:

1. `lowbit`: simple symmetric int8/int4 scalar quantization.
2. `tq`: TurboQuant first-stage with random rotation and low-bit quantization.
3. `tq-qjl`: TurboQuant/QJL prototype with random rotation and residual sign
   sketch correction.

## Supported Modes

```text
mode = none     bits = 0  no compression, stores FP32 bytes
mode = lowbit   bits = 8  symmetric int8 quantization
mode = lowbit   bits = 4  symmetric int4 quantization, packed two values per byte
mode = lowbit   bits = 2  symmetric int2 quantization, packed four values per byte
mode = tq       bits = 8  TurboQuant first-stage with int8 main codes
mode = tq       bits = 4  TurboQuant first-stage with int4 main codes
mode = tq       bits = 2  TurboQuant first-stage with int2 main codes
mode = tq-qjl   bits = 8  TurboQuant/QJL prototype with int8 main codes
mode = tq-qjl   bits = 4  TurboQuant/QJL prototype with int4 main codes
mode = tq-qjl   bits = 2  TurboQuant/QJL prototype with int2 main codes
```

The `lowbit` quantizer uses per-block symmetric scaling:

```text
scale = max(abs(x)) / qmax
q = clamp(round(x / scale), -qmax, qmax)
x_hat = q * scale
```

where:

```text
qmax = 127 for int8
qmax = 7   for int4
```

## API

```cpp
#include "../turboquant/turboquant.hpp"

turboquant::QuantizeOptions options =
    turboquant::make_quantize_options(bits, "tq-qjl", 256, 0.25f, 1234);

turboquant::CompressedBlock block =
    turboquant::quantize_fp32_block(values, rows, cols, options);

std::vector<float> reconstructed =
    turboquant::dequantize_fp32_block(block);
```

Useful metrics:

```cpp
block.payload_bytes();
block.compression_ratio_vs_fp32();
```

For GPU-side low-bit encoding:

```cpp
turboquant::QuantizeOptions options =
    turboquant::make_quantize_options(bits, "lowbit", 0, 1234);

turboquant::CompressedBlock block =
    turboquant::quantize_fp32_device_block(d_values, rows, cols, options);
```

## TurboQuant/QJL Prototype

The `tq` mode performs:

1. Random Rademacher sign preconditioning.
2. Normalized fast Walsh-Hadamard transform.
3. Low-bit scalar quantization in the rotated domain.
4. Inverse transform and inverse sign reconstruction.

The `tq-qjl` mode additionally performs:

5. Residual computation.
6. QJL residual sign sketch.
7. Approximate residual reconstruction from the sign sketch, scaled by
   `qjl_alpha`.

The QJL correction is a one-bit sketch reconstruction approximation. A more
faithful production version should integrate QJL directly into the
inner-product estimator used by `B_i = Q_i^T A_i`.

## Planned Extensions

1. GPU-side Hadamard rotation.
2. GPU-side QJL residual sketch.
3. Inner-product-level QJL correction for `B_i`.
4. Optional per-column or per-row block scaling.

For the randomized SVD v2 baseline, the primary compression target is the
`B_i = Q_i^T A_i` reduction.
