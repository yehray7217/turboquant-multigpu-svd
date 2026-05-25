#include "turboquant.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace turboquant {
namespace {

void check_cuda(cudaError_t status, const char* label);

int qmax_for_bits(int bits) {
    if (bits == 8) return 127;
    if (bits == 4) return 7;
    if (bits == 2) return 1;
    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

int next_power_of_two(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

__host__ __device__ std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__host__ __device__ float rademacher(unsigned seed, std::uint32_t a, std::uint32_t b) {
    std::uint32_t h = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU));
    return (h & 1U) ? 1.0f : -1.0f;
}

__host__ __device__ float uniform01_from_hash(std::uint32_t h) {
    return (static_cast<float>(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__host__ __device__ float gaussian_from_hash(unsigned seed, std::uint32_t a, std::uint32_t b) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    std::uint32_t h0 = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU) ^ 0x243f6a88U);
    std::uint32_t h1 = mix32(seed ^ (a * 0x85ebca6bU) ^ (b * 0xc2b2ae35U) ^ 0x9e3779b9U);
    float u1 = fmaxf(uniform01_from_hash(h0), 1.0e-7f);
    float u2 = uniform01_from_hash(h1);
    return sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
}

void apply_random_sign(std::vector<float>& x, unsigned seed) {
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] *= rademacher(seed, static_cast<std::uint32_t>(i), 0);
    }
}

void fwht_normalized(std::vector<float>& x) {
    const int n = static_cast<int>(x.size());
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                float a = x[i + j];
                float b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));
    for (float& v : x) v *= inv_sqrt_n;
}

void pack_sign_bit(std::vector<std::uint8_t>& signs, int idx, bool positive) {
    if (positive) signs[idx / 8] |= static_cast<std::uint8_t>(1U << (idx % 8));
}

bool unpack_sign_bit(const std::vector<std::uint8_t>& signs, int idx) {
    return ((signs[idx / 8] >> (idx % 8)) & 1U) != 0;
}

std::uint8_t encode_signed_to_unsigned(int q, int bits) {
    if (bits == 8) {
        return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
    }
    if (bits == 4) {
        return static_cast<std::uint8_t>(q + 8);
    }
    if (bits == 2) {
        return static_cast<std::uint8_t>(q + 1);
    }
    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

int decode_unsigned_to_signed(std::uint8_t code, int bits) {
    if (bits == 8) {
        return static_cast<int>(static_cast<std::int8_t>(code));
    }
    if (bits == 4) {
        return static_cast<int>(code & 0x0f) - 8;
    }
    if (bits == 2) {
        return static_cast<int>(code & 0x03) - 1;
    }
    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

std::vector<std::uint8_t> pack_codes(const std::vector<std::uint8_t>& codes, int bits) {
    if (bits == 8) return codes;

    if (bits == 4) {
        std::vector<std::uint8_t> packed((codes.size() + 1) / 2, 0);
        for (std::size_t i = 0; i < codes.size(); ++i) {
            std::uint8_t nibble = static_cast<std::uint8_t>(codes[i] & 0x0f);
            if ((i & 1) == 0) {
                packed[i / 2] = nibble;
            } else {
                packed[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
            }
        }
        return packed;
    }
    if (bits == 2) {
        std::vector<std::uint8_t> packed((codes.size() + 3) / 4, 0);
        for (std::size_t i = 0; i < codes.size(); ++i) {
            std::uint8_t two_bits = static_cast<std::uint8_t>(codes[i] & 0x03);
            packed[i / 4] |= static_cast<std::uint8_t>(two_bits << ((i % 4) * 2));
        }
        return packed;
    }

    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

std::uint8_t unpack_code_at(const std::vector<std::uint8_t>& packed, std::size_t i, int bits) {
    if (bits == 8) return packed[i];

    if (bits == 4) {
        std::uint8_t byte = packed[i / 2];
        if ((i & 1) == 0) return static_cast<std::uint8_t>(byte & 0x0f);
        return static_cast<std::uint8_t>((byte >> 4) & 0x0f);
    }
    if (bits == 2) {
        std::uint8_t byte = packed[i / 4];
        return static_cast<std::uint8_t>((byte >> ((i % 4) * 2)) & 0x03);
    }

    throw std::runtime_error("Unsupported quantization bit width: " + std::to_string(bits));
}

struct AbsValue {
    __host__ __device__ float operator()(const float& x) const {
        return fabsf(x);
    }
};

struct SquareValue {
    __host__ __device__ float operator()(const float& x) const {
        return x * x;
    }
};

__global__ void quantize_int8_kernel(
    const float* values,
    std::uint8_t* codes,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int q = static_cast<int>(nearbyintf(values[idx] / scale));
    q = max(-127, min(127, q));
    codes[idx] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

__global__ void quantize_int4_pack_kernel(
    const float* values,
    std::uint8_t* packed_codes,
    std::size_t count,
    float scale) {
    const std::size_t out_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t i0 = out_idx * 2;
    if (i0 >= count) return;

    int q0 = static_cast<int>(nearbyintf(values[i0] / scale));
    q0 = max(-7, min(7, q0));
    std::uint8_t c0 = static_cast<std::uint8_t>((q0 + 8) & 0x0f);

    std::uint8_t c1 = 0;
    if (i0 + 1 < count) {
        int q1 = static_cast<int>(nearbyintf(values[i0 + 1] / scale));
        q1 = max(-7, min(7, q1));
        c1 = static_cast<std::uint8_t>((q1 + 8) & 0x0f);
    }

    packed_codes[out_idx] = static_cast<std::uint8_t>(c0 | (c1 << 4));
}

__global__ void quantize_int2_pack_kernel(
    const float* values,
    std::uint8_t* packed_codes,
    std::size_t count,
    float scale) {
    const std::size_t out_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t i0 = out_idx * 4;
    if (i0 >= count) return;

    std::uint8_t packed = 0;
    for (int lane = 0; lane < 4; ++lane) {
        const std::size_t i = i0 + lane;
        if (i >= count) break;
        int q = static_cast<int>(nearbyintf(values[i] / scale));
        q = max(-1, min(1, q));
        std::uint8_t code = static_cast<std::uint8_t>((q + 1) & 0x03);
        packed |= static_cast<std::uint8_t>(code << (lane * 2));
    }
    packed_codes[out_idx] = packed;
}

__global__ void copy_pad_random_sign_kernel(
    const float* values,
    float* padded,
    std::size_t count,
    std::size_t padded_count,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= padded_count) return;
    float v = (idx < count) ? values[idx] : 0.0f;
    padded[idx] = v * rademacher(seed, static_cast<std::uint32_t>(idx), 0);
}

__global__ void copy_pad_random_sign_columns_kernel(
    const float* values,
    float* padded,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(padded_rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(padded_rows));
    float v = (row < rows) ? values[static_cast<std::size_t>(col) * rows + row] : 0.0f;
    padded[idx] = v * rademacher(seed, static_cast<std::uint32_t>(row),
                                 static_cast<std::uint32_t>(col));
}

__global__ void initialize_column_signs_kernel(
    signed char* signs,
    int padded_rows,
    int cols,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(padded_rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(padded_rows));
    signs[idx] = (rademacher(seed, static_cast<std::uint32_t>(row),
                             static_cast<std::uint32_t>(col)) > 0.0f) ? 1 : -1;
}

__global__ void copy_pad_apply_sign_columns_kernel(
    const float* values,
    const signed char* signs,
    float* padded,
    int rows,
    int cols,
    int padded_rows) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(padded_rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(padded_rows));
    float v = (row < rows) ? values[static_cast<std::size_t>(col) * rows + row] : 0.0f;
    padded[idx] = v * static_cast<float>(signs[idx]);
}

__global__ void apply_random_sign_truncate_kernel(
    const float* padded,
    float* values,
    std::size_t count,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    values[idx] = padded[idx] * rademacher(seed, static_cast<std::uint32_t>(idx), 0);
}

__global__ void apply_random_sign_truncate_add_kernel(
    const float* padded,
    float* values,
    std::size_t count,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    values[idx] += padded[idx] * rademacher(seed, static_cast<std::uint32_t>(idx), 0);
}

__global__ void apply_random_sign_truncate_columns_add_kernel(
    const float* padded,
    float* values,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    values[idx] += padded[static_cast<std::size_t>(col) * padded_rows + row] *
                   rademacher(seed, static_cast<std::uint32_t>(row),
                              static_cast<std::uint32_t>(col));
}

__global__ void apply_sign_mask_truncate_columns_add_kernel(
    const float* padded,
    const signed char* signs,
    float* values,
    int rows,
    int cols,
    int padded_rows) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    const std::size_t padded_idx = static_cast<std::size_t>(col) * padded_rows + row;
    values[idx] += padded[padded_idx] * static_cast<float>(signs[padded_idx]);
}

__global__ void add_plain_kernel(float* dst, const float* src, std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] += src[idx];
}

__global__ void residual_kernel(
    const float* values,
    const float* reconstructed,
    float* residual,
    std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    residual[idx] = values[idx] - reconstructed[idx];
}

__global__ void qjl_dot_partial_kernel(
    const float* residual,
    float* partials,
    std::size_t count,
    int qjl_dim,
    int blocks_per_sketch,
    unsigned seed) {
    extern __shared__ float smem[];
    const int s = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (std::size_t j = static_cast<std::size_t>(b) * blockDim.x + tid;
         j < count;
         j += static_cast<std::size_t>(blocks_per_sketch) * blockDim.x) {
        sum += gaussian_from_hash(seed, static_cast<std::uint32_t>(s),
                                  static_cast<std::uint32_t>(j)) * residual[j];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        partials[static_cast<std::size_t>(s) * blocks_per_sketch + b] = smem[0];
    }
}

__global__ void qjl_pack_signs_kernel(
    const float* partials,
    int* signs,
    int qjl_dim,
    int blocks_per_sketch) {
    extern __shared__ float smem[];
    const int s = blockIdx.x;
    const int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < blocks_per_sketch; i += blockDim.x) {
        sum += partials[static_cast<std::size_t>(s) * blocks_per_sketch + i];
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        signs[s] = (smem[0] >= 0.0f) ? 1 : 0;
    }
}

__global__ void qjl_reconstruct_kernel(
    float* reconstructed,
    const int* signs,
    std::size_t count,
    int qjl_dim,
    float coeff,
    unsigned seed) {
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= count) return;
    float accum = 0.0f;
    for (int s = 0; s < qjl_dim; ++s) {
        const float sign = signs[s] ? 1.0f : -1.0f;
        accum += sign * gaussian_from_hash(seed, static_cast<std::uint32_t>(s),
                                           static_cast<std::uint32_t>(j));
    }
    reconstructed[j] += coeff * accum;
}

__global__ void fwht_stage_kernel(float* values, std::size_t count, int len) {
    const std::size_t pair_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pairs = count / 2;
    if (pair_idx >= pairs) return;
    const std::size_t block = pair_idx / static_cast<std::size_t>(len);
    const std::size_t offset = pair_idx % static_cast<std::size_t>(len);
    const std::size_t i0 = block * static_cast<std::size_t>(len) * 2 + offset;
    const std::size_t i1 = i0 + static_cast<std::size_t>(len);
    float a = values[i0];
    float b = values[i1];
    values[i0] = a + b;
    values[i1] = a - b;
}

__global__ void fwht_columns_stage_kernel(float* values, int padded_rows, int cols, int len) {
    const std::size_t pair_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t pairs_per_col = static_cast<std::size_t>(padded_rows) / 2;
    const std::size_t total_pairs = pairs_per_col * cols;
    if (pair_idx >= total_pairs) return;
    const int col = static_cast<int>(pair_idx / pairs_per_col);
    const std::size_t pair_in_col = pair_idx % pairs_per_col;
    const std::size_t block = pair_in_col / static_cast<std::size_t>(len);
    const std::size_t offset = pair_in_col % static_cast<std::size_t>(len);
    const std::size_t base = static_cast<std::size_t>(col) * padded_rows;
    const std::size_t i0 = base + block * static_cast<std::size_t>(len) * 2 + offset;
    const std::size_t i1 = i0 + static_cast<std::size_t>(len);
    float a = values[i0];
    float b = values[i1];
    values[i0] = a + b;
    values[i1] = a - b;
}

__global__ void column_tq_forward_fwht_kernel(
    const float* values,
    float* transformed,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= padded_rows) return;

    float v = 0.0f;
    if (row < rows) {
        v = values[static_cast<std::size_t>(col) * rows + row] *
            rademacher(seed, static_cast<std::uint32_t>(row),
                       static_cast<std::uint32_t>(col));
    }
    smem[row] = v;
    __syncthreads();

    for (int len = 1; len < padded_rows; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < padded_rows) {
            const float a = smem[i0];
            const float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    const float inv_sqrt_n = rsqrtf(static_cast<float>(padded_rows));
    transformed[static_cast<std::size_t>(col) * padded_rows + row] = smem[row] * inv_sqrt_n;
}

__global__ void column_tq_inverse_fwht_add_kernel(
    const float* transformed,
    float* values,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= padded_rows) return;

    smem[row] = transformed[static_cast<std::size_t>(col) * padded_rows + row];
    __syncthreads();

    for (int len = 1; len < padded_rows; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < padded_rows) {
            const float a = smem[i0];
            const float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    if (row < rows) {
        const float inv_sqrt_n = rsqrtf(static_cast<float>(padded_rows));
        values[static_cast<std::size_t>(col) * rows + row] +=
            smem[row] * inv_sqrt_n *
            rademacher(seed, static_cast<std::uint32_t>(row),
                       static_cast<std::uint32_t>(col));
    }
}

__global__ void column_tq_inverse_fwht_store_kernel(
    const float* transformed,
    float* values,
    int rows,
    int cols,
    int padded_rows,
    unsigned seed) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int row = threadIdx.x;
    if (col >= cols || row >= padded_rows) return;

    smem[row] = transformed[static_cast<std::size_t>(col) * padded_rows + row];
    __syncthreads();

    for (int len = 1; len < padded_rows; len <<= 1) {
        const int pair = row >> 1;
        const int offset = pair & (len - 1);
        const int block = pair / len;
        const int i0 = block * len * 2 + offset;
        const int i1 = i0 + len;
        if ((row & 1) == 0 && i1 < padded_rows) {
            const float a = smem[i0];
            const float b = smem[i1];
            smem[i0] = a + b;
            smem[i1] = a - b;
        }
        __syncthreads();
    }

    if (row < rows) {
        const float inv_sqrt_n = rsqrtf(static_cast<float>(padded_rows));
        values[static_cast<std::size_t>(col) * rows + row] =
            smem[row] * inv_sqrt_n *
            rademacher(seed, static_cast<std::uint32_t>(row),
                       static_cast<std::uint32_t>(col));
    }
}

__global__ void scale_kernel(float* values, std::size_t count, float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    values[idx] *= scale;
}

__global__ void dequantize_int8_kernel(
    const std::uint8_t* codes,
    float* values,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int q = static_cast<int>(static_cast<std::int8_t>(codes[idx]));
    values[idx] = static_cast<float>(q) * scale;
}

__global__ void dequantize_int4_pack_kernel(
    const std::uint8_t* packed_codes,
    float* values,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    std::uint8_t byte = packed_codes[idx / 2];
    std::uint8_t code = ((idx & 1) == 0) ?
        static_cast<std::uint8_t>(byte & 0x0f) :
        static_cast<std::uint8_t>((byte >> 4) & 0x0f);
    int q = static_cast<int>(code) - 8;
    values[idx] = static_cast<float>(q) * scale;
}

__global__ void dequantize_int2_pack_kernel(
    const std::uint8_t* packed_codes,
    float* values,
    std::size_t count,
    float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    std::uint8_t byte = packed_codes[idx / 4];
    std::uint8_t code = static_cast<std::uint8_t>((byte >> ((idx % 4) * 2)) & 0x03);
    int q = static_cast<int>(code) - 1;
    values[idx] = static_cast<float>(q) * scale;
}

void fwht_normalized_device(float* d_values, std::size_t count, cudaStream_t stream) {
    const int threads = 256;
    const std::size_t pairs = count / 2;
    for (int len = 1; static_cast<std::size_t>(len) < count; len <<= 1) {
        const int blocks = static_cast<int>((pairs + threads - 1) / threads);
        fwht_stage_kernel<<<blocks, threads, 0, stream>>>(d_values, count, len);
        check_cuda(cudaGetLastError(), "launch fwht stage kernel");
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(count));
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    scale_kernel<<<blocks, threads, 0, stream>>>(d_values, count, inv_sqrt_n);
    check_cuda(cudaGetLastError(), "launch fwht scale kernel");
}

void fwht_columns_normalized_device(float* d_values, int padded_rows, int cols, cudaStream_t stream) {
    const int threads = 256;
    const std::size_t pairs = static_cast<std::size_t>(padded_rows) * cols / 2;
    for (int len = 1; len < padded_rows; len <<= 1) {
        const int blocks = static_cast<int>((pairs + threads - 1) / threads);
        fwht_columns_stage_kernel<<<blocks, threads, 0, stream>>>(d_values, padded_rows, cols, len);
        check_cuda(cudaGetLastError(), "launch column fwht stage kernel");
    }
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(padded_rows));
    const int blocks = static_cast<int>(((static_cast<std::size_t>(padded_rows) * cols) + threads - 1) / threads);
    scale_kernel<<<blocks, threads, 0, stream>>>(
        d_values, static_cast<std::size_t>(padded_rows) * cols, inv_sqrt_n);
    check_cuda(cudaGetLastError(), "launch column fwht scale kernel");
}

void check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

std::size_t CompressedBlock::value_count() const {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

std::size_t CompressedBlock::payload_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    return codes.size() + sizeof(scale) + qjl_signs.size() + sizeof(residual_norm);
}

double CompressedBlock::compression_ratio_vs_fp32() const {
    const double fp32_bytes = static_cast<double>(value_count() * sizeof(float));
    if (fp32_bytes == 0.0) return 1.0;
    return fp32_bytes / static_cast<double>(std::max<std::size_t>(payload_bytes(), 1));
}

std::size_t DeviceCompressedBlock::value_count() const {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

std::size_t DeviceCompressedBlock::code_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    const std::size_t count = static_cast<std::size_t>(padded_count);
    if (bits == 8) return count;
    if (bits == 4) return (count + 1) / 2;
    if (bits == 2) return (count + 3) / 4;
    return 0;
}

std::size_t DeviceCompressedBlock::payload_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    return code_bytes() + sizeof(scale) +
           ((mode == QuantizeMode::kTurboQuantQjl) ? static_cast<std::size_t>(qjl_dim) * sizeof(int) : 0) +
           sizeof(residual_norm);
}

QuantizeOptions make_quantize_options(int bits) {
    QuantizeOptions options;
    if (bits == 0) {
        options.mode = QuantizeMode::kNone;
        options.bits = 0;
        return options;
    }
    if (bits == 8 || bits == 4 || bits == 2) {
        options.mode = QuantizeMode::kLowBit;
        options.bits = bits;
        return options;
    }
    throw std::runtime_error("Unsupported --compress-b-bits value: " + std::to_string(bits));
}

QuantizeOptions make_quantize_options(
    int bits,
    const std::string& mode,
    int qjl_dim,
    float qjl_alpha,
    unsigned seed) {
    QuantizeOptions options;
    options.bits = bits;
    options.qjl_dim = qjl_dim;
    options.qjl_alpha = qjl_alpha;
    options.seed = seed;

    if (mode == "none") {
        if (bits != 0) {
            throw std::runtime_error("mode=none requires bits=0.");
        }
        options.mode = QuantizeMode::kNone;
        return options;
    }
    if (mode == "lowbit") {
        if (bits != 8 && bits != 4 && bits != 2) {
            throw std::runtime_error("mode=lowbit requires bits=8, bits=4, or bits=2.");
        }
        options.mode = QuantizeMode::kLowBit;
        return options;
    }
    if (mode == "tq") {
        if (bits != 8 && bits != 4 && bits != 2) {
            throw std::runtime_error("mode=tq requires bits=8, bits=4, or bits=2.");
        }
        options.mode = QuantizeMode::kTurboQuant;
        options.qjl_dim = 0;
        options.qjl_alpha = 0.0f;
        return options;
    }
    if (mode == "tq-qjl") {
        if (bits != 8 && bits != 4 && bits != 2) {
            throw std::runtime_error("mode=tq-qjl requires bits=8, bits=4, or bits=2.");
        }
        if (qjl_dim <= 0) {
            throw std::runtime_error("mode=tq-qjl requires qjl_dim > 0.");
        }
        options.mode = QuantizeMode::kTurboQuantQjl;
        return options;
    }
    throw std::runtime_error("Unsupported quantization mode: " + mode);
}

CompressedBlock quantize_fp32_block(
    const std::vector<float>& values,
    int rows,
    int cols,
    const QuantizeOptions& options) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (values.size() != count) {
        throw std::runtime_error("Input size does not match compressed block dimensions.");
    }

    CompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;

    if (options.mode == QuantizeMode::kNone) {
        block.bits = 0;
        block.scale = 1.0f;
        block.codes.resize(count * sizeof(float));
        std::memcpy(block.codes.data(), values.data(), block.codes.size());
        return block;
    }

    block.qjl_dim = options.qjl_dim;
    block.qjl_alpha = options.qjl_alpha;
    block.seed = options.seed;

    std::vector<float> values_for_quant = values;
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        block.padded_count = next_power_of_two(static_cast<int>(count));
        values_for_quant.assign(static_cast<std::size_t>(block.padded_count), 0.0f);
        std::copy(values.begin(), values.end(), values_for_quant.begin());
        apply_random_sign(values_for_quant, block.seed);
        fwht_normalized(values_for_quant);
    } else {
        block.padded_count = static_cast<int>(count);
    }

    const int qmax = qmax_for_bits(options.bits);
    float max_abs = 0.0f;
    for (float x : values_for_quant) max_abs = std::max(max_abs, std::fabs(x));

    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    std::vector<std::uint8_t> unpacked_codes(values_for_quant.size());
    for (std::size_t i = 0; i < values_for_quant.size(); ++i) {
        float scaled = values_for_quant[i] / block.scale;
        int q = static_cast<int>(std::lrintf(scaled));
        q = std::max(-qmax, std::min(qmax, q));
        unpacked_codes[i] = encode_signed_to_unsigned(q, options.bits);
    }
    block.codes = pack_codes(unpacked_codes, options.bits);

    if (options.mode == QuantizeMode::kTurboQuantQjl) {
        std::vector<float> reconstructed_rot(values_for_quant.size());
        for (std::size_t i = 0; i < values_for_quant.size(); ++i) {
            int q = decode_unsigned_to_signed(unpacked_codes[i], options.bits);
            reconstructed_rot[i] = static_cast<float>(q) * block.scale;
        }
        fwht_normalized(reconstructed_rot);
        apply_random_sign(reconstructed_rot, block.seed);

        std::vector<float> residual(static_cast<std::size_t>(block.padded_count), 0.0f);
        long double residual_norm2 = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
            residual[i] = values[i] - reconstructed_rot[i];
            residual_norm2 += static_cast<long double>(residual[i]) * residual[i];
        }
        block.residual_norm = std::sqrt(static_cast<float>(residual_norm2));
        block.qjl_signs.assign((static_cast<std::size_t>(options.qjl_dim) + 7) / 8, 0);
        for (int s = 0; s < options.qjl_dim; ++s) {
            float dot = 0.0f;
            for (std::size_t j = 0; j < count; ++j) {
                dot += rademacher(block.seed + 17U, static_cast<std::uint32_t>(s),
                                  static_cast<std::uint32_t>(j)) * residual[j];
            }
            pack_sign_bit(block.qjl_signs, s, dot >= 0.0f);
        }
    }
    return block;
}

CompressedBlock quantize_fp32_device_block(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    cudaStream_t stream) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    CompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;

    if (options.mode == QuantizeMode::kNone) {
        block.bits = 0;
        block.scale = 1.0f;
        block.codes.resize(count * sizeof(float));
        check_cuda(cudaMemcpyAsync(
                       block.codes.data(), d_values, block.codes.size(),
                       cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync FP32 block");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize FP32 block");
        return block;
    }
    if (options.mode != QuantizeMode::kLowBit) {
        throw std::runtime_error("Device-side quantization currently supports only none and lowbit modes.");
    }

    const int qmax = qmax_for_bits(options.bits);
    thrust::device_ptr<const float> begin(d_values);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize max_abs");

    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    const std::size_t code_bytes = (options.bits == 8) ? count :
                                   (options.bits == 4) ? (count + 1) / 2 :
                                   (count + 3) / 4;
    block.codes.resize(code_bytes);

    std::uint8_t* d_codes = nullptr;
    check_cuda(cudaMalloc(&d_codes, code_bytes), "cudaMalloc quantized codes");

    const int threads = 256;
    if (options.bits == 8) {
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_values, d_codes, count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (count + 1) / 2;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_values, d_codes, count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (count + 3) / 4;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_values, d_codes, count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch quantization kernel");
    check_cuda(cudaMemcpyAsync(
                   block.codes.data(), d_codes, code_bytes,
                   cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync quantized codes");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize quantized codes");
    cudaFree(d_codes);
    return block;
}

CompressedBlock quantize_dequant_fp32_device_block_to_device(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    float* d_reconstructed,
    cudaStream_t stream,
    bool copy_payload_to_host) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (!d_reconstructed) {
        throw std::runtime_error("Device reconstructed output pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    if (options.mode != QuantizeMode::kNone &&
        options.mode != QuantizeMode::kLowBit &&
        options.mode != QuantizeMode::kTurboQuant &&
        options.mode != QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Unsupported device-side quantization mode.");
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

    CompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_dim : 0;
    block.qjl_alpha = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_alpha : 0.0f;
    block.seed = options.seed;

    if (options.mode == QuantizeMode::kNone) {
        block.bits = 0;
        block.scale = 1.0f;
        block.codes.resize(count * sizeof(float));
        if (copy_payload_to_host) {
            check_cuda(cudaMemcpyAsync(
                           block.codes.data(), d_values, block.codes.size(),
                           cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync FP32 block");
        }
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed, d_values, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync FP32 reconstructed device block");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize FP32 block");
        return block;
    }

    qmax_for_bits(options.bits);
    const std::size_t work_count =
        (options.mode == QuantizeMode::kTurboQuant ||
         options.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(next_power_of_two(static_cast<int>(count))) :
        count;
    block.padded_count = static_cast<int>(work_count);

    float* d_work = nullptr;
    float* d_residual = nullptr;
    float* d_qjl_partials = nullptr;
    int* d_qjl_signs = nullptr;
    std::uint8_t* d_codes = nullptr;

    const std::size_t code_bytes = (options.bits == 8) ? work_count :
                                   (options.bits == 4) ? (work_count + 1) / 2 :
                                   (work_count + 3) / 4;
    block.codes.resize(code_bytes);

    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc quantization work");
    check_cuda(cudaMalloc(&d_codes, code_bytes), "cudaMalloc quantized codes");

    const int threads = 256;
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        copy_pad_random_sign_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_work, count, work_count, options.seed);
        check_cuda(cudaGetLastError(), "launch copy/pad/random-sign kernel");
        fwht_normalized_device(d_work, work_count, stream);
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_work, d_values, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit work block");
    }

    thrust::device_ptr<const float> begin(d_work);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + work_count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize max_abs");

    const int qmax = qmax_for_bits(options.bits);
    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    if (options.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (work_count + 1) / 2;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (work_count + 3) / 4;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch quantization kernel");

    if (options.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
    } else if (options.bits == 4) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
    } else if (options.bits == 2) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch dequantization kernel");

    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        fwht_normalized_device(d_work, work_count, stream);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        apply_random_sign_truncate_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_reconstructed, count, options.seed);
        check_cuda(cudaGetLastError(), "launch inverse random-sign/truncate kernel");
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed, d_work, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit reconstructed block");
    }

    if (options.mode == QuantizeMode::kTurboQuantQjl &&
        options.qjl_dim > 0 &&
        options.qjl_alpha != 0.0f) {
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        check_cuda(cudaMalloc(&d_residual, count * sizeof(float)), "cudaMalloc qjl residual");
        residual_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_reconstructed, d_residual, count);
        check_cuda(cudaGetLastError(), "launch qjl residual kernel");

        thrust::device_ptr<const float> residual_begin(d_residual);
        float residual_norm2 = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            residual_begin, residual_begin + count,
            SquareValue{},
            0.0f,
            thrust::plus<float>{});
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize qjl residual norm");
        block.residual_norm = std::sqrt(std::max(residual_norm2, 0.0f));

        const int qjl_threads = 256;
        const int blocks_per_sketch = 256;
        const std::size_t partial_count =
            static_cast<std::size_t>(options.qjl_dim) * blocks_per_sketch;
        check_cuda(cudaMalloc(&d_qjl_partials, partial_count * sizeof(float)),
                   "cudaMalloc qjl partials");
        check_cuda(cudaMalloc(&d_qjl_signs, static_cast<std::size_t>(options.qjl_dim) * sizeof(int)),
                   "cudaMalloc qjl signs");
        dim3 partial_grid(options.qjl_dim, blocks_per_sketch);
        qjl_dot_partial_kernel<<<partial_grid, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_residual, d_qjl_partials, count, options.qjl_dim, blocks_per_sketch,
            options.seed + 17U);
        check_cuda(cudaGetLastError(), "launch qjl dot partial kernel");
        qjl_pack_signs_kernel<<<options.qjl_dim, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_qjl_partials, d_qjl_signs, options.qjl_dim, blocks_per_sketch);
        check_cuda(cudaGetLastError(), "launch qjl pack signs kernel");

        const float coeff =
            options.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
            static_cast<float>(options.qjl_dim);
        qjl_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
            d_reconstructed, d_qjl_signs, count, options.qjl_dim, coeff, options.seed + 17U);
        check_cuda(cudaGetLastError(), "launch qjl reconstruction kernel");

        block.qjl_signs.assign((static_cast<std::size_t>(options.qjl_dim) + 7) / 8, 0);
        if (copy_payload_to_host) {
            std::vector<int> h_signs(options.qjl_dim, 0);
            check_cuda(cudaMemcpyAsync(
                           h_signs.data(), d_qjl_signs,
                           static_cast<std::size_t>(options.qjl_dim) * sizeof(int),
                           cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync qjl signs");
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize qjl signs");
            for (int s = 0; s < options.qjl_dim; ++s) {
                pack_sign_bit(block.qjl_signs, s, h_signs[s] != 0);
            }
        }
    } else if (options.mode == QuantizeMode::kTurboQuantQjl) {
        block.residual_norm = 0.0f;
        block.qjl_signs.assign((static_cast<std::size_t>(options.qjl_dim) + 7) / 8, 0);
    }

    if (copy_payload_to_host) {
        check_cuda(cudaMemcpyAsync(
                       block.codes.data(), d_codes, code_bytes,
                       cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync quantized codes");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize reconstructed quantized block");

    if (d_qjl_signs) cudaFree(d_qjl_signs);
    if (d_qjl_partials) cudaFree(d_qjl_partials);
    if (d_residual) cudaFree(d_residual);
    cudaFree(d_codes);
    cudaFree(d_work);
    return block;
}

std::size_t device_code_bytes(int rows, int cols, const QuantizeOptions& options) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (options.mode == QuantizeMode::kNone) return count * sizeof(float);
    qmax_for_bits(options.bits);
    const std::size_t work_count =
        (options.mode == QuantizeMode::kTurboQuant ||
         options.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(next_power_of_two(static_cast<int>(count))) :
        count;
    if (options.bits == 8) return work_count;
    if (options.bits == 4) return (work_count + 1) / 2;
    if (options.bits == 2) return (work_count + 3) / 4;
    return 0;
}

void initialize_column_tq_signs(
    int rows,
    int cols,
    unsigned seed,
    signed char* d_signs,
    cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Column sign dimensions must be positive.");
    }
    if (!d_signs) {
        throw std::runtime_error("Column sign output pointer is null.");
    }
    const int padded_rows = next_power_of_two(rows);
    const std::size_t total = static_cast<std::size_t>(padded_rows) * cols;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    initialize_column_signs_kernel<<<blocks, threads, 0, stream>>>(
        d_signs, padded_rows, cols, seed);
    check_cuda(cudaGetLastError(), "launch initialize column signs kernel");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize initialize column signs");
}

DeviceCompressedBlock quantize_fp32_device_block_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    int* d_qjl_signs,
    cudaStream_t stream) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (!d_codes) {
        throw std::runtime_error("Device code output pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    if (options.mode == QuantizeMode::kNone) {
        throw std::runtime_error("Device payload path does not handle mode=none.");
    }
    if (options.mode == QuantizeMode::kTurboQuantQjl && !d_qjl_signs) {
        throw std::runtime_error("tq-qjl device payload requires qjl sign storage.");
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    qmax_for_bits(options.bits);
    const std::size_t work_count =
        (options.mode == QuantizeMode::kTurboQuant ||
         options.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(next_power_of_two(static_cast<int>(count))) :
        count;

    DeviceCompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_dim : 0;
    block.qjl_alpha = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_alpha : 0.0f;
    block.seed = options.seed;
    block.padded_count = static_cast<int>(work_count);
    block.d_codes = d_codes;
    block.d_qjl_signs = d_qjl_signs;

    float* d_work = nullptr;
    float* d_reconstructed = nullptr;
    float* d_residual = nullptr;
    float* d_qjl_partials = nullptr;
    const std::size_t code_bytes = block.code_bytes();

    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc quantization work");

    const int threads = 256;
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        copy_pad_random_sign_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_work, count, work_count, options.seed);
        check_cuda(cudaGetLastError(), "launch copy/pad/random-sign kernel");
        fwht_normalized_device(d_work, work_count, stream);
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_work, d_values, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit work block");
    }

    thrust::device_ptr<const float> begin(d_work);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + work_count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize max_abs");

    const int qmax = qmax_for_bits(options.bits);
    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    if (options.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (work_count + 1) / 2;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (work_count + 3) / 4;
        const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch quantization kernel");

    if (options.mode == QuantizeMode::kTurboQuantQjl &&
        options.qjl_dim > 0 &&
        options.qjl_alpha != 0.0f) {
        if (options.bits == 8) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 4) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 2) {
            const int blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        }
        check_cuda(cudaGetLastError(), "launch qjl source dequantization kernel");

        if (options.mode == QuantizeMode::kTurboQuantQjl) {
            fwht_normalized_device(d_work, work_count, stream);
            check_cuda(cudaMalloc(&d_reconstructed, count * sizeof(float)), "cudaMalloc qjl reconstructed scratch");
            const int blocks = static_cast<int>((count + threads - 1) / threads);
            apply_random_sign_truncate_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_reconstructed, count, options.seed);
            check_cuda(cudaGetLastError(), "launch inverse random-sign/truncate kernel");

            check_cuda(cudaMalloc(&d_residual, count * sizeof(float)), "cudaMalloc qjl residual");
            residual_kernel<<<blocks, threads, 0, stream>>>(
                d_values, d_reconstructed, d_residual, count);
            check_cuda(cudaGetLastError(), "launch qjl residual kernel");

            thrust::device_ptr<const float> residual_begin(d_residual);
            float residual_norm2 = thrust::transform_reduce(
                thrust::cuda::par.on(stream),
                residual_begin, residual_begin + count,
                SquareValue{},
                0.0f,
                thrust::plus<float>{});
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize qjl residual norm");
            block.residual_norm = std::sqrt(std::max(residual_norm2, 0.0f));

            const int qjl_threads = 256;
            const int blocks_per_sketch = 256;
            const std::size_t partial_count =
                static_cast<std::size_t>(options.qjl_dim) * blocks_per_sketch;
            check_cuda(cudaMalloc(&d_qjl_partials, partial_count * sizeof(float)),
                       "cudaMalloc qjl partials");
            dim3 partial_grid(options.qjl_dim, blocks_per_sketch);
            qjl_dot_partial_kernel<<<partial_grid, qjl_threads, qjl_threads * sizeof(float), stream>>>(
                d_residual, d_qjl_partials, count, options.qjl_dim, blocks_per_sketch,
                options.seed + 17U);
            check_cuda(cudaGetLastError(), "launch qjl dot partial kernel");
            qjl_pack_signs_kernel<<<options.qjl_dim, qjl_threads, qjl_threads * sizeof(float), stream>>>(
                d_qjl_partials, d_qjl_signs, options.qjl_dim, blocks_per_sketch);
            check_cuda(cudaGetLastError(), "launch qjl pack signs kernel");
        }
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize device payload quantization");
    if (d_qjl_partials) cudaFree(d_qjl_partials);
    if (d_residual) cudaFree(d_residual);
    if (d_reconstructed) cudaFree(d_reconstructed);
    cudaFree(d_work);
    (void)code_bytes;
    return block;
}

DeviceCompressedBlock quantize_fp32_device_column_tq_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_external_work,
    const signed char* d_signs,
    int* d_qjl_signs,
    cudaStream_t stream) {
    if (!d_values) {
        throw std::runtime_error("Device input pointer is null.");
    }
    if (!d_codes) {
        throw std::runtime_error("Device code output pointer is null.");
    }
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    if (options.mode != QuantizeMode::kTurboQuant &&
        options.mode != QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Column TQ payload path requires mode=tq or mode=tq-qjl.");
    }
    qmax_for_bits(options.bits);

    const int padded_rows = next_power_of_two(rows);
    const std::size_t work_count = static_cast<std::size_t>(padded_rows) * cols;

    DeviceCompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_dim : 0;
    block.qjl_alpha = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_alpha : 0.0f;
    block.seed = options.seed;
    block.padded_count = static_cast<int>(work_count);
    block.d_codes = d_codes;
    block.d_qjl_signs = d_qjl_signs;

    float* d_work = d_external_work;
    if (!d_work) {
        check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc column TQ work");
    }

    const int threads = 256;
    int blocks = static_cast<int>((work_count + threads - 1) / threads);
    if (!d_signs && padded_rows <= 1024) {
        column_tq_forward_fwht_kernel<<<cols, padded_rows, padded_rows * sizeof(float), stream>>>(
            d_values, d_work, rows, cols, padded_rows, options.seed);
        check_cuda(cudaGetLastError(), "launch shared column TQ forward FWHT kernel");
    } else if (d_signs) {
        copy_pad_apply_sign_columns_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_signs, d_work, rows, cols, padded_rows);
        check_cuda(cudaGetLastError(), "launch column copy/pad/sign-mask kernel");
        fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
    } else {
        copy_pad_random_sign_columns_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_work, rows, cols, padded_rows, options.seed);
        check_cuda(cudaGetLastError(), "launch column copy/pad/random-sign kernel");
        fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
    }

    thrust::device_ptr<const float> begin(d_work);
    float max_abs = thrust::transform_reduce(
        thrust::cuda::par.on(stream),
        begin, begin + work_count,
        AbsValue{},
        0.0f,
        thrust::maximum<float>{});
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column max_abs");

    const int qmax = qmax_for_bits(options.bits);
    block.scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? max_abs / static_cast<float>(qmax) : 1.0f;

    if (options.bits == 8) {
        blocks = static_cast<int>((work_count + threads - 1) / threads);
        quantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 4) {
        const std::size_t packed_count = (work_count + 1) / 2;
        blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    } else if (options.bits == 2) {
        const std::size_t packed_count = (work_count + 3) / 4;
        blocks = static_cast<int>((packed_count + threads - 1) / threads);
        quantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_work, d_codes, work_count, block.scale);
    }
    check_cuda(cudaGetLastError(), "launch column TQ quantization kernel");

    float* d_reconstructed = nullptr;
    float* d_residual = nullptr;
    float* d_qjl_partials = nullptr;
    if (options.mode == QuantizeMode::kTurboQuantQjl &&
        options.qjl_dim > 0 &&
        options.qjl_alpha != 0.0f) {
        if (!block.d_qjl_signs) {
            throw std::runtime_error("Column tq-qjl payload requires qjl sign storage.");
        }
        if (options.bits == 8) {
            blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 4) {
            blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        } else if (options.bits == 2) {
            blocks = static_cast<int>((work_count + threads - 1) / threads);
            dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_work, work_count, block.scale);
        }
        check_cuda(cudaGetLastError(), "launch column tq-qjl source dequantization kernel");

        const std::size_t count = static_cast<std::size_t>(rows) * cols;
        check_cuda(cudaMalloc(&d_reconstructed, count * sizeof(float)),
                   "cudaMalloc column tq-qjl reconstructed scratch");
        if (!d_signs && padded_rows <= 1024) {
            column_tq_inverse_fwht_store_kernel<<<cols, padded_rows, padded_rows * sizeof(float), stream>>>(
                d_work, d_reconstructed, rows, cols, padded_rows, options.seed);
            check_cuda(cudaGetLastError(), "launch shared column tq-qjl inverse FWHT store kernel");
        } else if (d_signs) {
            check_cuda(cudaMemsetAsync(d_reconstructed, 0, count * sizeof(float), stream),
                       "cudaMemsetAsync column tq-qjl reconstructed");
            fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
            blocks = static_cast<int>((count + threads - 1) / threads);
            apply_sign_mask_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_signs, d_reconstructed, rows, cols, padded_rows);
            check_cuda(cudaGetLastError(), "launch column tq-qjl inverse sign-mask/truncate kernel");
        } else {
            check_cuda(cudaMemsetAsync(d_reconstructed, 0, count * sizeof(float), stream),
                       "cudaMemsetAsync column tq-qjl reconstructed");
            fwht_columns_normalized_device(d_work, padded_rows, cols, stream);
            blocks = static_cast<int>((count + threads - 1) / threads);
            apply_random_sign_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
                d_work, d_reconstructed, rows, cols, padded_rows, options.seed);
            check_cuda(cudaGetLastError(), "launch column tq-qjl inverse random-sign/truncate kernel");
        }

        check_cuda(cudaMalloc(&d_residual, count * sizeof(float)), "cudaMalloc column tq-qjl residual");
        blocks = static_cast<int>((count + threads - 1) / threads);
        residual_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_reconstructed, d_residual, count);
        check_cuda(cudaGetLastError(), "launch column tq-qjl residual kernel");

        thrust::device_ptr<const float> residual_begin(d_residual);
        float residual_norm2 = thrust::transform_reduce(
            thrust::cuda::par.on(stream),
            residual_begin, residual_begin + count,
            SquareValue{},
            0.0f,
            thrust::plus<float>{});
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column tq-qjl residual norm");
        block.residual_norm = std::sqrt(std::max(residual_norm2, 0.0f));

        const int qjl_threads = 256;
        const int blocks_per_sketch = 128;
        const std::size_t partial_count =
            static_cast<std::size_t>(options.qjl_dim) * blocks_per_sketch;
        check_cuda(cudaMalloc(&d_qjl_partials, partial_count * sizeof(float)),
                   "cudaMalloc column tq-qjl partials");
        dim3 partial_grid(options.qjl_dim, blocks_per_sketch);
        qjl_dot_partial_kernel<<<partial_grid, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_residual, d_qjl_partials, count, options.qjl_dim, blocks_per_sketch,
            options.seed + 17U);
        check_cuda(cudaGetLastError(), "launch column tq-qjl dot partial kernel");
        qjl_pack_signs_kernel<<<options.qjl_dim, qjl_threads, qjl_threads * sizeof(float), stream>>>(
            d_qjl_partials, block.d_qjl_signs, options.qjl_dim, blocks_per_sketch);
        check_cuda(cudaGetLastError(), "launch column tq-qjl pack signs kernel");
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column TQ quantization");

    if (d_qjl_partials) cudaFree(d_qjl_partials);
    if (d_residual) cudaFree(d_residual);
    if (d_reconstructed) cudaFree(d_reconstructed);

    if (!d_external_work) cudaFree(d_work);
    return block;
}

void dequantize_device_payload_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_reconstructed,
    cudaStream_t stream) {
    if (!block.d_codes) {
        throw std::runtime_error("Device compressed block has null codes.");
    }
    if (!d_reconstructed) {
        throw std::runtime_error("Device reconstructed output pointer is null.");
    }
    if (block.mode == QuantizeMode::kNone) {
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed,
                       reinterpret_cast<const float*>(block.d_codes),
                       block.value_count() * sizeof(float),
                       cudaMemcpyDeviceToDevice,
                       stream),
                   "cudaMemcpyAsync none payload to reconstructed");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize none payload");
        return;
    }

    const std::size_t count = block.value_count();
    const std::size_t work_count = static_cast<std::size_t>(block.padded_count);
    float* d_work = nullptr;
    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc payload decode work");

    const int threads = 256;
    if (block.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 4) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 2) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else {
        cudaFree(d_work);
        throw std::runtime_error("Unsupported device payload bit width.");
    }
    check_cuda(cudaGetLastError(), "launch payload dequantization kernel");

    if (block.mode == QuantizeMode::kTurboQuant ||
        block.mode == QuantizeMode::kTurboQuantQjl) {
        fwht_normalized_device(d_work, work_count, stream);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        apply_random_sign_truncate_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_reconstructed, count, block.seed);
        check_cuda(cudaGetLastError(), "launch payload inverse random-sign/truncate kernel");
    } else {
        check_cuda(cudaMemcpyAsync(
                       d_reconstructed, d_work, count * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream),
                   "cudaMemcpyAsync lowbit payload reconstructed block");
    }

    if (block.mode == QuantizeMode::kTurboQuantQjl &&
        block.qjl_dim > 0 &&
        block.qjl_alpha != 0.0f &&
        block.residual_norm > 0.0f) {
        if (!block.d_qjl_signs) {
            cudaFree(d_work);
            throw std::runtime_error("tq-qjl payload decode requires qjl signs.");
        }
        const float coeff =
            block.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
            static_cast<float>(block.qjl_dim);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        qjl_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
            d_reconstructed, block.d_qjl_signs, count, block.qjl_dim, coeff, block.seed + 17U);
        check_cuda(cudaGetLastError(), "launch payload qjl reconstruction kernel");
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize payload decode");
    cudaFree(d_work);
}

void dequantize_device_payload_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    cudaStream_t stream) {
    if (!block.d_codes) {
        throw std::runtime_error("Device compressed block has null codes.");
    }
    if (!d_accumulator || !d_work) {
        throw std::runtime_error("Device payload add path has null output/work pointer.");
    }
    if (block.mode == QuantizeMode::kNone) {
        const std::size_t count = block.value_count();
        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        add_plain_kernel<<<blocks, threads, 0, stream>>>(
            d_accumulator, reinterpret_cast<const float*>(block.d_codes), count);
        check_cuda(cudaGetLastError(), "launch none payload add kernel");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize none payload add");
        return;
    }
    if (block.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Fused payload add currently supports tq/lowbit, not tq-qjl.");
    }

    const std::size_t count = block.value_count();
    const std::size_t work_count = static_cast<std::size_t>(block.padded_count);
    const int threads = 256;
    if (block.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 4) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 2) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else {
        throw std::runtime_error("Unsupported device payload bit width.");
    }
    check_cuda(cudaGetLastError(), "launch payload add dequantization kernel");

    if (block.mode == QuantizeMode::kTurboQuant) {
        fwht_normalized_device(d_work, work_count, stream);
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        apply_random_sign_truncate_add_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_accumulator, count, block.seed);
        check_cuda(cudaGetLastError(), "launch payload inverse random-sign/truncate add kernel");
    } else {
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        add_plain_kernel<<<blocks, threads, 0, stream>>>(d_accumulator, d_work, count);
        check_cuda(cudaGetLastError(), "launch lowbit payload add kernel");
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize payload add decode");
}

void dequantize_column_tq_payload_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    const signed char* d_signs,
    cudaStream_t stream) {
    if (!block.d_codes) {
        throw std::runtime_error("Device compressed block has null codes.");
    }
    if (!d_accumulator || !d_work) {
        throw std::runtime_error("Column TQ payload add path has null output/work pointer.");
    }
    if (block.mode != QuantizeMode::kTurboQuant &&
        block.mode != QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error("Column TQ payload add path requires mode=tq or mode=tq-qjl.");
    }
    const int padded_rows = static_cast<int>(static_cast<std::size_t>(block.padded_count) / block.cols);
    const std::size_t work_count = static_cast<std::size_t>(block.padded_count);
    const int threads = 256;

    if (block.bits == 8) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int8_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 4) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int4_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else if (block.bits == 2) {
        const int blocks = static_cast<int>((work_count + threads - 1) / threads);
        dequantize_int2_pack_kernel<<<blocks, threads, 0, stream>>>(block.d_codes, d_work, work_count, block.scale);
    } else {
        throw std::runtime_error("Unsupported column TQ payload bit width.");
    }
    check_cuda(cudaGetLastError(), "launch column TQ payload dequantization kernel");

    if (!d_signs && padded_rows <= 1024) {
        column_tq_inverse_fwht_add_kernel<<<block.cols, padded_rows, padded_rows * sizeof(float), stream>>>(
            d_work, d_accumulator, block.rows, block.cols, padded_rows, block.seed);
        check_cuda(cudaGetLastError(), "launch shared column TQ inverse FWHT add kernel");
    } else if (d_signs) {
        fwht_columns_normalized_device(d_work, padded_rows, block.cols, stream);
        const int blocks = static_cast<int>((block.value_count() + threads - 1) / threads);
        apply_sign_mask_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_signs, d_accumulator, block.rows, block.cols, padded_rows);
        check_cuda(cudaGetLastError(), "launch column TQ inverse sign-mask/truncate add kernel");
    } else {
        fwht_columns_normalized_device(d_work, padded_rows, block.cols, stream);
        const int blocks = static_cast<int>((block.value_count() + threads - 1) / threads);
        apply_random_sign_truncate_columns_add_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_accumulator, block.rows, block.cols, padded_rows, block.seed);
        check_cuda(cudaGetLastError(), "launch column TQ inverse random-sign/truncate add kernel");
    }
    if (block.mode == QuantizeMode::kTurboQuantQjl &&
        block.qjl_dim > 0 &&
        block.qjl_alpha != 0.0f &&
        block.residual_norm > 0.0f) {
        if (!block.d_qjl_signs) {
            throw std::runtime_error("Column tq-qjl payload decode requires qjl signs.");
        }
        const float coeff =
            block.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
            static_cast<float>(block.qjl_dim);
        const int blocks = static_cast<int>((block.value_count() + threads - 1) / threads);
        qjl_reconstruct_kernel<<<blocks, threads, 0, stream>>>(
            d_accumulator, block.d_qjl_signs, block.value_count(), block.qjl_dim, coeff, block.seed + 17U);
        check_cuda(cudaGetLastError(), "launch column tq-qjl reconstruction add kernel");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column TQ payload add decode");
}

CompressedBlock quantize_dequant_fp32_device_block(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::vector<float>* reconstructed,
    cudaStream_t stream) {
    if (!reconstructed) {
        throw std::runtime_error("Reconstructed output pointer is null.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    reconstructed->assign(count, 0.0f);

    float* d_reconstructed = nullptr;
    check_cuda(cudaMalloc(&d_reconstructed, count * sizeof(float)), "cudaMalloc reconstructed block");
    CompressedBlock block = quantize_dequant_fp32_device_block_to_device(
        d_values, rows, cols, options, d_reconstructed, stream);
    check_cuda(cudaMemcpyAsync(
                   reconstructed->data(), d_reconstructed, count * sizeof(float),
                   cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync reconstructed block");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize reconstructed block");
    cudaFree(d_reconstructed);
    return block;
}

std::vector<float> dequantize_fp32_block(const CompressedBlock& block) {
    if (block.rows <= 0 || block.cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    const std::size_t count = block.value_count();
    std::vector<float> values(count);

    if (block.mode == QuantizeMode::kNone) {
        if (block.codes.size() != count * sizeof(float)) {
            throw std::runtime_error("Invalid uncompressed payload size.");
        }
        std::memcpy(values.data(), block.codes.data(), block.codes.size());
        return values;
    }

    qmax_for_bits(block.bits);
    const std::size_t decode_count =
        (block.mode == QuantizeMode::kTurboQuant ||
         block.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(block.padded_count) : count;
    std::vector<float> decoded(decode_count, 0.0f);
    for (std::size_t i = 0; i < decode_count; ++i) {
        std::uint8_t code = unpack_code_at(block.codes, i, block.bits);
        int q = decode_unsigned_to_signed(code, block.bits);
        decoded[i] = static_cast<float>(q) * block.scale;
    }

    if (block.mode == QuantizeMode::kTurboQuant ||
        block.mode == QuantizeMode::kTurboQuantQjl) {
        fwht_normalized(decoded);
        apply_random_sign(decoded, block.seed);
        std::copy(decoded.begin(), decoded.begin() + count, values.begin());

        if (block.qjl_dim > 0 && !block.qjl_signs.empty() && block.residual_norm > 0.0f) {
            const float coeff =
                block.qjl_alpha * block.residual_norm * std::sqrt(3.14159265358979323846f / 2.0f) /
                static_cast<float>(block.qjl_dim);
            for (std::size_t j = 0; j < count; ++j) {
                float accum = 0.0f;
                for (int s = 0; s < block.qjl_dim; ++s) {
                    float sign = unpack_sign_bit(block.qjl_signs, s) ? 1.0f : -1.0f;
                    accum += sign * rademacher(block.seed + 17U, static_cast<std::uint32_t>(s),
                                               static_cast<std::uint32_t>(j));
                }
                values[j] += coeff * accum;
            }
        }
    } else {
        std::copy(decoded.begin(), decoded.end(), values.begin());
    }
    return values;
}

std::string mode_name(QuantizeMode mode) {
    switch (mode) {
        case QuantizeMode::kNone: return "none";
        case QuantizeMode::kLowBit: return "lowbit";
        case QuantizeMode::kTurboQuant: return "tq";
        case QuantizeMode::kTurboQuantQjl: return "tq-qjl";
        default: return "unknown";
    }
}

}  // namespace turboquant
