#include "turboquant.hpp"
#include "tq_codebooks_generated.hpp"

#include <cublas_v2.h>
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
#include <vector>

namespace turboquant {
namespace {

void check_cuda(cudaError_t status, const char* label);
void check_cublas(cublasStatus_t status, const char* label);

// (README) 量化範圍設定: 根據 bit 數決定對稱 signed quantization 可使用的最大整數值。
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

// (README) 隨機符號矩陣 D: 用 hash 產生 deterministic 的 Rademacher 正負號來模擬隨機對角矩陣。
__host__ __device__ float rademacher(unsigned seed, std::uint32_t a, std::uint32_t b) {
    std::uint32_t h = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU));
    return (h & 1U) ? 1.0f : -1.0f;
}

__host__ __device__ float uniform01_from_hash(std::uint32_t h) {
    return (static_cast<float>(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

// (README) QJL 隨機投影係數: 用 hash 產生 deterministic Gaussian 樣本供 QJL residual sketch 使用。
__host__ __device__ float gaussian_from_hash(unsigned seed, std::uint32_t a, std::uint32_t b) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    std::uint32_t h0 = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU) ^ 0x243f6a88U);
    std::uint32_t h1 = mix32(seed ^ (a * 0x85ebca6bU) ^ (b * 0xc2b2ae35U) ^ 0x9e3779b9U);
    float u1 = fmaxf(uniform01_from_hash(h0), 1.0e-7f);
    float u2 = uniform01_from_hash(h1);
    return sqrtf(-2.0f * logf(u1)) * cosf(kTwoPi * u2);
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

// (README) Low-bit payload 打包: 將 4-bit 或 2-bit 量化碼壓進 byte array 以減少通訊 payload。
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

// (README) Lloyd-Max codebooks: precomputed scalar quantizers for RHT coordinates of unit vectors.


__host__ __device__ bool is_supported_lloyd_tq_bits(int bits) {
    return bits >= 2 && bits <= 8;
}

__host__ __device__ bool is_supported_lloyd_tq_dim(int dim) {
    return dim == 256 || dim == 512 || dim == 1024 || dim == 2048;
}

int tq_mse_bits_for_options(const QuantizeOptions& options) {
    if (options.mode == QuantizeMode::kTurboQuant) return options.bits;
    if (options.mode == QuantizeMode::kTurboQuantQjl) return options.bits - 1;
    return options.bits;
}

int effective_qjl_dim(int rows, const QuantizeOptions& options) {
    if (options.mode != QuantizeMode::kTurboQuantQjl) return 0;
    return (options.qjl_dim > 0) ? options.qjl_dim : rows;
}

std::size_t bitpacked_code_bytes(int rows, int cols, int bits) {
    const std::size_t bit_count =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
        static_cast<std::size_t>(bits);
    return ((bit_count + 31U) / 32U) * sizeof(unsigned int);
}

std::size_t lloyd_tq_code_bytes(int rows, int cols, int bits) {
    if (!is_supported_lloyd_tq_dim(rows)) {
        throw std::runtime_error("Lloyd-Max TQ codebook path requires vector dimension d in {256, 512, 1024, 2048}.");
    }
    if (!is_supported_lloyd_tq_bits(bits)) {
        throw std::runtime_error("Lloyd-Max TQ codebook path requires MSE bits in [2, 8].");
    }
    return bitpacked_code_bytes(rows, cols, bits);
}

std::size_t qjl_sign_bytes(int rows, int cols, int qjl_dim) {
    if (!is_supported_lloyd_tq_dim(rows)) {
        throw std::runtime_error("QJL sign path requires vector dimension d in {256, 512, 1024, 2048}.");
    }
    if (qjl_dim <= 0) {
        throw std::runtime_error("QJL sign path requires positive qjl_dim.");
    }
    return bitpacked_code_bytes(qjl_dim, cols, 1);
}

void validate_column_tq_options(int rows, int cols, const QuantizeOptions& options, const char* context) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error(std::string(context) + " requires positive dimensions.");
    }
    if (!is_supported_lloyd_tq_dim(rows)) {
        throw std::runtime_error(
            std::string(context) + " requires vector dimension d in {256, 512, 1024, 2048}; got d=" +
            std::to_string(rows) + ".");
    }
    if (options.mode == QuantizeMode::kTurboQuant) {
        if (!is_supported_lloyd_tq_bits(options.bits)) {
            throw std::runtime_error(std::string(context) + " mode=tq requires bits in [2, 8].");
        }
        return;
    }
    if (options.mode == QuantizeMode::kTurboQuantQjl) {
        if (options.bits < 3 || options.bits > 8) {
            throw std::runtime_error(std::string(context) + " mode=tq-qjl requires total bits in [3, 8].");
        }
        if (options.qjl_dim < 0) {
            throw std::runtime_error(std::string(context) + " mode=tq-qjl requires qjl_dim >= 0.");
        }
        if (!is_supported_lloyd_tq_bits(options.bits - 1)) {
            throw std::runtime_error(std::string(context) + " mode=tq-qjl requires MSE bits in [2, 7].");
        }
        return;
    }
    throw std::runtime_error(std::string(context) + " requires mode=tq or mode=tq-qjl.");
}

__device__ int lloyd_tq_bucket(float value, int dim, int bits) {
    const int levels = 1 << bits;
    const TQDeviceCodebook codebook = get_tq_codebook_device(dim, bits);
    const float* boundaries = codebook.boundaries;
    if (!boundaries || codebook.levels != levels || isnan(value)) return 0;
    if (value >= boundaries[levels]) return levels - 1;
    for (int i = 0; i < levels; ++i) {
        if (value < boundaries[i + 1]) return i;
    }
    return levels - 1;
}

__device__ void bitpack_write_code(std::uint8_t* packed, std::size_t idx, int bits, int code) {
    unsigned int* words = reinterpret_cast<unsigned int*>(packed);
    const std::size_t bit_offset = idx * static_cast<std::size_t>(bits);
    const std::size_t word_idx = bit_offset >> 5;
    const int shift = static_cast<int>(bit_offset & 31U);
    const unsigned int mask = (bits == 32) ? 0xffffffffU : ((1U << bits) - 1U);
    const unsigned int value = static_cast<unsigned int>(code) & mask;
    atomicOr(words + word_idx, value << shift);
    if (shift + bits > 32) {
        atomicOr(words + word_idx + 1, value >> (32 - shift));
    }
}

__device__ int bitpack_read_code(const std::uint8_t* packed, std::size_t idx, int bits) {
    const unsigned int* words = reinterpret_cast<const unsigned int*>(packed);
    const std::size_t bit_offset = idx * static_cast<std::size_t>(bits);
    const std::size_t word_idx = bit_offset >> 5;
    const int shift = static_cast<int>(bit_offset & 31U);
    const unsigned int mask = (bits == 32) ? 0xffffffffU : ((1U << bits) - 1U);
    unsigned int value = words[word_idx] >> shift;
    if (shift + bits > 32) {
        value |= words[word_idx + 1] << (32 - shift);
    }
    return static_cast<int>(value & mask);
}

struct QjlMatrixCacheEntry {
    int device = -1;
    int rows = 0;
    int qjl_dim = 0;
    unsigned seed = 0;
    float* d_S = nullptr;
};

struct QjlScratchCacheEntry {
    int device = -1;
    std::size_t projection_count = 0;
    std::size_t signs_float_count = 0;
    std::size_t residual_hat_count = 0;
    std::size_t reconstructed_count = 0;
    std::size_t residual_count = 0;
    float* d_projection = nullptr;
    float* d_signs_float = nullptr;
    float* d_residual_hat = nullptr;
    float* d_reconstructed = nullptr;
    float* d_residual = nullptr;
};

struct CublasHandleCacheEntry {
    int device = -1;
    cublasHandle_t handle = nullptr;
};

std::vector<QjlMatrixCacheEntry>& qjl_matrix_cache() {
    static std::vector<QjlMatrixCacheEntry> cache;
    return cache;
}

std::vector<QjlScratchCacheEntry>& qjl_scratch_cache() {
    static std::vector<QjlScratchCacheEntry> cache;
    return cache;
}

std::vector<CublasHandleCacheEntry>& cublas_handle_cache() {
    static std::vector<CublasHandleCacheEntry> cache;
    return cache;
}

float* reserve_float_buffer(float*& ptr, std::size_t& capacity, std::size_t required, const char* label) {
    if (required == 0) return ptr;
    if (capacity >= required && ptr) return ptr;
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
        capacity = 0;
    }
    check_cuda(cudaMalloc(&ptr, required * sizeof(float)), label);
    capacity = required;
    return ptr;
}

float* get_qjl_matrix_device(int rows, int qjl_dim, unsigned seed) {
    if (!is_supported_lloyd_tq_dim(rows) || qjl_dim <= 0) {
        throw std::runtime_error("QJL matrix requires supported dim and positive qjl_dim.");
    }
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice QJL matrix cache");
    auto& cache = qjl_matrix_cache();
    for (auto& entry : cache) {
        if (entry.device == device && entry.rows == rows &&
            entry.qjl_dim == qjl_dim && entry.seed == seed) {
            return entry.d_S;
        }
    }

    std::vector<float> h_S(static_cast<std::size_t>(qjl_dim) * rows);
    for (int row = 0; row < rows; ++row) {
        for (int s = 0; s < qjl_dim; ++s) {
            h_S[static_cast<std::size_t>(row) * qjl_dim + s] =
                gaussian_from_hash(seed, static_cast<std::uint32_t>(s),
                                   static_cast<std::uint32_t>(row));
        }
    }

    QjlMatrixCacheEntry entry;
    entry.device = device;
    entry.rows = rows;
    entry.qjl_dim = qjl_dim;
    entry.seed = seed;
    check_cuda(cudaMalloc(&entry.d_S, h_S.size() * sizeof(float)), "cudaMalloc QJL matrix S");
    check_cuda(cudaMemcpy(entry.d_S, h_S.data(), h_S.size() * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy QJL matrix S");
    cache.push_back(entry);
    return cache.back().d_S;
}

QjlScratchCacheEntry& get_qjl_scratch(
    int rows,
    int cols,
    int qjl_dim,
    bool need_reconstructed,
    bool need_residual) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice QJL scratch cache");
    auto& cache = qjl_scratch_cache();
    for (auto& entry : cache) {
        if (entry.device == device) {
            reserve_float_buffer(
                entry.d_projection, entry.projection_count,
                static_cast<std::size_t>(qjl_dim) * cols,
                "cudaMalloc QJL projection scratch");
            reserve_float_buffer(
                entry.d_signs_float, entry.signs_float_count,
                static_cast<std::size_t>(qjl_dim) * cols,
                "cudaMalloc QJL sign float scratch");
            reserve_float_buffer(
                entry.d_residual_hat, entry.residual_hat_count,
                static_cast<std::size_t>(rows) * cols,
                "cudaMalloc QJL residual reconstruction scratch");
            if (need_reconstructed) {
                reserve_float_buffer(
                    entry.d_reconstructed, entry.reconstructed_count,
                    static_cast<std::size_t>(rows) * cols,
                    "cudaMalloc QJL MSE reconstruction scratch");
            }
            if (need_residual) {
                reserve_float_buffer(
                    entry.d_residual, entry.residual_count,
                    static_cast<std::size_t>(rows) * cols,
                    "cudaMalloc QJL residual scratch");
            }
            return entry;
        }
    }

    QjlScratchCacheEntry entry;
    entry.device = device;
    cache.push_back(entry);
    return get_qjl_scratch(rows, cols, qjl_dim, need_reconstructed, need_residual);
}

cublasHandle_t get_cached_cublas_handle(cudaStream_t stream) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice QJL cuBLAS handle cache");
    auto& cache = cublas_handle_cache();
    for (auto& entry : cache) {
        if (entry.device == device) {
            check_cublas(cublasSetStream(entry.handle, stream), "cublasSetStream QJL");
            return entry.handle;
        }
    }

    CublasHandleCacheEntry entry;
    entry.device = device;
    check_cublas(cublasCreate(&entry.handle), "cublasCreate QJL");
    check_cublas(cublasSetStream(entry.handle, stream), "cublasSetStream QJL");
    cache.push_back(entry);
    return cache.back().handle;
}



// (README) 8-bit 均勻量化: 將 FP32 數值依照 shared scale 四捨五入到 int8 範圍。
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

// (README) 4-bit 均勻量化與打包: 將 FP32 數值量化到 signed 4-bit 並把兩個 code 打包成一個 byte。
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

// (README) 2-bit 均勻量化與打包: 將 FP32 數值量化到 signed 2-bit 並把四個 code 打包成一個 byte。
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

// (README) Column-wise FWHT 單階段 butterfly: 對多個 column 各自執行 Hadamard transform 的一層 butterfly。
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

// (README) Column-wise norm 計算: 對每個 column vector 計算原始 L2 norm，供 TQ 解碼時乘回尺度。
__global__ void column_norms_kernel(
    const float* values,
    float* norms,
    int rows,
    int cols) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (int row = tid; row < rows; row += blockDim.x) {
        const float v = values[static_cast<std::size_t>(col) * rows + row];
        sum += v * v;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) norms[col] = sqrtf(fmaxf(smem[0], 0.0f));
}

// (README) Column-wise TQ 正規化與 D 符號: 將每個 column 除以 norm 後套 deterministic Rademacher signs。
__global__ void column_tq_normalize_sign_kernel(
    const float* values,
    const float* norms,
    float* work,
    int rows,
    int cols,
    unsigned seed,
    float eps) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    const float inv_norm = 1.0f / (norms[col] + eps);
    work[idx] = values[idx] * inv_norm *
        rademacher(seed, static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(col));
}

// (README) Column-wise Lloyd-Max bucket 壓縮: 將 RHT 後的 coordinate 查 codebook bucket 並 bit-pack。
__global__ void column_tq_lloyd_quantize_kernel(
    const float* transformed,
    std::uint8_t* codes,
    int rows,
    int cols,
    int bits) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int code = lloyd_tq_bucket(transformed[idx], rows, bits);
    bitpack_write_code(codes, idx, bits, code);
}

// (README) Column-wise Lloyd-Max centroid 解碼: 將 bit-packed bucket index 查回 rotated-domain centroid。
__global__ void column_tq_lloyd_dequantize_kernel(
    const std::uint8_t* codes,
    float* transformed,
    int rows,
    int cols,
    int bits) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int code = bitpack_read_code(codes, idx, bits);
    const TQDeviceCodebook codebook = get_tq_codebook_device(rows, bits);
    transformed[idx] = (codebook.centroids && code >= 0 && code < codebook.levels) ?
        codebook.centroids[code] : 0.0f;
}

// (README) Column-wise TQ 反旋轉與 rescale 累加: inverse RHT 後乘回原 norm 與 D 符號並加到 accumulator。
__global__ void column_tq_apply_sign_norm_add_kernel(
    const float* work,
    const float* norms,
    float* values,
    int rows,
    int cols,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    values[idx] += norms[col] * work[idx] *
        rademacher(seed, static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(col));
}

// (README) Column-wise TQ 反旋轉與 rescale 儲存: inverse RHT 後乘回原 norm 與 D 符號並寫出 MSE reconstruction。
__global__ void column_tq_apply_sign_norm_store_kernel(
    const float* work,
    const float* norms,
    float* values,
    int rows,
    int cols,
    unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int row = static_cast<int>(idx % static_cast<std::size_t>(rows));
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    values[idx] = norms[col] * work[idx] *
        rademacher(seed, static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(col));
}

// (README) Column-wise residual norm 計算: 對每個 TQ-QJL residual vector 計算 L2 norm。
__global__ void column_residual_norms_kernel(
    const float* residual,
    float* residual_norms,
    int rows,
    int cols) {
    extern __shared__ float smem[];
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (int row = tid; row < rows; row += blockDim.x) {
        const float v = residual[static_cast<std::size_t>(col) * rows + row];
        sum += v * v;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    if (tid == 0) residual_norms[col] = sqrtf(fmaxf(smem[0], 0.0f));
}

// (README) Column-wise QJL sign packing: 將 GEMM 得到的 projected = S R 正負號以 1 bit per coordinate 打包。
__global__ void column_qjl_pack_projected_signs_kernel(
    const float* projected,
    std::uint8_t* signs,
    int qjl_dim,
    int cols) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(qjl_dim) * cols;
    if (idx >= total) return;
    bitpack_write_code(signs, idx, 1, projected[idx] >= 0.0f ? 1 : 0);
}

// (README) Column-wise QJL sign unpacking: 將 packed sign bits 展開成 GEMM 可使用的 dense {-1,+1} matrix。
__global__ void column_qjl_unpack_signs_to_float_kernel(
    const std::uint8_t* signs,
    float* signs_float,
    int qjl_dim,
    int cols) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(qjl_dim) * cols;
    if (idx >= total) return;
    signs_float[idx] = bitpack_read_code(signs, idx, 1) ? 1.0f : -1.0f;
}

// (README) Column-wise QJL residual rescale: 將 GEMM 得到的 S^T q 乘上 sqrt(pi/2)/qjl_dim 與 residual norm 後補回 accumulator。
__global__ void column_qjl_scale_residual_add_kernel(
    float* values,
    const float* residual_hat,
    const float* residual_norms,
    int rows,
    int cols,
    int qjl_dim,
    float alpha,
    float base_coeff) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    const int col = static_cast<int>(idx / static_cast<std::size_t>(rows));
    const float coeff = alpha * base_coeff * residual_norms[col] / static_cast<float>(qjl_dim);
    values[idx] += coeff * residual_hat[idx];
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

// (README) GPU Column-wise 正規化 Hadamard 轉換: 對每個 column 分別執行 normalized FWHT。
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

void check_cublas(cublasStatus_t status, const char* label) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(label) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
    }
}

}  // namespace

std::size_t CompressedBlock::value_count() const {
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
}

std::size_t CompressedBlock::payload_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    if (mode == QuantizeMode::kLowBit) return codes.size() + sizeof(scale);
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
    if (mode == QuantizeMode::kTurboQuant || mode == QuantizeMode::kTurboQuantQjl) {
        const int mse_bits = (mode == QuantizeMode::kTurboQuantQjl) ? bits - 1 : bits;
        return lloyd_tq_code_bytes(rows, cols, mse_bits);
    }
    const std::size_t count = static_cast<std::size_t>(padded_count);
    if (bits == 8) return count;
    if (bits == 4) return (count + 1) / 2;
    if (bits == 2) return (count + 3) / 4;
    return 0;
}

std::size_t DeviceCompressedBlock::payload_bytes() const {
    if (mode == QuantizeMode::kNone) return value_count() * sizeof(float);
    if (mode == QuantizeMode::kTurboQuant || mode == QuantizeMode::kTurboQuantQjl) {
        const std::size_t norm_bytes = static_cast<std::size_t>(cols) * sizeof(float);
        const std::size_t residual_norm_bytes =
            (mode == QuantizeMode::kTurboQuantQjl) ? static_cast<std::size_t>(cols) * sizeof(float) : 0;
        const std::size_t sign_bytes =
            (mode == QuantizeMode::kTurboQuantQjl) ? qjl_sign_bytes(rows, cols, qjl_dim) : 0;
        return code_bytes() + norm_bytes + residual_norm_bytes + sign_bytes;
    }
    return code_bytes() + sizeof(scale);
}

// (README) 預設量化參數建立: 只用 bit 數建立 none 或 lowbit 模式的 QuantizeOptions。
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

// (README) CLI 量化參數解析: 將 mode、bit 數、QJL 參數與 seed 轉成 codec 會使用的 QuantizeOptions。
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
        if (!is_supported_lloyd_tq_bits(bits)) {
            throw std::runtime_error("mode=tq requires bits in [2, 8].");
        }
        options.mode = QuantizeMode::kTurboQuant;
        options.qjl_dim = 0;
        options.qjl_alpha = 0.0f;
        return options;
    }
    if (mode == "tq-qjl") {
        if (bits < 3 || bits > 8) {
            throw std::runtime_error("mode=tq-qjl requires total bits in [3, 8]; bits=2 is unsupported.");
        }
        if (qjl_dim < 0) {
            throw std::runtime_error("mode=tq-qjl requires qjl_dim >= 0; use 0 for vector dimension.");
        }
        options.mode = QuantizeMode::kTurboQuantQjl;
        return options;
    }
    throw std::runtime_error("Unsupported quantization mode: " + mode);
}

// (README) CPU 端 lowbit codec: 在 host vector 上執行 legacy lowbit max-abs uniform quantization。
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
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error(
            "mode=tq and mode=tq-qjl use Lloyd-Max column-vector quantization and are supported only by the device column path.");
    }

    block.qjl_dim = options.qjl_dim;
    block.qjl_alpha = options.qjl_alpha;
    block.seed = options.seed;

    std::vector<float> values_for_quant = values;
    block.padded_count = static_cast<int>(count);

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

    return block;
}

// (README) GPU block lowbit 壓縮: 將 device 上的 FP32 block 直接量化成 host-side compressed payload。
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

// (README) GPU block lowbit 壓縮並重建: 對 flattened block 做 legacy lowbit quantize-dequantize。
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
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error(
            "mode=tq and mode=tq-qjl use Lloyd-Max column-vector quantization and are supported only by the device column path.");
    }

    qmax_for_bits(options.bits);
    const std::size_t work_count = count;
    block.padded_count = static_cast<int>(work_count);

    float* d_work = nullptr;
    std::uint8_t* d_codes = nullptr;

    const std::size_t code_bytes = (options.bits == 8) ? work_count :
                                   (options.bits == 4) ? (work_count + 1) / 2 :
                                   (work_count + 3) / 4;
    block.codes.resize(code_bytes);

    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc quantization work");
    check_cuda(cudaMalloc(&d_codes, code_bytes), "cudaMalloc quantized codes");

    const int threads = 256;
    check_cuda(cudaMemcpyAsync(
                   d_work, d_values, count * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync lowbit work block");

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

    check_cuda(cudaMemcpyAsync(
                   d_reconstructed, d_work, count * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync lowbit reconstructed block");

    if (copy_payload_to_host) {
        check_cuda(cudaMemcpyAsync(
                       block.codes.data(), d_codes, code_bytes,
                       cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync quantized codes");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize reconstructed quantized block");

    cudaFree(d_codes);
    cudaFree(d_work);
    return block;
}

// (README) Device payload 大小估算: 根據矩陣大小、padding 與 bit 數計算 compressed code buffer 需要的 bytes。
std::size_t device_code_bytes(int rows, int cols, const QuantizeOptions& options) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Compressed block dimensions must be positive.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (options.mode == QuantizeMode::kNone) return count * sizeof(float);
    if (options.mode == QuantizeMode::kTurboQuant ||
        options.mode == QuantizeMode::kTurboQuantQjl) {
        validate_column_tq_options(rows, cols, options, "device_code_bytes");
        return lloyd_tq_code_bytes(rows, cols, tq_mse_bits_for_options(options));
    }
    qmax_for_bits(options.bits);
    const std::size_t work_count =
        (options.mode == QuantizeMode::kTurboQuantQjl) ?
        static_cast<std::size_t>(next_power_of_two(static_cast<int>(count))) :
        count;
    if (options.bits == 8) return work_count;
    if (options.bits == 4) return (work_count + 1) / 2;
    if (options.bits == 2) return (work_count + 3) / 4;
    return 0;
}

std::size_t device_norm_bytes(int rows, int cols, const QuantizeOptions& options) {
    if (options.mode == QuantizeMode::kNone || options.mode == QuantizeMode::kLowBit) return 0;
    validate_column_tq_options(rows, cols, options, "device_norm_bytes");
    const std::size_t vectors = static_cast<std::size_t>(cols);
    const std::size_t norm_sets = (options.mode == QuantizeMode::kTurboQuantQjl) ? 2U : 1U;
    return vectors * norm_sets * sizeof(float);
}

std::size_t device_qjl_sign_bytes(int rows, int cols, const QuantizeOptions& options) {
    if (options.mode != QuantizeMode::kTurboQuantQjl) return 0;
    validate_column_tq_options(rows, cols, options, "device_qjl_sign_bytes");
    return qjl_sign_bytes(rows, cols, effective_qjl_dim(rows, options));
}

// (README) Column-wise TQ 符號預產生: 預先建立每個 column/padded row 的 D 符號矩陣以便重複使用。
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

// (README) GPU block lowbit payload 壓縮: 將 flattened device block 做 legacy lowbit payload；TQ/QJL 走 column-wise payload。
DeviceCompressedBlock quantize_fp32_device_block_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    std::uint8_t* d_qjl_signs,
    cudaStream_t stream) {
    (void)d_qjl_signs;
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
    if (options.mode == QuantizeMode::kTurboQuant) {
        throw std::runtime_error(
            "mode=tq uses Lloyd-Max codebook quantization and is supported only by the device column path.");
    }
    if (options.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error(
            "mode=tq-qjl uses Lloyd-Max + QJL column-vector quantization and is supported only by the device column path.");
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    qmax_for_bits(options.bits);
    const std::size_t work_count = count;

    DeviceCompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = 0;
    block.qjl_alpha = 0.0f;
    block.seed = options.seed;
    block.padded_count = static_cast<int>(work_count);
    block.d_codes = d_codes;
    block.d_qjl_signs = nullptr;

    float* d_work = nullptr;
    const std::size_t code_bytes = block.code_bytes();

    check_cuda(cudaMalloc(&d_work, work_count * sizeof(float)), "cudaMalloc quantization work");

    const int threads = 256;
    check_cuda(cudaMemcpyAsync(
                   d_work, d_values, count * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync lowbit work block");

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

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize device payload quantization");
    cudaFree(d_work);
    (void)code_bytes;
    return block;
}

// (README) Column-wise TQ payload 壓縮: mode=tq 使用 Lloyd-Max TQ-MSE，mode=tq-qjl 使用 (bits-1)-bit TQ-MSE 加 1-bit QJL residual sketch。
DeviceCompressedBlock quantize_fp32_device_column_tq_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_norms,
    float* d_external_work,
    const signed char* d_signs,
    std::uint8_t* d_qjl_signs,
    float* d_qjl_reconstructed_ext,
    float* d_qjl_residual_ext,
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
    validate_column_tq_options(rows, cols, options, "Column TQ payload path");
    if (!d_norms) {
        throw std::runtime_error("Column TQ payload path requires per-vector norm storage.");
    }
    if (d_signs) {
        throw std::runtime_error("Column TQ payload path owns its Rademacher signs; external signs are unsupported.");
    }
    if (options.mode == QuantizeMode::kTurboQuantQjl && !d_qjl_signs) {
        throw std::runtime_error("Column tq-qjl payload requires packed QJL sign storage.");
    }

    const int mse_bits = tq_mse_bits_for_options(options);
    const int qjl_dim = effective_qjl_dim(rows, options);
    const std::size_t count = static_cast<std::size_t>(rows) * cols;

    DeviceCompressedBlock block;
    block.rows = rows;
    block.cols = cols;
    block.bits = options.bits;
    block.mode = options.mode;
    block.qjl_dim = qjl_dim;
    block.qjl_alpha = (options.mode == QuantizeMode::kTurboQuantQjl) ? options.qjl_alpha : 0.0f;
    block.seed = options.seed;
    block.padded_count = static_cast<int>(count);
    block.scale = 1.0f;
    block.residual_norm = 0.0f;
    block.d_codes = d_codes;
    block.d_norms = d_norms;
    block.d_residual_norms =
        (options.mode == QuantizeMode::kTurboQuantQjl) ? d_norms + cols : nullptr;
    block.d_qjl_signs = d_qjl_signs;

    float* d_work = d_external_work;
    if (!d_work) {
        check_cuda(cudaMalloc(&d_work, count * sizeof(float)), "cudaMalloc column TQ work");
    }

    const int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    check_cuda(cudaMemsetAsync(d_codes, 0, block.code_bytes(), stream),
               "cudaMemsetAsync column TQ Lloyd-Max codes");
    if (options.mode == QuantizeMode::kTurboQuantQjl) {
        check_cuda(cudaMemsetAsync(d_qjl_signs, 0, qjl_sign_bytes(rows, cols, qjl_dim), stream),
                   "cudaMemsetAsync column tq-qjl signs");
    }
    column_norms_kernel<<<cols, threads, threads * sizeof(float), stream>>>(
        d_values, d_norms, rows, cols);
    check_cuda(cudaGetLastError(), "launch column TQ norm kernel");
    column_tq_normalize_sign_kernel<<<blocks, threads, 0, stream>>>(
        d_values, d_norms, d_work, rows, cols, options.seed, 1.0e-12f);
    check_cuda(cudaGetLastError(), "launch column TQ normalize/sign kernel");
    fwht_columns_normalized_device(d_work, rows, cols, stream);
    column_tq_lloyd_quantize_kernel<<<blocks, threads, 0, stream>>>(
        d_work, d_codes, rows, cols, mse_bits);
    check_cuda(cudaGetLastError(), "launch column TQ Lloyd-Max quantize kernel");

    if (options.mode == QuantizeMode::kTurboQuantQjl) {
        QjlScratchCacheEntry& qjl_scratch = get_qjl_scratch(rows, cols, qjl_dim, !d_qjl_reconstructed_ext, !d_qjl_residual_ext);
        float* d_reconstructed = d_qjl_reconstructed_ext ? d_qjl_reconstructed_ext : qjl_scratch.d_reconstructed;
        float* d_residual = d_qjl_residual_ext ? d_qjl_residual_ext : qjl_scratch.d_residual;
        column_tq_lloyd_dequantize_kernel<<<blocks, threads, 0, stream>>>(
            d_codes, d_work, rows, cols, mse_bits);
        check_cuda(cudaGetLastError(), "launch column tq-qjl MSE centroid decode kernel");
        fwht_columns_normalized_device(d_work, rows, cols, stream);
        column_tq_apply_sign_norm_store_kernel<<<blocks, threads, 0, stream>>>(
            d_work, d_norms, d_reconstructed, rows, cols, options.seed);
        check_cuda(cudaGetLastError(), "launch column tq-qjl MSE inverse/store kernel");

        residual_kernel<<<blocks, threads, 0, stream>>>(
            d_values, d_reconstructed, d_residual, count);
        check_cuda(cudaGetLastError(), "launch column tq-qjl residual kernel");
        column_residual_norms_kernel<<<cols, threads, threads * sizeof(float), stream>>>(
            d_residual, block.d_residual_norms, rows, cols);
        check_cuda(cudaGetLastError(), "launch column tq-qjl residual norm kernel");

        float* d_S = get_qjl_matrix_device(rows, qjl_dim, options.seed + 17U);
        cublasHandle_t handle = get_cached_cublas_handle(stream);
        const float gemm_alpha = 1.0f;
        const float gemm_beta = 0.0f;
        check_cublas(cublasSgemm(
                         handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         qjl_dim,
                         cols,
                         rows,
                         &gemm_alpha,
                         d_S,
                         qjl_dim,
                         d_residual,
                         rows,
                         &gemm_beta,
                         qjl_scratch.d_projection,
                         qjl_dim),
                     "cublasSgemm QJL projection S*R");
        int qjl_blocks = static_cast<int>(((static_cast<std::size_t>(qjl_dim) * cols) + threads - 1) / threads);
        column_qjl_pack_projected_signs_kernel<<<qjl_blocks, threads, 0, stream>>>(
            qjl_scratch.d_projection, d_qjl_signs, qjl_dim, cols);
        check_cuda(cudaGetLastError(), "launch column tq-qjl packed sign kernel");
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column TQ quantization");

    if (!d_external_work) cudaFree(d_work);
    return block;
}

// (README) Device payload 解碼重建: 將 lowbit/TQ payload 解回完整 FP32 reconstructed matrix。
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
    if (block.mode == QuantizeMode::kTurboQuant ||
        block.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error(
            "mode=tq and mode=tq-qjl payloads are column-wise with per-vector norms; use dequantize_column_tq_payload_add_to_fp32.");
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

    check_cuda(cudaMemcpyAsync(
                   d_reconstructed, d_work, count * sizeof(float),
                   cudaMemcpyDeviceToDevice, stream),
               "cudaMemcpyAsync lowbit payload reconstructed block");

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize payload decode");
    cudaFree(d_work);
}

// (README) Flattened payload 解碼累加: 將 lowbit payload 解碼後直接加到 FP32 accumulator；正式 mode=tq 需走 column-wise norm-aware decoder。
void dequantize_device_payload_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    const signed char* d_signs,
    cudaStream_t stream) {
    (void)d_signs;
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
    if (block.mode == QuantizeMode::kTurboQuant ||
        block.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error(
            "mode=tq and mode=tq-qjl payloads are column-wise with per-vector norms; use dequantize_column_tq_payload_add_to_fp32.");
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

    const int blocks = static_cast<int>((count + threads - 1) / threads);
    add_plain_kernel<<<blocks, threads, 0, stream>>>(d_accumulator, d_work, count);
    check_cuda(cudaGetLastError(), "launch lowbit payload add kernel");

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize payload add decode");
}

// (README) Column-wise TQ payload 解碼累加: mode=tq 解碼 TQ-MSE，mode=tq-qjl 再用 QJL residual sketch 補回估計 residual。
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
    if (!is_supported_lloyd_tq_dim(block.rows)) {
        throw std::runtime_error("Column TQ decode requires vector dimension d in {256, 512, 1024, 2048}.");
    }
    if (!block.d_norms) {
        throw std::runtime_error("Column TQ decode requires per-vector norm storage.");
    }
    if (d_signs) {
        throw std::runtime_error("Column TQ decode owns its Rademacher signs; external signs are unsupported.");
    }

    const int mse_bits = (block.mode == QuantizeMode::kTurboQuantQjl) ? block.bits - 1 : block.bits;
    if (!is_supported_lloyd_tq_bits(mse_bits)) {
        throw std::runtime_error("Column TQ decode has unsupported MSE bit width.");
    }
    const std::size_t count = block.value_count();
    const int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    column_tq_lloyd_dequantize_kernel<<<blocks, threads, 0, stream>>>(
        block.d_codes, d_work, block.rows, block.cols, mse_bits);
    check_cuda(cudaGetLastError(), "launch column TQ Lloyd-Max centroid decode kernel");
    fwht_columns_normalized_device(d_work, block.rows, block.cols, stream);
    column_tq_apply_sign_norm_add_kernel<<<blocks, threads, 0, stream>>>(
        d_work, block.d_norms, d_accumulator, block.rows, block.cols, block.seed);
    check_cuda(cudaGetLastError(), "launch column TQ inverse/rescale add kernel");

    if (block.mode == QuantizeMode::kTurboQuantQjl && block.qjl_alpha != 0.0f) {
        if (!block.d_qjl_signs) {
            throw std::runtime_error("Column tq-qjl payload decode requires qjl signs.");
        }
        if (block.qjl_dim <= 0) {
            throw std::runtime_error("Column tq-qjl payload decode requires positive qjl_dim.");
        }
        const float* residual_norms = block.d_residual_norms ?
            block.d_residual_norms : (block.d_norms + block.cols);

        QjlScratchCacheEntry& qjl_scratch = get_qjl_scratch(block.rows, block.cols, block.qjl_dim, false, false);
        const std::size_t sign_count = static_cast<std::size_t>(block.qjl_dim) * block.cols;
        int qjl_blocks = static_cast<int>((sign_count + threads - 1) / threads);
        column_qjl_unpack_signs_to_float_kernel<<<qjl_blocks, threads, 0, stream>>>(
            block.d_qjl_signs,
            qjl_scratch.d_signs_float,
            block.qjl_dim,
            block.cols);
        check_cuda(cudaGetLastError(), "launch column tq-qjl sign unpack kernel");

        float* d_S = get_qjl_matrix_device(block.rows, block.qjl_dim, block.seed + 17U);
        cublasHandle_t handle = get_cached_cublas_handle(stream);
        const float gemm_alpha = 1.0f;
        const float gemm_beta = 0.0f;
        check_cublas(cublasSgemm(
                         handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         block.rows,
                         block.cols,
                         block.qjl_dim,
                         &gemm_alpha,
                         d_S,
                         block.qjl_dim,
                         qjl_scratch.d_signs_float,
                         block.qjl_dim,
                         &gemm_beta,
                         qjl_scratch.d_residual_hat,
                         block.rows),
                     "cublasSgemm QJL reconstruction S^T*q");

        constexpr float kSqrtPiOverTwo = 1.25331413731550025121f;
        column_qjl_scale_residual_add_kernel<<<blocks, threads, 0, stream>>>(
            d_accumulator,
            qjl_scratch.d_residual_hat,
            residual_norms,
            block.rows,
            block.cols,
            block.qjl_dim,
            block.qjl_alpha,
            kSqrtPiOverTwo);
        check_cuda(cudaGetLastError(), "launch column tq-qjl residual reconstruction add kernel");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize column TQ payload add decode");
}

// (README) GPU 壓縮重建測試入口: 將 device FP32 block 壓縮後解碼回 host vector 以便檢查誤差。
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

// (README) CPU payload 解碼: 將 host-side compressed payload 解回 FP32 vector，主要用於 prototype 與驗證。
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
    if (block.mode == QuantizeMode::kTurboQuant ||
        block.mode == QuantizeMode::kTurboQuantQjl) {
        throw std::runtime_error(
            "mode=tq and mode=tq-qjl host payload decode is not implemented; use the device column path.");
    }

    qmax_for_bits(block.bits);
    const std::size_t decode_count = count;
    std::vector<float> decoded(decode_count, 0.0f);
    for (std::size_t i = 0; i < decode_count; ++i) {
        std::uint8_t code = unpack_code_at(block.codes, i, block.bits);
        int q = decode_unsigned_to_signed(code, block.bits);
        decoded[i] = static_cast<float>(q) * block.scale;
    }

    std::copy(decoded.begin(), decoded.end(), values.begin());
    return values;
}

// (README) 量化模式名稱: 將 QuantizeMode enum 轉成 log 會印出的文字名稱。
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
