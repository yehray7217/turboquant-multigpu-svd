#ifndef TURBOQUANT_TURBOQUANT_HPP_
#define TURBOQUANT_TURBOQUANT_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace turboquant {

enum class QuantizeMode {
    kNone,
    kLowBit,
    kTurboQuant,
    kTurboQuantQjl,
};

struct QuantizeOptions {
    QuantizeMode mode = QuantizeMode::kNone;
    int bits = 0;
    int qjl_dim = 0;
    float qjl_alpha = 1.0f;
    unsigned seed = 1234;
};

struct CompressedBlock {
    int rows = 0;
    int cols = 0;
    int bits = 0;
    int qjl_dim = 0;
    int padded_count = 0;
    float scale = 1.0f;
    float residual_norm = 0.0f;
    float qjl_alpha = 1.0f;
    unsigned seed = 1234;
    QuantizeMode mode = QuantizeMode::kNone;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint8_t> qjl_signs;

    std::size_t value_count() const;
    std::size_t payload_bytes() const;
    double compression_ratio_vs_fp32() const;
};

struct DeviceCompressedBlock {
    int rows = 0;
    int cols = 0;
    int bits = 0;
    int qjl_dim = 0;
    int padded_count = 0;
    float scale = 1.0f;
    float residual_norm = 0.0f;
    float qjl_alpha = 1.0f;
    unsigned seed = 1234;
    QuantizeMode mode = QuantizeMode::kNone;
    std::uint8_t* d_codes = nullptr;
    float* d_norms = nullptr;
    float* d_residual_norms = nullptr;
    std::uint8_t* d_qjl_signs = nullptr;

    std::size_t value_count() const;
    std::size_t code_bytes() const;
    std::size_t payload_bytes() const;
};

struct TqColumnProfileTimings {
    float clear_codes_ms = 0.0f;
    float norm_ms = 0.0f;
    float transform_ms = 0.0f;
    float quantize_ms = 0.0f;
    float decode_add_ms = 0.0f;
};

QuantizeOptions make_quantize_options(int bits);
QuantizeOptions make_quantize_options(
    int bits,
    const std::string& mode,
    int qjl_dim,
    float qjl_alpha,
    unsigned seed);

CompressedBlock quantize_fp32_block(
    const std::vector<float>& values,
    int rows,
    int cols,
    const QuantizeOptions& options);

CompressedBlock quantize_fp32_device_block(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    cudaStream_t stream = 0);

CompressedBlock quantize_dequant_fp32_device_block(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::vector<float>* reconstructed,
    cudaStream_t stream = 0);

CompressedBlock quantize_dequant_fp32_device_block_to_device(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    float* d_reconstructed,
    cudaStream_t stream = 0,
    bool copy_payload_to_host = true);

std::size_t device_code_bytes(int rows, int cols, const QuantizeOptions& options);
std::size_t device_norm_bytes(int rows, int cols, const QuantizeOptions& options);
std::size_t device_qjl_sign_bytes(int rows, int cols, const QuantizeOptions& options);

void initialize_column_tq_signs(
    int rows,
    int cols,
    unsigned seed,
    signed char* d_signs,
    cudaStream_t stream = 0);

DeviceCompressedBlock quantize_fp32_device_block_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    std::uint8_t* d_qjl_signs,
    cudaStream_t stream = 0);

DeviceCompressedBlock quantize_fp32_device_column_tq_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_norms = nullptr,
    float* d_work = nullptr,
    const signed char* d_signs = nullptr,
    std::uint8_t* d_qjl_signs = nullptr,
    // Optional pre-allocated QJL scratch buffers. If null, turboquant reuses
    // cached per-device scratch buffers instead of allocating in the hot loop.
    float* d_qjl_reconstructed = nullptr, // size: rows * cols floats
    float* d_qjl_residual = nullptr,      // size: rows * cols floats
    cudaStream_t stream = 0);

inline DeviceCompressedBlock quantize_fp32_device_column_tq_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_work,
    const signed char* d_signs,
    std::uint8_t* d_qjl_signs,
    float* d_qjl_reconstructed = nullptr,
    float* d_qjl_residual = nullptr,
    cudaStream_t stream = 0) {
    return quantize_fp32_device_column_tq_to_device_payload(
        d_values,
        rows,
        cols,
        options,
        d_codes,
        nullptr,
        d_work,
        d_signs,
        d_qjl_signs,
        d_qjl_reconstructed,
        d_qjl_residual,
        stream);
}

void dequantize_device_payload_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_reconstructed,
    cudaStream_t stream = 0);

void dequantize_device_payload_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    const signed char* d_signs = nullptr,
    cudaStream_t stream = 0);

void dequantize_column_tq_payload_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    const signed char* d_signs = nullptr,
    cudaStream_t stream = 0);

DeviceCompressedBlock profile_tq_column_encode_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_norms,
    float* d_work,
    TqColumnProfileTimings* timings,
    cudaStream_t stream = 0);

void profile_tq_column_decode_add_to_fp32(
    const DeviceCompressedBlock& block,
    float* d_accumulator,
    float* d_work,
    TqColumnProfileTimings* timings,
    cudaStream_t stream = 0);

float profile_tq4_column_quantize_pack4_alt(
    const float* d_transformed,
    int rows,
    int cols,
    std::uint8_t* d_codes,
    cudaStream_t stream = 0);

float profile_tq4_column_quantize_branchless_alt(
    const float* d_transformed,
    int rows,
    int cols,
    std::uint8_t* d_codes,
    cudaStream_t stream = 0);

float profile_tq4_column_quantize_pack4_branchless_alt(
    const float* d_transformed,
    int rows,
    int cols,
    std::uint8_t* d_codes,
    cudaStream_t stream = 0);

std::vector<float> dequantize_fp32_block(const CompressedBlock& block);

std::string mode_name(QuantizeMode mode);

}  // namespace turboquant

#endif  // TURBOQUANT_TURBOQUANT_HPP_
