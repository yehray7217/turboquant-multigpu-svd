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
    int* d_qjl_signs = nullptr;

    std::size_t value_count() const;
    std::size_t code_bytes() const;
    std::size_t payload_bytes() const;
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
    int* d_qjl_signs,
    cudaStream_t stream = 0);

// blocks_per_sketch is a fixed internal constant (128) used by the QJL dot-product
// kernel in the column TQ path. Pre-allocate d_qjl_partials as
// (qjl_dim * kQjlColumnBlocksPerSketch) floats to avoid per-call cudaMalloc.
static constexpr int kQjlColumnBlocksPerSketch = 128;

DeviceCompressedBlock quantize_fp32_device_column_tq_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_norms = nullptr,
    float* d_work = nullptr,
    const signed char* d_signs = nullptr,
    int* d_qjl_signs = nullptr,
    // Optional pre-allocated QJL scratch buffers (avoid per-call cudaMalloc in
    // the hot loop). If null, the function allocates and frees them internally.
    float* d_qjl_reconstructed = nullptr, // size: rows * cols floats
    float* d_qjl_residual = nullptr,      // size: rows * cols floats
    float* d_qjl_partials = nullptr,      // size: qjl_dim * kQjlColumnBlocksPerSketch floats
    cudaStream_t stream = 0);

inline DeviceCompressedBlock quantize_fp32_device_column_tq_to_device_payload(
    const float* d_values,
    int rows,
    int cols,
    const QuantizeOptions& options,
    std::uint8_t* d_codes,
    float* d_work,
    const signed char* d_signs,
    int* d_qjl_signs,
    float* d_qjl_reconstructed = nullptr,
    float* d_qjl_residual = nullptr,
    float* d_qjl_partials = nullptr,
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
        d_qjl_partials,
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

std::vector<float> dequantize_fp32_block(const CompressedBlock& block);

std::string mode_name(QuantizeMode mode);

}  // namespace turboquant

#endif  // TURBOQUANT_TURBOQUANT_HPP_
