#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../turboquant/turboquant.hpp"

namespace {

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status__ = (call);                                          \
        if (status__ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string(#call) + ": " +               \
                                     cudaGetErrorString(status__));             \
        }                                                                       \
    } while (0)

struct Options {
    int rows = 256;
    int cols = 16384;
    int bits = 4;
    int repeat = 100;
    int warmup = 10;
    unsigned seed = 1234;
};

__device__ std::uint32_t mix_u32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__global__ void fill_input_kernel(float* values, std::size_t count, unsigned seed) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const std::uint32_t h = mix_u32(static_cast<std::uint32_t>(idx) ^ seed);
    const float u = static_cast<float>(h & 0x00ffffffU) * (1.0f / 16777216.0f);
    values[idx] = 2.0f * u - 1.0f;
}

__global__ void squared_error_kernel(
    const float* input,
    const float* reconstructed,
    double* sums,
    std::size_t count) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const double x = static_cast<double>(input[idx]);
    const double diff = x - static_cast<double>(reconstructed[idx]);
    atomicAdd(sums, diff * diff);
    atomicAdd(sums + 1, x * x);
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
               clock::now().time_since_epoch())
        .count();
}

int parse_int(const char* text, const char* name) {
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (!end || *end != '\0') {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--rows") {
            opt.rows = parse_int(require_value("--rows"), "--rows");
        } else if (arg == "--cols") {
            opt.cols = parse_int(require_value("--cols"), "--cols");
        } else if (arg == "--bits") {
            opt.bits = parse_int(require_value("--bits"), "--bits");
        } else if (arg == "--repeat") {
            opt.repeat = parse_int(require_value("--repeat"), "--repeat");
        } else if (arg == "--warmup") {
            opt.warmup = parse_int(require_value("--warmup"), "--warmup");
        } else if (arg == "--seed") {
            opt.seed = static_cast<unsigned>(parse_int(require_value("--seed"), "--seed"));
        } else if (arg == "--help") {
            std::printf(
                "Usage: tq_kernel_microbench [--rows 256] [--cols 16384] "
                "[--bits 4] [--warmup 10] [--repeat 100] [--seed 1234]\n");
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (opt.rows <= 0 || opt.cols <= 0 || opt.repeat <= 0 || opt.warmup < 0) {
        throw std::runtime_error("rows/cols/repeat must be positive and warmup must be nonnegative.");
    }
    return opt;
}

struct Stats {
    double mean = 0.0;
    double min = 0.0;
    double stddev = 0.0;
};

Stats summarize(const std::vector<double>& values) {
    if (values.empty()) return {};
    Stats s;
    s.min = values[0];
    for (double v : values) {
        s.mean += v;
        if (v < s.min) s.min = v;
    }
    s.mean /= static_cast<double>(values.size());
    for (double v : values) {
        const double d = v - s.mean;
        s.stddev += d * d;
    }
    s.stddev = std::sqrt(s.stddev / static_cast<double>(values.size()));
    return s;
}

void print_stats(const char* label, const std::vector<double>& values) {
    const Stats s = summarize(values);
    std::printf("%-20s mean=%10.4f ms  min=%10.4f ms  stddev=%10.4f ms\n",
                label, s.mean, s.min, s.stddev);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opt = parse_args(argc, argv);
        CHECK_CUDA(cudaSetDevice(0));
        cudaStream_t stream = nullptr;
        CHECK_CUDA(cudaStreamCreate(&stream));

        const std::size_t count = static_cast<std::size_t>(opt.rows) * opt.cols;
        const std::size_t value_bytes = count * sizeof(float);

        turboquant::QuantizeOptions tq =
            turboquant::make_quantize_options(opt.bits, "tq", 0, 0.0f, opt.seed);
        const std::size_t code_bytes =
            turboquant::device_code_bytes(opt.rows, opt.cols, tq);
        const std::size_t norm_bytes =
            turboquant::device_norm_bytes(opt.rows, opt.cols, tq);

        float* d_input = nullptr;
        float* d_reconstructed = nullptr;
        float* d_work = nullptr;
        float* d_norms = nullptr;
        std::uint8_t* d_codes = nullptr;
        double* d_sums = nullptr;

        CHECK_CUDA(cudaMalloc(&d_input, value_bytes));
        CHECK_CUDA(cudaMalloc(&d_reconstructed, value_bytes));
        CHECK_CUDA(cudaMalloc(&d_work, value_bytes));
        CHECK_CUDA(cudaMalloc(&d_norms, norm_bytes));
        CHECK_CUDA(cudaMalloc(&d_codes, code_bytes));
        CHECK_CUDA(cudaMalloc(&d_sums, 2 * sizeof(double)));

        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        fill_input_kernel<<<blocks, threads, 0, stream>>>(d_input, count, opt.seed);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));

        turboquant::DeviceCompressedBlock block;
        std::vector<double> encode_ms;
        std::vector<double> decode_ms;
        std::vector<double> roundtrip_ms;
        std::vector<double> clear_codes_ms;
        std::vector<double> norm_ms;
        std::vector<double> transform_ms;
        std::vector<double> quantize_ms;
        std::vector<double> decode_event_ms;
        std::vector<double> encode_event_total_ms;
        std::vector<double> roundtrip_event_total_ms;
        std::vector<double> quantize_pack4_alt_ms;
        std::vector<double> quantize_branchless4_alt_ms;
        std::vector<double> quantize_pack4_branchless_alt_ms;
        encode_ms.reserve(opt.repeat);
        decode_ms.reserve(opt.repeat);
        roundtrip_ms.reserve(opt.repeat);
        clear_codes_ms.reserve(opt.repeat);
        norm_ms.reserve(opt.repeat);
        transform_ms.reserve(opt.repeat);
        quantize_ms.reserve(opt.repeat);
        decode_event_ms.reserve(opt.repeat);
        encode_event_total_ms.reserve(opt.repeat);
        roundtrip_event_total_ms.reserve(opt.repeat);
        quantize_pack4_alt_ms.reserve(opt.repeat);
        quantize_branchless4_alt_ms.reserve(opt.repeat);
        quantize_pack4_branchless_alt_ms.reserve(opt.repeat);

        for (int iter = 0; iter < opt.warmup + opt.repeat; ++iter) {
            CHECK_CUDA(cudaMemsetAsync(d_reconstructed, 0, value_bytes, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));

            turboquant::TqColumnProfileTimings phase{};
            const double t0 = now_ms();
            block = turboquant::profile_tq_column_encode_to_device_payload(
                d_input,
                opt.rows,
                opt.cols,
                tq,
                d_codes,
                d_norms,
                d_work,
                &phase,
                stream);
            const double t1 = now_ms();
            turboquant::profile_tq_column_decode_add_to_fp32(
                block,
                d_reconstructed,
                d_work,
                &phase,
                stream);
            const double t2 = now_ms();
            float alt_pack_ms = 0.0f;
            if (opt.bits == 4) {
                alt_pack_ms = turboquant::profile_tq4_column_quantize_pack4_alt(
                    d_work,
                    opt.rows,
                    opt.cols,
                    d_codes,
                    stream);
            }
            float alt_branchless_ms = 0.0f;
            if (opt.bits == 4) {
                alt_branchless_ms = turboquant::profile_tq4_column_quantize_branchless_alt(
                    d_work,
                    opt.rows,
                    opt.cols,
                    d_codes,
                    stream);
            }
            float alt_pack_branchless_ms = 0.0f;
            if (opt.bits == 4) {
                alt_pack_branchless_ms = turboquant::profile_tq4_column_quantize_pack4_branchless_alt(
                    d_work,
                    opt.rows,
                    opt.cols,
                    d_codes,
                    stream);
            }

            if (iter >= opt.warmup) {
                encode_ms.push_back(t1 - t0);
                decode_ms.push_back(t2 - t1);
                roundtrip_ms.push_back(t2 - t0);
                clear_codes_ms.push_back(phase.clear_codes_ms);
                norm_ms.push_back(phase.norm_ms);
                transform_ms.push_back(phase.transform_ms);
                quantize_ms.push_back(phase.quantize_ms);
                decode_event_ms.push_back(phase.decode_add_ms);
                const double encode_event =
                    static_cast<double>(phase.clear_codes_ms) +
                    static_cast<double>(phase.norm_ms) +
                    static_cast<double>(phase.transform_ms) +
                    static_cast<double>(phase.quantize_ms);
                encode_event_total_ms.push_back(encode_event);
                roundtrip_event_total_ms.push_back(
                    encode_event + static_cast<double>(phase.decode_add_ms));
                if (opt.bits == 4) {
                    quantize_pack4_alt_ms.push_back(alt_pack_ms);
                    quantize_branchless4_alt_ms.push_back(alt_branchless_ms);
                    quantize_pack4_branchless_alt_ms.push_back(alt_pack_branchless_ms);
                }
            }
        }

        CHECK_CUDA(cudaMemsetAsync(d_sums, 0, 2 * sizeof(double), stream));
        squared_error_kernel<<<blocks, threads, 0, stream>>>(
            d_input, d_reconstructed, d_sums, count);
        CHECK_CUDA(cudaGetLastError());
        double h_sums[2] = {0.0, 0.0};
        CHECK_CUDA(cudaMemcpyAsync(h_sums, d_sums, 2 * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        const double relative_error =
            std::sqrt(h_sums[0] / (h_sums[1] > 0.0 ? h_sums[1] : 1.0));

        std::printf("TQ kernel microbenchmark\n");
        std::printf("rows=%d cols=%d bits=%d warmup=%d repeat=%d seed=%u\n",
                    opt.rows, opt.cols, opt.bits, opt.warmup, opt.repeat, opt.seed);
        std::printf("value_count=%zu fp32_bytes=%.3f MiB code_bytes=%.3f MiB norm_bytes=%.3f MiB\n",
                    count,
                    static_cast<double>(value_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(code_bytes) / (1024.0 * 1024.0),
                    static_cast<double>(norm_bytes) / (1024.0 * 1024.0));
        print_stats("encode", encode_ms);
        print_stats("decode_add", decode_ms);
        print_stats("roundtrip", roundtrip_ms);
        std::printf("CUDA event phase breakdown\n");
        print_stats("clear_codes", clear_codes_ms);
        print_stats("norm", norm_ms);
        print_stats("transform", transform_ms);
        print_stats("quantize_bitpack", quantize_ms);
        print_stats("decode_event", decode_event_ms);
        print_stats("encode_event_total", encode_event_total_ms);
        print_stats("roundtrip_event_total", roundtrip_event_total_ms);
        if (opt.bits == 4) {
            print_stats("quantize_pack4_alt", quantize_pack4_alt_ms);
            print_stats("quantize_branchless4_alt", quantize_branchless4_alt_ms);
            print_stats("quantize_pack4_branchless_alt", quantize_pack4_branchless_alt_ms);
        }
        std::printf("relative_error=%0.8f\n", relative_error);

        cudaFree(d_input);
        cudaFree(d_reconstructed);
        cudaFree(d_work);
        cudaFree(d_norms);
        cudaFree(d_codes);
        cudaFree(d_sums);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 1;
    }
}
