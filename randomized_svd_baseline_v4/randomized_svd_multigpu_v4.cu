// randomized_svd_multigpu_v3.cu
//
// Baseline B v3: MPI + TSQR-style multi-node/multi-GPU randomized SVD.
//
// Compared with randomized_svd_baseline, this version keeps the large matrix
// blocks A_i on their owning GPUs after the first projection. It computes:
//
//   1. Y_i = A_i * Omega on GPU i
//   2. local QR: Y_i = Qbar_i * R_i on GPU i
//   3. host/GPU0 TSQR reduction: stack(R_i) = T * R
//   4. Q_i = Qbar_i * T_i on GPU i
//   5. B_i = Q_i^T * A_i on GPU i
//   6. B = sum_i B_i on GPU 0, then SVD(B) on GPU 0
//   7. U_i = Q_i * U_tilde_k on GPU i
//
// This is still a research baseline, not a production NCCL implementation.
// Each MPI rank controls the visible GPUs on one node. Cross-rank communication
// uses host MPI for TSQR metadata and compressed B payload collectives.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <mpi.h>
#include <nvtx3/nvToolsExt.h>

#include "../turboquant/turboquant.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status__ = (call);                                           \
        if (status__ != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " -> " +  \
                                     cudaGetErrorString(status__));              \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status__ = (call);                                        \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ +\
                                     ":" + std::to_string(__LINE__) + " -> " +  \
                                     cublas_status_to_string(status__));          \
        }                                                                       \
    } while (0)

#define CHECK_CUSOLVER(call)                                                    \
    do {                                                                        \
        cusolverStatus_t status__ = (call);                                      \
        if (status__ != CUSOLVER_STATUS_SUCCESS) {                               \
            throw std::runtime_error(std::string("cuSOLVER error at ") + __FILE__+\
                                     ":" + std::to_string(__LINE__) + " -> " +  \
                                     cusolver_status_to_string(status__));        \
        }                                                                       \
    } while (0)

#define CHECK_MPI(call)                                                         \
    do {                                                                        \
        int status__ = (call);                                                   \
        if (status__ != MPI_SUCCESS) {                                           \
            char errstr__[MPI_MAX_ERROR_STRING];                                \
            int len__ = 0;                                                       \
            MPI_Error_string(status__, errstr__, &len__);                       \
            throw std::runtime_error(std::string("MPI error at ") + __FILE__ +  \
                                     ":" + std::to_string(__LINE__) + " -> " +  \
                                     std::string(errstr__, len__));              \
        }                                                                       \
    } while (0)

static const char* cublas_status_to_string(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if defined(CUBLAS_STATUS_NOT_SUPPORTED)
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

static const char* cusolver_status_to_string(cusolverStatus_t s) {
    switch (s) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
#if defined(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
#endif
#if defined(CUSOLVER_STATUS_NOT_SUPPORTED)
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
#endif
        default: return "CUSOLVER_STATUS_UNKNOWN";
    }
}

struct Options {
    int m = 4096;
    int n = 2048;
    int k = 64;
    int oversample = 16;
    int ngpus = 1;
    int gpus_per_rank = 0;
    unsigned seed = 1234;
    bool check_error = true;
    bool check_b_error = false;
    std::string compress_b_mode = "none";
    int compress_b_bits = 0;
    std::string compress_subspace_mode = "none";
    int compress_subspace_bits = 0;
    int qjl_dim = 256;
    float qjl_alpha = 1.0f;
    bool device_random_input = false;
    bool skip_form_u = false;
    int subspace_iter = 0;
    bool stabilize_subspace_z = false;
    std::string spectrum_decay_mode = "random";
    double spectrum_decay_param = 1.0;
    int spectrum_rank = 0;
    int repeat = 1;
    bool summary_only = false;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --m <int>             Number of rows of A. Default: 4096\n"
        << "  --n <int>             Number of cols of A. Default: 2048\n"
        << "  --k <int>             Target rank. Default: 64\n"
        << "  --oversample <int>    Oversampling p. l = k + p. Default: 16\n"
        << "  --ngpus <int>         Number of GPUs to use. Default: 1\n"
        << "                        In v3 this is total GPUs across all MPI ranks.\n"
        << "  --gpus-per-rank <int> Local GPUs controlled by each MPI rank.\n"
        << "                        Default: ceil(ngpus / mpi_size), capped by visible GPUs.\n"
        << "  --seed <int>          RNG seed. Default: 1234\n"
        << "  --compress-b-mode <none|lowbit|tq|tq-qjl>\n"
        << "                        B_i compression mode. Default: none\n"
        << "  --compress-b-bits <0|2..8>\n"
        << "                        B_i quantization bits. mode=tq supports 2..8; lowbit/tq-qjl support 2/4/8. Default: 0\n"
        << "  --compress-subspace-mode <none|lowbit|tq|tq-qjl>\n"
        << "                        Z = A^T Q compression mode inside subspace iteration. Default: none\n"
        << "  --compress-subspace-bits <0|2..8>\n"
        << "                        Z = A^T Q quantization bits. mode=tq supports 2..8; lowbit/tq-qjl support 2/4/8. Default: 0\n"
        << "  --qjl-dim <int>       QJL residual sketch dimension for tq-qjl. Default: 256\n"
        << "  --qjl-alpha <float>   QJL residual correction strength. Default: 1.0\n"
        << "  --device-random-input\n"
        << "                        Generate A_i and Omega directly on each GPU.\n"
        << "  --skip-form-u         Skip distributed U_i formation after SVD(B).\n"
        << "  --subspace-iter <int> Number of randomized SVD power/subspace iterations. Default: 0\n"
        << "  --stabilize-subspace-z\n"
        << "                        QR-orthogonalize Z = A^T Q inside each subspace iteration.\n"
        << "  --spectrum-decay-mode <random|polynomial|exponential>\n"
        << "                        Test matrix singular spectrum. Default: random\n"
        << "  --spectrum-decay-param <float>\n"
        << "                        p for polynomial, alpha for exponential. Default: 1.0\n"
        << "  --spectrum-rank <int> Number of synthetic singular values. Default: min(m,n)\n"
        << "  --repeat <int>        Reuse setup and run the compute pipeline N times. Default: 1\n"
        << "  --summary-only        Suppress per-repeat details and print compact summaries only.\n"
        << "  --no-check-error      Skip final reconstruction error metric.\n"
        << "  --check-b-error       Enable B_i compression error copy/check. Default: off\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const std::string& name) -> char* {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };
        if (a == "--m") opt.m = std::stoi(need_value(a));
        else if (a == "--n") opt.n = std::stoi(need_value(a));
        else if (a == "--k") opt.k = std::stoi(need_value(a));
        else if (a == "--oversample") opt.oversample = std::stoi(need_value(a));
        else if (a == "--ngpus") opt.ngpus = std::stoi(need_value(a));
        else if (a == "--gpus-per-rank") opt.gpus_per_rank = std::stoi(need_value(a));
        else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need_value(a)));
        else if (a == "--compress-b-mode") opt.compress_b_mode = need_value(a);
        else if (a == "--compress-b-bits") opt.compress_b_bits = std::stoi(need_value(a));
        else if (a == "--compress-subspace-mode") opt.compress_subspace_mode = need_value(a);
        else if (a == "--compress-subspace-bits") opt.compress_subspace_bits = std::stoi(need_value(a));
        else if (a == "--qjl-dim") opt.qjl_dim = std::stoi(need_value(a));
        else if (a == "--qjl-alpha") opt.qjl_alpha = std::stof(need_value(a));
        else if (a == "--repeat") opt.repeat = std::stoi(need_value(a));
        else if (a == "--device-random-input") opt.device_random_input = true;
        else if (a == "--skip-form-u") opt.skip_form_u = true;
        else if (a == "--subspace-iter") opt.subspace_iter = std::stoi(need_value(a));
        else if (a == "--stabilize-subspace-z") opt.stabilize_subspace_z = true;
        else if (a == "--spectrum-decay-mode") opt.spectrum_decay_mode = need_value(a);
        else if (a == "--spectrum-decay-param") opt.spectrum_decay_param = std::stod(need_value(a));
        else if (a == "--spectrum-rank") opt.spectrum_rank = std::stoi(need_value(a));
        else if (a == "--summary-only") opt.summary_only = true;
        else if (a == "--no-check-error") opt.check_error = false;
        else if (a == "--check-b-error") opt.check_b_error = true;
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + a);
        }
    }
    if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.oversample < 0 || opt.ngpus <= 0 || opt.gpus_per_rank < 0) {
        throw std::runtime_error("m, n, k, ngpus must be positive; gpus-per-rank and oversample must be non-negative.");
    }
    if (opt.repeat <= 0) {
        throw std::runtime_error("repeat must be positive.");
    }
    if (opt.subspace_iter < 0) {
        throw std::runtime_error("subspace-iter must be non-negative.");
    }
    if (opt.spectrum_decay_mode != "random" &&
        opt.spectrum_decay_mode != "polynomial" &&
        opt.spectrum_decay_mode != "exponential") {
        throw std::runtime_error("spectrum-decay-mode must be one of: random, polynomial, exponential.");
    }
    if (opt.spectrum_decay_mode != "random" && opt.spectrum_decay_param <= 0.0) {
        throw std::runtime_error("spectrum-decay-param must be positive for polynomial/exponential modes.");
    }
    if (opt.spectrum_rank < 0) {
        throw std::runtime_error("spectrum-rank must be non-negative.");
    }
    if (opt.k + opt.oversample > std::min(opt.m, opt.n)) {
        throw std::runtime_error("Require k + oversample <= min(m, n).");
    }
    turboquant::QuantizeOptions qopt =
        turboquant::make_quantize_options(opt.compress_b_bits, opt.compress_b_mode, opt.qjl_dim, opt.qjl_alpha, opt.seed);
    turboquant::QuantizeOptions subspace_qopt =
        turboquant::make_quantize_options(opt.compress_subspace_bits, opt.compress_subspace_mode, opt.qjl_dim, opt.qjl_alpha, opt.seed + 7919u);
    return opt;
}

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    void tic() { t0 = std::chrono::high_resolution_clock::now(); }
    double toc_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

static std::vector<int> split_rows(int m, int ngpus) {
    std::vector<int> rows(ngpus, m / ngpus);
    for (int i = 0; i < m % ngpus; ++i) rows[i]++;
    return rows;
}

static std::vector<int> prefix_offsets(const std::vector<int>& rows) {
    std::vector<int> offsets(rows.size(), 0);
    int sum = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        offsets[i] = sum;
        sum += rows[i];
    }
    return offsets;
}

static int checked_mpi_count(std::size_t count, const char* label) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds MPI int count limit.");
    }
    return static_cast<int>(count);
}

struct MpiRuntime {
    int rank = 0;
    int size = 1;
    MpiRuntime(int* argc, char*** argv) {
        CHECK_MPI(MPI_Init(argc, argv));
        CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    }
    ~MpiRuntime() {
        MPI_Finalize();
    }
    MpiRuntime(const MpiRuntime&) = delete;
    MpiRuntime& operator=(const MpiRuntime&) = delete;
};

static std::vector<float> make_random_matrix(int rows, int cols, unsigned seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> a(static_cast<size_t>(rows) * cols);
    for (float& x : a) x = dist(gen);
    return a;
}

static std::vector<std::vector<float>> make_random_row_blocks(
    const std::vector<int>& rows,
    int cols,
    unsigned seed,
    float scale,
    double* norm2_out) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<std::vector<float>> blocks(rows.size());
    long double norm2 = 0.0L;

    for (size_t g = 0; g < rows.size(); ++g) {
        blocks[g].resize(static_cast<size_t>(rows[g]) * cols);
    }
    for (int col = 0; col < cols; ++col) {
        for (size_t g = 0; g < rows.size(); ++g) {
            float* dst = blocks[g].data() + static_cast<size_t>(col) * rows[g];
            for (int row = 0; row < rows[g]; ++row) {
                float x = dist(gen);
                dst[row] = x;
                norm2 += static_cast<long double>(x) * x;
            }
        }
    }
    if (norm2_out) *norm2_out = static_cast<double>(norm2);
    return blocks;
}

static void insert_row_block_colmajor(
    std::vector<float>& A, const std::vector<float>& Ai,
    int m, int cols, int row0, int mi) {
    for (int col = 0; col < cols; ++col) {
        float* dst = A.data() + static_cast<size_t>(col) * m + row0;
        const float* src = Ai.data() + static_cast<size_t>(col) * mi;
        std::copy(src, src + mi, dst);
    }
}

static double fast_reconstruction_relative_error(
    double a_norm2,
    const std::vector<float>& S,
    int k) {
    long double top_singular_energy = 0.0L;
    for (int i = 0; i < k; ++i) {
        top_singular_energy += static_cast<long double>(S[i]) * static_cast<long double>(S[i]);
    }
    const long double err2 = std::max(
        static_cast<long double>(a_norm2) - top_singular_energy,
        0.0L);
    return std::sqrt(static_cast<double>(err2 / std::max(static_cast<long double>(a_norm2), 1e-30L)));
}

static double synthetic_spectrum_theoretical_error(
    const std::string& decay_mode,
    double decay_param,
    int spectrum_rank,
    int k) {
    if (decay_mode == "random" || spectrum_rank <= 0) return -1.0;

    long double total = 0.0L;
    long double tail = 0.0L;
    for (int i = 0; i < spectrum_rank; ++i) {
        long double sigma = 1.0L;
        if (decay_mode == "polynomial") {
            sigma = std::pow(static_cast<long double>(i + 1), -static_cast<long double>(decay_param));
        } else if (decay_mode == "exponential") {
            sigma = std::exp(-static_cast<long double>(decay_param) * static_cast<long double>(i));
        } else {
            return -1.0;
        }
        const long double sigma2 = sigma * sigma;
        total += sigma2;
        if (i >= k) {
            tail += sigma2;
        }
    }
    if (total <= 0.0L) return -1.0;
    return std::sqrt(static_cast<double>(tail / total));
}

struct SummaryStats {
    double mean = 0.0;
    double min = 0.0;
    double stddev = 0.0;
};

static SummaryStats summarize_samples(const std::vector<double>& xs, size_t begin) {
    if (begin >= xs.size()) return {};
    const size_t count = xs.size() - begin;
    double sum = 0.0;
    double min_value = xs[begin];
    for (size_t i = begin; i < xs.size(); ++i) {
        sum += xs[i];
        min_value = std::min(min_value, xs[i]);
    }
    const double mean = sum / static_cast<double>(count);
    double variance_sum = 0.0;
    for (size_t i = begin; i < xs.size(); ++i) {
        const double diff = xs[i] - mean;
        variance_sum += diff * diff;
    }
    const double stddev = (count > 1) ?
        std::sqrt(variance_sum / static_cast<double>(count - 1)) :
        0.0;
    return SummaryStats{mean, min_value, stddev};
}

static void print_summary_stats_row(
    const std::string& label,
    const SummaryStats& stats,
    int label_width = 32,
    double value_scale = 1.0,
    const std::string& value_suffix = "") {
    constexpr int value_width = 16;
    auto format_value = [&](double value) {
        std::ostringstream oss;
        oss << (value * value_scale) << value_suffix;
        return oss.str();
    };
    std::cout <<  std::setw(label_width) << std::right << label
              <<  std::setw(value_width) << std::right << format_value(stats.mean)
              <<  std::setw(value_width) << std::right << format_value(stats.min)
              <<  std::setw(value_width) << std::right << format_value(stats.stddev) << "\n";
}

static void print_reconstruction_error_row(
    const std::string& label,
    const SummaryStats& stats,
    double theoretical,
    int label_width = 32) {
    constexpr int value_width = 16;
    auto format_percent = [](double value) {
        std::ostringstream oss;
        oss << (value * 100.0) << "%";
        return oss.str();
    };
    auto format_ratio = [](double value) {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    };

    const bool has_theoretical = theoretical > 0.0;
    const double ratio = has_theoretical ? stats.mean / theoretical : 0.0;
    std::cout <<  std::setw(label_width) << std::right << label
              <<  std::setw(value_width) << std::right << format_percent(stats.mean)
              <<  std::setw(value_width) << std::right << format_percent(stats.min)
              <<  std::setw(value_width) << std::right << format_percent(stats.stddev)
              <<  std::setw(value_width) << std::right << (has_theoretical ? format_percent(theoretical) : std::string("n/a"))
              <<  std::setw(value_width) << std::right << (has_theoretical ? format_ratio(ratio) : std::string("n/a"))
              << "\n";
}

__global__ void transpose_colmajor_kernel(const float* src, float* dst, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        dst[static_cast<size_t>(row) * cols + col] = src[static_cast<size_t>(col) * rows + row];
    }
}

__device__ unsigned long long splitmix64_device(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ float hash_uniform_centered(int row, int col, unsigned seed, float scale) {
    unsigned long long key = static_cast<unsigned long long>(row + 1) * 0x9e3779b97f4a7c15ULL;
    key ^= static_cast<unsigned long long>(col + 1) * 0xbf58476d1ce4e5b9ULL;
    key ^= static_cast<unsigned long long>(seed);
    unsigned long long rnd = splitmix64_device(key);
    float u = static_cast<float>((rnd >> 40) & 0xffffffULL) * (1.0f / 16777216.0f);
    return (2.0f * u - 1.0f) * scale;
}

__global__ void fill_random_colmajor_kernel(
    float* dst,
    int rows,
    int cols,
    int global_row0,
    unsigned seed,
    float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        dst[static_cast<size_t>(col) * rows + row] =
            hash_uniform_centered(global_row0 + row, col, seed, scale);
    }
}

__global__ void fill_dct_basis_colmajor_kernel(
    float* dst,
    int rows,
    int cols,
    int global_row0,
    int total_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        const int global_row = global_row0 + row;
        float value = 0.0f;
        if (col == 0) {
            value = rsqrtf(static_cast<float>(total_rows));
        } else {
            const float angle =
                3.14159265358979323846f *
                (static_cast<float>(global_row) + 0.5f) *
                static_cast<float>(col) /
                static_cast<float>(total_rows);
            value = sqrtf(2.0f / static_cast<float>(total_rows)) * cosf(angle);
        }
        dst[static_cast<size_t>(col) * rows + row] = value;
    }
}

__global__ void fill_dct_basis_scaled_colmajor_kernel(
    float* dst,
    int rows,
    int cols,
    int total_rows,
    int decay_mode,
    double decay_param) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        float basis = 0.0f;
        if (col == 0) {
            basis = rsqrtf(static_cast<float>(total_rows));
        } else {
            const float angle =
                3.14159265358979323846f *
                (static_cast<float>(row) + 0.5f) *
                static_cast<float>(col) /
                static_cast<float>(total_rows);
            basis = sqrtf(2.0f / static_cast<float>(total_rows)) * cosf(angle);
        }
        double sigma = 1.0;
        if (decay_mode == 1) {
            sigma = pow(static_cast<double>(col + 1), -decay_param);
        } else if (decay_mode == 2) {
            sigma = exp(-decay_param * static_cast<double>(col));
        }
        dst[static_cast<size_t>(col) * rows + row] = basis * static_cast<float>(sigma);
    }
}

__global__ void add_kernel(float* dst, const float* src, size_t count) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] += src[idx];
}

__global__ void subtract_kernel(float* dst, const float* src, size_t count) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] -= src[idx];
}

__global__ void extract_vt_rows_scaled_kernel(
    const float* vt,
    const float* s,
    float* v_scaled,
    int l,
    int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < l && col < k) {
        v_scaled[static_cast<size_t>(col) * l + row] =
            vt[static_cast<size_t>(row) * l + col] * s[col];
    }
}

__global__ void extract_upper_kernel(const float* src, int src_ld, float* dst, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        dst[static_cast<size_t>(col) * n + row] =
            (row <= col) ? src[static_cast<size_t>(col) * src_ld + row] : 0.0f;
    }
}

struct DeviceWork {
    int dev = 0;
    int row0 = 0;
    int mi = 0;
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    float* d_Ai = nullptr;       // mi x n
    float* d_Ubasis = nullptr;   // mi x spectrum_rank synthetic left basis
    float* d_Vsigma = nullptr;   // n x spectrum_rank synthetic right basis scaled by singular values
    float* d_Omega = nullptr;    // n x l
    float* d_Qbar = nullptr;     // mi x l, first Y_i then local Qbar_i
    float* d_tau = nullptr;      // l
    float* d_qr_work = nullptr;  // local QR workspace
    float* d_Ri = nullptr;       // l x l local TSQR R block
    int qr_lwork = 0;
    float* d_Ti = nullptr;       // l x l
    float* d_Qi = nullptr;       // mi x l
    float* d_Zi = nullptr;       // n x l local A_i^T Q_i contribution
    float* d_Z = nullptr;        // n x l global reduced subspace iterate
    float* d_Bi = nullptr;       // l x n
    float* d_Bi_hat = nullptr;   // l x n reconstructed B_i after optional compression
    std::uint8_t* d_Bi_codes = nullptr; // compressed B_i payload
    float* d_Bi_norms = nullptr;        // per-column norms for Lloyd-Max TQ
    int* d_Bi_qjl_signs = nullptr;      // tq-qjl residual signs
    float* d_Bi_tq_work = nullptr;      // column-wise tq encode scratch
    float* d_Uti_k = nullptr;    // l x k
    float* d_Ui = nullptr;       // mi x k
    int* d_info = nullptr;
};

static void destroy_work(DeviceWork& w) {
    if (w.dev >= 0) cudaSetDevice(w.dev);
    if (w.d_Ai) cudaFree(w.d_Ai);
    if (w.d_Ubasis) cudaFree(w.d_Ubasis);
    if (w.d_Vsigma) cudaFree(w.d_Vsigma);
    if (w.d_Omega) cudaFree(w.d_Omega);
    if (w.d_Qbar) cudaFree(w.d_Qbar);
    if (w.d_tau) cudaFree(w.d_tau);
    if (w.d_qr_work) cudaFree(w.d_qr_work);
    if (w.d_Ri) cudaFree(w.d_Ri);
    if (w.d_Ti) cudaFree(w.d_Ti);
    if (w.d_Qi) cudaFree(w.d_Qi);
    if (w.d_Zi) cudaFree(w.d_Zi);
    if (w.d_Z) cudaFree(w.d_Z);
    if (w.d_Bi) cudaFree(w.d_Bi);
    if (w.d_Bi_hat) cudaFree(w.d_Bi_hat);
    if (w.d_Bi_codes) cudaFree(w.d_Bi_codes);
    if (w.d_Bi_norms) cudaFree(w.d_Bi_norms);
    if (w.d_Bi_qjl_signs) cudaFree(w.d_Bi_qjl_signs);
    if (w.d_Bi_tq_work) cudaFree(w.d_Bi_tq_work);
    if (w.d_Uti_k) cudaFree(w.d_Uti_k);
    if (w.d_Ui) cudaFree(w.d_Ui);
    if (w.d_info) cudaFree(w.d_info);
    if (w.blas) cublasDestroy(w.blas);
    if (w.solver) cusolverDnDestroy(w.solver);
    w = DeviceWork{};
}

static void check_solver_info(int* d_info, const std::string& label) {
    int info = 0;
    CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) throw std::runtime_error(label + " failed, info=" + std::to_string(info));
}

int main(int argc, char** argv) {
    try {
        MpiRuntime mpi(&argc, &argv);
        const bool is_root = mpi.rank == 0;
        Options opt = parse_args(argc, argv);
        int device_count = 0;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) throw std::runtime_error("No CUDA device found.");
        const int requested_total_gpus = opt.ngpus;
        const int requested_gpus_per_rank =
            opt.gpus_per_rank > 0 ?
            opt.gpus_per_rank :
            (requested_total_gpus + mpi.size - 1) / mpi.size;
        opt.ngpus = std::min(requested_gpus_per_rank, device_count);
        int min_local_gpus = 0;
        CHECK_MPI(MPI_Allreduce(&opt.ngpus, &min_local_gpus, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        opt.ngpus = min_local_gpus;
        const int global_ngpus = opt.ngpus * mpi.size;
        if (global_ngpus <= 0) throw std::runtime_error("No MPI rank has a usable CUDA device.");
        if (global_ngpus < requested_total_gpus && is_root) {
            std::cerr << "Warning: requested " << requested_total_gpus
                      << " total GPUs but only using " << global_ngpus
                      << " visible GPUs across MPI ranks.\n";
        }

        const int m = opt.m;
        const int n = opt.n;
        const int k = opt.k;
        const int l = opt.k + opt.oversample;
        const bool use_spectrum_decay = opt.spectrum_decay_mode != "random";
        if (opt.spectrum_rank == 0) {
            opt.spectrum_rank = std::min(m, n);
        }
        if (opt.spectrum_rank > std::min(m, n)) {
            throw std::runtime_error("spectrum-rank must be <= min(m, n).");
        }
        const std::vector<int> global_rows = split_rows(m, global_ngpus);
        const std::vector<int> global_row0 = prefix_offsets(global_rows);
        std::vector<int> rows(opt.ngpus);
        std::vector<int> row0s(opt.ngpus);
        for (int g = 0; g < opt.ngpus; ++g) {
            const int global_g = mpi.rank * opt.ngpus + g;
            rows[g] = global_rows[global_g];
            row0s[g] = global_row0[global_g];
        }
        for (int mi : rows) {
            if (mi < l) throw std::runtime_error("Each row block must have at least k + oversample rows.");
        }

        if (is_root) {
            std::cout << "Randomized multi-GPU SVD baseline v4 (MPI TSQR/distributed B + optional subspace iteration)\n"
                      << "  m=" << m << " n=" << n << " k=" << k
                      << " oversample=" << opt.oversample << " l=" << l
                      << " mpi_size=" << mpi.size
                      << " local_gpus_per_rank=" << opt.ngpus
                      << " global_ngpus=" << global_ngpus
                      << " compress_b_mode=" << opt.compress_b_mode
                      << " compress_b_bits=" << opt.compress_b_bits
                      << " compress_subspace_mode=" << opt.compress_subspace_mode
                      << " compress_subspace_bits=" << opt.compress_subspace_bits
                      << " qjl_alpha=" << opt.qjl_alpha
                      << " device_random_input=" << (opt.device_random_input ? "yes" : "no")
                      << " skip_form_u=" << (opt.skip_form_u ? "yes" : "no")
                      << " subspace_iter=" << opt.subspace_iter
                      << " stabilize_subspace_z=" << (opt.stabilize_subspace_z ? "yes" : "no")
                      << " spectrum_decay_mode=" << opt.spectrum_decay_mode
                      << " spectrum_decay_param=" << opt.spectrum_decay_param
                      << " spectrum_rank=" << opt.spectrum_rank
                      << " repeat=" << opt.repeat
                      << "\n";
        }

        Timer timer;
        timer.tic();
        std::vector<std::vector<float>> h_A_blocks;
        std::vector<float> h_Omega;
        double h_A_norm2 = 0.0;
        {
            NvtxRange range("init_host_random");
            if (!opt.device_random_input) {
                if (!use_spectrum_decay) {
                    h_A_blocks = make_random_row_blocks(
                        rows, n, opt.seed, 1.0f / std::sqrt(static_cast<float>(m)), &h_A_norm2);
                }
                h_Omega = make_random_matrix(n, l, opt.seed + 1, 1.0f);
            }
            if (use_spectrum_decay) {
                long double norm2 = 0.0L;
                for (int i = 0; i < opt.spectrum_rank; ++i) {
                    long double sigma = 1.0L;
                    if (opt.spectrum_decay_mode == "polynomial") {
                        sigma = std::pow(static_cast<long double>(i + 1), -static_cast<long double>(opt.spectrum_decay_param));
                    } else if (opt.spectrum_decay_mode == "exponential") {
                        sigma = std::exp(-static_cast<long double>(opt.spectrum_decay_param) * static_cast<long double>(i));
                    }
                    norm2 += sigma * sigma;
                }
                h_A_norm2 = static_cast<double>(norm2);
            }
        }
        double t_init_ms = timer.toc_ms();

        std::vector<DeviceWork> works(opt.ngpus);
        cublasHandle_t blas0 = nullptr;
        cusolverDnHandle_t solver0 = nullptr;
        int* d_info0 = nullptr;
        const int local_r_rows = opt.ngpus * l;
        const int r_rows = global_ngpus * l;
        const size_t b_count = static_cast<size_t>(l) * n;
        const size_t z_count = static_cast<size_t>(n) * l;
        const size_t bi_fp32_bytes = b_count * sizeof(float);
        const size_t z_fp32_bytes = z_count * sizeof(float);
        const turboquant::QuantizeOptions b_quant_options =
            turboquant::make_quantize_options(opt.compress_b_bits, opt.compress_b_mode, opt.qjl_dim, opt.qjl_alpha, opt.seed);
        const turboquant::QuantizeOptions subspace_quant_options =
            turboquant::make_quantize_options(
                opt.compress_subspace_bits,
                opt.compress_subspace_mode,
                opt.qjl_dim,
                opt.qjl_alpha,
                opt.seed + 7919u);
        const bool compress_b_none = b_quant_options.mode == turboquant::QuantizeMode::kNone;
        const bool compress_b_tq = b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant;
        const bool compress_b_qjl = b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl;
        const bool need_compressed_safe_error = !compress_b_none && opt.check_error;
        const bool need_global_b_metric = !compress_b_none && (opt.check_error || opt.check_b_error);
        const bool need_svd_vectors = !opt.skip_form_u || need_compressed_safe_error;
        const bool compress_subspace_none = subspace_quant_options.mode == turboquant::QuantizeMode::kNone;
        const bool compress_subspace_lloyd_tq =
            subspace_quant_options.mode == turboquant::QuantizeMode::kTurboQuant;
        const bool compress_subspace_tq =
            subspace_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
            subspace_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl;
        const bool compress_subspace_qjl = subspace_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl;
        const size_t bi_code_bytes = compress_b_none ? 0 :
            turboquant::device_code_bytes(l, n, b_quant_options);
        const size_t z_code_bytes = compress_subspace_none ? 0 :
            turboquant::device_code_bytes(l, n, subspace_quant_options);
        const size_t bi_qjl_sign_bytes = compress_b_qjl ?
            static_cast<size_t>(b_quant_options.qjl_dim) * sizeof(int) : 0;
        const size_t z_qjl_sign_bytes = compress_subspace_qjl ?
            static_cast<size_t>(subspace_quant_options.qjl_dim) * sizeof(int) : 0;
        const size_t bi_norm_bytes = compress_b_tq ? static_cast<size_t>(n) * sizeof(float) : 0;
        const size_t z_norm_bytes = compress_subspace_lloyd_tq ? static_cast<size_t>(n) * sizeof(float) : 0;
        const size_t bi_payload_work_count = compress_b_none ? 0 :
            ((b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
              b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl) ?
             static_cast<size_t>(1) << static_cast<size_t>(std::ceil(std::log2(static_cast<double>(b_count)))) :
             b_count);
        const size_t bi_column_tq_work_count = static_cast<size_t>(1) << static_cast<size_t>(std::ceil(std::log2(static_cast<double>(l))));
        const size_t bi_column_tq_sign_count = bi_column_tq_work_count * static_cast<size_t>(n);
        const size_t z_row_tq_work_count = static_cast<size_t>(1) << static_cast<size_t>(std::ceil(std::log2(static_cast<double>(l))));
        const size_t z_row_tq_sign_count = z_row_tq_work_count * static_cast<size_t>(n);
        const size_t z_payload_work_count = compress_subspace_none ? 0 :
            (compress_subspace_tq ? z_row_tq_sign_count : z_count);
        float* d_Rstack = nullptr;
        float* d_Z_qr = nullptr;
        float* d_tau_r = nullptr;
        float* d_tau_z = nullptr;
        float* d_work_r = nullptr;
        float* d_work_z = nullptr;
        float* d_B = nullptr;
        float* d_B_exact = nullptr;
        float* d_metric_Bk = nullptr;
        float* d_metric_VkS = nullptr;
        float* d_BT = nullptr;
        float* d_V = nullptr;
        float* d_S = nullptr;
        float* d_UtT = nullptr;
        float* d_work_svd = nullptr;
        float* d_rwork = nullptr;
        float* d_Ut_k = nullptr;
        float* d_payload_decode_work = nullptr;
        float* d_mpi_Z_local = nullptr;
        float* d_mpi_Z_global = nullptr;
        float* d_mpi_ZT_local = nullptr;
        float* d_mpi_ZT_global = nullptr;
        float* d_mpi_Z_work = nullptr;
        std::uint8_t* d_mpi_Z_codes = nullptr;
        std::uint8_t* d_mpi_Z_codes_recv = nullptr;
        float* d_mpi_Z_norms = nullptr;
        float* d_mpi_Z_norms_recv = nullptr;
        int* d_mpi_Z_qjl_signs = nullptr;
        int* d_mpi_Z_qjl_signs_recv = nullptr;
        std::uint8_t* d_mpi_B_codes = nullptr;
        std::uint8_t* d_mpi_B_codes_recv = nullptr;
        float* d_mpi_B_norms = nullptr;
        float* d_mpi_B_norms_recv = nullptr;
        int* d_mpi_B_qjl_signs = nullptr;
        int* d_mpi_B_qjl_signs_recv = nullptr;
        float* d_mpi_B_tq_work = nullptr;
        std::vector<float*> d_Bi_reduce_on_gpu0(opt.ngpus, nullptr);
        std::vector<std::uint8_t*> d_Bi_codes_on_gpu0(opt.ngpus, nullptr);
        std::vector<float*> d_Bi_norms_on_gpu0(opt.ngpus, nullptr);
        std::vector<int*> d_Bi_qjl_signs_on_gpu0(opt.ngpus, nullptr);
        int lwork_r = 0;
        int lwork_z = 0;
        int lwork_svd = 0;

        timer.tic();
        {
            NvtxRange range("create_gpu_handles");
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                w.dev = g;
                w.row0 = row0s[g];
                w.mi = rows[g];

                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUBLAS(cublasCreate(&w.blas));
                CHECK_CUSOLVER(cusolverDnCreate(&w.solver));
            }
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUBLAS(cublasCreate(&blas0));
            CHECK_CUSOLVER(cusolverDnCreate(&solver0));
        }
        double t_create_gpu_handles_ms = timer.toc_ms();

        timer.tic();
        {
            NvtxRange range("allocate_gpu_buffers");
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                CHECK_CUDA(cudaMalloc(&w.d_Ai, static_cast<size_t>(w.mi) * n * sizeof(float)));
                if (use_spectrum_decay) {
                    CHECK_CUDA(cudaMalloc(&w.d_Ubasis, static_cast<size_t>(w.mi) * opt.spectrum_rank * sizeof(float)));
                    CHECK_CUDA(cudaMalloc(&w.d_Vsigma, static_cast<size_t>(n) * opt.spectrum_rank * sizeof(float)));
                }
                CHECK_CUDA(cudaMalloc(&w.d_Omega, static_cast<size_t>(n) * l * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_Qbar, static_cast<size_t>(w.mi) * l * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_tau, static_cast<size_t>(l) * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_info, sizeof(int)));
                int lwork_geqrf = 0;
                int lwork_orgqr = 0;
                CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(w.solver, w.mi, l, w.d_Qbar, w.mi, &lwork_geqrf));
                CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(w.solver, w.mi, l, l, w.d_Qbar, w.mi, w.d_tau, &lwork_orgqr));
                w.qr_lwork = std::max(lwork_geqrf, lwork_orgqr);
                CHECK_CUDA(cudaMalloc(&w.d_qr_work, static_cast<size_t>(w.qr_lwork) * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_Ri, static_cast<size_t>(l) * l * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_Ti, static_cast<size_t>(l) * l * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_Qi, static_cast<size_t>(w.mi) * l * sizeof(float)));
                if (opt.subspace_iter > 0) {
                    CHECK_CUDA(cudaMalloc(&w.d_Zi, z_fp32_bytes));
                    CHECK_CUDA(cudaMalloc(&w.d_Z, z_fp32_bytes));
                }
                CHECK_CUDA(cudaMalloc(&w.d_Bi, static_cast<size_t>(l) * n * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_Bi_hat, bi_fp32_bytes));
                if (!compress_b_none) {
                    CHECK_CUDA(cudaMalloc(&w.d_Bi_codes, bi_code_bytes));
                    if (compress_b_tq) {
                        CHECK_CUDA(cudaMalloc(&w.d_Bi_norms, bi_norm_bytes));
                    }
                    if (compress_b_tq || compress_b_qjl) {
                        CHECK_CUDA(cudaMalloc(&w.d_Bi_tq_work, bi_column_tq_sign_count * sizeof(float)));
                    }
                    if (compress_b_qjl) {
                        CHECK_CUDA(cudaMalloc(&w.d_Bi_qjl_signs, bi_qjl_sign_bytes));
                    }
                }
            }
            CHECK_CUDA(cudaSetDevice(0));
            for (int g = 1; g < opt.ngpus; ++g) {
                CHECK_CUDA(cudaMalloc(&d_Bi_reduce_on_gpu0[g], bi_fp32_bytes));
                if (!compress_b_none) {
                    CHECK_CUDA(cudaMalloc(&d_Bi_codes_on_gpu0[g], bi_code_bytes));
                    if (compress_b_tq) {
                        CHECK_CUDA(cudaMalloc(&d_Bi_norms_on_gpu0[g], bi_norm_bytes));
                    }
                    if (compress_b_qjl) {
                        CHECK_CUDA(cudaMalloc(&d_Bi_qjl_signs_on_gpu0[g], bi_qjl_sign_bytes));
                    }
                }
            }
            CHECK_CUDA(cudaMalloc(&d_info0, sizeof(int)));
            CHECK_CUDA(cudaMalloc(&d_Rstack, static_cast<size_t>(r_rows) * l * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_tau_r, static_cast<size_t>(l) * sizeof(float)));
            int lwork_r_geqrf = 0;
            int lwork_r_orgqr = 0;
            CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver0, r_rows, l, d_Rstack, r_rows, &lwork_r_geqrf));
            CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(solver0, r_rows, l, l, d_Rstack, r_rows, d_tau_r, &lwork_r_orgqr));
            lwork_r = std::max(lwork_r_geqrf, lwork_r_orgqr);
            CHECK_CUDA(cudaMalloc(&d_work_r, static_cast<size_t>(lwork_r) * sizeof(float)));

            if (opt.stabilize_subspace_z && opt.subspace_iter > 0) {
                CHECK_CUDA(cudaMalloc(&d_Z_qr, z_fp32_bytes));
                CHECK_CUDA(cudaMalloc(&d_tau_z, static_cast<size_t>(l) * sizeof(float)));
                int lwork_z_geqrf = 0;
                int lwork_z_orgqr = 0;
                CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver0, n, l, d_Z_qr, n, &lwork_z_geqrf));
                CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(solver0, n, l, l, d_Z_qr, n, d_tau_z, &lwork_z_orgqr));
                lwork_z = std::max(lwork_z_geqrf, lwork_z_orgqr);
                CHECK_CUDA(cudaMalloc(&d_work_z, static_cast<size_t>(lwork_z) * sizeof(float)));
            }

            if (opt.subspace_iter > 0 && !compress_subspace_none) {
                CHECK_CUDA(cudaMalloc(&d_mpi_Z_local, z_fp32_bytes));
                CHECK_CUDA(cudaMalloc(&d_mpi_Z_global, z_fp32_bytes));
                CHECK_CUDA(cudaMalloc(&d_mpi_ZT_local, z_fp32_bytes));
                CHECK_CUDA(cudaMalloc(&d_mpi_ZT_global, z_fp32_bytes));
                CHECK_CUDA(cudaMalloc(&d_mpi_Z_work, z_payload_work_count * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_mpi_Z_codes, z_code_bytes));
                CHECK_CUDA(cudaMalloc(&d_mpi_Z_codes_recv, z_code_bytes));
                if (compress_subspace_lloyd_tq) {
                    CHECK_CUDA(cudaMalloc(&d_mpi_Z_norms, z_norm_bytes));
                    CHECK_CUDA(cudaMalloc(&d_mpi_Z_norms_recv, z_norm_bytes));
                }
                if (compress_subspace_qjl) {
                    CHECK_CUDA(cudaMalloc(&d_mpi_Z_qjl_signs, z_qjl_sign_bytes));
                    CHECK_CUDA(cudaMalloc(&d_mpi_Z_qjl_signs_recv, z_qjl_sign_bytes));
                }
            }

            CHECK_CUDA(cudaMalloc(&d_B, bi_fp32_bytes));
            if (need_global_b_metric) {
                CHECK_CUDA(cudaMalloc(&d_B_exact, bi_fp32_bytes));
            }
            if (!compress_b_none) {
                CHECK_CUDA(cudaMalloc(&d_payload_decode_work, bi_payload_work_count * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_mpi_B_codes, bi_code_bytes));
                CHECK_CUDA(cudaMalloc(&d_mpi_B_tq_work, bi_column_tq_sign_count * sizeof(float)));
                if (compress_b_tq) {
                    CHECK_CUDA(cudaMalloc(&d_mpi_B_norms, bi_norm_bytes));
                }
                if (compress_b_qjl) {
                    CHECK_CUDA(cudaMalloc(&d_mpi_B_qjl_signs, bi_qjl_sign_bytes));
                }
                if (is_root) {
                    CHECK_CUDA(cudaMalloc(&d_mpi_B_codes_recv, bi_code_bytes));
                    if (compress_b_tq) {
                        CHECK_CUDA(cudaMalloc(&d_mpi_B_norms_recv, bi_norm_bytes));
                    }
                    if (compress_b_qjl) {
                        CHECK_CUDA(cudaMalloc(&d_mpi_B_qjl_signs_recv, bi_qjl_sign_bytes));
                    }
                }
            }
            CHECK_CUDA(cudaMalloc(&d_BT, static_cast<size_t>(n) * l * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_S, static_cast<size_t>(l) * sizeof(float)));
            if (need_svd_vectors) {
                CHECK_CUDA(cudaMalloc(&d_V, static_cast<size_t>(n) * l * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_UtT, static_cast<size_t>(l) * l * sizeof(float)));
            }
            if (!opt.skip_form_u) {
                CHECK_CUDA(cudaMalloc(&d_Ut_k, static_cast<size_t>(l) * k * sizeof(float)));
            }
            if (need_global_b_metric && is_root) {
                CHECK_CUDA(cudaMalloc(&d_metric_Bk, bi_fp32_bytes));
            }
            if (need_compressed_safe_error && is_root) {
                CHECK_CUDA(cudaMalloc(&d_metric_VkS, static_cast<size_t>(l) * k * sizeof(float)));
            }
            CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(solver0, n, l, &lwork_svd));
            CHECK_CUDA(cudaMalloc(&d_work_svd, static_cast<size_t>(lwork_svd) * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_rwork, static_cast<size_t>(5) * l * sizeof(float)));
        }
        double t_allocate_gpu_buffers_ms = timer.toc_ms();

        double t_generate_test_matrix_ms = 0.0;
        if (use_spectrum_decay) {
            timer.tic();
            NvtxRange range("generate_spectrum_test_matrix");
            const int decay_mode_id =
                (opt.spectrum_decay_mode == "polynomial") ? 1 :
                (opt.spectrum_decay_mode == "exponential") ? 2 : 0;
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                dim3 block(16, 16);
                dim3 grid_u((w.mi + block.x - 1) / block.x,
                            (opt.spectrum_rank + block.y - 1) / block.y);
                fill_dct_basis_colmajor_kernel<<<grid_u, block>>>(
                    w.d_Ubasis, w.mi, opt.spectrum_rank, w.row0, m);
                CHECK_CUDA(cudaGetLastError());

                dim3 grid_v((n + block.x - 1) / block.x,
                            (opt.spectrum_rank + block.y - 1) / block.y);
                fill_dct_basis_scaled_colmajor_kernel<<<grid_v, block>>>(
                    w.d_Vsigma, n, opt.spectrum_rank, n, decay_mode_id, opt.spectrum_decay_param);
                CHECK_CUDA(cudaGetLastError());

                const float alpha = 1.0f;
                const float beta = 0.0f;
                CHECK_CUBLAS(cublasSgemm(
                    w.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                    w.mi, n, opt.spectrum_rank,
                    &alpha,
                    w.d_Ubasis, w.mi,
                    w.d_Vsigma, n,
                    &beta,
                    w.d_Ai, w.mi));
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                CHECK_CUDA(cudaSetDevice(works[g].dev));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            t_generate_test_matrix_ms = timer.toc_ms();
        }
        double t_setup_gpu_resources_ms =
            t_create_gpu_handles_ms + t_allocate_gpu_buffers_ms + t_generate_test_matrix_ms;

        std::vector<double> repeat_compute_ms;
        std::vector<double> repeat_pipeline_ms;
        std::vector<double> repeat_b_relative_error;
        std::vector<double> repeat_global_b_relative_error;
        std::vector<double> repeat_final_error;
        repeat_compute_ms.reserve(opt.repeat);
        repeat_pipeline_ms.reserve(opt.repeat);
        repeat_b_relative_error.reserve(opt.repeat);
        repeat_global_b_relative_error.reserve(opt.repeat);
        repeat_final_error.reserve(opt.repeat);
        constexpr int kStageCount = 8;
        const std::array<const char*, kStageCount> stage_names = {
            "Local Projection (Yi)",
            "Local QR Yi",
            "TSQR R Reduce (to GPU0)",
            "Form Distributed Qi",
            "Subspace Iteration",
            "Build Reduce Bi",
            "SVD B (on GPU0)",
            "Form Distributed Ui",
        };
        std::array<std::vector<double>, kStageCount> repeat_stage_ms;
        if (is_root) {
            for (auto& samples : repeat_stage_ms) {
                samples.reserve(opt.repeat);
            }
        }
        const bool vary_seed_per_repeat = opt.check_error || opt.check_b_error;

        for (int repeat_idx = 0; repeat_idx < opt.repeat; ++repeat_idx) {
        const unsigned repeat_seed = vary_seed_per_repeat ?
            opt.seed + static_cast<unsigned>(repeat_idx) :
            opt.seed;
        const bool print_repeat_detail =
            !opt.summary_only &&
            (repeat_idx == 0 ||
            repeat_idx + 1 == opt.repeat);
        if (is_root && print_repeat_detail) {
            std::cout << "\nRepeat " << (repeat_idx + 1) << "/" << opt.repeat << "\n";
        }

        timer.tic();
        double current_A_norm2 = h_A_norm2;
        {
            NvtxRange range("local_projection_Yi");
            std::vector<std::vector<float>> h_A_blocks_repeat;
            std::vector<float> h_Omega_repeat;
            const std::vector<std::vector<float>>* h_A_blocks_src = &h_A_blocks;
            const std::vector<float>* h_Omega_src = &h_Omega;
            if (!opt.device_random_input && vary_seed_per_repeat) {
                if (!use_spectrum_decay) {
                    h_A_blocks_repeat = make_random_row_blocks(
                        rows, n, repeat_seed, 1.0f / std::sqrt(static_cast<float>(m)), nullptr);
                    h_A_blocks_src = &h_A_blocks_repeat;
                }
                h_Omega_repeat = make_random_matrix(n, l, repeat_seed + 1, 1.0f);
                h_Omega_src = &h_Omega_repeat;
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                if (opt.device_random_input) {
                    dim3 block(16, 16);
                    if (!use_spectrum_decay) {
                        dim3 grid_a((w.mi + block.x - 1) / block.x, (n + block.y - 1) / block.y);
                        const float a_scale = std::sqrt(3.0f) / std::sqrt(static_cast<float>(m));
                        fill_random_colmajor_kernel<<<grid_a, block>>>(
                            w.d_Ai, w.mi, n, w.row0, repeat_seed, a_scale);
                        CHECK_CUDA(cudaGetLastError());
                    }

                    dim3 grid_omega((n + block.x - 1) / block.x, (l + block.y - 1) / block.y);
                    fill_random_colmajor_kernel<<<grid_omega, block>>>(
                        w.d_Omega, n, l, 0, repeat_seed + 1, std::sqrt(3.0f));
                    CHECK_CUDA(cudaGetLastError());
                } else {
                    if (!use_spectrum_decay) {
                        CHECK_CUDA(cudaMemcpy(w.d_Ai, (*h_A_blocks_src)[g].data(), static_cast<size_t>(w.mi) * n * sizeof(float), cudaMemcpyHostToDevice));
                    }
                    CHECK_CUDA(cudaMemcpy(w.d_Omega, h_Omega_src->data(), static_cast<size_t>(n) * l * sizeof(float), cudaMemcpyHostToDevice));
                }

                const float alpha = 1.0f;
                const float beta = 0.0f;
                CHECK_CUBLAS(cublasSgemm(
                    w.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    w.mi, l, n,
                    &alpha,
                    w.d_Ai, w.mi,
                    w.d_Omega, n,
                    &beta,
                    w.d_Qbar, w.mi));
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
            if (opt.check_error && !use_spectrum_decay) {
                double local_A_norm2 = 0.0;
                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    float block_norm = 0.0f;
                    CHECK_CUBLAS(cublasSnrm2(
                        w.blas,
                        static_cast<int>(static_cast<size_t>(w.mi) * n),
                        w.d_Ai,
                        1,
                        &block_norm));
                    local_A_norm2 += static_cast<double>(block_norm) * block_norm;
                }
                double global_A_norm2 = 0.0;
                CHECK_MPI(MPI_Reduce(
                    &local_A_norm2,
                    &global_A_norm2,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    0,
                    MPI_COMM_WORLD));
                if (is_root) {
                    current_A_norm2 = global_A_norm2;
                }
            }
        }
        double t_local_projection_ms = timer.toc_ms();

        timer.tic();
        std::vector<float> h_Rstack_local(static_cast<size_t>(local_r_rows) * l, 0.0f);
        std::vector<float> h_Rstack;
        if (is_root) {
            h_Rstack.resize(static_cast<size_t>(r_rows) * l, 0.0f);
        }
        size_t r_payload_bytes = 0;
        {
            NvtxRange range("local_qr_Yi");
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));

                CHECK_CUSOLVER(cusolverDnSgeqrf(
                    w.solver, w.mi, l, w.d_Qbar, w.mi, w.d_tau, w.d_qr_work, w.qr_lwork, w.d_info));
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                CHECK_CUDA(cudaSetDevice(works[g].dev));
                CHECK_CUDA(cudaDeviceSynchronize());
                check_solver_info(works[g].d_info, "local geqrf");
            }

            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                dim3 block(16, 16);
                dim3 grid((l + block.x - 1) / block.x, (l + block.y - 1) / block.y);
                extract_upper_kernel<<<grid, block>>>(w.d_Qbar, w.mi, w.d_Ri, l);
                CHECK_CUDA(cudaGetLastError());
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                std::vector<float> h_Ri(static_cast<size_t>(l) * l);
                CHECK_CUDA(cudaMemcpy(h_Ri.data(), w.d_Ri, static_cast<size_t>(l) * l * sizeof(float), cudaMemcpyDeviceToHost));
                for (int col = 0; col < l; ++col) {
                    for (int row = 0; row <= col; ++row) {
                        h_Rstack_local[static_cast<size_t>(col) * local_r_rows + g * l + row] =
                            h_Ri[static_cast<size_t>(col) * l + row];
                    }
                }
                r_payload_bytes += static_cast<size_t>(l) * l * sizeof(float);
            }

            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                CHECK_CUSOLVER(cusolverDnSorgqr(
                    w.solver, w.mi, l, l, w.d_Qbar, w.mi, w.d_tau, w.d_qr_work, w.qr_lwork, w.d_info));
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                CHECK_CUDA(cudaSetDevice(works[g].dev));
                CHECK_CUDA(cudaDeviceSynchronize());
                check_solver_info(works[g].d_info, "local orgqr");
            }
        }
        double t_local_qr_ms = timer.toc_ms();

        timer.tic();
        std::vector<float> h_Tstack;
        {
            NvtxRange range("tsqr_R_reduce_gpu0");
            CHECK_CUDA(cudaSetDevice(0));

            CHECK_MPI(MPI_Gather(
                h_Rstack_local.data(),
                local_r_rows * l,
                MPI_FLOAT,
                is_root ? h_Rstack.data() : nullptr,
                local_r_rows * l,
                MPI_FLOAT,
                0,
                MPI_COMM_WORLD));

            if (is_root) {
            CHECK_CUDA(cudaMemcpy(d_Rstack, h_Rstack.data(), static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyHostToDevice));

            CHECK_CUSOLVER(cusolverDnSgeqrf(solver0, r_rows, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
            CHECK_CUDA(cudaDeviceSynchronize());
            check_solver_info(d_info0, "TSQR geqrf");
            CHECK_CUSOLVER(cusolverDnSorgqr(solver0, r_rows, l, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
            CHECK_CUDA(cudaDeviceSynchronize());
            check_solver_info(d_info0, "TSQR orgqr");

            h_Tstack.resize(static_cast<size_t>(r_rows) * l);
            CHECK_CUDA(cudaMemcpy(h_Tstack.data(), d_Rstack, static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyDeviceToHost));
            } else {
                h_Tstack.resize(static_cast<size_t>(r_rows) * l);
            }
            CHECK_MPI(MPI_Bcast(h_Tstack.data(), r_rows * l, MPI_FLOAT, 0, MPI_COMM_WORLD));
        }
        double t_tsqr_reduce_ms = timer.toc_ms();

        timer.tic();
        {
            NvtxRange range("form_distributed_Qi");
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                std::vector<float> h_Ti(static_cast<size_t>(l) * l);
                for (int col = 0; col < l; ++col) {
                    const int global_g = mpi.rank * opt.ngpus + g;
                    const float* src = h_Tstack.data() + static_cast<size_t>(col) * r_rows + global_g * l;
                    float* dst = h_Ti.data() + static_cast<size_t>(col) * l;
                    std::copy(src, src + l, dst);
                }
                CHECK_CUDA(cudaMemcpy(w.d_Ti, h_Ti.data(), static_cast<size_t>(l) * l * sizeof(float), cudaMemcpyHostToDevice));

                const float alpha = 1.0f;
                const float beta = 0.0f;
                CHECK_CUBLAS(cublasSgemm(
                    w.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    w.mi, l, l,
                    &alpha,
                    w.d_Qbar, w.mi,
                    w.d_Ti, l,
                    &beta,
                    w.d_Qi, w.mi));
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                CHECK_CUDA(cudaSetDevice(g));
                CHECK_CUDA(cudaDeviceSynchronize());
            }
        }
        double t_form_distributed_q_ms = timer.toc_ms();

        size_t z_fp32_payload_bytes = 0;
        size_t z_transmitted_payload_bytes = 0;
        double t_compress_subspace_ms = 0.0;
        double t_subspace_iter_ms = 0.0;
        if (opt.subspace_iter > 0) {
            timer.tic();
            NvtxRange range("subspace_iteration");
            std::vector<float> h_Z_local(z_count);
            std::vector<float> h_Z_global(z_count);
            for (int iter = 0; iter < opt.subspace_iter; ++iter) {
                std::fill(h_Z_local.begin(), h_Z_local.end(), 0.0f);

                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    CHECK_CUBLAS(cublasSgemm(
                        w.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, l, w.mi,
                        &alpha,
                        w.d_Ai, w.mi,
                        w.d_Qi, w.mi,
                        &beta,
                        w.d_Zi, n));
                }
                for (int g = 0; g < opt.ngpus; ++g) {
                    CHECK_CUDA(cudaSetDevice(works[g].dev));
                    CHECK_CUDA(cudaDeviceSynchronize());
                }

                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    std::vector<float> h_Zi(z_count);
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    CHECK_CUDA(cudaMemcpy(h_Zi.data(), w.d_Zi, z_fp32_bytes, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < z_count; ++i) {
                        h_Z_local[i] += h_Zi[i];
                    }
                }

                z_fp32_payload_bytes += z_fp32_bytes;
                if (compress_subspace_none) {
                    CHECK_MPI(MPI_Allreduce(
                        h_Z_local.data(),
                        h_Z_global.data(),
                        checked_mpi_count(z_count, "subspace Z MPI allreduce count"),
                        MPI_FLOAT,
                        MPI_SUM,
                        MPI_COMM_WORLD));
                    z_transmitted_payload_bytes += z_fp32_bytes;
                } else {
                    Timer subspace_compress_timer;
                    subspace_compress_timer.tic();
                    CHECK_CUDA(cudaSetDevice(0));
                    CHECK_CUDA(cudaMemcpy(d_mpi_Z_local, h_Z_local.data(), z_fp32_bytes, cudaMemcpyHostToDevice));
                    {
                        dim3 block(16, 16);
                        dim3 grid((n + block.x - 1) / block.x, (l + block.y - 1) / block.y);
                        transpose_colmajor_kernel<<<grid, block>>>(d_mpi_Z_local, d_mpi_ZT_local, n, l);
                        CHECK_CUDA(cudaGetLastError());
                    }

                    turboquant::DeviceCompressedBlock local_Z_compressed;
                    if (compress_subspace_tq) {
                        local_Z_compressed = turboquant::quantize_fp32_device_column_tq_to_device_payload(
                            d_mpi_ZT_local,
                            l,
                            n,
                            subspace_quant_options,
                            d_mpi_Z_codes,
                            compress_subspace_lloyd_tq ? d_mpi_Z_norms : nullptr,
                            d_mpi_Z_work,
                            nullptr,
                            d_mpi_Z_qjl_signs);
                    } else {
                        local_Z_compressed = turboquant::quantize_fp32_device_block_to_device_payload(
                            d_mpi_ZT_local,
                            l,
                            n,
                            subspace_quant_options,
                            d_mpi_Z_codes,
                            d_mpi_Z_qjl_signs);
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());
                    t_compress_subspace_ms += subspace_compress_timer.toc_ms();
                    z_transmitted_payload_bytes += local_Z_compressed.payload_bytes();

                    std::vector<std::uint8_t> h_Z_codes(z_code_bytes);
                    std::vector<std::uint8_t> h_Z_codes_all(z_code_bytes * static_cast<size_t>(mpi.size));
                    CHECK_CUDA(cudaMemcpy(h_Z_codes.data(), d_mpi_Z_codes, z_code_bytes, cudaMemcpyDeviceToHost));
                    CHECK_MPI(MPI_Allgather(
                        h_Z_codes.data(),
                        checked_mpi_count(z_code_bytes, "compressed subspace Z code MPI allgather send count"),
                        MPI_UNSIGNED_CHAR,
                        h_Z_codes_all.data(),
                        checked_mpi_count(z_code_bytes, "compressed subspace Z code MPI allgather receive count"),
                        MPI_UNSIGNED_CHAR,
                        MPI_COMM_WORLD));

                    std::vector<float> h_Z_norms;
                    std::vector<float> h_Z_norms_all;
                    if (compress_subspace_lloyd_tq) {
                        h_Z_norms.resize(static_cast<size_t>(n));
                        h_Z_norms_all.resize(static_cast<size_t>(n) * mpi.size);
                        CHECK_CUDA(cudaMemcpy(
                            h_Z_norms.data(),
                            d_mpi_Z_norms,
                            z_norm_bytes,
                            cudaMemcpyDeviceToHost));
                        CHECK_MPI(MPI_Allgather(
                            h_Z_norms.data(),
                            n,
                            MPI_FLOAT,
                            h_Z_norms_all.data(),
                            n,
                            MPI_FLOAT,
                            MPI_COMM_WORLD));
                    }

                    std::vector<int> h_Z_qjl_signs;
                    std::vector<int> h_Z_qjl_signs_all;
                    if (compress_subspace_qjl) {
                        h_Z_qjl_signs.resize(static_cast<size_t>(subspace_quant_options.qjl_dim));
                        h_Z_qjl_signs_all.resize(static_cast<size_t>(subspace_quant_options.qjl_dim) * mpi.size);
                        CHECK_CUDA(cudaMemcpy(
                            h_Z_qjl_signs.data(),
                            d_mpi_Z_qjl_signs,
                            z_qjl_sign_bytes,
                            cudaMemcpyDeviceToHost));
                        CHECK_MPI(MPI_Allgather(
                            h_Z_qjl_signs.data(),
                            subspace_quant_options.qjl_dim,
                            MPI_INT,
                            h_Z_qjl_signs_all.data(),
                            subspace_quant_options.qjl_dim,
                            MPI_INT,
                            MPI_COMM_WORLD));
                    }

                    const float local_Z_scale = local_Z_compressed.scale;
                    const float local_Z_residual_norm = local_Z_compressed.residual_norm;
                    std::vector<float> h_Z_scales(mpi.size);
                    std::vector<float> h_Z_residual_norms(mpi.size);
                    CHECK_MPI(MPI_Allgather(
                        &local_Z_scale,
                        1,
                        MPI_FLOAT,
                        h_Z_scales.data(),
                        1,
                        MPI_FLOAT,
                        MPI_COMM_WORLD));
                    CHECK_MPI(MPI_Allgather(
                        &local_Z_residual_norm,
                        1,
                        MPI_FLOAT,
                        h_Z_residual_norms.data(),
                        1,
                        MPI_FLOAT,
                        MPI_COMM_WORLD));

                    CHECK_CUDA(cudaSetDevice(0));
                    CHECK_CUDA(cudaMemset(d_mpi_ZT_global, 0, z_fp32_bytes));
                    for (int r = 0; r < mpi.size; ++r) {
                        CHECK_CUDA(cudaMemcpy(
                            d_mpi_Z_codes_recv,
                            h_Z_codes_all.data() + static_cast<size_t>(r) * z_code_bytes,
                            z_code_bytes,
                            cudaMemcpyHostToDevice));
                        turboquant::DeviceCompressedBlock rank_Z = local_Z_compressed;
                        rank_Z.d_codes = d_mpi_Z_codes_recv;
                        rank_Z.scale = h_Z_scales[r];
                        rank_Z.residual_norm = h_Z_residual_norms[r];
                        if (compress_subspace_lloyd_tq) {
                            CHECK_CUDA(cudaMemcpy(
                                d_mpi_Z_norms_recv,
                                h_Z_norms_all.data() + static_cast<size_t>(r) * n,
                                z_norm_bytes,
                                cudaMemcpyHostToDevice));
                            rank_Z.d_norms = d_mpi_Z_norms_recv;
                        }
                        if (compress_subspace_qjl) {
                            CHECK_CUDA(cudaMemcpy(
                                d_mpi_Z_qjl_signs_recv,
                                h_Z_qjl_signs_all.data() + static_cast<size_t>(r) * subspace_quant_options.qjl_dim,
                                z_qjl_sign_bytes,
                                cudaMemcpyHostToDevice));
                            rank_Z.d_qjl_signs = d_mpi_Z_qjl_signs_recv;
                        }
                        if (compress_subspace_tq) {
                            turboquant::dequantize_column_tq_payload_add_to_fp32(
                                rank_Z,
                                d_mpi_ZT_global,
                                d_mpi_Z_work);
                        } else {
                            turboquant::dequantize_device_payload_to_fp32(
                                rank_Z,
                                d_mpi_Z_work);
                            const int threads = 256;
                            const int blocks = static_cast<int>((z_count + threads - 1) / threads);
                            add_kernel<<<blocks, threads>>>(d_mpi_ZT_global, d_mpi_Z_work, z_count);
                            CHECK_CUDA(cudaGetLastError());
                        }
                    }
                    {
                        dim3 block(16, 16);
                        dim3 grid((l + block.x - 1) / block.x, (n + block.y - 1) / block.y);
                        transpose_colmajor_kernel<<<grid, block>>>(d_mpi_ZT_global, d_mpi_Z_global, l, n);
                        CHECK_CUDA(cudaGetLastError());
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());
                    CHECK_CUDA(cudaMemcpy(h_Z_global.data(), d_mpi_Z_global, z_fp32_bytes, cudaMemcpyDeviceToHost));
                }

                if (opt.stabilize_subspace_z) {
                    NvtxRange z_qr_range("orthogonalize_subspace_Z");
                    CHECK_CUDA(cudaSetDevice(0));
                    CHECK_CUDA(cudaMemcpy(d_Z_qr, h_Z_global.data(), z_fp32_bytes, cudaMemcpyHostToDevice));
                    CHECK_CUSOLVER(cusolverDnSgeqrf(
                        solver0, n, l, d_Z_qr, n, d_tau_z, d_work_z, lwork_z, d_info0));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    check_solver_info(d_info0, "subspace Z geqrf");
                    CHECK_CUSOLVER(cusolverDnSorgqr(
                        solver0, n, l, l, d_Z_qr, n, d_tau_z, d_work_z, lwork_z, d_info0));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    check_solver_info(d_info0, "subspace Z orgqr");
                    CHECK_CUDA(cudaMemcpy(h_Z_global.data(), d_Z_qr, z_fp32_bytes, cudaMemcpyDeviceToHost));
                }

                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    CHECK_CUDA(cudaMemcpy(w.d_Z, h_Z_global.data(), z_fp32_bytes, cudaMemcpyHostToDevice));
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    CHECK_CUBLAS(cublasSgemm(
                        w.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        w.mi, l, n,
                        &alpha,
                        w.d_Ai, w.mi,
                        w.d_Z, n,
                        &beta,
                        w.d_Qbar, w.mi));
                }
                for (int g = 0; g < opt.ngpus; ++g) {
                    CHECK_CUDA(cudaSetDevice(works[g].dev));
                    CHECK_CUDA(cudaDeviceSynchronize());
                }

                std::fill(h_Rstack_local.begin(), h_Rstack_local.end(), 0.0f);
                if (is_root) {
                    std::fill(h_Rstack.begin(), h_Rstack.end(), 0.0f);
                }
                h_Tstack.clear();

                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    CHECK_CUSOLVER(cusolverDnSgeqrf(
                        w.solver, w.mi, l, w.d_Qbar, w.mi, w.d_tau, w.d_qr_work, w.qr_lwork, w.d_info));
                }
                for (int g = 0; g < opt.ngpus; ++g) {
                    CHECK_CUDA(cudaSetDevice(works[g].dev));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    check_solver_info(works[g].d_info, "subspace local geqrf");
                }

                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    dim3 block(16, 16);
                    dim3 grid((l + block.x - 1) / block.x, (l + block.y - 1) / block.y);
                    extract_upper_kernel<<<grid, block>>>(w.d_Qbar, w.mi, w.d_Ri, l);
                    CHECK_CUDA(cudaGetLastError());
                }
                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    std::vector<float> h_Ri(static_cast<size_t>(l) * l);
                    CHECK_CUDA(cudaMemcpy(h_Ri.data(), w.d_Ri, static_cast<size_t>(l) * l * sizeof(float), cudaMemcpyDeviceToHost));
                    for (int col = 0; col < l; ++col) {
                        for (int row = 0; row <= col; ++row) {
                            h_Rstack_local[static_cast<size_t>(col) * local_r_rows + g * l + row] =
                                h_Ri[static_cast<size_t>(col) * l + row];
                        }
                    }
                    r_payload_bytes += static_cast<size_t>(l) * l * sizeof(float);
                }

                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    CHECK_CUSOLVER(cusolverDnSorgqr(
                        w.solver, w.mi, l, l, w.d_Qbar, w.mi, w.d_tau, w.d_qr_work, w.qr_lwork, w.d_info));
                }
                for (int g = 0; g < opt.ngpus; ++g) {
                    CHECK_CUDA(cudaSetDevice(works[g].dev));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    check_solver_info(works[g].d_info, "subspace local orgqr");
                }

                CHECK_CUDA(cudaSetDevice(0));
                CHECK_MPI(MPI_Gather(
                    h_Rstack_local.data(),
                    local_r_rows * l,
                    MPI_FLOAT,
                    is_root ? h_Rstack.data() : nullptr,
                    local_r_rows * l,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD));

                if (is_root) {
                    CHECK_CUDA(cudaMemcpy(d_Rstack, h_Rstack.data(), static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyHostToDevice));
                    CHECK_CUSOLVER(cusolverDnSgeqrf(solver0, r_rows, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    check_solver_info(d_info0, "subspace TSQR geqrf");
                    CHECK_CUSOLVER(cusolverDnSorgqr(solver0, r_rows, l, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    check_solver_info(d_info0, "subspace TSQR orgqr");

                    h_Tstack.resize(static_cast<size_t>(r_rows) * l);
                    CHECK_CUDA(cudaMemcpy(h_Tstack.data(), d_Rstack, static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyDeviceToHost));
                } else {
                    h_Tstack.resize(static_cast<size_t>(r_rows) * l);
                }
                CHECK_MPI(MPI_Bcast(h_Tstack.data(), r_rows * l, MPI_FLOAT, 0, MPI_COMM_WORLD));

                for (int g = 0; g < opt.ngpus; ++g) {
                    DeviceWork& w = works[g];
                    CHECK_CUDA(cudaSetDevice(w.dev));
                    std::vector<float> h_Ti(static_cast<size_t>(l) * l);
                    for (int col = 0; col < l; ++col) {
                        const int global_g = mpi.rank * opt.ngpus + g;
                        const float* src = h_Tstack.data() + static_cast<size_t>(col) * r_rows + global_g * l;
                        float* dst = h_Ti.data() + static_cast<size_t>(col) * l;
                        std::copy(src, src + l, dst);
                    }
                    CHECK_CUDA(cudaMemcpy(w.d_Ti, h_Ti.data(), static_cast<size_t>(l) * l * sizeof(float), cudaMemcpyHostToDevice));

                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    CHECK_CUBLAS(cublasSgemm(
                        w.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        w.mi, l, l,
                        &alpha,
                        w.d_Qbar, w.mi,
                        w.d_Ti, l,
                        &beta,
                        w.d_Qi, w.mi));
                }
                for (int g = 0; g < opt.ngpus; ++g) {
                    CHECK_CUDA(cudaSetDevice(g));
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            }
            t_subspace_iter_ms = timer.toc_ms();
        }

        timer.tic();
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMemset(d_B, 0, bi_fp32_bytes));
        if (need_global_b_metric) {
            CHECK_CUDA(cudaMemset(d_B_exact, 0, bi_fp32_bytes));
        }

        size_t b_fp32_payload_bytes = 0;
        size_t b_transmitted_payload_bytes = 0;
        size_t mpi_b_fp32_payload_bytes = 0;
        size_t mpi_b_transmitted_payload_bytes = 0;
        double t_compress_b_ms = 0.0;
        long double b_error_norm2 = 0.0L;
        long double b_ref_norm2 = 0.0L;
        {
            NvtxRange range("build_reduce_Bi");
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                const float alpha = 1.0f;
                const float beta = 0.0f;
                CHECK_CUBLAS(cublasSgemm(
                    w.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    l, n, w.mi,
                    &alpha,
                    w.d_Qi, w.mi,
                    w.d_Ai, w.mi,
                    &beta,
                    w.d_Bi, l));
            }
            for (int g = 0; g < opt.ngpus; ++g) {
                CHECK_CUDA(cudaSetDevice(works[g].dev));
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                b_fp32_payload_bytes += bi_fp32_bytes;
                std::vector<float> h_Bi_ref;
                if (opt.check_b_error) {
                    h_Bi_ref.resize(b_count);
                    CHECK_CUDA(cudaMemcpy(h_Bi_ref.data(), w.d_Bi, bi_fp32_bytes, cudaMemcpyDeviceToHost));
                }
                if (need_global_b_metric) {
                    CHECK_CUDA(cudaSetDevice(0));
                    float* d_Bi_exact_on_gpu0 = w.d_Bi;
                    if (w.dev != 0) {
                        CHECK_CUDA(cudaMemcpyPeer(
                            d_Bi_reduce_on_gpu0[g], 0, w.d_Bi, w.dev, bi_fp32_bytes));
                        d_Bi_exact_on_gpu0 = d_Bi_reduce_on_gpu0[g];
                    }
                    const int threads = 256;
                    const int blocks = static_cast<int>((b_count + threads - 1) / threads);
                    add_kernel<<<blocks, threads>>>(d_B_exact, d_Bi_exact_on_gpu0, b_count);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    CHECK_CUDA(cudaSetDevice(w.dev));
                }

                float* d_reconstructed_Bi = w.d_Bi_hat;
                bool decoded_payload_on_gpu0 = false;
                Timer compress_timer;
                compress_timer.tic();
                size_t compressed_payload_bytes = bi_fp32_bytes;
                if (compress_b_none) {
                    d_reconstructed_Bi = w.d_Bi;
                    compressed_payload_bytes = bi_fp32_bytes;
                } else if (!compress_b_none) {
                    turboquant::DeviceCompressedBlock compressed;
                    {
                        NvtxRange compress_range("compress_Bi_payload");
                        if (b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
                            b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl) {
                            compressed = turboquant::quantize_fp32_device_column_tq_to_device_payload(
                                w.d_Bi,
                                l,
                                n,
                                b_quant_options,
                                w.d_Bi_codes,
                                compress_b_tq ? w.d_Bi_norms : nullptr,
                                w.d_Bi_tq_work,
                                nullptr,
                                w.d_Bi_qjl_signs);
                        } else {
                            compressed = turboquant::quantize_fp32_device_block_to_device_payload(
                                w.d_Bi,
                                l,
                                n,
                                b_quant_options,
                                w.d_Bi_codes,
                                w.d_Bi_qjl_signs);
                        }
                    }
                    compressed_payload_bytes = compressed.payload_bytes();

                    CHECK_CUDA(cudaSetDevice(0));
                    turboquant::DeviceCompressedBlock compressed_on_gpu0 = compressed;
                    float* d_payload_decode_output = w.d_Bi_hat;
                    if (w.dev != 0) {
                        CHECK_CUDA(cudaMemcpyPeer(
                            d_Bi_codes_on_gpu0[g], 0, w.d_Bi_codes, w.dev, bi_code_bytes));
                        compressed_on_gpu0.d_codes = d_Bi_codes_on_gpu0[g];
                        d_payload_decode_output = d_Bi_reduce_on_gpu0[g];
                        decoded_payload_on_gpu0 = true;
                        if (compress_b_tq) {
                            CHECK_CUDA(cudaMemcpyPeer(
                                d_Bi_norms_on_gpu0[g], 0, w.d_Bi_norms, w.dev, bi_norm_bytes));
                            compressed_on_gpu0.d_norms = d_Bi_norms_on_gpu0[g];
                        }
                        if (compress_b_qjl) {
                            CHECK_CUDA(cudaMemcpyPeer(
                                d_Bi_qjl_signs_on_gpu0[g], 0, w.d_Bi_qjl_signs, w.dev, bi_qjl_sign_bytes));
                            compressed_on_gpu0.d_qjl_signs = d_Bi_qjl_signs_on_gpu0[g];
                        }
                    }
                    {
                        NvtxRange decode_range("decode_Bi_payload_gpu0");
                        if (b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
                            b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl) {
                            if (opt.check_b_error) {
                                CHECK_CUDA(cudaMemset(d_payload_decode_output, 0, bi_fp32_bytes));
                                turboquant::dequantize_column_tq_payload_add_to_fp32(
                                    compressed_on_gpu0,
                                    d_payload_decode_output,
                                    d_payload_decode_work);
                            } else {
                                turboquant::dequantize_column_tq_payload_add_to_fp32(
                                    compressed_on_gpu0,
                                    d_B,
                                    d_payload_decode_work);
                                decoded_payload_on_gpu0 = true;
                                d_payload_decode_output = d_B;
                            }
                        } else {
                            turboquant::dequantize_device_payload_to_fp32(
                                compressed_on_gpu0,
                                d_payload_decode_output);
                        }
                    }
                    d_reconstructed_Bi = d_payload_decode_output;
                }
                t_compress_b_ms += compress_timer.toc_ms();

                b_transmitted_payload_bytes += compressed_payload_bytes;

                if (opt.check_b_error) {
                    std::vector<float> h_Bi_hat(b_count);
                    CHECK_CUDA(cudaMemcpy(h_Bi_hat.data(), d_reconstructed_Bi, bi_fp32_bytes, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < b_count; ++i) {
                        const long double ref = h_Bi_ref[i];
                        const long double diff = static_cast<long double>(h_Bi_hat[i]) - ref;
                        b_error_norm2 += diff * diff;
                        b_ref_norm2 += ref * ref;
                    }
                }

                if (decoded_payload_on_gpu0) {
                    continue;
                }

                CHECK_CUDA(cudaSetDevice(0));
                float* d_Bi_hat_on_gpu0 = compress_b_none ? w.d_Bi : d_reconstructed_Bi;
                if (w.dev != 0 && !decoded_payload_on_gpu0) {
                    d_Bi_hat_on_gpu0 = d_Bi_reduce_on_gpu0[g];
                    CHECK_CUDA(cudaMemcpyPeer(
                        d_Bi_hat_on_gpu0,
                        0,
                        compress_b_none ? w.d_Bi : d_reconstructed_Bi,
                        w.dev,
                        bi_fp32_bytes));
                }
                const int threads = 256;
                const int blocks = static_cast<int>((b_count + threads - 1) / threads);
                add_kernel<<<blocks, threads>>>(d_B, d_Bi_hat_on_gpu0, b_count);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());
            }

            mpi_b_fp32_payload_bytes = bi_fp32_bytes;
            if (compress_b_none) {
                mpi_b_transmitted_payload_bytes = bi_fp32_bytes;
                std::vector<float> h_B_local(b_count);
                std::vector<float> h_B_global;
                if (is_root) {
                    h_B_global.resize(b_count);
                }
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemcpy(h_B_local.data(), d_B, bi_fp32_bytes, cudaMemcpyDeviceToHost));
                CHECK_MPI(MPI_Reduce(
                    h_B_local.data(),
                    is_root ? h_B_global.data() : nullptr,
                    checked_mpi_count(b_count, "B FP32 MPI reduce count"),
                    MPI_FLOAT,
                    MPI_SUM,
                    0,
                    MPI_COMM_WORLD));
                if (is_root) {
                    CHECK_CUDA(cudaMemcpy(d_B, h_B_global.data(), bi_fp32_bytes, cudaMemcpyHostToDevice));
                }
            } else {
                Timer mpi_compress_timer;
                mpi_compress_timer.tic();
                CHECK_CUDA(cudaSetDevice(0));
                turboquant::DeviceCompressedBlock mpi_B_compressed;
                if (b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
                    b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl) {
                    mpi_B_compressed = turboquant::quantize_fp32_device_column_tq_to_device_payload(
                        d_B,
                        l,
                        n,
                        b_quant_options,
                        d_mpi_B_codes,
                        compress_b_tq ? d_mpi_B_norms : nullptr,
                        d_mpi_B_tq_work,
                        nullptr,
                        d_mpi_B_qjl_signs);
                } else {
                    mpi_B_compressed = turboquant::quantize_fp32_device_block_to_device_payload(
                        d_B,
                        l,
                        n,
                        b_quant_options,
                        d_mpi_B_codes,
                        d_mpi_B_qjl_signs);
                }
                CHECK_CUDA(cudaDeviceSynchronize());
                t_compress_b_ms += mpi_compress_timer.toc_ms();

                std::vector<std::uint8_t> h_mpi_B_codes(bi_code_bytes);
                CHECK_CUDA(cudaMemcpy(h_mpi_B_codes.data(), d_mpi_B_codes, bi_code_bytes, cudaMemcpyDeviceToHost));
                std::vector<std::uint8_t> h_mpi_B_codes_all;
                if (is_root) {
                    h_mpi_B_codes_all.resize(bi_code_bytes * static_cast<size_t>(mpi.size));
                }
                CHECK_MPI(MPI_Gather(
                    h_mpi_B_codes.data(),
                    checked_mpi_count(bi_code_bytes, "compressed B code MPI gather count"),
                    MPI_UNSIGNED_CHAR,
                    is_root ? h_mpi_B_codes_all.data() : nullptr,
                    checked_mpi_count(bi_code_bytes, "compressed B code MPI receive count"),
                    MPI_UNSIGNED_CHAR,
                    0,
                    MPI_COMM_WORLD));

                std::vector<float> h_mpi_B_norms;
                std::vector<float> h_mpi_B_norms_all;
                if (compress_b_tq) {
                    h_mpi_B_norms.resize(static_cast<size_t>(n));
                    CHECK_CUDA(cudaMemcpy(
                        h_mpi_B_norms.data(),
                        d_mpi_B_norms,
                        bi_norm_bytes,
                        cudaMemcpyDeviceToHost));
                    if (is_root) {
                        h_mpi_B_norms_all.resize(static_cast<size_t>(n) * mpi.size);
                    }
                    CHECK_MPI(MPI_Gather(
                        h_mpi_B_norms.data(),
                        n,
                        MPI_FLOAT,
                        is_root ? h_mpi_B_norms_all.data() : nullptr,
                        n,
                        MPI_FLOAT,
                        0,
                        MPI_COMM_WORLD));
                }

                std::vector<int> h_mpi_B_qjl_signs;
                std::vector<int> h_mpi_B_qjl_signs_all;
                if (compress_b_qjl) {
                    h_mpi_B_qjl_signs.resize(static_cast<size_t>(b_quant_options.qjl_dim));
                    CHECK_CUDA(cudaMemcpy(
                        h_mpi_B_qjl_signs.data(),
                        d_mpi_B_qjl_signs,
                        bi_qjl_sign_bytes,
                        cudaMemcpyDeviceToHost));
                    if (is_root) {
                        h_mpi_B_qjl_signs_all.resize(static_cast<size_t>(b_quant_options.qjl_dim) * mpi.size);
                    }
                    CHECK_MPI(MPI_Gather(
                        h_mpi_B_qjl_signs.data(),
                        b_quant_options.qjl_dim,
                        MPI_INT,
                        is_root ? h_mpi_B_qjl_signs_all.data() : nullptr,
                        b_quant_options.qjl_dim,
                        MPI_INT,
                        0,
                        MPI_COMM_WORLD));
                }

                const float local_mpi_B_scale = mpi_B_compressed.scale;
                const float local_mpi_B_residual_norm = mpi_B_compressed.residual_norm;
                std::vector<float> h_mpi_B_scales;
                std::vector<float> h_mpi_B_residual_norms;
                if (is_root) {
                    h_mpi_B_scales.resize(mpi.size);
                    h_mpi_B_residual_norms.resize(mpi.size);
                }
                CHECK_MPI(MPI_Gather(
                    &local_mpi_B_scale,
                    1,
                    MPI_FLOAT,
                    is_root ? h_mpi_B_scales.data() : nullptr,
                    1,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD));
                CHECK_MPI(MPI_Gather(
                    &local_mpi_B_residual_norm,
                    1,
                    MPI_FLOAT,
                    is_root ? h_mpi_B_residual_norms.data() : nullptr,
                    1,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD));

                mpi_b_transmitted_payload_bytes = mpi_B_compressed.payload_bytes();
                if (is_root) {
                    CHECK_CUDA(cudaSetDevice(0));
                    CHECK_CUDA(cudaMemset(d_B, 0, bi_fp32_bytes));
                    for (int r = 0; r < mpi.size; ++r) {
                        CHECK_CUDA(cudaMemcpy(
                            d_mpi_B_codes_recv,
                            h_mpi_B_codes_all.data() + static_cast<size_t>(r) * bi_code_bytes,
                            bi_code_bytes,
                            cudaMemcpyHostToDevice));
                        turboquant::DeviceCompressedBlock rank_B = mpi_B_compressed;
                        rank_B.d_codes = d_mpi_B_codes_recv;
                        rank_B.scale = h_mpi_B_scales[r];
                        rank_B.residual_norm = h_mpi_B_residual_norms[r];
                        if (compress_b_tq) {
                            CHECK_CUDA(cudaMemcpy(
                                d_mpi_B_norms_recv,
                                h_mpi_B_norms_all.data() + static_cast<size_t>(r) * n,
                                bi_norm_bytes,
                                cudaMemcpyHostToDevice));
                            rank_B.d_norms = d_mpi_B_norms_recv;
                        }
                        if (compress_b_qjl) {
                            CHECK_CUDA(cudaMemcpy(
                                d_mpi_B_qjl_signs_recv,
                                h_mpi_B_qjl_signs_all.data() + static_cast<size_t>(r) * b_quant_options.qjl_dim,
                                bi_qjl_sign_bytes,
                                cudaMemcpyHostToDevice));
                            rank_B.d_qjl_signs = d_mpi_B_qjl_signs_recv;
                        }
                        if (b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
                            b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl) {
                            turboquant::dequantize_column_tq_payload_add_to_fp32(
                                rank_B,
                                d_B,
                                d_payload_decode_work);
                        } else {
                            turboquant::dequantize_device_payload_to_fp32(
                                rank_B,
                                d_mpi_B_tq_work);
                            const int threads = 256;
                            const int blocks = static_cast<int>((b_count + threads - 1) / threads);
                            add_kernel<<<blocks, threads>>>(d_B, d_mpi_B_tq_work, b_count);
                            CHECK_CUDA(cudaGetLastError());
                        }
                    }
                    CHECK_CUDA(cudaDeviceSynchronize());
                }
            }
        }
        double t_build_b_reduce_ms = timer.toc_ms();
        long double global_b_error_norm2 = 0.0L;
        long double global_b_ref_norm2 = 0.0L;
        if (opt.check_b_error) {
            CHECK_MPI(MPI_Reduce(&b_error_norm2, &global_b_error_norm2, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
            CHECK_MPI(MPI_Reduce(&b_ref_norm2, &global_b_ref_norm2, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
        }
        const double b_relative_error = (opt.check_b_error && is_root) ?
            std::sqrt(static_cast<double>(global_b_error_norm2 / std::max(global_b_ref_norm2, 1e-30L))) :
            -1.0;
        double global_b_relative_error = -1.0;
        double global_b_exact_norm2 = -1.0;
        if (need_global_b_metric) {
            std::vector<float> h_B_exact_local(b_count);
            std::vector<float> h_B_exact_global;
            if (is_root) {
                h_B_exact_global.resize(b_count);
            }
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaMemcpy(h_B_exact_local.data(), d_B_exact, bi_fp32_bytes, cudaMemcpyDeviceToHost));
            CHECK_MPI(MPI_Reduce(
                h_B_exact_local.data(),
                is_root ? h_B_exact_global.data() : nullptr,
                checked_mpi_count(b_count, "B exact MPI reduce count"),
                MPI_FLOAT,
                MPI_SUM,
                0,
                MPI_COMM_WORLD));
            if (is_root) {
                CHECK_CUDA(cudaMemcpy(d_B_exact, h_B_exact_global.data(), bi_fp32_bytes, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_metric_Bk, d_B, bi_fp32_bytes, cudaMemcpyDeviceToDevice));
                const int threads = 256;
                const int blocks = static_cast<int>((b_count + threads - 1) / threads);
                subtract_kernel<<<blocks, threads>>>(d_metric_Bk, d_B_exact, b_count);
                CHECK_CUDA(cudaGetLastError());
                float global_b_diff_norm = 0.0f;
                float global_b_exact_norm = 0.0f;
                CHECK_CUBLAS(cublasSnrm2(
                    blas0,
                    checked_mpi_count(b_count, "global B relative error vector length"),
                    d_metric_Bk,
                    1,
                    &global_b_diff_norm));
                CHECK_CUBLAS(cublasSnrm2(
                    blas0,
                    checked_mpi_count(b_count, "global B exact norm vector length"),
                    d_B_exact,
                    1,
                    &global_b_exact_norm));
                CHECK_CUDA(cudaDeviceSynchronize());
                global_b_exact_norm2 =
                    static_cast<double>(global_b_exact_norm) * static_cast<double>(global_b_exact_norm);
                global_b_relative_error =
                    static_cast<double>(global_b_diff_norm) /
                    std::max(static_cast<double>(global_b_exact_norm), 1e-30);
            }
        }

        timer.tic();
        if (is_root) {
            NvtxRange range("svd_B_on_gpu0");
            CHECK_CUDA(cudaSetDevice(0));

            dim3 block(16, 16);
            dim3 grid((l + block.x - 1) / block.x, (n + block.y - 1) / block.y);
            transpose_colmajor_kernel<<<grid, block>>>(d_B, d_BT, l, n);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            signed char jobu = opt.skip_form_u ? 'N' : 'S';
            signed char jobvt = opt.skip_form_u ? 'N' : 'A';
            CHECK_CUSOLVER(cusolverDnSgesvd(
                solver0, jobu, jobvt,
                n, l,
                d_BT, n,
                d_S,
                d_V, n,
                d_UtT, l,
                d_work_svd, lwork_svd,
                d_rwork,
                d_info0));
            CHECK_CUDA(cudaDeviceSynchronize());
            check_solver_info(d_info0, "gesvd(B)");
            if (!opt.skip_form_u) {
                CHECK_CUDA(cudaMemcpy(d_Ut_k, d_UtT, static_cast<size_t>(l) * k * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
        double t_svd_b_ms = timer.toc_ms();
        if (!is_root) {
            t_svd_b_ms = 0.0;
        }

        double t_form_distributed_u_ms = 0.0;
        if (!opt.skip_form_u && is_root) {
            timer.tic();
            std::vector<float> h_Uk(static_cast<size_t>(m) * k);
            NvtxRange range("form_distributed_Ui");
            std::vector<float> h_Ut_k(static_cast<size_t>(l) * k);
            CHECK_CUDA(cudaMemcpy(h_Ut_k.data(), d_Ut_k, static_cast<size_t>(l) * k * sizeof(float), cudaMemcpyDeviceToHost));

            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                CHECK_CUDA(cudaMalloc(&w.d_Uti_k, static_cast<size_t>(l) * k * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_Ui, static_cast<size_t>(w.mi) * k * sizeof(float)));
                CHECK_CUDA(cudaMemcpy(w.d_Uti_k, h_Ut_k.data(), static_cast<size_t>(l) * k * sizeof(float), cudaMemcpyHostToDevice));
                const float alpha = 1.0f;
                const float beta = 0.0f;
                CHECK_CUBLAS(cublasSgemm(
                    w.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                    w.mi, k, l,
                    &alpha,
                    w.d_Qi, w.mi,
                    w.d_Uti_k, l,
                    &beta,
                    w.d_Ui, w.mi));
                CHECK_CUDA(cudaDeviceSynchronize());

                std::vector<float> h_Ui(static_cast<size_t>(w.mi) * k);
                CHECK_CUDA(cudaMemcpy(h_Ui.data(), w.d_Ui, static_cast<size_t>(w.mi) * k * sizeof(float), cudaMemcpyDeviceToHost));
                insert_row_block_colmajor(h_Uk, h_Ui, m, k, w.row0, w.mi);
            }
            t_form_distributed_u_ms = timer.toc_ms();
        }

        std::vector<float> h_S(l);
        if (is_root) {
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, static_cast<size_t>(l) * sizeof(float), cudaMemcpyDeviceToHost));
        }

        double rel_err = -1.0;
        double fast_rel_err = -1.0;
        double t_err_ms = 0.0;
        if (opt.check_error && is_root) {
            timer.tic();
            fast_rel_err = fast_reconstruction_relative_error(current_A_norm2, h_S, k);
            rel_err = fast_rel_err;
            if (need_compressed_safe_error) {
                if (global_b_exact_norm2 < 0.0) {
                    throw std::runtime_error("Compressed-safe reconstruction error requires global B_exact.");
                }
                CHECK_CUDA(cudaSetDevice(0));
                if (opt.skip_form_u) {
                    dim3 block(16, 16);
                    dim3 grid((l + block.x - 1) / block.x, (n + block.y - 1) / block.y);
                    transpose_colmajor_kernel<<<grid, block>>>(d_B, d_BT, l, n);
                    CHECK_CUDA(cudaGetLastError());
                    CHECK_CUDA(cudaDeviceSynchronize());
                    CHECK_CUSOLVER(cusolverDnSgesvd(
                        solver0, 'S', 'A',
                        n, l,
                        d_BT, n,
                        d_S,
                        d_V, n,
                        d_UtT, l,
                        d_work_svd, lwork_svd,
                        d_rwork,
                        d_info0));
                    CHECK_CUDA(cudaDeviceSynchronize());
                    check_solver_info(d_info0, "gesvd(B) for compressed-safe error");
                    CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, static_cast<size_t>(l) * sizeof(float), cudaMemcpyDeviceToHost));
                    fast_rel_err = fast_reconstruction_relative_error(current_A_norm2, h_S, k);
                }

                {
                    dim3 block(16, 16);
                    dim3 grid((l + block.x - 1) / block.x, (k + block.y - 1) / block.y);
                    extract_vt_rows_scaled_kernel<<<grid, block>>>(d_UtT, d_S, d_metric_VkS, l, k);
                    CHECK_CUDA(cudaGetLastError());
                }
                const float alpha = 1.0f;
                const float beta = 0.0f;
                CHECK_CUBLAS(cublasSgemm(
                    blas0, CUBLAS_OP_N, CUBLAS_OP_T,
                    l, n, k,
                    &alpha,
                    d_metric_VkS, l,
                    d_V, n,
                    &beta,
                    d_metric_Bk, l));
                {
                    const int threads = 256;
                    const int blocks = static_cast<int>((b_count + threads - 1) / threads);
                    subtract_kernel<<<blocks, threads>>>(d_metric_Bk, d_B_exact, b_count);
                    CHECK_CUDA(cudaGetLastError());
                }
                float projected_b_diff_norm = 0.0f;
                CHECK_CUBLAS(cublasSnrm2(
                    blas0,
                    checked_mpi_count(b_count, "compressed-safe B difference vector length"),
                    d_metric_Bk,
                    1,
                    &projected_b_diff_norm));
                CHECK_CUDA(cudaDeviceSynchronize());
                const long double projected_b_diff_norm2 =
                    static_cast<long double>(projected_b_diff_norm) *
                    static_cast<long double>(projected_b_diff_norm);
                const long double numerator = std::max(
                    static_cast<long double>(current_A_norm2) -
                        static_cast<long double>(global_b_exact_norm2) +
                        projected_b_diff_norm2,
                    0.0L);
                rel_err = std::sqrt(static_cast<double>(
                    numerator / std::max(static_cast<long double>(current_A_norm2), 1e-30L)));
            }
            t_err_ms = timer.toc_ms();
            if (!opt.summary_only) {
                std::cout << "  reconstruction_error_time_ms=" << t_err_ms << "\n";
                std::cout << "  reconstruction_error_metric="
                          << (need_compressed_safe_error ? "compressed_safe_projected_space" : "fast_energy_proxy")
                          << "\n";
                if (need_compressed_safe_error) {
                    std::cout << "  fast_energy_proxy_error_diagnostic=" << fast_rel_err << "\n"
                              << "  global_B_relative_error=" << global_b_relative_error << "\n";
                }
            }
        }

        const double t_gpu_compute_reported =
            t_local_projection_ms + t_local_qr_ms + t_tsqr_reduce_ms + t_form_distributed_q_ms +
            t_subspace_iter_ms + t_build_b_reduce_ms + t_svd_b_ms + t_form_distributed_u_ms;
        const double t_iter_init_ms = (repeat_idx == 0) ? t_init_ms : 0.0;
        const double t_iter_setup_ms = (repeat_idx == 0) ? t_setup_gpu_resources_ms : 0.0;
        const double t_total_reported = t_iter_init_ms + t_iter_setup_ms + t_gpu_compute_reported;
        const double t_gpu_pipeline_reported = t_iter_setup_ms + t_gpu_compute_reported;
        double t_gpu_compute_reported_max = 0.0;
        double t_gpu_pipeline_reported_max = 0.0;
        double stage_local[kStageCount] = {
            t_local_projection_ms,
            t_local_qr_ms,
            t_tsqr_reduce_ms,
            t_form_distributed_q_ms,
            t_subspace_iter_ms,
            t_build_b_reduce_ms,
            t_svd_b_ms,
            t_form_distributed_u_ms,
        };
        double stage_max[kStageCount] = {};
        CHECK_MPI(MPI_Reduce(&t_gpu_compute_reported, &t_gpu_compute_reported_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&t_gpu_pipeline_reported, &t_gpu_pipeline_reported_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(stage_local, stage_max, kStageCount, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        unsigned long long r_payload_bytes_global = 0;
        unsigned long long z_fp32_payload_bytes_global = 0;
        unsigned long long z_transmitted_payload_bytes_global = 0;
        unsigned long long b_fp32_payload_bytes_global = 0;
        unsigned long long b_transmitted_payload_bytes_global = 0;
        unsigned long long mpi_b_fp32_payload_bytes_global = 0;
        unsigned long long mpi_b_transmitted_payload_bytes_global = 0;
        const unsigned long long r_payload_bytes_local_ull = static_cast<unsigned long long>(r_payload_bytes);
        const unsigned long long z_fp32_payload_bytes_local_ull = static_cast<unsigned long long>(z_fp32_payload_bytes);
        const unsigned long long z_transmitted_payload_bytes_local_ull = static_cast<unsigned long long>(z_transmitted_payload_bytes);
        const unsigned long long b_fp32_payload_bytes_local_ull = static_cast<unsigned long long>(b_fp32_payload_bytes);
        const unsigned long long b_transmitted_payload_bytes_local_ull = static_cast<unsigned long long>(b_transmitted_payload_bytes);
        const unsigned long long mpi_b_fp32_payload_bytes_local_ull = static_cast<unsigned long long>(mpi_b_fp32_payload_bytes);
        const unsigned long long mpi_b_transmitted_payload_bytes_local_ull = static_cast<unsigned long long>(mpi_b_transmitted_payload_bytes);
        double t_compress_subspace_ms_max = 0.0;
        double t_compress_b_ms_max = 0.0;
        CHECK_MPI(MPI_Reduce(&r_payload_bytes_local_ull, &r_payload_bytes_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&z_fp32_payload_bytes_local_ull, &z_fp32_payload_bytes_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&z_transmitted_payload_bytes_local_ull, &z_transmitted_payload_bytes_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&b_fp32_payload_bytes_local_ull, &b_fp32_payload_bytes_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&b_transmitted_payload_bytes_local_ull, &b_transmitted_payload_bytes_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&mpi_b_fp32_payload_bytes_local_ull, &mpi_b_fp32_payload_bytes_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&mpi_b_transmitted_payload_bytes_local_ull, &mpi_b_transmitted_payload_bytes_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&t_compress_subspace_ms, &t_compress_subspace_ms_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        CHECK_MPI(MPI_Reduce(&t_compress_b_ms, &t_compress_b_ms_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        if (is_root) {
            repeat_pipeline_ms.push_back(t_gpu_pipeline_reported_max);
            repeat_compute_ms.push_back(t_gpu_compute_reported_max);
            for (int i = 0; i < kStageCount; ++i) {
                repeat_stage_ms[i].push_back(stage_max[i]);
            }
            if (opt.check_b_error) {
                repeat_b_relative_error.push_back(b_relative_error);
            }
            if (global_b_relative_error >= 0.0) {
                repeat_global_b_relative_error.push_back(global_b_relative_error);
            }
            if (opt.check_error) {
                repeat_final_error.push_back(rel_err);
            }
        }

        if (is_root && print_repeat_detail) {
            std::cout << "\nTimings (ms)\n"
                      << "  init_host_random          " << t_iter_init_ms << "\n"
                      << "  create_gpu_handles        " << ((repeat_idx == 0) ? t_create_gpu_handles_ms : 0.0) << "\n"
                      << "  allocate_gpu_buffers      " << ((repeat_idx == 0) ? t_allocate_gpu_buffers_ms : 0.0) << "\n"
                      << "  generate_test_matrix      " << ((repeat_idx == 0) ? t_generate_test_matrix_ms : 0.0) << "\n"
                      << "  setup_gpu_resources       " << t_iter_setup_ms << "\n"
                      << "  local_projection_Yi       " << t_local_projection_ms << "\n"
                      << "  local_qr_Yi               " << t_local_qr_ms << "\n"
                      << "  tsqr_R_reduce_gpu0        " << t_tsqr_reduce_ms << "\n"
                      << "  form_distributed_Qi       " << t_form_distributed_q_ms << "\n"
                      << "  subspace_iteration        " << t_subspace_iter_ms << "\n"
                      << "  build_reduce_Bi           " << t_build_b_reduce_ms << "\n"
                      << "  svd_B_on_gpu0             " << t_svd_b_ms << "\n"
                      << "  form_distributed_Ui       " << t_form_distributed_u_ms << "\n"
                      << "  gpu_pipeline_reported     " << t_gpu_pipeline_reported_max << "\n"
                      << "  gpu_compute_reported      " << t_gpu_compute_reported_max << "\n"
                      << "  subtotal_reported         " << t_total_reported << "\n";

            std::cout << "\nCommunication baseline\n"
                      << "  tsqr_R_payload_bytes      " << r_payload_bytes_global << "\n"
                      << "  tsqr_R_payload_MiB        " << (r_payload_bytes_global / 1024.0 / 1024.0) << "\n"
                      << "  subspace_Z_quant_mode     " << turboquant::mode_name(subspace_quant_options.mode) << "\n"
                      << "  subspace_Z_quant_bits     " << opt.compress_subspace_bits << "\n"
                      << "  subspace_Z_tq_codec       "
                      << (compress_subspace_lloyd_tq ? "rht_lloyd_max_d256_norm" : "n/a") << "\n"
                      << "  subspace_Z_qjl_dim        " << subspace_quant_options.qjl_dim << "\n"
                      << "  subspace_Z_qjl_alpha      " << subspace_quant_options.qjl_alpha << "\n"
                      << "  subspace_Z_vector_layout  " << (compress_subspace_none ? "n/a" : "row_vectors_length_l") << "\n"
                      << "  subspace_Z_fp32_bytes     " << z_fp32_payload_bytes_global << "\n"
                      << "  subspace_Z_fp32_MiB       " << (z_fp32_payload_bytes_global / 1024.0 / 1024.0) << "\n"
                      << "  subspace_Z_payload_bytes  " << z_transmitted_payload_bytes_global << "\n"
                      << "  subspace_Z_payload_MiB    " << (z_transmitted_payload_bytes_global / 1024.0 / 1024.0) << "\n"
                      << "  subspace_Z_compression_ratio "
                      << (static_cast<double>(z_fp32_payload_bytes_global) /
                          static_cast<double>(std::max<unsigned long long>(z_transmitted_payload_bytes_global, 1))) << "\n"
                      << "  subspace_Z_compress_time_ms " << t_compress_subspace_ms_max << "\n"
                      << "  subspace_Z_collective_mode "
                      << (compress_subspace_none ? "fp32_allreduce" : "compressed_allgather_decode_rowwise") << "\n"
                      << "  reduce_B_quant_mode       " << turboquant::mode_name(b_quant_options.mode) << "\n"
                      << "  reduce_B_quant_bits       " << opt.compress_b_bits << "\n"
                      << "  reduce_B_tq_codec         "
                      << (compress_b_tq ? "rht_lloyd_max_d256_norm" : "n/a") << "\n"
                      << "  reduce_B_qjl_dim          " << b_quant_options.qjl_dim << "\n"
                      << "  reduce_B_qjl_alpha        " << b_quant_options.qjl_alpha << "\n"
                      << "  reduce_B_fp32_bytes       " << b_fp32_payload_bytes_global << "\n"
                      << "  reduce_B_fp32_MiB         " << (b_fp32_payload_bytes_global / 1024.0 / 1024.0) << "\n"
                      << "  reduce_B_payload_bytes    " << b_transmitted_payload_bytes_global << "\n"
                      << "  reduce_B_payload_MiB      " << (b_transmitted_payload_bytes_global / 1024.0 / 1024.0) << "\n"
                      << "  reduce_B_compression_ratio "
                      << (static_cast<double>(b_fp32_payload_bytes_global) /
                          static_cast<double>(std::max<unsigned long long>(b_transmitted_payload_bytes_global, 1))) << "\n"
                      << "  reduce_B_relative_error   "
                      << (opt.check_b_error ? std::to_string(b_relative_error) : std::string("skipped")) << "\n"
                      << "  global_B_relative_error   "
                      << (global_b_relative_error >= 0.0 ? std::to_string(global_b_relative_error) : std::string("skipped")) << "\n"
                      << "  reduce_B_compress_time_ms " << t_compress_b_ms_max << "\n"
                      << "  mpi_B_collective_mode     " << (compress_b_none ? "fp32_reduce" : "compressed_gather_decode") << "\n"
                      << "  mpi_B_fp32_bytes          " << mpi_b_fp32_payload_bytes_global << "\n"
                      << "  mpi_B_fp32_MiB            " << (mpi_b_fp32_payload_bytes_global / 1024.0 / 1024.0) << "\n"
                      << "  mpi_B_payload_bytes       " << mpi_b_transmitted_payload_bytes_global << "\n"
                      << "  mpi_B_payload_MiB         " << (mpi_b_transmitted_payload_bytes_global / 1024.0 / 1024.0) << "\n"
                      << "  mpi_B_compression_ratio   "
                      << (static_cast<double>(mpi_b_fp32_payload_bytes_global) /
                          static_cast<double>(std::max<unsigned long long>(mpi_b_transmitted_payload_bytes_global, 1))) << "\n";

            std::cout << "\nLeading singular values\n  ";
            for (int i = 0; i < std::min(k, 10); ++i) {
                std::cout << h_S[i] << (i + 1 == std::min(k, 10) ? "" : ", ");
            }
            std::cout << "\n";
            if (opt.check_error) {
                std::cout << "\nRelative Frobenius reconstruction error\n"
                          << "  metric = "
                          << (need_compressed_safe_error ? "compressed_safe_projected_space" : "fast_energy_proxy")
                          << "\n"
                          << "  final_reconstruction_error = " << rel_err << "\n";
                if (need_compressed_safe_error) {
                    std::cout << "  fast_energy_proxy_error_diagnostic = " << fast_rel_err << "\n";
                }
            }
        }

        CHECK_CUDA(cudaSetDevice(0));
        for (auto& w : works) {
            CHECK_CUDA(cudaSetDevice(w.dev));
            if (w.d_Uti_k) {
                CHECK_CUDA(cudaFree(w.d_Uti_k));
                w.d_Uti_k = nullptr;
            }
            if (w.d_Ui) {
                CHECK_CUDA(cudaFree(w.d_Ui));
                w.d_Ui = nullptr;
            }
        }
        }

        if (is_root) {
            const size_t cold_count = (opt.repeat > 1) ? 1 : 0;
            const size_t warm_begin = (opt.repeat > 1) ? 1 : 0;
            const size_t warm_count = opt.repeat - cold_count;
            const SummaryStats total_stats = summarize_samples(repeat_compute_ms, warm_begin);
            const int label_width = 32;
            const bool timing_only = !opt.check_error && !opt.check_b_error;
            const bool accuracy_summary = opt.check_error || opt.check_b_error;

            std::cout << "\nSummary\n"
                      << "Repeat Count: " << opt.repeat
                      << " (warm: " << warm_count
                      << " | cold: " << cold_count << ")\n";

            if (timing_only) {
                std::cout <<  std::setw(label_width) << std::right << ""
                          <<  std::setw(16) << std::right << "mean"
                          <<  std::setw(16) << std::right << "min"
                          <<  std::setw(16) << std::right << "stddev" << "\n";
                print_summary_stats_row("Total Time", total_stats, label_width);
            }

            if (accuracy_summary) {
                if (!repeat_global_b_relative_error.empty()) {
                    std::cout <<  std::setw(label_width) << std::right << ""
                              <<  std::setw(16) << std::right << "mean"
                              <<  std::setw(16) << std::right << "min"
                              <<  std::setw(16) << std::right << "stddev" << "\n";
                    print_summary_stats_row(
                        "Global B Relative Error",
                        summarize_samples(repeat_global_b_relative_error, 0),
                        label_width,
                        100.0,
                        "%");
                }
                if (opt.check_b_error) {
                    std::cout <<  std::setw(label_width) << std::right << ""
                              <<  std::setw(16) << std::right << "mean"
                              <<  std::setw(16) << std::right << "min"
                              <<  std::setw(16) << std::right << "stddev" << "\n";
                    print_summary_stats_row(
                        "Local B_i Relative Error",
                        summarize_samples(repeat_b_relative_error, 0),
                        label_width,
                        100.0,
                        "%");
                }
                if (opt.check_error) {
                    const double theoretical_error = synthetic_spectrum_theoretical_error(
                        opt.spectrum_decay_mode,
                        opt.spectrum_decay_param,
                        opt.spectrum_rank,
                        k);
                    std::cout <<  std::setw(label_width) << std::right << ""
                              <<  std::setw(16) << std::right << "mean"
                              <<  std::setw(16) << std::right << "min"
                              <<  std::setw(16) << std::right << "stddev"
                              <<  std::setw(16) << std::right << "theoretical"
                              <<  std::setw(16) << std::right << "err ratio" << "\n";
                    print_reconstruction_error_row(
                        "Final Reconstruction Error",
                        summarize_samples(repeat_final_error, 0),
                        theoretical_error,
                        label_width);
                }
            }

            std::cout << "\nStages\n"
                      <<  std::setw(label_width) << std::right << "stage"
                      <<  std::setw(16) << std::right << "mean"
                      <<  std::setw(16) << std::right << "min"
                      <<  std::setw(16) << std::right << "stddev" << "\n";
            for (int i = 0; i < kStageCount; ++i) {
                const SummaryStats stats = summarize_samples(repeat_stage_ms[i], warm_begin);
                print_summary_stats_row(stage_names[i], stats, label_width);
            }
        }

        CHECK_CUDA(cudaSetDevice(0));
        cudaFree(d_B);
        cudaFree(d_B_exact);
        cudaFree(d_metric_Bk);
        cudaFree(d_metric_VkS);
        cudaFree(d_BT);
        cudaFree(d_V);
        cudaFree(d_S);
        cudaFree(d_UtT);
        cudaFree(d_Ut_k);
        cudaFree(d_payload_decode_work);
        cudaFree(d_mpi_Z_local);
        cudaFree(d_mpi_Z_global);
        cudaFree(d_mpi_ZT_local);
        cudaFree(d_mpi_ZT_global);
        cudaFree(d_mpi_Z_work);
        cudaFree(d_mpi_Z_codes);
        cudaFree(d_mpi_Z_codes_recv);
        cudaFree(d_mpi_Z_norms);
        cudaFree(d_mpi_Z_norms_recv);
        cudaFree(d_mpi_Z_qjl_signs);
        cudaFree(d_mpi_Z_qjl_signs_recv);
        cudaFree(d_mpi_B_codes);
        cudaFree(d_mpi_B_codes_recv);
        cudaFree(d_mpi_B_norms);
        cudaFree(d_mpi_B_norms_recv);
        cudaFree(d_mpi_B_qjl_signs);
        cudaFree(d_mpi_B_qjl_signs_recv);
        cudaFree(d_mpi_B_tq_work);
        cudaFree(d_work_svd);
        cudaFree(d_rwork);
        cudaFree(d_work_z);
        cudaFree(d_tau_z);
        cudaFree(d_Z_qr);
        cudaFree(d_work_r);
        cudaFree(d_tau_r);
        cudaFree(d_Rstack);
        for (float* p : d_Bi_reduce_on_gpu0) {
            if (p) cudaFree(p);
        }
        for (std::uint8_t* p : d_Bi_codes_on_gpu0) {
            if (p) cudaFree(p);
        }
        for (float* p : d_Bi_norms_on_gpu0) {
            if (p) cudaFree(p);
        }
        for (int* p : d_Bi_qjl_signs_on_gpu0) {
            if (p) cudaFree(p);
        }
        cudaFree(d_info0);
        cublasDestroy(blas0);
        cusolverDnDestroy(solver0);
        for (auto& w : works) destroy_work(w);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "!Error: " << e.what() << "\n";
        return 1;
    }
}
