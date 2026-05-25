// randomized_svd_multigpu_v2.cu
//
// Baseline B v2: TSQR-style multi-GPU randomized SVD.
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
// The communication hooks are the R_i TSQR gather and the B_i reduction.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nvtx3/nvToolsExt.h>

#include "../turboquant/turboquant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
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
    unsigned seed = 1234;
    bool check_error = true;
    bool check_b_error = true;
    std::string compress_b_mode = "none";
    int compress_b_bits = 0;
    int qjl_dim = 256;
    float qjl_alpha = 1.0f;
    bool allow_host_tq_prototype = false;
    bool device_random_input = false;
    bool skip_form_u = false;
    bool host_reduce_b = false;
    int repeat = 1;
    int repeat_print_every = 1;
};

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --m <int>             Number of rows of A. Default: 4096\n"
        << "  --n <int>             Number of cols of A. Default: 2048\n"
        << "  --k <int>             Target rank. Default: 64\n"
        << "  --oversample <int>    Oversampling p. l = k + p. Default: 16\n"
        << "  --ngpus <int>         Number of GPUs to use. Default: 1\n"
        << "  --seed <int>          RNG seed. Default: 1234\n"
        << "  --compress-b-mode <none|lowbit|tq|tq-qjl>\n"
        << "                        B_i compression mode. Default: none\n"
        << "  --compress-b-bits <0|8|4|2>\n"
        << "                        B_i quantization bits. Default: 0\n"
        << "  --qjl-dim <int>       QJL residual sketch dimension for tq-qjl. Default: 256\n"
        << "  --qjl-alpha <float>   QJL residual correction strength. Default: 1.0\n"
        << "  --allow-host-tq-prototype\n"
        << "                        Deprecated compatibility flag; tq-qjl is GPU-side now.\n"
        << "  --device-random-input\n"
        << "                        Generate A_i and Omega directly on each GPU.\n"
        << "  --skip-form-u         Skip distributed U_i formation after SVD(B).\n"
        << "  --host-reduce-b       Use the original host-side B_i sum for all compress-b modes.\n"
        << "  --repeat <int>        Reuse setup and run the compute pipeline N times. Default: 1\n"
        << "  --repeat-print-every <int>\n"
        << "                        Print detailed timings every N repeats. 0 means first/last only. Default: 1\n"
        << "  --no-check-error      Skip fast reconstruction error.\n"
        << "  --no-check-b-error    Skip B_i compression error copy/check.\n";
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
        else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need_value(a)));
        else if (a == "--compress-b-mode") opt.compress_b_mode = need_value(a);
        else if (a == "--compress-b-bits") opt.compress_b_bits = std::stoi(need_value(a));
        else if (a == "--qjl-dim") opt.qjl_dim = std::stoi(need_value(a));
        else if (a == "--qjl-alpha") opt.qjl_alpha = std::stof(need_value(a));
        else if (a == "--repeat") opt.repeat = std::stoi(need_value(a));
        else if (a == "--repeat-print-every") opt.repeat_print_every = std::stoi(need_value(a));
        else if (a == "--allow-host-tq-prototype") opt.allow_host_tq_prototype = true;
        else if (a == "--device-random-input") opt.device_random_input = true;
        else if (a == "--skip-form-u") opt.skip_form_u = true;
        else if (a == "--host-reduce-b") opt.host_reduce_b = true;
        else if (a == "--no-check-error") opt.check_error = false;
        else if (a == "--no-check-b-error") opt.check_b_error = false;
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + a);
        }
    }
    if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.oversample < 0 || opt.ngpus <= 0) {
        throw std::runtime_error("m, n, k, ngpus must be positive and oversample must be non-negative.");
    }
    if (opt.repeat <= 0) {
        throw std::runtime_error("repeat must be positive.");
    }
    if (opt.repeat_print_every < 0) {
        throw std::runtime_error("repeat-print-every must be non-negative.");
    }
    if (opt.k + opt.oversample > std::min(opt.m, opt.n)) {
        throw std::runtime_error("Require k + oversample <= min(m, n).");
    }
    turboquant::QuantizeOptions qopt =
        turboquant::make_quantize_options(opt.compress_b_bits, opt.compress_b_mode, opt.qjl_dim, opt.qjl_alpha, opt.seed);
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
    long double captured = 0.0L;
    for (int i = 0; i < k; ++i) {
        captured += static_cast<long double>(S[i]) * static_cast<long double>(S[i]);
    }
    const long double err2 = std::max(
        static_cast<long double>(a_norm2) - captured,
        0.0L);
    return std::sqrt(static_cast<double>(err2 / std::max(static_cast<long double>(a_norm2), 1e-30L)));
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

__global__ void add_kernel(float* dst, const float* src, size_t count) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] += src[idx];
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
    float* d_Omega = nullptr;    // n x l
    float* d_Qbar = nullptr;     // mi x l, first Y_i then local Qbar_i
    float* d_tau = nullptr;      // l
    float* d_qr_work = nullptr;  // local QR workspace
    float* d_Ri = nullptr;       // l x l local TSQR R block
    int qr_lwork = 0;
    float* d_Ti = nullptr;       // l x l
    float* d_Qi = nullptr;       // mi x l
    float* d_Bi = nullptr;       // l x n
    float* d_Bi_hat = nullptr;   // l x n reconstructed B_i after optional compression
    std::uint8_t* d_Bi_codes = nullptr; // compressed B_i payload
    int* d_Bi_qjl_signs = nullptr;      // tq-qjl residual signs
    float* d_Bi_tq_work = nullptr;      // column-wise tq encode scratch
    float* d_Uti_k = nullptr;    // l x k
    float* d_Ui = nullptr;       // mi x k
    int* d_info = nullptr;
};

static void destroy_work(DeviceWork& w) {
    if (w.dev >= 0) cudaSetDevice(w.dev);
    if (w.d_Ai) cudaFree(w.d_Ai);
    if (w.d_Omega) cudaFree(w.d_Omega);
    if (w.d_Qbar) cudaFree(w.d_Qbar);
    if (w.d_tau) cudaFree(w.d_tau);
    if (w.d_qr_work) cudaFree(w.d_qr_work);
    if (w.d_Ri) cudaFree(w.d_Ri);
    if (w.d_Ti) cudaFree(w.d_Ti);
    if (w.d_Qi) cudaFree(w.d_Qi);
    if (w.d_Bi) cudaFree(w.d_Bi);
    if (w.d_Bi_hat) cudaFree(w.d_Bi_hat);
    if (w.d_Bi_codes) cudaFree(w.d_Bi_codes);
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
        Options opt = parse_args(argc, argv);
        int device_count = 0;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) throw std::runtime_error("No CUDA device found.");
        opt.ngpus = std::min(opt.ngpus, device_count);

        const int m = opt.m;
        const int n = opt.n;
        const int k = opt.k;
        const int l = opt.k + opt.oversample;
        const std::vector<int> rows = split_rows(m, opt.ngpus);
        for (int mi : rows) {
            if (mi < l) throw std::runtime_error("Each row block must have at least k + oversample rows.");
        }

        std::cout << "Randomized multi-GPU SVD baseline v2 (TSQR/distributed B)\n"
                  << "  m=" << m << " n=" << n << " k=" << k
                  << " oversample=" << opt.oversample << " l=" << l
                  << " ngpus=" << opt.ngpus
                  << " compress_b_mode=" << opt.compress_b_mode
                  << " compress_b_bits=" << opt.compress_b_bits
                  << " qjl_alpha=" << opt.qjl_alpha
                  << " device_random_input=" << (opt.device_random_input ? "yes" : "no")
                  << " skip_form_u=" << (opt.skip_form_u ? "yes" : "no")
                  << " host_reduce_b=" << (opt.host_reduce_b ? "yes" : "no")
                  << " repeat=" << opt.repeat
                  << " repeat_print_every=" << opt.repeat_print_every
                  << " allow_host_tq_prototype=" << (opt.allow_host_tq_prototype ? "yes" : "no")
                  << "\n";

        Timer timer;
        timer.tic();
        std::vector<std::vector<float>> h_A_blocks;
        std::vector<float> h_Omega;
        double h_A_norm2 = 0.0;
        {
            NvtxRange range("init_host_random");
            if (!opt.device_random_input) {
                h_A_blocks = make_random_row_blocks(
                    rows, n, opt.seed, 1.0f / std::sqrt(static_cast<float>(m)), &h_A_norm2);
                h_Omega = make_random_matrix(n, l, opt.seed + 1, 1.0f);
            }
        }
        double t_init_ms = timer.toc_ms();

        std::vector<DeviceWork> works(opt.ngpus);
        cublasHandle_t blas0 = nullptr;
        cusolverDnHandle_t solver0 = nullptr;
        int* d_info0 = nullptr;
        const int r_rows = opt.ngpus * l;
        const size_t b_count = static_cast<size_t>(l) * n;
        const size_t bi_fp32_bytes = b_count * sizeof(float);
        const turboquant::QuantizeOptions b_quant_options =
            turboquant::make_quantize_options(opt.compress_b_bits, opt.compress_b_mode, opt.qjl_dim, opt.qjl_alpha, opt.seed);
        const bool compress_b_none = b_quant_options.mode == turboquant::QuantizeMode::kNone;
        const bool compress_b_tq = b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant;
        const bool compress_b_qjl = b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl;
        const size_t bi_code_bytes = compress_b_none ? 0 :
            turboquant::device_code_bytes(l, n, b_quant_options);
        const size_t bi_qjl_sign_bytes = compress_b_qjl ?
            static_cast<size_t>(b_quant_options.qjl_dim) * sizeof(int) : 0;
        const size_t bi_payload_work_count = compress_b_none ? 0 :
            ((b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
              b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl) ?
             static_cast<size_t>(1) << static_cast<size_t>(std::ceil(std::log2(static_cast<double>(b_count)))) :
             b_count);
        const size_t bi_column_tq_work_count = static_cast<size_t>(1) << static_cast<size_t>(std::ceil(std::log2(static_cast<double>(l))));
        const size_t bi_column_tq_sign_count = bi_column_tq_work_count * static_cast<size_t>(n);
        float* d_Rstack = nullptr;
        float* d_tau_r = nullptr;
        float* d_work_r = nullptr;
        float* d_B = nullptr;
        float* d_BT = nullptr;
        float* d_V = nullptr;
        float* d_S = nullptr;
        float* d_UtT = nullptr;
        float* d_work_svd = nullptr;
        float* d_rwork = nullptr;
        float* d_Ut_k = nullptr;
        float* d_payload_decode_work = nullptr;
        std::vector<float*> d_Bi_reduce_on_gpu0(opt.ngpus, nullptr);
        std::vector<std::uint8_t*> d_Bi_codes_on_gpu0(opt.ngpus, nullptr);
        std::vector<int*> d_Bi_qjl_signs_on_gpu0(opt.ngpus, nullptr);
        int lwork_r = 0;
        int lwork_svd = 0;

        timer.tic();
        {
            NvtxRange range("create_gpu_handles");
            int row0 = 0;
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                w.dev = g;
                w.row0 = row0;
                w.mi = rows[g];
                row0 += rows[g];

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
                CHECK_CUDA(cudaMalloc(&w.d_Bi, static_cast<size_t>(l) * n * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&w.d_Bi_hat, bi_fp32_bytes));
                if (!compress_b_none) {
                    CHECK_CUDA(cudaMalloc(&w.d_Bi_codes, bi_code_bytes));
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

            CHECK_CUDA(cudaMalloc(&d_B, bi_fp32_bytes));
            if (!compress_b_none) {
                CHECK_CUDA(cudaMalloc(&d_payload_decode_work, bi_payload_work_count * sizeof(float)));
            }
            CHECK_CUDA(cudaMalloc(&d_BT, static_cast<size_t>(n) * l * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_S, static_cast<size_t>(l) * sizeof(float)));
            if (!opt.skip_form_u) {
                CHECK_CUDA(cudaMalloc(&d_V, static_cast<size_t>(n) * l * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_UtT, static_cast<size_t>(l) * l * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_Ut_k, static_cast<size_t>(l) * k * sizeof(float)));
            }
            CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(solver0, n, l, &lwork_svd));
            CHECK_CUDA(cudaMalloc(&d_work_svd, static_cast<size_t>(lwork_svd) * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_rwork, static_cast<size_t>(5) * l * sizeof(float)));
        }
        double t_allocate_gpu_buffers_ms = timer.toc_ms();
        double t_setup_gpu_resources_ms = t_create_gpu_handles_ms + t_allocate_gpu_buffers_ms;

        std::vector<double> repeat_compute_ms;
        std::vector<double> repeat_pipeline_ms;
        repeat_compute_ms.reserve(opt.repeat);
        repeat_pipeline_ms.reserve(opt.repeat);

        for (int repeat_idx = 0; repeat_idx < opt.repeat; ++repeat_idx) {
        const bool print_repeat_detail =
            repeat_idx == 0 ||
            repeat_idx + 1 == opt.repeat ||
            (opt.repeat_print_every > 0 && (repeat_idx + 1) % opt.repeat_print_every == 0);
        if (print_repeat_detail) {
            std::cout << "\nRepeat " << (repeat_idx + 1) << "/" << opt.repeat << "\n";
        }

        timer.tic();
        {
            NvtxRange range("local_projection_Yi");
            for (int g = 0; g < opt.ngpus; ++g) {
                DeviceWork& w = works[g];
                CHECK_CUDA(cudaSetDevice(w.dev));
                if (opt.device_random_input) {
                    dim3 block(16, 16);
                    dim3 grid_a((w.mi + block.x - 1) / block.x, (n + block.y - 1) / block.y);
                    const float a_scale = std::sqrt(3.0f) / std::sqrt(static_cast<float>(m));
                    fill_random_colmajor_kernel<<<grid_a, block>>>(
                        w.d_Ai, w.mi, n, w.row0, opt.seed, a_scale);
                    CHECK_CUDA(cudaGetLastError());

                    dim3 grid_omega((n + block.x - 1) / block.x, (l + block.y - 1) / block.y);
                    fill_random_colmajor_kernel<<<grid_omega, block>>>(
                        w.d_Omega, n, l, 0, opt.seed + 1, std::sqrt(3.0f));
                    CHECK_CUDA(cudaGetLastError());
                } else {
                    CHECK_CUDA(cudaMemcpy(w.d_Ai, h_A_blocks[g].data(), static_cast<size_t>(w.mi) * n * sizeof(float), cudaMemcpyHostToDevice));
                    CHECK_CUDA(cudaMemcpy(w.d_Omega, h_Omega.data(), static_cast<size_t>(n) * l * sizeof(float), cudaMemcpyHostToDevice));
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
            if (opt.device_random_input && opt.check_error) {
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
                    h_A_norm2 += static_cast<double>(block_norm) * block_norm;
                }
            }
        }
        double t_local_projection_ms = timer.toc_ms();

        timer.tic();
        std::vector<float> h_Rstack(static_cast<size_t>(opt.ngpus) * l * l, 0.0f);
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
                        h_Rstack[static_cast<size_t>(col) * (opt.ngpus * l) + g * l + row] =
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

            CHECK_CUDA(cudaMemcpy(d_Rstack, h_Rstack.data(), static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyHostToDevice));

            CHECK_CUSOLVER(cusolverDnSgeqrf(solver0, r_rows, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
            CHECK_CUDA(cudaDeviceSynchronize());
            check_solver_info(d_info0, "TSQR geqrf");
            CHECK_CUSOLVER(cusolverDnSorgqr(solver0, r_rows, l, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
            CHECK_CUDA(cudaDeviceSynchronize());
            check_solver_info(d_info0, "TSQR orgqr");

            h_Tstack.resize(static_cast<size_t>(r_rows) * l);
            CHECK_CUDA(cudaMemcpy(h_Tstack.data(), d_Rstack, static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyDeviceToHost));
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
                    const float* src = h_Tstack.data() + static_cast<size_t>(col) * r_rows + g * l;
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

        timer.tic();
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMemset(d_B, 0, bi_fp32_bytes));

        const bool use_host_reduce_b = opt.host_reduce_b;
        std::vector<float> h_B_host;
        if (use_host_reduce_b) {
            h_B_host.assign(b_count, 0.0f);
        }

        size_t b_fp32_payload_bytes = 0;
        size_t b_transmitted_payload_bytes = 0;
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

                    if (use_host_reduce_b) {
                        CHECK_CUDA(cudaSetDevice(w.dev));
                        if (b_quant_options.mode == turboquant::QuantizeMode::kTurboQuant ||
                            b_quant_options.mode == turboquant::QuantizeMode::kTurboQuantQjl) {
                            CHECK_CUDA(cudaMemset(w.d_Bi_hat, 0, bi_fp32_bytes));
                            turboquant::dequantize_column_tq_payload_add_to_fp32(
                                compressed,
                                w.d_Bi_hat,
                                w.d_Bi_tq_work);
                        } else {
                            turboquant::dequantize_device_payload_to_fp32(
                                compressed,
                                w.d_Bi_hat);
                        }
                        d_reconstructed_Bi = w.d_Bi_hat;
                    } else {
                        CHECK_CUDA(cudaSetDevice(0));
                        turboquant::DeviceCompressedBlock compressed_on_gpu0 = compressed;
                        float* d_payload_decode_output = w.d_Bi_hat;
                        if (w.dev != 0) {
                            CHECK_CUDA(cudaMemcpyPeer(
                                d_Bi_codes_on_gpu0[g], 0, w.d_Bi_codes, w.dev, bi_code_bytes));
                            compressed_on_gpu0.d_codes = d_Bi_codes_on_gpu0[g];
                            d_payload_decode_output = d_Bi_reduce_on_gpu0[g];
                            decoded_payload_on_gpu0 = true;
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

                if (use_host_reduce_b) {
                    std::vector<float> h_Bi(b_count);
                    CHECK_CUDA(cudaMemcpy(h_Bi.data(), d_reconstructed_Bi, bi_fp32_bytes, cudaMemcpyDeviceToHost));
                    for (size_t i = 0; i < b_count; ++i) {
                        h_B_host[i] += h_Bi[i];
                    }
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

            if (use_host_reduce_b) {
                CHECK_CUDA(cudaSetDevice(0));
                CHECK_CUDA(cudaMemcpy(d_B, h_B_host.data(), bi_fp32_bytes, cudaMemcpyHostToDevice));
            }
        }
        double t_build_b_reduce_ms = timer.toc_ms();
        const double b_relative_error = opt.check_b_error ?
            std::sqrt(static_cast<double>(b_error_norm2 / std::max(b_ref_norm2, 1e-30L))) :
            -1.0;

        timer.tic();
        {
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

        double t_form_distributed_u_ms = 0.0;
        if (!opt.skip_form_u) {
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
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, static_cast<size_t>(l) * sizeof(float), cudaMemcpyDeviceToHost));

        double rel_err = -1.0;
        double t_err_ms = 0.0;
        if (opt.check_error) {
            timer.tic();
            rel_err = fast_reconstruction_relative_error(h_A_norm2, h_S, k);
            t_err_ms = timer.toc_ms();
            std::cout << "  reconstruction_error_fast_time_ms=" << t_err_ms << "\n";
        }

        const double t_gpu_compute_reported =
            t_local_projection_ms + t_local_qr_ms + t_tsqr_reduce_ms + t_form_distributed_q_ms +
            t_build_b_reduce_ms + t_svd_b_ms + t_form_distributed_u_ms;
        const double t_iter_init_ms = (repeat_idx == 0) ? t_init_ms : 0.0;
        const double t_iter_setup_ms = (repeat_idx == 0) ? t_setup_gpu_resources_ms : 0.0;
        const double t_total_reported = t_iter_init_ms + t_iter_setup_ms + t_gpu_compute_reported;
        const double t_gpu_pipeline_reported = t_iter_setup_ms + t_gpu_compute_reported;
        repeat_pipeline_ms.push_back(t_gpu_pipeline_reported);
        repeat_compute_ms.push_back(t_gpu_compute_reported);

        if (print_repeat_detail) {
            std::cout << "\nTimings (ms)\n"
                      << "  init_host_random          " << t_iter_init_ms << "\n"
                      << "  create_gpu_handles        " << ((repeat_idx == 0) ? t_create_gpu_handles_ms : 0.0) << "\n"
                      << "  allocate_gpu_buffers      " << ((repeat_idx == 0) ? t_allocate_gpu_buffers_ms : 0.0) << "\n"
                      << "  setup_gpu_resources       " << t_iter_setup_ms << "\n"
                      << "  local_projection_Yi       " << t_local_projection_ms << "\n"
                      << "  local_qr_Yi               " << t_local_qr_ms << "\n"
                      << "  tsqr_R_reduce_gpu0        " << t_tsqr_reduce_ms << "\n"
                      << "  form_distributed_Qi       " << t_form_distributed_q_ms << "\n"
                      << "  build_reduce_Bi           " << t_build_b_reduce_ms << "\n"
                      << "  svd_B_on_gpu0             " << t_svd_b_ms << "\n"
                      << "  form_distributed_Ui       " << t_form_distributed_u_ms << "\n"
                      << "  gpu_pipeline_reported     " << t_gpu_pipeline_reported << "\n"
                      << "  gpu_compute_reported      " << t_gpu_compute_reported << "\n"
                      << "  subtotal_reported         " << t_total_reported << "\n";

            std::cout << "\nCommunication baseline\n"
                      << "  tsqr_R_payload_bytes      " << r_payload_bytes << "\n"
                      << "  tsqr_R_payload_MiB        " << (r_payload_bytes / 1024.0 / 1024.0) << "\n"
                      << "  reduce_B_quant_mode       " << turboquant::mode_name(b_quant_options.mode) << "\n"
                      << "  reduce_B_quant_bits       " << opt.compress_b_bits << "\n"
                      << "  reduce_B_qjl_dim          " << b_quant_options.qjl_dim << "\n"
                      << "  reduce_B_qjl_alpha        " << b_quant_options.qjl_alpha << "\n"
                      << "  reduce_B_fp32_bytes       " << b_fp32_payload_bytes << "\n"
                      << "  reduce_B_fp32_MiB         " << (b_fp32_payload_bytes / 1024.0 / 1024.0) << "\n"
                      << "  reduce_B_payload_bytes    " << b_transmitted_payload_bytes << "\n"
                      << "  reduce_B_payload_MiB      " << (b_transmitted_payload_bytes / 1024.0 / 1024.0) << "\n"
                      << "  reduce_B_compression_ratio "
                      << (static_cast<double>(b_fp32_payload_bytes) /
                          static_cast<double>(std::max<size_t>(b_transmitted_payload_bytes, 1))) << "\n"
                      << "  reduce_B_relative_error   "
                      << (opt.check_b_error ? std::to_string(b_relative_error) : std::string("skipped")) << "\n"
                      << "  reduce_B_compress_time_ms " << t_compress_b_ms << "\n";

            std::cout << "\nLeading singular values\n  ";
            for (int i = 0; i < std::min(k, 10); ++i) {
                std::cout << h_S[i] << (i + 1 == std::min(k, 10) ? "" : ", ");
            }
            std::cout << "\n";
            if (opt.check_error) {
                std::cout << "\nFast relative Frobenius reconstruction error\n"
                          << "  sqrt((||A||_F^2 - sum(S_k^2)) / ||A||_F^2) = " << rel_err << "\n";
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

        if (opt.repeat > 1) {
            const size_t warm_begin = 1;
            const auto avg_from = [](const std::vector<double>& xs, size_t begin) {
                double sum = 0.0;
                for (size_t i = begin; i < xs.size(); ++i) sum += xs[i];
                return sum / static_cast<double>(xs.size() - begin);
            };
            const auto min_it_compute = std::min_element(repeat_compute_ms.begin() + warm_begin, repeat_compute_ms.end());
            const auto min_it_pipeline = std::min_element(repeat_pipeline_ms.begin() + warm_begin, repeat_pipeline_ms.end());
            std::cout << "\nRepeat summary\n"
                      << "  repeat_count              " << opt.repeat << "\n"
                      << "  warm_count                " << (opt.repeat - 1) << "\n"
                      << "  cold_pipeline_ms          " << repeat_pipeline_ms.front() << "\n"
                      << "  warm_compute_avg_ms       " << avg_from(repeat_compute_ms, warm_begin) << "\n"
                      << "  warm_compute_min_ms       " << *min_it_compute << "\n"
                      << "  warm_pipeline_avg_ms      " << avg_from(repeat_pipeline_ms, warm_begin) << "\n"
                      << "  warm_pipeline_min_ms      " << *min_it_pipeline << "\n";
        }

        CHECK_CUDA(cudaSetDevice(0));
        cudaFree(d_B);
        cudaFree(d_BT);
        cudaFree(d_V);
        cudaFree(d_S);
        cudaFree(d_UtT);
        cudaFree(d_Ut_k);
        cudaFree(d_payload_decode_work);
        cudaFree(d_work_svd);
        cudaFree(d_rwork);
        cudaFree(d_work_r);
        cudaFree(d_tau_r);
        cudaFree(d_Rstack);
        for (float* p : d_Bi_reduce_on_gpu0) {
            if (p) cudaFree(p);
        }
        for (std::uint8_t* p : d_Bi_codes_on_gpu0) {
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
