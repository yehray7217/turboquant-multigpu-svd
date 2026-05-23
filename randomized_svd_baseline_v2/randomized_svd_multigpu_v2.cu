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
//   6. B = sum_i B_i on host, then SVD(B) on GPU 0
//   7. U_i = Q_i * U_tilde_k on GPU i
//
// This is still a research baseline, not a production NCCL implementation.
// The communication hooks are the R_i TSQR gather and the B_i reduction.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

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
        << "  --no-check-error      Skip host reconstruction error.\n";
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
        else if (a == "--no-check-error") opt.check_error = false;
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
    if (opt.k + opt.oversample > std::min(opt.m, opt.n)) {
        throw std::runtime_error("Require k + oversample <= min(m, n).");
    }
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

static std::vector<float> extract_row_block_colmajor(
    const std::vector<float>& A, int m, int n, int row0, int mi) {
    std::vector<float> Ai(static_cast<size_t>(mi) * n);
    for (int col = 0; col < n; ++col) {
        const float* src = A.data() + static_cast<size_t>(col) * m + row0;
        float* dst = Ai.data() + static_cast<size_t>(col) * mi;
        std::copy(src, src + mi, dst);
    }
    return Ai;
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

static double reconstruction_relative_error(
    const std::vector<float>& A, int m, int n,
    const std::vector<float>& U, const std::vector<float>& S, const std::vector<float>& VT, int k) {
    long double err2 = 0.0L;
    long double norm2 = 0.0L;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            long double ahat = 0.0L;
            for (int r = 0; r < k; ++r) {
                ahat += static_cast<long double>(U[static_cast<size_t>(r) * m + row]) *
                        static_cast<long double>(S[r]) *
                        static_cast<long double>(VT[static_cast<size_t>(col) * k + r]);
            }
            float a = A[static_cast<size_t>(col) * m + row];
            long double diff = static_cast<long double>(a) - ahat;
            err2 += diff * diff;
            norm2 += static_cast<long double>(a) * a;
        }
    }
    return std::sqrt(static_cast<double>(err2 / std::max(norm2, 1e-30L)));
}

__global__ void transpose_colmajor_kernel(const float* src, float* dst, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        dst[static_cast<size_t>(row) * cols + col] = src[static_cast<size_t>(col) * rows + row];
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
    float* d_Ti = nullptr;       // l x l
    float* d_Qi = nullptr;       // mi x l
    float* d_Bi = nullptr;       // l x n
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
    if (w.d_Ti) cudaFree(w.d_Ti);
    if (w.d_Qi) cudaFree(w.d_Qi);
    if (w.d_Bi) cudaFree(w.d_Bi);
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
                  << " ngpus=" << opt.ngpus << "\n";

        Timer timer;
        timer.tic();
        std::vector<float> h_A = make_random_matrix(m, n, opt.seed, 1.0f / std::sqrt(static_cast<float>(m)));
        std::vector<float> h_Omega = make_random_matrix(n, l, opt.seed + 1, 1.0f);
        double t_init_ms = timer.toc_ms();

        std::vector<DeviceWork> works(opt.ngpus);

        timer.tic();
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

            std::vector<float> h_Ai = extract_row_block_colmajor(h_A, m, n, w.row0, w.mi);

            CHECK_CUDA(cudaMalloc(&w.d_Ai, static_cast<size_t>(w.mi) * n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&w.d_Omega, static_cast<size_t>(n) * l * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&w.d_Qbar, static_cast<size_t>(w.mi) * l * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&w.d_tau, static_cast<size_t>(l) * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&w.d_info, sizeof(int)));

            CHECK_CUDA(cudaMemcpy(w.d_Ai, h_Ai.data(), static_cast<size_t>(w.mi) * n * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(w.d_Omega, h_Omega.data(), static_cast<size_t>(n) * l * sizeof(float), cudaMemcpyHostToDevice));

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
        double t_local_projection_ms = timer.toc_ms();

        timer.tic();
        std::vector<float> h_Rstack(static_cast<size_t>(opt.ngpus) * l * l, 0.0f);
        size_t r_payload_bytes = 0;
        for (int g = 0; g < opt.ngpus; ++g) {
            DeviceWork& w = works[g];
            CHECK_CUDA(cudaSetDevice(w.dev));

            int lwork_geqrf = 0;
            int lwork_orgqr = 0;
            CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(w.solver, w.mi, l, w.d_Qbar, w.mi, &lwork_geqrf));
            CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(w.solver, w.mi, l, l, w.d_Qbar, w.mi, w.d_tau, &lwork_orgqr));
            int lwork = std::max(lwork_geqrf, lwork_orgqr);
            float* d_work = nullptr;
            CHECK_CUDA(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(float)));

            CHECK_CUSOLVER(cusolverDnSgeqrf(w.solver, w.mi, l, w.d_Qbar, w.mi, w.d_tau, d_work, lwork, w.d_info));
            CHECK_CUDA(cudaDeviceSynchronize());
            check_solver_info(w.d_info, "local geqrf");

            std::vector<float> h_factored(static_cast<size_t>(w.mi) * l);
            CHECK_CUDA(cudaMemcpy(h_factored.data(), w.d_Qbar, static_cast<size_t>(w.mi) * l * sizeof(float), cudaMemcpyDeviceToHost));
            for (int col = 0; col < l; ++col) {
                for (int row = 0; row <= col; ++row) {
                    h_Rstack[static_cast<size_t>(col) * (opt.ngpus * l) + g * l + row] =
                        h_factored[static_cast<size_t>(col) * w.mi + row];
                }
            }
            r_payload_bytes += static_cast<size_t>(l) * l * sizeof(float);

            CHECK_CUSOLVER(cusolverDnSorgqr(w.solver, w.mi, l, l, w.d_Qbar, w.mi, w.d_tau, d_work, lwork, w.d_info));
            CHECK_CUDA(cudaDeviceSynchronize());
            check_solver_info(w.d_info, "local orgqr");

            CHECK_CUDA(cudaFree(d_work));
        }
        double t_local_qr_ms = timer.toc_ms();

        timer.tic();
        CHECK_CUDA(cudaSetDevice(0));
        cublasHandle_t blas0 = nullptr;
        cusolverDnHandle_t solver0 = nullptr;
        CHECK_CUBLAS(cublasCreate(&blas0));
        CHECK_CUSOLVER(cusolverDnCreate(&solver0));

        const int r_rows = opt.ngpus * l;
        float* d_Rstack = nullptr;
        float* d_tau_r = nullptr;
        float* d_work_r = nullptr;
        int* d_info0 = nullptr;
        CHECK_CUDA(cudaMalloc(&d_Rstack, static_cast<size_t>(r_rows) * l * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tau_r, static_cast<size_t>(l) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_info0, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_Rstack, h_Rstack.data(), static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyHostToDevice));

        int lwork_r_geqrf = 0;
        int lwork_r_orgqr = 0;
        CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver0, r_rows, l, d_Rstack, r_rows, &lwork_r_geqrf));
        CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(solver0, r_rows, l, l, d_Rstack, r_rows, d_tau_r, &lwork_r_orgqr));
        int lwork_r = std::max(lwork_r_geqrf, lwork_r_orgqr);
        CHECK_CUDA(cudaMalloc(&d_work_r, static_cast<size_t>(lwork_r) * sizeof(float)));
        CHECK_CUSOLVER(cusolverDnSgeqrf(solver0, r_rows, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
        CHECK_CUDA(cudaDeviceSynchronize());
        check_solver_info(d_info0, "TSQR geqrf");
        CHECK_CUSOLVER(cusolverDnSorgqr(solver0, r_rows, l, l, d_Rstack, r_rows, d_tau_r, d_work_r, lwork_r, d_info0));
        CHECK_CUDA(cudaDeviceSynchronize());
        check_solver_info(d_info0, "TSQR orgqr");

        std::vector<float> h_Tstack(static_cast<size_t>(r_rows) * l);
        CHECK_CUDA(cudaMemcpy(h_Tstack.data(), d_Rstack, static_cast<size_t>(r_rows) * l * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(d_work_r));
        CHECK_CUDA(cudaFree(d_tau_r));
        CHECK_CUDA(cudaFree(d_Rstack));
        double t_tsqr_reduce_ms = timer.toc_ms();

        timer.tic();
        for (int g = 0; g < opt.ngpus; ++g) {
            DeviceWork& w = works[g];
            CHECK_CUDA(cudaSetDevice(w.dev));
            std::vector<float> h_Ti(static_cast<size_t>(l) * l);
            for (int col = 0; col < l; ++col) {
                const float* src = h_Tstack.data() + static_cast<size_t>(col) * r_rows + g * l;
                float* dst = h_Ti.data() + static_cast<size_t>(col) * l;
                std::copy(src, src + l, dst);
            }
            CHECK_CUDA(cudaMalloc(&w.d_Ti, static_cast<size_t>(l) * l * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&w.d_Qi, static_cast<size_t>(w.mi) * l * sizeof(float)));
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
        double t_form_distributed_q_ms = timer.toc_ms();

        timer.tic();
        std::vector<float> h_B(static_cast<size_t>(l) * n, 0.0f);
        size_t b_payload_bytes = 0;
        for (int g = 0; g < opt.ngpus; ++g) {
            DeviceWork& w = works[g];
            CHECK_CUDA(cudaSetDevice(w.dev));
            CHECK_CUDA(cudaMalloc(&w.d_Bi, static_cast<size_t>(l) * n * sizeof(float)));
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
            CHECK_CUDA(cudaDeviceSynchronize());

            std::vector<float> h_Bi(static_cast<size_t>(l) * n);
            CHECK_CUDA(cudaMemcpy(h_Bi.data(), w.d_Bi, static_cast<size_t>(l) * n * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < h_B.size(); ++i) h_B[i] += h_Bi[i];
            b_payload_bytes += static_cast<size_t>(l) * n * sizeof(float);
        }
        double t_build_b_reduce_ms = timer.toc_ms();

        timer.tic();
        CHECK_CUDA(cudaSetDevice(0));
        float* d_B = nullptr;
        float* d_BT = nullptr;
        float* d_V = nullptr;
        float* d_S = nullptr;
        float* d_UtT = nullptr;
        float* d_work_svd = nullptr;
        float* d_rwork = nullptr;
        float* d_Ut_k = nullptr;
        CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(l) * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_BT, static_cast<size_t>(n) * l * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, static_cast<size_t>(n) * l * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_S, static_cast<size_t>(l) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_UtT, static_cast<size_t>(l) * l * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Ut_k, static_cast<size_t>(l) * k * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), static_cast<size_t>(l) * n * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((l + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        transpose_colmajor_kernel<<<grid, block>>>(d_B, d_BT, l, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        int lwork_svd = 0;
        CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(solver0, n, l, &lwork_svd));
        CHECK_CUDA(cudaMalloc(&d_work_svd, static_cast<size_t>(lwork_svd) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_rwork, static_cast<size_t>(5) * l * sizeof(float)));
        signed char jobu = 'S';
        signed char jobvt = 'A';
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
        CHECK_CUDA(cudaMemcpy(d_Ut_k, d_UtT, static_cast<size_t>(l) * k * sizeof(float), cudaMemcpyDeviceToDevice));
        double t_svd_b_ms = timer.toc_ms();

        timer.tic();
        std::vector<float> h_Ut_k(static_cast<size_t>(l) * k);
        CHECK_CUDA(cudaMemcpy(h_Ut_k.data(), d_Ut_k, static_cast<size_t>(l) * k * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_Uk(static_cast<size_t>(m) * k);
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
        double t_form_distributed_u_ms = timer.toc_ms();

        std::vector<float> h_S(l);
        std::vector<float> h_V_nl(static_cast<size_t>(n) * l);
        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, static_cast<size_t>(l) * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_V_nl.data(), d_V, static_cast<size_t>(n) * l * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<float> h_VT_k(static_cast<size_t>(k) * n);
        for (int col = 0; col < n; ++col) {
            for (int r = 0; r < k; ++r) {
                h_VT_k[static_cast<size_t>(col) * k + r] = h_V_nl[static_cast<size_t>(r) * n + col];
            }
        }

        double rel_err = -1.0;
        double t_err_ms = 0.0;
        if (opt.check_error) {
            timer.tic();
            rel_err = reconstruction_relative_error(h_A, m, n, h_Uk, h_S, h_VT_k, k);
            t_err_ms = timer.toc_ms();
            std::cout << "  reconstruction_error_time_ms=" << t_err_ms << "\n";
        }

        double t_total_reported = t_init_ms + t_local_projection_ms + t_local_qr_ms +
                                  t_tsqr_reduce_ms + t_form_distributed_q_ms +
                                  t_build_b_reduce_ms + t_svd_b_ms + t_form_distributed_u_ms;

        std::cout << "\nTimings (ms)\n"
                  << "  init_host_random          " << t_init_ms << "\n"
                  << "  local_projection_Yi       " << t_local_projection_ms << "\n"
                  << "  local_qr_Yi               " << t_local_qr_ms << "\n"
                  << "  tsqr_R_reduce_gpu0        " << t_tsqr_reduce_ms << "\n"
                  << "  form_distributed_Qi       " << t_form_distributed_q_ms << "\n"
                  << "  build_reduce_Bi           " << t_build_b_reduce_ms << "\n"
                  << "  svd_B_on_gpu0             " << t_svd_b_ms << "\n"
                  << "  form_distributed_Ui       " << t_form_distributed_u_ms << "\n"
                  << "  subtotal_reported         " << t_total_reported << "\n";

        std::cout << "\nCommunication baseline\n"
                  << "  tsqr_R_payload_bytes      " << r_payload_bytes << "\n"
                  << "  tsqr_R_payload_MiB        " << (r_payload_bytes / 1024.0 / 1024.0) << "\n"
                  << "  reduce_B_payload_bytes    " << b_payload_bytes << "\n"
                  << "  reduce_B_payload_MiB      " << (b_payload_bytes / 1024.0 / 1024.0) << "\n";

        std::cout << "\nLeading singular values\n  ";
        for (int i = 0; i < std::min(k, 10); ++i) {
            std::cout << h_S[i] << (i + 1 == std::min(k, 10) ? "" : ", ");
        }
        std::cout << "\n";
        if (opt.check_error) {
            std::cout << "\nRelative Frobenius reconstruction error\n"
                      << "  ||A - Uk Sk Vk^T||_F / ||A||_F = " << rel_err << "\n";
        }

        CHECK_CUDA(cudaSetDevice(0));
        cudaFree(d_B);
        cudaFree(d_BT);
        cudaFree(d_V);
        cudaFree(d_S);
        cudaFree(d_UtT);
        cudaFree(d_Ut_k);
        cudaFree(d_work_svd);
        cudaFree(d_rwork);
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
