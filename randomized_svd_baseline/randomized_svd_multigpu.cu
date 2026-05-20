// randomized_svd_multigpu.cu
//
// Baseline B: custom multi-GPU Randomized SVD baseline.
//
// What this file implements:
//   1. Generate a dense column-major matrix A on host.
//   2. Split A by row blocks across multiple GPUs.
//   3. Each GPU computes local randomized projection: Y_i = A_i * Omega.
//   4. Gather full-precision Y_i back to host, assemble Y.
//   5. On GPU 0, compute QR: Y = Q R.
//   6. On GPU 0, compute B = Q^T A.
//   7. On GPU 0, compute SVD(B) = U_tilde S V^T.
//   8. Compute U_k = Q U_tilde_k.
//   9. Optionally estimate reconstruction error on host.
//
// The communication compression hook is intentionally marked around the Y_i gather.
// TurboQuant/QJL should be inserted after Y_i is produced and before it is copied/gathered.
//
// Build example:
//   nvcc -O3 -std=c++17 randomized_svd_multigpu.cu \
//     -lcublas -lcusolver -o randomized_svd_multigpu
//
// Run example:
//   ./randomized_svd_multigpu --m 4096 --n 2048 --k 64 --oversample 16 --ngpus 2
//
// Notes:
//   * This is a readable research baseline, not yet the fastest possible implementation.
//   * It gathers Y through host memory for portability. Replacing this with NCCL/MPI is the next step.
//   * All matrices are column-major because cuBLAS/cuSOLVER use Fortran-style layout.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
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
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
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

// Extract a row block from full column-major A(m x n) into contiguous column-major Ai(mi x n).
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

// Place Yi(mi x l) into Y(m x l), both column-major.
static void insert_row_block_colmajor(
    std::vector<float>& Y, const std::vector<float>& Yi,
    int m, int l, int row0, int mi) {
    for (int col = 0; col < l; ++col) {
        float* dst = Y.data() + static_cast<size_t>(col) * m + row0;
        const float* src = Yi.data() + static_cast<size_t>(col) * mi;
        std::copy(src, src + mi, dst);
    }
}

static double frobenius_norm(const std::vector<float>& A) {
    long double sum = 0.0L;
    for (float x : A) sum += static_cast<long double>(x) * x;
    return std::sqrt(static_cast<double>(sum));
}

static double reconstruction_relative_error(
    const std::vector<float>& A, int m, int n,
    const std::vector<float>& U, const std::vector<float>& S, const std::vector<float>& VT, int k) {
    // A_hat = U(:,1:k) * diag(S) * VT(1:k,:)
    // Column-major U(m x k), VT(k x n).
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

struct DeviceWork {
    int dev = 0;
    int row0 = 0;
    int mi = 0;
    cublasHandle_t blas = nullptr;
    float* d_Ai = nullptr;     // mi x n
    float* d_Omega = nullptr;  // n x l
    float* d_Yi = nullptr;     // mi x l
};

static void free_device_work(DeviceWork& w) {
    if (w.blas) {
        cudaSetDevice(w.dev);
        cublasDestroy(w.blas);
        w.blas = nullptr;
    }
    if (w.d_Ai) { cudaSetDevice(w.dev); cudaFree(w.d_Ai); w.d_Ai = nullptr; }
    if (w.d_Omega) { cudaSetDevice(w.dev); cudaFree(w.d_Omega); w.d_Omega = nullptr; }
    if (w.d_Yi) { cudaSetDevice(w.dev); cudaFree(w.d_Yi); w.d_Yi = nullptr; }
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

        std::cout << "Randomized multi-GPU SVD baseline\n"
                  << "  m=" << m << " n=" << n << " k=" << k
                  << " oversample=" << opt.oversample << " l=" << l
                  << " ngpus=" << opt.ngpus << "\n";

        Timer timer;
        timer.tic();
        std::vector<float> h_A = make_random_matrix(m, n, opt.seed, 1.0f / std::sqrt(static_cast<float>(m)));
        std::vector<float> h_Omega = make_random_matrix(n, l, opt.seed + 1, 1.0f);
        double t_init_ms = timer.toc_ms();

        // ------------------------------------------------------------------
        // Multi-GPU local randomized projection: Y_i = A_i * Omega.
        // ------------------------------------------------------------------
        timer.tic();
        std::vector<DeviceWork> works(opt.ngpus);
        int row0 = 0;
        for (int g = 0; g < opt.ngpus; ++g) {
            DeviceWork& w = works[g];
            w.dev = g;
            w.row0 = row0;
            w.mi = rows[g];
            row0 += rows[g];

            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUBLAS(cublasCreate(&w.blas));

            std::vector<float> h_Ai = extract_row_block_colmajor(h_A, m, n, w.row0, w.mi);

            CHECK_CUDA(cudaMalloc(&w.d_Ai, static_cast<size_t>(w.mi) * n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&w.d_Omega, static_cast<size_t>(n) * l * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&w.d_Yi, static_cast<size_t>(w.mi) * l * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(w.d_Ai, h_Ai.data(), static_cast<size_t>(w.mi) * n * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(w.d_Omega, h_Omega.data(), static_cast<size_t>(n) * l * sizeof(float), cudaMemcpyHostToDevice));

            const float alpha = 1.0f;
            const float beta = 0.0f;
            // d_Yi(mi x l) = d_Ai(mi x n) * d_Omega(n x l)
            CHECK_CUBLAS(cublasSgemm(
                w.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                w.mi, l, n,
                &alpha,
                w.d_Ai, w.mi,
                w.d_Omega, n,
                &beta,
                w.d_Yi, w.mi));
        }
        for (int g = 0; g < opt.ngpus; ++g) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        double t_local_projection_ms = timer.toc_ms();

        // ------------------------------------------------------------------
        // Communication hook / gather Y.
        //
        // Baseline: copy Y_i in FP32 from each GPU to host and assemble Y.
        //
        // TurboQuant + QJL insertion point:
        //   encode d_Yi -> compressed payload on each GPU,
        //   send payload through NCCL/MPI/host,
        //   decode/reconstruct approximate Y_i on receiver,
        //   then assemble approximate Y.
        // ------------------------------------------------------------------
        timer.tic();
        std::vector<float> h_Y(static_cast<size_t>(m) * l);
        size_t fp32_payload_bytes = 0;
        for (int g = 0; g < opt.ngpus; ++g) {
            DeviceWork& w = works[g];
            CHECK_CUDA(cudaSetDevice(w.dev));
            std::vector<float> h_Yi(static_cast<size_t>(w.mi) * l);
            CHECK_CUDA(cudaMemcpy(h_Yi.data(), w.d_Yi,
                                  static_cast<size_t>(w.mi) * l * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            insert_row_block_colmajor(h_Y, h_Yi, m, l, w.row0, w.mi);
            fp32_payload_bytes += static_cast<size_t>(w.mi) * l * sizeof(float);
        }
        double t_gather_y_ms = timer.toc_ms();

        for (auto& w : works) free_device_work(w);

        // ------------------------------------------------------------------
        // GPU 0: QR(Y) -> Q, B = Q^T A, SVD(B), U = Q U_tilde.
        // ------------------------------------------------------------------
        CHECK_CUDA(cudaSetDevice(0));
        cublasHandle_t blas0 = nullptr;
        cusolverDnHandle_t solver = nullptr;
        CHECK_CUBLAS(cublasCreate(&blas0));
        CHECK_CUSOLVER(cusolverDnCreate(&solver));

        float* d_A = nullptr;
        float* d_Q = nullptr;
        float* d_tau = nullptr;
        float* d_B = nullptr;
        float* d_BT = nullptr;  // B transposed (n x l) for SVD when l < n
        float* d_Ut = nullptr;
        float* d_S = nullptr;
        float* d_VT = nullptr;
        float* d_Uk = nullptr;
        float* d_Ut_k = nullptr;
        int* d_info = nullptr;
        float* d_work = nullptr;
        float* d_rwork = nullptr;

        CHECK_CUDA(cudaMalloc(&d_A, static_cast<size_t>(m) * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Q, static_cast<size_t>(m) * l * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tau, static_cast<size_t>(l) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(l) * n * sizeof(float)));
        // B^T (n x l): cuSOLVER SVD requires rows >= cols; since l <= min(m,n) <= n, n x l is safe.
        CHECK_CUDA(cudaMalloc(&d_BT, static_cast<size_t>(n) * l * sizeof(float)));
        // d_Ut holds V (n x l) from SVD(B^T)
        CHECK_CUDA(cudaMalloc(&d_Ut, static_cast<size_t>(n) * l * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_S, static_cast<size_t>(l) * sizeof(float)));
        // d_VT holds U_tilde^T (l x l) from SVD(B^T)
        CHECK_CUDA(cudaMalloc(&d_VT, static_cast<size_t>(l) * l * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Uk, static_cast<size_t>(m) * k * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Ut_k, static_cast<size_t>(l) * k * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), static_cast<size_t>(m) * n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_Q, h_Y.data(), static_cast<size_t>(m) * l * sizeof(float), cudaMemcpyHostToDevice));

        // QR: geqrf overwrites d_Q with Householder vectors, then orgqr forms explicit Q.
        timer.tic();
        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(solver, m, l, d_Q, m, &lwork_geqrf));
        CHECK_CUSOLVER(cusolverDnSorgqr_bufferSize(solver, m, l, l, d_Q, m, d_tau, &lwork_orgqr));
        int lwork_qr = std::max(lwork_geqrf, lwork_orgqr);
        CHECK_CUDA(cudaMalloc(&d_work, static_cast<size_t>(lwork_qr) * sizeof(float)));
        CHECK_CUSOLVER(cusolverDnSgeqrf(solver, m, l, d_Q, m, d_tau, d_work, lwork_qr, d_info));
        CHECK_CUDA(cudaDeviceSynchronize());
        int info = 0;
        CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) throw std::runtime_error("geqrf failed, info=" + std::to_string(info));
        CHECK_CUSOLVER(cusolverDnSorgqr(solver, m, l, l, d_Q, m, d_tau, d_work, lwork_qr, d_info));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) throw std::runtime_error("orgqr failed, info=" + std::to_string(info));
        CHECK_CUDA(cudaFree(d_work));
        d_work = nullptr;
        double t_qr_ms = timer.toc_ms();

        // B(l x n) = Q^T(l x m) * A(m x n).
        timer.tic();
        const float alpha = 1.0f;
        const float beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(
            blas0, CUBLAS_OP_T, CUBLAS_OP_N,
            l, n, m,
            &alpha,
            d_Q, m,
            d_A, m,
            &beta,
            d_B, l));
        CHECK_CUDA(cudaDeviceSynchronize());
        double t_build_b_ms = timer.toc_ms();

        // Transpose B(l x n) -> B^T(n x l) using cublasSgeam.
        // B is col-major (l x n), B^T is col-major (n x l).
        timer.tic();
        {
            const float one = 1.0f, zero = 0.0f;
            CHECK_CUBLAS(cublasSgeam(
                blas0,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, l,           // output dimensions
                &one,
                d_B, l,         // input A (l x n col-major), transposed
                &zero,
                nullptr, n,
                d_BT, n));      // output (n x l col-major)
        }

        // SVD of B^T(n x l): B^T = V * S * U_tilde^T
        //   d_Ut  <- V       (n x l)
        //   d_S   <- S       (l)
        //   d_VT  <- U_tilde^T (l x l)
        int lwork_svd = 0;
        CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(solver, n, l, &lwork_svd));
        CHECK_CUDA(cudaMalloc(&d_work, static_cast<size_t>(lwork_svd) * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_rwork, static_cast<size_t>(5 * l) * sizeof(float)));
        // jobu  = 'S': first l columns of V  (left singular vectors of B^T)
        // jobvt = 'A': all l rows of U_tilde^T
        signed char jobu  = 'S';  // first min(n,l)=l left singular vectors of B^T → V
        signed char jobvt = 'A';  // all l right singular vectors of B^T → U_tilde^T
        CHECK_CUSOLVER(cusolverDnSgesvd(
            solver, jobu, jobvt,
            n, l,           // n x l matrix (n >= l guaranteed since l = k+p <= min(m,n))
            d_BT, n,        // leading dim of BT is n
            d_S,
            d_Ut, n,        // V: n x l, leading dim n
            d_VT, l,        // U_tilde^T: l x l, leading dim l
            d_work, lwork_svd,
            d_rwork,
            d_info));
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) throw std::runtime_error("gesvd failed, info=" + std::to_string(info));
        CHECK_CUDA(cudaFree(d_work));
        d_work = nullptr;
        CHECK_CUDA(cudaFree(d_rwork));
        d_rwork = nullptr;
        double t_svd_b_ms = timer.toc_ms();

        // U_tilde is stored as U_tilde^T in d_VT (l x l, col-major).
        // First k singular vectors of U_tilde = first k rows of U_tilde^T = first k columns of d_VT.
        // So d_Ut_k = first k columns of d_VT (l x k, col-major), contiguous.
        CHECK_CUDA(cudaMemcpy(d_Ut_k, d_VT, static_cast<size_t>(l) * k * sizeof(float), cudaMemcpyDeviceToDevice));

        // U_k(m x k) = Q(m x l) * U_tilde_k(l x k).
        // d_Ut_k holds the first k columns of d_VT = U_tilde_k already in col-major (l x k).
        timer.tic();
        CHECK_CUBLAS(cublasSgemm(
            blas0, CUBLAS_OP_N, CUBLAS_OP_N,
            m, k, l,
            &alpha,
            d_Q, m,
            d_Ut_k, l,
            &beta,
            d_Uk, m));
        CHECK_CUDA(cudaDeviceSynchronize());
        double t_form_u_ms = timer.toc_ms();

        std::vector<float> h_S(l);
        std::vector<float> h_Uk(static_cast<size_t>(m) * k);
        std::vector<float> h_VT_k(static_cast<size_t>(k) * n);
        CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, static_cast<size_t>(l) * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_Uk.data(), d_Uk, static_cast<size_t>(m) * k * sizeof(float), cudaMemcpyDeviceToHost));

        // d_Ut holds V (n x l, col-major) from SVD(B^T).
        // VT_k = first k rows of V^T = first k columns of V (n x k, col-major).
        // For the error check we need VT_k stored as (k x n, col-major) — i.e., each column j
        // contains the k entries VT_k[:,j] = V[j, 0:k].
        std::vector<float> h_V_nl(static_cast<size_t>(n) * l);
        CHECK_CUDA(cudaMemcpy(h_V_nl.data(), d_Ut, static_cast<size_t>(n) * l * sizeof(float), cudaMemcpyDeviceToHost));
        // h_VT_k is col-major (k x n): for column j of A, VT_k[:, j] = V[j, 0:k]
        // V is col-major (n x l): V[row, col] = h_V_nl[col * n + row]
        // So V[j, r] = h_V_nl[r * n + j]
        for (int col = 0; col < n; ++col) {
            for (int r = 0; r < k; ++r) {
                // V[col, r] in col-major(n x l): index = r * n + col
                h_VT_k[static_cast<size_t>(col) * k + r] = h_V_nl[static_cast<size_t>(r) * n + col];
            }
        }

        double rel_err = -1.0;
        if (opt.check_error) {
            timer.tic();
            rel_err = reconstruction_relative_error(h_A, m, n, h_Uk, h_S, h_VT_k, k);
            double t_err_ms = timer.toc_ms();
            std::cout << "  reconstruction_error_time_ms=" << t_err_ms << "\n";
        }

        double t_total_reported = t_init_ms + t_local_projection_ms + t_gather_y_ms +
                                  t_qr_ms + t_build_b_ms + t_svd_b_ms + t_form_u_ms;

        std::cout << "\nTimings (ms)\n"
                  << "  init_host_random          " << t_init_ms << "\n"
                  << "  local_projection_Yi       " << t_local_projection_ms << "\n"
                  << "  gather_Y_fp32             " << t_gather_y_ms << "\n"
                  << "  qr_Y_on_gpu0              " << t_qr_ms << "\n"
                  << "  build_B_QtA_on_gpu0       " << t_build_b_ms << "\n"
                  << "  svd_B_on_gpu0             " << t_svd_b_ms << "\n"
                  << "  form_Uk_QUtilde_on_gpu0   " << t_form_u_ms << "\n"
                  << "  subtotal_reported         " << t_total_reported << "\n";

        std::cout << "\nCommunication baseline\n"
                  << "  gathered_tensor           Y = A * Omega\n"
                  << "  fp32_payload_bytes        " << fp32_payload_bytes << "\n"
                  << "  fp32_payload_MiB          " << (fp32_payload_bytes / 1024.0 / 1024.0) << "\n";

        std::cout << "\nLeading singular values\n  ";
        for (int i = 0; i < std::min(k, 10); ++i) std::cout << h_S[i] << (i + 1 == std::min(k, 10) ? "" : ", ");
        std::cout << "\n";
        if (opt.check_error) {
            std::cout << "\nRelative Frobenius reconstruction error\n"
                      << "  ||A - Uk Sk Vk^T||_F / ||A||_F = " << rel_err << "\n";
        }

        cudaFree(d_A);
        cudaFree(d_Q);
        cudaFree(d_tau);
        cudaFree(d_B);
        cudaFree(d_BT);
        cudaFree(d_Ut);
        cudaFree(d_S);
        cudaFree(d_VT);
        cudaFree(d_Uk);
        cudaFree(d_Ut_k);
        cudaFree(d_info);
        cublasDestroy(blas0);
        cusolverDnDestroy(solver);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "!Error: " << e.what() << "\n";
        return 1;
    }
}
