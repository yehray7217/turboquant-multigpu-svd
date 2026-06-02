#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status__ = (call);                                          \
        if (status__ != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(status__));             \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status__ = (call);                                       \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                \
            throw std::runtime_error("cuBLAS error");                          \
        }                                                                       \
    } while (0)

struct Options {
    int m = 4096;
    int n = 1024;
    int l = 256;
    int sketch_dim = 256;
    int block_rows = 4096;
    unsigned seed = 1234;
};

class GpuTimer {
public:
    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_));
        CHECK_CUDA(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void tic() { CHECK_CUDA(cudaEventRecord(start_)); }
    float toc_ms() {
        CHECK_CUDA(cudaEventRecord(stop_));
        CHECK_CUDA(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

__host__ __device__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__host__ __device__ float uniform_centered(unsigned int seed, unsigned int a, unsigned int b) {
    unsigned int h = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU));
    float u = static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
    return 2.0f * u - 1.0f;
}

__host__ __device__ float rademacher(unsigned int seed, unsigned int a, unsigned int b) {
    unsigned int h = mix32(seed ^ (a * 0x9e3779b9U) ^ (b * 0x85ebca6bU));
    return (h & 1U) ? 1.0f : -1.0f;
}

__global__ void fill_random_colmajor_kernel(
    float* dst,
    int rows,
    int cols,
    unsigned seed,
    float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        dst[static_cast<size_t>(col) * rows + row] =
            uniform_centered(seed, static_cast<unsigned int>(row),
                             static_cast<unsigned int>(col)) * scale;
    }
}

__global__ void implicit_block_rademacher_sketch_kernel(
    const float* x,
    float* sx,
    int rows,
    int cols,
    int sketch_dim,
    int block_rows,
    int num_blocks,
    unsigned seed) {
    int total_s = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int total_sketch_dim = sketch_dim * num_blocks;
    if (total_s >= total_sketch_dim || col >= cols) return;

    int block_id = total_s / sketch_dim;
    int s = total_s - block_id * sketch_dim;
    int row_begin = block_id * block_rows;
    int row_end = min(rows, row_begin + block_rows);

    float sum = 0.0f;
    const float* x_col = x + static_cast<size_t>(col) * rows;
    for (int row = row_begin; row < row_end; ++row) {
        sum += rademacher(seed ^ static_cast<unsigned int>(block_id * 0x27d4eb2dU),
                          static_cast<unsigned int>(s),
                          static_cast<unsigned int>(row)) * x_col[row];
    }
    sx[static_cast<size_t>(col) * total_sketch_dim + total_s] = sum;
}

__global__ void sign_inplace_kernel(float* x, size_t count) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        x[idx] = (x[idx] >= 0.0f) ? 1.0f : -1.0f;
    }
}

__global__ void diff_kernel(const float* a, const float* b, float* out, size_t count) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) out[idx] = a[idx] - b[idx];
}

Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const std::string& name) -> char* {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };
        if (a == "--m") opt.m = std::stoi(need(a));
        else if (a == "--n") opt.n = std::stoi(need(a));
        else if (a == "--l") opt.l = std::stoi(need(a));
        else if (a == "--sketch-dim") opt.sketch_dim = std::stoi(need(a));
        else if (a == "--block-rows") opt.block_rows = std::stoi(need(a));
        else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need(a)));
        else throw std::runtime_error("Unknown option: " + a);
    }
    if (opt.m <= 0 || opt.n <= 0 || opt.l <= 0 || opt.sketch_dim <= 0 || opt.block_rows <= 0) {
        throw std::runtime_error("m, n, l, sketch-dim, and block-rows must be positive.");
    }
    opt.block_rows = std::min(opt.block_rows, opt.m);
    return opt;
}

double device_norm(cublasHandle_t handle, const float* d_x, size_t count) {
    constexpr size_t kMaxChunk = 1u << 30;
    long double norm2 = 0.0L;
    for (size_t offset = 0; offset < count; offset += kMaxChunk) {
        size_t chunk = std::min(kMaxChunk, count - offset);
        float chunk_norm = 0.0f;
        CHECK_CUBLAS(cublasSnrm2(handle, static_cast<int>(chunk), d_x + offset, 1, &chunk_norm));
        norm2 += static_cast<long double>(chunk_norm) * static_cast<long double>(chunk_norm);
    }
    return std::sqrt(static_cast<double>(norm2));
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        cublasHandle_t blas = nullptr;
        CHECK_CUBLAS(cublasCreate(&blas));

        float* d_Q = nullptr;
        float* d_A = nullptr;
        float* d_B = nullptr;
        float* d_B_hat = nullptr;
        float* d_B_sign = nullptr;
        float* d_diff = nullptr;
        float* d_SQ = nullptr;
        float* d_SA = nullptr;
        float* d_SQ_sign = nullptr;
        float* d_SA_sign = nullptr;

        const size_t q_count = static_cast<size_t>(opt.m) * opt.l;
        const size_t a_count = static_cast<size_t>(opt.m) * opt.n;
        const size_t b_count = static_cast<size_t>(opt.l) * opt.n;
        const int num_blocks = (opt.m + opt.block_rows - 1) / opt.block_rows;
        const int total_sketch_dim = opt.sketch_dim * num_blocks;
        const size_t sq_count = static_cast<size_t>(total_sketch_dim) * opt.l;
        const size_t sa_count = static_cast<size_t>(total_sketch_dim) * opt.n;

        CHECK_CUDA(cudaMalloc(&d_Q, q_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_A, a_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, b_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_hat, b_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B_sign, b_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_diff, b_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_SQ, sq_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_SA, sa_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_SQ_sign, sq_count * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_SA_sign, sa_count * sizeof(float)));

        dim3 block2d(16, 16);
        dim3 q_grid((opt.m + block2d.x - 1) / block2d.x, (opt.l + block2d.y - 1) / block2d.y);
        dim3 a_grid((opt.m + block2d.x - 1) / block2d.x, (opt.n + block2d.y - 1) / block2d.y);
        fill_random_colmajor_kernel<<<q_grid, block2d>>>(
            d_Q, opt.m, opt.l, opt.seed, 1.0f / std::sqrt(static_cast<float>(opt.m)));
        fill_random_colmajor_kernel<<<a_grid, block2d>>>(
            d_A, opt.m, opt.n, opt.seed + 1U, 1.0f / std::sqrt(static_cast<float>(opt.m)));
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        const float one = 1.0f;
        const float zero = 0.0f;
        GpuTimer timer;

        timer.tic();
        CHECK_CUBLAS(cublasSgemm(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            opt.l, opt.n, opt.m,
            &one,
            d_Q, opt.m,
            d_A, opt.m,
            &zero,
            d_B, opt.l));
        CHECK_CUDA(cudaDeviceSynchronize());
        const float exact_gemm_ms = timer.toc_ms();

        dim3 sketch_block(16, 16);
        dim3 sq_grid((total_sketch_dim + sketch_block.x - 1) / sketch_block.x,
                     (opt.l + sketch_block.y - 1) / sketch_block.y);
        dim3 sa_grid((total_sketch_dim + sketch_block.x - 1) / sketch_block.x,
                     (opt.n + sketch_block.y - 1) / sketch_block.y);

        timer.tic();
        implicit_block_rademacher_sketch_kernel<<<sq_grid, sketch_block>>>(
            d_Q, d_SQ, opt.m, opt.l, opt.sketch_dim, opt.block_rows, num_blocks, opt.seed + 17U);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const float sketch_q_ms = timer.toc_ms();

        timer.tic();
        implicit_block_rademacher_sketch_kernel<<<sa_grid, sketch_block>>>(
            d_A, d_SA, opt.m, opt.n, opt.sketch_dim, opt.block_rows, num_blocks, opt.seed + 17U);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const float sketch_a_ms = timer.toc_ms();

        const float inv_s = 1.0f / static_cast<float>(opt.sketch_dim);
        timer.tic();
        CHECK_CUBLAS(cublasSgemm(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            opt.l, opt.n, total_sketch_dim,
            &inv_s,
            d_SQ, total_sketch_dim,
            d_SA, total_sketch_dim,
            &zero,
            d_B_hat, opt.l));
        CHECK_CUDA(cudaDeviceSynchronize());
        const float sketch_gemm_ms = timer.toc_ms();

        CHECK_CUDA(cudaMemcpy(d_SQ_sign, d_SQ, sq_count * sizeof(float), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_SA_sign, d_SA, sa_count * sizeof(float), cudaMemcpyDeviceToDevice));
        int threads = 256;
        sign_inplace_kernel<<<static_cast<int>((sq_count + threads - 1) / threads), threads>>>(d_SQ_sign, sq_count);
        sign_inplace_kernel<<<static_cast<int>((sa_count + threads - 1) / threads), threads>>>(d_SA_sign, sa_count);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        timer.tic();
        CHECK_CUBLAS(cublasSgemm(
            blas, CUBLAS_OP_T, CUBLAS_OP_N,
            opt.l, opt.n, total_sketch_dim,
            &inv_s,
            d_SQ_sign, total_sketch_dim,
            d_SA_sign, total_sketch_dim,
            &zero,
            d_B_sign, opt.l));
        CHECK_CUDA(cudaDeviceSynchronize());
        const float sign_gemm_ms = timer.toc_ms();

        diff_kernel<<<static_cast<int>((b_count + threads - 1) / threads), threads>>>(
            d_B_hat, d_B, d_diff, b_count);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const double b_norm = device_norm(blas, d_B, b_count);
        const double jl_diff_norm = device_norm(blas, d_diff, b_count);

        diff_kernel<<<static_cast<int>((b_count + threads - 1) / threads), threads>>>(
            d_B_sign, d_B, d_diff, b_count);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        const double sign_diff_norm = device_norm(blas, d_diff, b_count);

        const double jl_relative_error = jl_diff_norm / std::max(b_norm, 1.0e-30);
        const double sign_proxy_error = sign_diff_norm / std::max(b_norm, 1.0e-30);
        const double jl_total_ms = sketch_q_ms + sketch_a_ms + sketch_gemm_ms;

        std::cout << "QJL_INNER_PRODUCT_RESULT"
                  << ",m=" << opt.m
                  << ",n=" << opt.n
                  << ",l=" << opt.l
                  << ",sketch_dim=" << opt.sketch_dim
                  << ",block_rows=" << opt.block_rows
                  << ",num_blocks=" << num_blocks
                  << ",total_sketch_dim=" << total_sketch_dim
                  << ",exact_gemm_ms=" << exact_gemm_ms
                  << ",sketch_Q_ms=" << sketch_q_ms
                  << ",sketch_A_ms=" << sketch_a_ms
                  << ",sketch_gemm_ms=" << sketch_gemm_ms
                  << ",direct_jl_total_ms=" << jl_total_ms
                  << ",direct_jl_relative_error=" << jl_relative_error
                  << ",sign_proxy_gemm_ms=" << sign_gemm_ms
                  << ",sign_proxy_relative_error=" << sign_proxy_error
                  << "\n";

        cudaFree(d_Q);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_B_hat);
        cudaFree(d_B_sign);
        cudaFree(d_diff);
        cudaFree(d_SQ);
        cudaFree(d_SA);
        cudaFree(d_SQ_sign);
        cudaFree(d_SA_sign);
        cublasDestroy(blas);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
