#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER Error code: " << err << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main(int argc, char* argv[]) {
    int M = 1024;
    int N = 1024;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M" && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (arg == "-N" && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./cusolver [-M rows] [-N cols]" << std::endl;
            return 0;
        }
    }

    const int lda = M; 

    std::cout << "[INFO] Starting cuSOLVER Jacobi SVD test..." << std::endl;
    std::cout << "[INFO] Matrix size: " << M << " x " << N << std::endl;

    size_t total_elements = static_cast<size_t>(M) * N;
    size_t u_elements     = static_cast<size_t>(M) * M;
    size_t v_elements     = static_cast<size_t>(N) * N;

    std::vector<float> h_A(total_elements);
    std::vector<float> h_S(std::min(M, N));     
    std::vector<float> h_U(u_elements);              
    std::vector<float> h_V(v_elements);              
    std::vector<int> h_info(1);                  

    for (size_t i = 0; i < total_elements; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_S, *d_U, *d_V;
    int *d_info;
    
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(float) * total_elements));
    CHECK_CUDA(cudaMalloc((void**)&d_S, sizeof(float) * std::min(M, N)));
    CHECK_CUDA(cudaMalloc((void**)&d_U, sizeof(float) * u_elements));
    CHECK_CUDA(cudaMalloc((void**)&d_V, sizeof(float) * v_elements));
    CHECK_CUDA(cudaMalloc((void**)&d_info, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeof(float) * total_elements, cudaMemcpyHostToDevice));

    cusolverDnHandle_t cusolverH = NULL;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    gesvdjInfo_t gesvdj_params = NULL;
    CHECK_CUSOLVER(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    const float tol = 1e-7;
    const int max_sweeps = 15; 
    CHECK_CUSOLVER(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
    CHECK_CUSOLVER(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSgesvdj_bufferSize(
        cusolverH, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
        d_A, lda, d_S, d_U, lda, d_V, lda, 
        &lwork, gesvdj_params));

    float *d_work;
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(float) * lwork));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUSOLVER(cusolverDnSgesvdj(
        cusolverH, 
        CUSOLVER_EIG_MODE_VECTOR, 
        1,                        
        M, N, 
        d_A, lda, 
        d_S,                      
        d_U, lda,                 
        d_V, lda,                 
        d_work, lwork, d_info, gesvdj_params));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop)); 

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "=> GPU SVD Execution Time: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_info.data(), d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info[0] == 0) {
        std::cout << "[SUCCESS] SVD computed successfully." << std::endl;
        CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, sizeof(float) * std::min(M, N), cudaMemcpyDeviceToHost));
        std::cout << "Top 3 Singular Values: " << h_S[0] << ", " << h_S[1] << ", " << h_S[2] << std::endl;
    } else if (h_info[0] > 0) {
        std::cout << "[WARNING] SVD did not converge. Info = " << h_info[0] << std::endl;
    } else {
        std::cout << "[ERROR] Invalid parameter at index " << -h_info[0] << std::endl;
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUSOLVER(cusolverDnDestroyGesvdjInfo(gesvdj_params));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));

    return 0;
}