#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <mpi.h> 

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER Error code: " << err << " at line " << __LINE__ << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    int num_devices;
    CHECK_CUDA(cudaGetDeviceCount(&num_devices));
    
    int local_gpu_id = rank % num_devices; 
    CHECK_CUDA(cudaSetDevice(local_gpu_id));

    int M = 1024;
    int N = 1024;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M" && i + 1 < argc) {
            M = std::atoi(argv[++i]);
        } else if (arg == "-N" && i + 1 < argc) {
            N = std::atoi(argv[++i]);
        }
    }

    if (rank == 0) {
        std::cout << "=================================================" << std::endl;
        std::cout << "[INFO] Distributed Multi-GPU SVD Proxy App" << std::endl;
        std::cout << "[INFO] Total MPI Ranks (GPUs): " << size << std::endl;
        std::cout << "[INFO] Local Matrix size per GPU: " << M << " x " << N << std::endl;
        std::cout << "=================================================" << std::endl;
    }

    const int lda = M; 
    size_t total_elements = static_cast<size_t>(M) * N;
    size_t u_elements     = static_cast<size_t>(M) * M;
    size_t v_elements     = static_cast<size_t>(N) * N;

    srand(12345 + rank); 
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

    // Phase 1: Local Computation 
    MPI_Barrier(MPI_COMM_WORLD); 
    double start_time = MPI_Wtime();

    CHECK_CUSOLVER(cusolverDnSgesvdj(
        cusolverH, CUSOLVER_EIG_MODE_VECTOR, 1, M, N, 
        d_A, lda, d_S, d_U, lda, d_V, lda, 
        d_work, lwork, d_info, gesvdj_params));
    
    cudaDeviceSynchronize();
    double compute_time = MPI_Wtime() - start_time;

    CHECK_CUDA(cudaMemcpy(h_V.data(), d_V, sizeof(float) * v_elements, cudaMemcpyDeviceToHost));

    // Phase 2: MPI Communication
    std::vector<float> local_vector_to_send(h_V.begin(), h_V.begin() + N);
    std::vector<float> global_gathered_vectors(N * size);

    double comm_start = MPI_Wtime();
    MPI_Allgather(
        local_vector_to_send.data(), N, MPI_FLOAT,      
        global_gathered_vectors.data(), N, MPI_FLOAT,   
        MPI_COMM_WORLD
    );

    double comm_time = MPI_Wtime() - comm_start;
    double total_time = compute_time + comm_time;

    if (rank == 0) {
        std::cout << "=> Compute Time (Local SVD) : " << compute_time * 1000.0 << " ms" << std::endl;
        std::cout << "=> Comm Time (MPI Gather)   : " << comm_time * 1000.0 << " ms" << std::endl;
        std::cout << "=> Total Pipeline Time      : " << total_time * 1000.0 << " ms" << std::endl;
        
        if (size > 1) {
            std::cout << "[Verify] Received value from Rank 1: " << global_gathered_vectors[N] << std::endl;
        }
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUSOLVER(cusolverDnDestroyGesvdjInfo(gesvdj_params));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    
    MPI_Finalize(); 
    return 0;
}