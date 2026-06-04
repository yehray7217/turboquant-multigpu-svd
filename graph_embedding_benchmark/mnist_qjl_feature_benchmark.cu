// Real MNIST/Fashion-MNIST giant-feature QJL benchmark.
//
// Reads IDX ubyte files prepared by prepare_mnist_data.py. The large
// N x feature_dim matrix is generated implicitly in chunks:
//   phi(x) = sqrt(2/D) * cos(W^T x + b)
// then projected with a Rademacher QJL matrix. Metrics compare pairwise
// distances in the expanded feature space against the QJL projection and report
// nearest-centroid classification accuracy.
//
// This is the "real data" side of the experiment: real images -> huge implicit
// feature matrix -> QJL compression. It then calls the original TurboQuant
// library on the QJL features, so the final metrics compare:
//   QJL-only representation
//   QJL + TurboQuant-compressed/decompressed representation

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../turboquant/turboquant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                        \
        cublasStatus_t status__ = (call);                                        \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                                \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " -> " +  \
                                     std::to_string(static_cast<int>(status__))); \
        }                                                                       \
    } while (0)

namespace {

struct Options {
    std::string dataset_dir = "./data/fashion-mnist";
    std::string output = "results/fashion_mnist_qjl_tq.csv";
    int train_samples = 60000;
    int test_samples = 10000;
    int feature_dim = 100000;
    int feature_chunk = 2048;
    int tq_bits = 4;
    int geometry_samples = 1024;
    int geometry_pairs = 4096;
    unsigned seed = 1234;
    std::vector<int> qjl_dims = {256, 512, 1024};
};

std::uint32_t read_be32(const std::vector<unsigned char>& b, std::size_t off) {
    return (static_cast<std::uint32_t>(b[off]) << 24) |
           (static_cast<std::uint32_t>(b[off + 1]) << 16) |
           (static_cast<std::uint32_t>(b[off + 2]) << 8) |
           static_cast<std::uint32_t>(b[off + 3]);
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void load_idx(
    const std::string& image_path,
    const std::string& label_path,
    int limit,
    unsigned seed,
    std::vector<float>& x_col_major,
    std::vector<int>& y) {

    // Load the standard MNIST/Fashion-MNIST IDX files and keep images as
    // column-major 784 x N, matching cuBLAS GEMM layout below.
    std::vector<unsigned char> img = read_file(image_path);
    std::vector<unsigned char> lab = read_file(label_path);
    if (read_be32(img, 0) != 2051 || read_be32(lab, 0) != 2049) {
        throw std::runtime_error("Invalid IDX magic in " + image_path + " or " + label_path);
    }
    int n = static_cast<int>(read_be32(img, 4));
    int rows = static_cast<int>(read_be32(img, 8));
    int cols = static_cast<int>(read_be32(img, 12));
    int nlab = static_cast<int>(read_be32(lab, 4));
    if (n != nlab || rows != 28 || cols != 28) {
        throw std::runtime_error("Unexpected IDX dimensions.");
    }
    limit = std::min(limit, n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(limit);

    const int dim = 784;
    x_col_major.assign(static_cast<std::size_t>(dim) * limit, 0.0f);
    y.assign(limit, 0);
    for (int j = 0; j < limit; ++j) {
        int src = idx[j];
        const unsigned char* pix = img.data() + 16 + static_cast<std::size_t>(src) * dim;
        for (int r = 0; r < dim; ++r) {
            x_col_major[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * dim] =
                static_cast<float>(pix[r]) / 255.0f;
        }
        y[j] = static_cast<int>(lab[8 + src]);
    }
}

Options parse_args(int argc, char** argv) {
    Options opt;
    auto need = [&](int& i, const std::string& a) -> const char* {
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dataset-dir") opt.dataset_dir = need(i, a);
        else if (a == "--output") opt.output = need(i, a);
        else if (a == "--train-samples") opt.train_samples = std::stoi(need(i, a));
        else if (a == "--test-samples") opt.test_samples = std::stoi(need(i, a));
        else if (a == "--feature-dim") opt.feature_dim = std::stoi(need(i, a));
        else if (a == "--feature-chunk") opt.feature_chunk = std::stoi(need(i, a));
        else if (a == "--tq-bits") opt.tq_bits = std::stoi(need(i, a));
        else if (a == "--geometry-samples") opt.geometry_samples = std::stoi(need(i, a));
        else if (a == "--geometry-pairs") opt.geometry_pairs = std::stoi(need(i, a));
        else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need(i, a)));
        else if (a == "--qjl-dims") {
            opt.qjl_dims.clear();
            std::stringstream ss(need(i, a));
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) opt.qjl_dims.push_back(std::stoi(item));
            }
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    if (opt.qjl_dims.empty()) throw std::runtime_error("No QJL dimensions requested.");
    std::sort(opt.qjl_dims.begin(), opt.qjl_dims.end());
    opt.qjl_dims.erase(std::unique(opt.qjl_dims.begin(), opt.qjl_dims.end()), opt.qjl_dims.end());
    for (int q : opt.qjl_dims) {
        if (q != 256 && q != 512 && q != 1024 && q != 2048) {
            throw std::runtime_error("QJL+TurboQuant benchmark requires qjl-dims in {256,512,1024,2048}.");
        }
    }
    return opt;
}

void normalize_from_train(std::vector<float>& train, std::vector<float>& test, int ntrain, int ntest) {
    const int dim = 784;
    // Standardize each pixel using train-set statistics only.
    for (int r = 0; r < dim; ++r) {
        double sum = 0.0;
        for (int j = 0; j < ntrain; ++j) sum += train[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * dim];
        double mean = sum / ntrain;
        double var = 0.0;
        for (int j = 0; j < ntrain; ++j) {
            double d = train[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * dim] - mean;
            var += d * d;
        }
        double inv = 1.0 / std::sqrt(var / ntrain + 1e-6);
        for (int j = 0; j < ntrain; ++j) train[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * dim] = static_cast<float>((train[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * dim] - mean) * inv);
        for (int j = 0; j < ntest; ++j) test[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * dim] = static_cast<float>((test[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * dim] - mean) * inv);
    }
}

__global__ void add_bias_cos_kernel(float* phi, const float* bias, int rows, int cols, float scale) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    int r = static_cast<int>(idx % rows);
    // Convert W^T x + b into random Fourier-style expanded features.
    phi[idx] = scale * cosf(phi[idx] + bias[r]);
}

__global__ void accumulate_pair_dist_kernel(
    const float* phi,
    const int* pair_i,
    const int* pair_j,
    float* dot,
    float* norm_i,
    float* norm_j,
    int chunk_rows,
    int pairs) {
    int p = blockIdx.x;
    int t = threadIdx.x;
    if (p >= pairs) return;
    int i = pair_i[p];
    int j = pair_j[p];
    float sd = 0.0f, si = 0.0f, sj = 0.0f;
    for (int r = t; r < chunk_rows; r += blockDim.x) {
        float a = phi[r + i * chunk_rows];
        float b = phi[r + j * chunk_rows];
        sd += a * b;
        si += a * a;
        sj += b * b;
    }
    __shared__ float buf_d[256], buf_i[256], buf_j[256];
    buf_d[t] = sd;
    buf_i[t] = si;
    buf_j[t] = sj;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            buf_d[t] += buf_d[t + stride];
            buf_i[t] += buf_i[t + stride];
            buf_j[t] += buf_j[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        dot[p] += buf_d[0];
        norm_i[p] += buf_i[0];
        norm_j[p] += buf_j[0];
    }
}

__global__ void copy_prefix_rows_kernel(const float* src, int src_ld, float* dst, int rows, int cols) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(rows) * cols;
    if (idx >= total) return;
    int r = static_cast<int>(idx % rows);
    int c = static_cast<int>(idx / rows);
    dst[idx] = src[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * src_ld];
}

void fill_random_chunk(std::vector<float>& w, std::vector<float>& b, std::vector<float>& s, int width, int max_q, int start, unsigned seed) {
    // Generate one feature chunk and the matching Rademacher QJL signs.
    // Chunking avoids materializing the full N x 100000 matrix.
    std::mt19937 rng(seed + 1000003u * static_cast<unsigned>(start));
    std::normal_distribution<float> normal(0.0f, 1.0f / std::sqrt(784.0f));
    std::uniform_real_distribution<float> uniform(0.0f, 2.0f * 3.14159265358979323846f);
    w.resize(static_cast<std::size_t>(784) * width);
    b.resize(width);
    for (float& v : w) v = normal(rng);
    for (float& v : b) v = uniform(rng);

    std::mt19937 srng(seed + 2000003u * static_cast<unsigned>(start));
    std::uniform_int_distribution<int> bit(0, 1);
    s.resize(static_cast<std::size_t>(width) * max_q);
    for (float& v : s) v = bit(srng) ? 1.0f : -1.0f;
}

double centroid_accuracy(const std::vector<float>& x, const std::vector<int>& y, int train_n, int test_n, int dim, int ld, float scale) {
    // Simple accuracy check: nearest class centroid, not a tuned classifier.
    std::vector<double> cent(static_cast<std::size_t>(10) * dim, 0.0);
    std::vector<int> cnt(10, 0);
    for (int j = 0; j < train_n; ++j) {
        int c = y[j];
        cnt[c]++;
        for (int r = 0; r < dim; ++r) cent[static_cast<std::size_t>(c) * dim + r] += scale * x[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * ld];
    }
    for (int c = 0; c < 10; ++c) {
        if (cnt[c] == 0) continue;
        for (int r = 0; r < dim; ++r) cent[static_cast<std::size_t>(c) * dim + r] /= cnt[c];
    }
    int correct = 0;
    for (int j = 0; j < test_n; ++j) {
        int col = train_n + j;
        int best = 0;
        double best_d = std::numeric_limits<double>::infinity();
        for (int c = 0; c < 10; ++c) {
            double d = 0.0;
            for (int r = 0; r < dim; ++r) {
                double z = scale * x[static_cast<std::size_t>(r) + static_cast<std::size_t>(col) * ld] - cent[static_cast<std::size_t>(c) * dim + r];
                d += z * z;
            }
            if (d < best_d) {
                best_d = d;
                best = c;
            }
        }
        if (best == y[col]) correct++;
    }
    return static_cast<double>(correct) / test_n;
}

void evaluate_representation(
    const std::vector<float>& rep,
    const std::vector<int>& labels,
    int train_n,
    int test_n,
    int q,
    int ld,
    float scale,
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    const std::vector<float>& dot,
    const std::vector<float>& ni,
    const std::vector<float>& nj,
    double& mean_rel,
    double& p95_rel,
    double& acc) {

    std::vector<float> rel(pair_i.size());
    for (std::size_t p = 0; p < pair_i.size(); ++p) {
        int a = pair_i[p], b = pair_j[p];
        double pd = 0.0;
        for (int r = 0; r < q; ++r) {
            double d = scale * (rep[static_cast<std::size_t>(r) + static_cast<std::size_t>(a) * ld] -
                                rep[static_cast<std::size_t>(r) + static_cast<std::size_t>(b) * ld]);
            pd += d * d;
        }
        double exact = std::sqrt(std::max(0.0f, ni[p] + nj[p] - 2.0f * dot[p]));
        rel[p] = static_cast<float>(std::fabs(std::sqrt(pd) - exact) / std::max(1e-8, exact));
    }
    std::sort(rel.begin(), rel.end());
    mean_rel = std::accumulate(rel.begin(), rel.end(), 0.0) / rel.size();
    p95_rel = rel[static_cast<std::size_t>(0.95 * (rel.size() - 1))];
    acc = centroid_accuracy(rep, labels, train_n, test_n, q, ld, scale);
}

std::size_t next_power_of_two(std::size_t x) {
    std::size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);
        int max_q = opt.qjl_dims.back();
        std::string prefix = opt.dataset_dir + "/";

        std::vector<float> train_x, test_x;
        std::vector<int> train_y, test_y;
        load_idx(prefix + "train-images-idx3-ubyte", prefix + "train-labels-idx1-ubyte", opt.train_samples, opt.seed, train_x, train_y);
        load_idx(prefix + "t10k-images-idx3-ubyte", prefix + "t10k-labels-idx1-ubyte", opt.test_samples, opt.seed + 1, test_x, test_y);
        int train_n = static_cast<int>(train_y.size());
        int test_n = static_cast<int>(test_y.size());
        normalize_from_train(train_x, test_x, train_n, test_n);

        int n = train_n + test_n;
        std::vector<float> x(static_cast<std::size_t>(784) * n);
        std::vector<int> y(n);
        std::copy(train_x.begin(), train_x.end(), x.begin());
        std::copy(test_x.begin(), test_x.end(), x.begin() + static_cast<std::size_t>(784) * train_n);
        std::copy(train_y.begin(), train_y.end(), y.begin());
        std::copy(test_y.begin(), test_y.end(), y.begin() + train_n);
        double raw_acc = centroid_accuracy(x, y, train_n, test_n, 784, 784, 1.0f);

        int geom_n = std::min(opt.geometry_samples, n);
        std::mt19937 rng(opt.seed + 77);
        std::uniform_int_distribution<int> pick(0, geom_n - 1);
        std::vector<int> h_pair_i(opt.geometry_pairs), h_pair_j(opt.geometry_pairs);
        for (int p = 0; p < opt.geometry_pairs; ++p) {
            h_pair_i[p] = pick(rng);
            h_pair_j[p] = pick(rng);
            if (h_pair_i[p] == h_pair_j[p]) h_pair_j[p] = (h_pair_j[p] + 1) % geom_n;
        }

        float *d_x = nullptr, *d_w = nullptr, *d_phi = nullptr, *d_s = nullptr, *d_y = nullptr;
        float *d_dot = nullptr, *d_ni = nullptr, *d_nj = nullptr, *d_bias = nullptr;
        int *d_pair_i = nullptr, *d_pair_j = nullptr;
        CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_w, static_cast<std::size_t>(784) * opt.feature_chunk * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_phi, static_cast<std::size_t>(opt.feature_chunk) * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_s, static_cast<std::size_t>(opt.feature_chunk) * max_q * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, static_cast<std::size_t>(max_q) * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_bias, opt.feature_chunk * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_dot, opt.geometry_pairs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ni, opt.geometry_pairs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_nj, opt.geometry_pairs * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_pair_i, opt.geometry_pairs * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_pair_j, opt.geometry_pairs * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_pair_i, h_pair_i.data(), h_pair_i.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_pair_j, h_pair_j.data(), h_pair_j.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_y, 0, static_cast<std::size_t>(max_q) * n * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_dot, 0, opt.geometry_pairs * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_ni, 0, opt.geometry_pairs * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_nj, 0, opt.geometry_pairs * sizeof(float)));

        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));
        std::vector<float> h_w, h_b, h_s;
        const float one = 1.0f, zero = 0.0f;
        auto t0 = std::chrono::steady_clock::now();
        for (int start = 0; start < opt.feature_dim; start += opt.feature_chunk) {
            int width = std::min(opt.feature_chunk, opt.feature_dim - start);
            fill_random_chunk(h_w, h_b, h_s, width, max_q, start, opt.seed);
            CHECK_CUDA(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_bias, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_s, h_s.data(), h_s.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, width, n, 784, &one, d_w, 784, d_x, 784, &zero, d_phi, width));
            int threads = 256;
            int blocks = static_cast<int>(((static_cast<std::size_t>(width) * n) + threads - 1) / threads);
            float scale = std::sqrt(2.0f / static_cast<float>(opt.feature_dim));
            add_bias_cos_kernel<<<blocks, threads>>>(d_phi, d_bias, width, n, scale);
            CHECK_CUDA(cudaGetLastError());
            accumulate_pair_dist_kernel<<<opt.geometry_pairs, 256>>>(d_phi, d_pair_i, d_pair_j, d_dot, d_ni, d_nj, width, opt.geometry_pairs);
            CHECK_CUDA(cudaGetLastError());
            // Accumulate QJL projection: Y += S^T phi_chunk.
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, max_q, n, width, &one, d_s, width, d_phi, width, &one, d_y, max_q));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        double projection_seconds = std::chrono::duration<double>(t1 - t0).count();

        std::vector<float> proj(static_cast<std::size_t>(max_q) * n);
        std::vector<float> dot(opt.geometry_pairs), ni(opt.geometry_pairs), nj(opt.geometry_pairs);
        CHECK_CUDA(cudaMemcpy(proj.data(), d_y, proj.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(dot.data(), d_dot, dot.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(ni.data(), d_ni, ni.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(nj.data(), d_nj, nj.size() * sizeof(float), cudaMemcpyDeviceToHost));

        std::ofstream out(opt.output);
        if (!out) throw std::runtime_error("Cannot write " + opt.output);
        out << "dataset_dir,train_samples,test_samples,expanded_feature_dim,qjl_dim,tq_bits,"
            << "qjl_compression_vs_fp32_features,tq_payload_mib,tq_compression_vs_qjl_fp32,total_compression_vs_expanded_fp32,"
            << "qjl_mean_pair_distance_relative_error,qjl_p95_pair_distance_relative_error,qjl_centroid_accuracy,"
            << "tq_mean_pair_distance_relative_error,tq_p95_pair_distance_relative_error,tq_centroid_accuracy,"
            << "raw_pixel_centroid_accuracy,projection_seconds,tq_seconds,device\n";
        std::cout << "dataset_dir,expanded_feature_dim,qjl_dim,tq_bits,"
                  << "qjl_compression_vs_fp32_features,tq_payload_mib,tq_compression_vs_qjl_fp32,total_compression_vs_expanded_fp32,"
                  << "qjl_mean_pair_distance_relative_error,qjl_p95_pair_distance_relative_error,qjl_centroid_accuracy,"
                  << "tq_mean_pair_distance_relative_error,tq_p95_pair_distance_relative_error,tq_centroid_accuracy,"
                  << "raw_pixel_centroid_accuracy,projection_seconds,tq_seconds,device\n";
        for (int q : opt.qjl_dims) {
            // Report how much geometry and classification signal remains after
            // QJL, then after TurboQuant compress/decompress of the QJL matrix.
            float inv_sqrt_q = 1.0f / std::sqrt(static_cast<float>(q));
            double qjl_mean = 0.0, qjl_p95 = 0.0, qjl_acc = 0.0;
            evaluate_representation(
                proj, y, train_n, test_n, q, max_q, inv_sqrt_q,
                h_pair_i, h_pair_j, dot, ni, nj,
                qjl_mean, qjl_p95, qjl_acc);

            turboquant::QuantizeOptions qopt =
                turboquant::make_quantize_options(opt.tq_bits, "tq", 0, 0.0f, opt.seed);
            std::size_t code_bytes = turboquant::device_code_bytes(q, n, qopt);
            std::size_t norm_bytes = turboquant::device_norm_bytes(q, n, qopt);
            std::size_t tq_payload_bytes = code_bytes + norm_bytes;

            float* d_qjl_contig = nullptr;
            float* d_tq_recon = nullptr;
            float* d_tq_work = nullptr;
            std::uint8_t* d_tq_codes = nullptr;
            float* d_tq_norms = nullptr;
            CHECK_CUDA(cudaMalloc(&d_qjl_contig, static_cast<std::size_t>(q) * n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_tq_recon, static_cast<std::size_t>(q) * n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_tq_work, next_power_of_two(static_cast<std::size_t>(q)) * static_cast<std::size_t>(n) * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_tq_codes, code_bytes));
            CHECK_CUDA(cudaMalloc(&d_tq_norms, norm_bytes));

            int threads = 256;
            int blocks = static_cast<int>(((static_cast<std::size_t>(q) * n) + threads - 1) / threads);
            copy_prefix_rows_kernel<<<blocks, threads>>>(d_y, max_q, d_qjl_contig, q, n);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaMemset(d_tq_recon, 0, static_cast<std::size_t>(q) * n * sizeof(float)));

            auto tq_t0 = std::chrono::steady_clock::now();
            turboquant::DeviceCompressedBlock tq_block =
                turboquant::quantize_fp32_device_column_tq_to_device_payload(
                    d_qjl_contig,
                    q,
                    n,
                    qopt,
                    d_tq_codes,
                    d_tq_norms,
                    d_tq_work,
                    nullptr,
                    nullptr);
            turboquant::dequantize_column_tq_payload_add_to_fp32(tq_block, d_tq_recon, d_tq_work);
            CHECK_CUDA(cudaDeviceSynchronize());
            auto tq_t1 = std::chrono::steady_clock::now();
            double tq_seconds = std::chrono::duration<double>(tq_t1 - tq_t0).count();

            std::vector<float> tq_proj(static_cast<std::size_t>(q) * n);
            CHECK_CUDA(cudaMemcpy(tq_proj.data(), d_tq_recon, tq_proj.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double tq_mean = 0.0, tq_p95 = 0.0, tq_acc = 0.0;
            evaluate_representation(
                tq_proj, y, train_n, test_n, q, q, inv_sqrt_q,
                h_pair_i, h_pair_j, dot, ni, nj,
                tq_mean, tq_p95, tq_acc);

            cudaFree(d_qjl_contig);
            cudaFree(d_tq_recon);
            cudaFree(d_tq_work);
            cudaFree(d_tq_codes);
            cudaFree(d_tq_norms);

            double qjl_comp = static_cast<double>(opt.feature_dim) / q;
            double tq_comp_vs_qjl = static_cast<double>(q) * n * sizeof(float) / static_cast<double>(tq_payload_bytes);
            double total_comp = static_cast<double>(opt.feature_dim) * n * sizeof(float) / static_cast<double>(tq_payload_bytes);
            double tq_payload_mib = static_cast<double>(tq_payload_bytes) / (1024.0 * 1024.0);

            out << opt.dataset_dir << "," << train_n << "," << test_n << "," << opt.feature_dim << "," << q << "," << opt.tq_bits << ","
                << qjl_comp << "," << tq_payload_mib << "," << tq_comp_vs_qjl << "," << total_comp << ","
                << qjl_mean << "," << qjl_p95 << "," << qjl_acc << ","
                << tq_mean << "," << tq_p95 << "," << tq_acc << ","
                << raw_acc << "," << projection_seconds << "," << tq_seconds << ",cuda\n";
            std::cout << opt.dataset_dir << "," << opt.feature_dim << "," << q << "," << opt.tq_bits << ","
                      << std::fixed << std::setprecision(4)
                      << qjl_comp << "," << tq_payload_mib << "," << tq_comp_vs_qjl << "," << total_comp << ","
                      << qjl_mean << "," << qjl_p95 << "," << qjl_acc << ","
                      << tq_mean << "," << tq_p95 << "," << tq_acc << ","
                      << raw_acc << "," << projection_seconds << "," << tq_seconds << ",cuda\n";
        }

        cublasDestroy(handle);
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_phi); cudaFree(d_s); cudaFree(d_y);
        cudaFree(d_dot); cudaFree(d_ni); cudaFree(d_nj); cudaFree(d_bias);
        cudaFree(d_pair_i); cudaFree(d_pair_j);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
