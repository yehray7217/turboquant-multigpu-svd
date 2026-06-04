// Synthetic GraphSAGE-style embedding exchange benchmark.
//
// This does not load a real graph dataset. It creates graph-like node embedding
// tensors and measures the communication pattern that appears after graph
// partitioning: each MPI rank exchanges dense halo-node embeddings with all
// other ranks.
//
// Important: the compressed modes call the original project TurboQuant library:
//   ../turboquant/turboquant.hpp
//   ../turboquant/turboquant.cu
// so this benchmark is testing your TurboQuant implementation on a GNN-style
// communication workload.

#include <cuda_runtime.h>
#include <mpi.h>

#include "../turboquant/turboquant.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
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

namespace {

struct MpiScope {
    int rank = 0;
    int size = 1;

    MpiScope(int argc, char** argv) {
        CHECK_MPI(MPI_Init(&argc, &argv));
        CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
    }

    ~MpiScope() {
        MPI_Finalize();
    }
};

struct Options {
    int halo_nodes = 65536;
    int embedding_dim = 256;
    int repeats = 20;
    int warmup = 3;
    int bits = 4;
    int qjl_dim = 256;
    float qjl_alpha = 0.25f;
    unsigned seed = 1234;
    std::string mode = "sweep";
};

struct RunConfig {
    std::string label;
    std::string mode;
    int bits = 0;
};

struct RunResult {
    std::string label;
    std::string mode;
    int bits = 0;
    std::size_t payload_bytes = 0;
    double compression_ratio = 1.0;
    double mean_ms = 0.0;
    double min_ms = 0.0;
    double stddev_ms = 0.0;
    double relative_rmse = 0.0;
};

void usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --halo-nodes <int>       fixed exchanged nodes per rank (default: 65536)\n"
        << "  --embedding-dim <int>    embedding width; TQ supports 256/512/1024/2048 (default: 256)\n"
        << "  --mode <none|lowbit|tq|tq-qjl|sweep> (default: sweep)\n"
        << "  --bits <int>             compression bits for non-sweep mode (default: 4)\n"
        << "  --qjl-dim <int>          QJL sketch dim for tq-qjl (default: 256)\n"
        << "  --qjl-alpha <float>      QJL residual scale (default: 0.25)\n"
        << "  --repeats <int>          measured repeats (default: 20)\n"
        << "  --warmup <int>           warmup repeats (default: 3)\n"
        << "  --seed <int>             deterministic embedding seed (default: 1234)\n";
}

Options parse_args(int argc, char** argv) {
    Options opt;
    auto need_value = [&](int& i, const std::string& arg) -> const char* {
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + arg);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--halo-nodes") opt.halo_nodes = std::stoi(need_value(i, a));
        else if (a == "--embedding-dim") opt.embedding_dim = std::stoi(need_value(i, a));
        else if (a == "--mode") opt.mode = need_value(i, a);
        else if (a == "--bits") opt.bits = std::stoi(need_value(i, a));
        else if (a == "--qjl-dim") opt.qjl_dim = std::stoi(need_value(i, a));
        else if (a == "--qjl-alpha") opt.qjl_alpha = std::stof(need_value(i, a));
        else if (a == "--repeats") opt.repeats = std::stoi(need_value(i, a));
        else if (a == "--warmup") opt.warmup = std::stoi(need_value(i, a));
        else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need_value(i, a)));
        else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }

    if (opt.halo_nodes <= 0 || opt.embedding_dim <= 0 || opt.repeats <= 0 || opt.warmup < 0) {
        throw std::runtime_error("halo-nodes, embedding-dim, and repeats must be positive.");
    }
    return opt;
}

int checked_mpi_count(std::size_t count, const char* label) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " exceeds MPI int count limit.");
    }
    return static_cast<int>(count);
}

std::size_t next_power_of_two(std::size_t x) {
    std::size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

__global__ void fill_embeddings_kernel(float* out, int rows, int cols, int rank, unsigned seed) {
    std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int feature = static_cast<int>(idx % rows);
    int node = static_cast<int>(idx / rows);
    unsigned h = seed ^ static_cast<unsigned>(rank * 1103515245u);
    h ^= static_cast<unsigned>(node * 2654435761u);
    h ^= static_cast<unsigned>(feature * 2246822519u);
    h = (h ^ (h >> 16)) * 2246822519u;
    h = (h ^ (h >> 13)) * 3266489917u;
    h ^= (h >> 16);
    float u = static_cast<float>(h & 0x00ffffffu) / static_cast<float>(0x01000000u);
    float centered = 2.0f * u - 1.0f;
    float feature_wave = sinf(0.013f * static_cast<float>(feature + 1) *
                              static_cast<float>((node % 97) + 1));
    // Synthetic but structured embeddings: random noise plus a feature pattern.
    // This gives a more realistic distribution than pure uniform noise.
    out[idx] = 0.08f * centered + 0.04f * feature_wave + 0.001f * static_cast<float>(rank);
}

std::vector<RunConfig> make_run_configs(const Options& opt) {
    if (opt.mode == "sweep") {
        // Run one baseline and three compressed variants in the same job.
        return {
            {"none", "none", 0},
            {"lowbit8", "lowbit", 8},
            {"tq4", "tq", 4},
            {"tq-qjl4", "tq-qjl", 4},
        };
    }
    if (opt.mode != "none" && opt.mode != "lowbit" && opt.mode != "tq" && opt.mode != "tq-qjl") {
        throw std::runtime_error("Unsupported mode: " + opt.mode);
    }
    return {{opt.mode + ((opt.mode == "none") ? "" : std::to_string(opt.bits)), opt.mode, opt.bits}};
}

turboquant::QuantizeOptions make_quant_options(const RunConfig& cfg, const Options& opt) {
    if (cfg.mode == "none") {
        return turboquant::make_quantize_options(0, "none", 0, 0.0f, opt.seed);
    }
    return turboquant::make_quantize_options(
        cfg.bits,
        cfg.mode,
        opt.qjl_dim,
        opt.qjl_alpha,
        opt.seed);
}

void allgather_bytes(const std::uint8_t* send, std::size_t send_bytes, std::vector<std::uint8_t>& recv) {
    CHECK_MPI(MPI_Allgather(
        send,
        checked_mpi_count(send_bytes, "byte allgather send count"),
        MPI_UNSIGNED_CHAR,
        recv.data(),
        checked_mpi_count(send_bytes, "byte allgather receive count"),
        MPI_UNSIGNED_CHAR,
        MPI_COMM_WORLD));
}

void allgather_floats(const float* send, std::size_t send_count, std::vector<float>& recv) {
    CHECK_MPI(MPI_Allgather(
        send,
        checked_mpi_count(send_count, "float allgather send count"),
        MPI_FLOAT,
        recv.data(),
        checked_mpi_count(send_count, "float allgather receive count"),
        MPI_FLOAT,
        MPI_COMM_WORLD));
}

double relative_rmse(const std::vector<float>& ref, const std::vector<float>& got) {
    if (ref.size() != got.size()) {
        throw std::runtime_error("relative_rmse input sizes differ.");
    }
    long double diff2 = 0.0L;
    long double ref2 = 0.0L;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        long double d = static_cast<long double>(got[i]) - static_cast<long double>(ref[i]);
        diff2 += d * d;
        ref2 += static_cast<long double>(ref[i]) * static_cast<long double>(ref[i]);
    }
    if (ref2 == 0.0L) return 0.0;
    return static_cast<double>(std::sqrt(diff2 / ref2));
}

RunResult run_once_mode(
    const RunConfig& cfg,
    const Options& opt,
    int mpi_rank,
    int mpi_size,
    float* d_local,
    float* d_recv_all,
    const std::vector<float>& h_ref_all) {

    const int rows = opt.embedding_dim;
    const int cols = opt.halo_nodes;
    const std::size_t value_count = static_cast<std::size_t>(rows) * cols;
    const std::size_t fp32_bytes = value_count * sizeof(float);
    const std::size_t all_value_count = value_count * static_cast<std::size_t>(mpi_size);
    const std::size_t all_fp32_bytes = all_value_count * sizeof(float);

    RunResult result;
    result.label = cfg.label;
    result.mode = cfg.mode;
    result.bits = cfg.bits;

    std::vector<double> measured_ms;
    measured_ms.reserve(static_cast<std::size_t>(opt.repeats));

    if (cfg.mode == "none") {
        std::vector<float> h_local(value_count);
        std::vector<float> h_all(all_value_count);
        result.payload_bytes = fp32_bytes;
        result.compression_ratio = 1.0;

        // Baseline: gather raw FP32 embeddings without compression.
        for (int rep = -opt.warmup; rep < opt.repeats; ++rep) {
            CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
            double t0 = now_ms();
            CHECK_CUDA(cudaMemcpy(h_local.data(), d_local, fp32_bytes, cudaMemcpyDeviceToHost));
            allgather_floats(h_local.data(), value_count, h_all);
            CHECK_CUDA(cudaMemcpy(d_recv_all, h_all.data(), all_fp32_bytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
            double elapsed = now_ms() - t0;
            if (rep >= 0) measured_ms.push_back(elapsed);
        }

        std::vector<float> h_got(all_value_count);
        CHECK_CUDA(cudaMemcpy(h_got.data(), d_recv_all, all_fp32_bytes, cudaMemcpyDeviceToHost));
        result.relative_rmse = relative_rmse(h_ref_all, h_got);
    } else {
        // Compressed path: use the original TurboQuant API to encode the local
        // GPU tensor, allgather the compact payload, then decode on GPU.
        turboquant::QuantizeOptions qopt = make_quant_options(cfg, opt);
        const bool is_tq = qopt.mode == turboquant::QuantizeMode::kTurboQuant ||
                           qopt.mode == turboquant::QuantizeMode::kTurboQuantQjl;
        const bool is_qjl = qopt.mode == turboquant::QuantizeMode::kTurboQuantQjl;
        const std::size_t code_bytes = turboquant::device_code_bytes(rows, cols, qopt);
        const std::size_t norm_bytes = turboquant::device_norm_bytes(rows, cols, qopt);
        const std::size_t qjl_sign_bytes = turboquant::device_qjl_sign_bytes(rows, cols, qopt);
        const std::size_t metadata_bytes = is_tq ? 0 : sizeof(float);
        result.payload_bytes = code_bytes + norm_bytes + qjl_sign_bytes + metadata_bytes;
        result.compression_ratio = static_cast<double>(fp32_bytes) /
                                   static_cast<double>(std::max<std::size_t>(1, result.payload_bytes));

        std::uint8_t* d_codes = nullptr;
        std::uint8_t* d_codes_recv = nullptr;
        std::uint8_t* d_qjl_signs = nullptr;
        std::uint8_t* d_qjl_signs_recv = nullptr;
        float* d_norms = nullptr;
        float* d_norms_recv = nullptr;
        float* d_work = nullptr;
        CHECK_CUDA(cudaMalloc(&d_codes, code_bytes));
        CHECK_CUDA(cudaMalloc(&d_codes_recv, code_bytes));
        if (norm_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_norms, norm_bytes));
            CHECK_CUDA(cudaMalloc(&d_norms_recv, norm_bytes));
        }
        if (qjl_sign_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_qjl_signs, qjl_sign_bytes));
            CHECK_CUDA(cudaMalloc(&d_qjl_signs_recv, qjl_sign_bytes));
        }
        std::size_t work_count = is_tq ? next_power_of_two(static_cast<std::size_t>(rows)) * static_cast<std::size_t>(cols)
                                       : value_count;
        CHECK_CUDA(cudaMalloc(&d_work, work_count * sizeof(float)));

        std::vector<std::uint8_t> h_codes(code_bytes);
        std::vector<std::uint8_t> h_codes_all(code_bytes * static_cast<std::size_t>(mpi_size));
        std::vector<float> h_norms(norm_bytes / sizeof(float));
        std::vector<float> h_norms_all((norm_bytes / sizeof(float)) * static_cast<std::size_t>(mpi_size));
        std::vector<std::uint8_t> h_qjl_signs(qjl_sign_bytes);
        std::vector<std::uint8_t> h_qjl_signs_all(qjl_sign_bytes * static_cast<std::size_t>(mpi_size));
        std::vector<float> h_scales(static_cast<std::size_t>(mpi_size), 1.0f);
        std::vector<float> h_got(all_value_count);

        turboquant::DeviceCompressedBlock local_block;
        for (int rep = -opt.warmup; rep < opt.repeats; ++rep) {
            CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
            double t0 = now_ms();

            if (is_tq) {
                // TQ and TQ+QJL use the column-vector TurboQuant path.
                local_block = turboquant::quantize_fp32_device_column_tq_to_device_payload(
                    d_local,
                    rows,
                    cols,
                    qopt,
                    d_codes,
                    d_norms,
                    d_work,
                    nullptr,
                    d_qjl_signs);
            } else {
                // lowbit uses the simpler block quantization path.
                local_block = turboquant::quantize_fp32_device_block_to_device_payload(
                    d_local,
                    rows,
                    cols,
                    qopt,
                    d_codes,
                    nullptr);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_codes.data(), d_codes, code_bytes, cudaMemcpyDeviceToHost));
            allgather_bytes(h_codes.data(), code_bytes, h_codes_all);
            if (norm_bytes > 0) {
                CHECK_CUDA(cudaMemcpy(h_norms.data(), d_norms, norm_bytes, cudaMemcpyDeviceToHost));
                allgather_floats(h_norms.data(), h_norms.size(), h_norms_all);
            }
            if (qjl_sign_bytes > 0) {
                CHECK_CUDA(cudaMemcpy(h_qjl_signs.data(), d_qjl_signs, qjl_sign_bytes, cudaMemcpyDeviceToHost));
                allgather_bytes(h_qjl_signs.data(), qjl_sign_bytes, h_qjl_signs_all);
            }
            if (!is_tq) {
                float local_scale = local_block.scale;
                CHECK_MPI(MPI_Allgather(
                    &local_scale,
                    1,
                    MPI_FLOAT,
                    h_scales.data(),
                    1,
                    MPI_FLOAT,
                    MPI_COMM_WORLD));
            }

            CHECK_CUDA(cudaMemset(d_recv_all, 0, all_fp32_bytes));
            for (int r = 0; r < mpi_size; ++r) {
                CHECK_CUDA(cudaMemcpy(
                    d_codes_recv,
                    h_codes_all.data() + static_cast<std::size_t>(r) * code_bytes,
                    code_bytes,
                    cudaMemcpyHostToDevice));
                turboquant::DeviceCompressedBlock block = local_block;
                block.d_codes = d_codes_recv;
                if (!is_tq) {
                    block.scale = h_scales[static_cast<std::size_t>(r)];
                }

                if (norm_bytes > 0) {
                    CHECK_CUDA(cudaMemcpy(
                        d_norms_recv,
                        h_norms_all.data() + static_cast<std::size_t>(r) * h_norms.size(),
                        norm_bytes,
                        cudaMemcpyHostToDevice));
                    block.d_norms = d_norms_recv;
                    block.d_residual_norms = is_qjl ? d_norms_recv + cols : nullptr;
                }
                if (qjl_sign_bytes > 0) {
                    CHECK_CUDA(cudaMemcpy(
                        d_qjl_signs_recv,
                        h_qjl_signs_all.data() + static_cast<std::size_t>(r) * qjl_sign_bytes,
                        qjl_sign_bytes,
                        cudaMemcpyHostToDevice));
                    block.d_qjl_signs = d_qjl_signs_recv;
                }

                float* d_segment = d_recv_all + static_cast<std::size_t>(r) * value_count;
                if (is_tq) {
                    turboquant::dequantize_column_tq_payload_add_to_fp32(block, d_segment, d_work);
                } else {
                    turboquant::dequantize_device_payload_to_fp32(block, d_segment);
                }
            }
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
            double elapsed = now_ms() - t0;
            if (rep >= 0) measured_ms.push_back(elapsed);
        }

        CHECK_CUDA(cudaMemcpy(h_got.data(), d_recv_all, all_fp32_bytes, cudaMemcpyDeviceToHost));
        result.relative_rmse = relative_rmse(h_ref_all, h_got);

        cudaFree(d_codes);
        cudaFree(d_codes_recv);
        cudaFree(d_qjl_signs);
        cudaFree(d_qjl_signs_recv);
        cudaFree(d_norms);
        cudaFree(d_norms_recv);
        cudaFree(d_work);
    }

    double local_sum = 0.0;
    double local_min = std::numeric_limits<double>::infinity();
    for (double v : measured_ms) {
        local_sum += v;
        local_min = std::min(local_min, v);
    }
    double local_mean = local_sum / static_cast<double>(measured_ms.size());
    double local_var = 0.0;
    for (double v : measured_ms) {
        double d = v - local_mean;
        local_var += d * d;
    }
    local_var /= static_cast<double>(measured_ms.size());

    CHECK_MPI(MPI_Reduce(&local_mean, &result.mean_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    CHECK_MPI(MPI_Reduce(&local_min, &result.min_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    double local_stddev = std::sqrt(local_var);
    CHECK_MPI(MPI_Reduce(&local_stddev, &result.stddev_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    double global_rmse = 0.0;
    CHECK_MPI(MPI_Reduce(&result.relative_rmse, &global_rmse, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    result.relative_rmse = global_rmse;

    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        MpiScope mpi(argc, argv);
        Options opt = parse_args(argc, argv);

        int device_count = 0;
        CHECK_CUDA(cudaGetDeviceCount(&device_count));
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA device visible.");
        }
        int dev = mpi.rank % device_count;
        CHECK_CUDA(cudaSetDevice(dev));

        const int rows = opt.embedding_dim;
        const int cols = opt.halo_nodes;
        const std::size_t value_count = static_cast<std::size_t>(rows) * cols;
        const std::size_t fp32_bytes = value_count * sizeof(float);
        const std::size_t all_value_count = value_count * static_cast<std::size_t>(mpi.size);
        const std::size_t all_fp32_bytes = all_value_count * sizeof(float);

        float* d_local = nullptr;
        float* d_recv_all = nullptr;
        CHECK_CUDA(cudaMalloc(&d_local, fp32_bytes));
        CHECK_CUDA(cudaMalloc(&d_recv_all, all_fp32_bytes));

        const int threads = 256;
        const int blocks = static_cast<int>((value_count + threads - 1) / threads);
        fill_embeddings_kernel<<<blocks, threads>>>(d_local, rows, cols, mpi.rank, opt.seed);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> h_local(value_count);
        std::vector<float> h_ref_all(all_value_count);
        CHECK_CUDA(cudaMemcpy(h_local.data(), d_local, fp32_bytes, cudaMemcpyDeviceToHost));
        allgather_floats(h_local.data(), value_count, h_ref_all);

        if (mpi.rank == 0) {
            std::cout << "Graph embedding communication benchmark\n"
                      << "  ranks             : " << mpi.size << "\n"
                      << "  halo_nodes/rank   : " << opt.halo_nodes << "\n"
                      << "  embedding_dim     : " << opt.embedding_dim << "\n"
                      << "  fp32 payload/rank : " << std::fixed << std::setprecision(2)
                      << static_cast<double>(fp32_bytes) / (1024.0 * 1024.0) << " MiB\n"
                      << "  repeats/warmup    : " << opt.repeats << "/" << opt.warmup << "\n\n";
            std::cout << "label,mode,bits,payload_per_rank_mib,compression_vs_fp32,mean_ms,min_ms,stddev_ms,relative_rmse,speedup_vs_none\n";
        }

        std::vector<RunResult> results;
        for (const RunConfig& cfg : make_run_configs(opt)) {
            results.push_back(run_once_mode(cfg, opt, mpi.rank, mpi.size, d_local, d_recv_all, h_ref_all));
        }

        if (mpi.rank == 0) {
            double baseline_ms = results.empty() ? 0.0 : results.front().mean_ms;
            for (const RunResult& r : results) {
                double speedup = (baseline_ms > 0.0) ? (baseline_ms / r.mean_ms) : 1.0;
                std::cout << r.label << ","
                          << r.mode << ","
                          << r.bits << ","
                          << std::fixed << std::setprecision(4)
                          << static_cast<double>(r.payload_bytes) / (1024.0 * 1024.0) << ","
                          << r.compression_ratio << ","
                          << r.mean_ms << ","
                          << r.min_ms << ","
                          << r.stddev_ms << ","
                          << std::scientific << r.relative_rmse << std::fixed << ","
                          << speedup << "\n";
            }
        }

        cudaFree(d_local);
        cudaFree(d_recv_all);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
