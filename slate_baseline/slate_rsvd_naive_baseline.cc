#include <slate/slate.hh>

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kTileSize = 256;

struct Config {
    int64_t m = 65536;
    int64_t n = 4096;
    int64_t k = 128;
    int64_t oversample = 32;
    uint64_t seed = 12345;
    int expected_mpi_size = 16;
    int grid_p = 4;
    int grid_q = 4;
};

void print_usage(const char* prog)
{
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --m <int>            Global rows of A (default: 65536)\n"
        << "  --n <int>            Global cols of A (default: 4096)\n"
        << "  --k <int>            Target rank k (default: 128)\n"
        << "  --oversample <int>   Oversampling p, l = k + p (default: 32)\n"
        << "  --seed <int>         Random seed (default: 12345)\n"
        << "  --expected-mpi-size  Required MPI size (default: 16)\n"
        << "  --grid-p <int>       Process grid rows p (default: 4)\n"
        << "  --grid-q <int>       Process grid cols q (default: 4)\n"
        << "  --help               Print this help\n";
}

Config parse_args(int argc, char** argv)
{
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--m") {
            cfg.m = std::stoll(require_value(arg));
        }
        else if (arg == "--n") {
            cfg.n = std::stoll(require_value(arg));
        }
        else if (arg == "--k") {
            cfg.k = std::stoll(require_value(arg));
        }
        else if (arg == "--oversample") {
            cfg.oversample = std::stoll(require_value(arg));
        }
        else if (arg == "--seed") {
            cfg.seed = static_cast<uint64_t>(std::stoull(require_value(arg)));
        }
        else if (arg == "--expected-mpi-size") {
            cfg.expected_mpi_size = std::stoi(require_value(arg));
        }
        else if (arg == "--grid-p") {
            cfg.grid_p = std::stoi(require_value(arg));
        }
        else if (arg == "--grid-q") {
            cfg.grid_q = std::stoi(require_value(arg));
        }
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (cfg.m <= 0 || cfg.n <= 0 || cfg.k <= 0 || cfg.oversample < 0) {
        throw std::runtime_error("Require m,n,k > 0 and oversample >= 0.");
    }
    if (cfg.expected_mpi_size <= 0 || cfg.grid_p <= 0 || cfg.grid_q <= 0) {
        throw std::runtime_error("Require expected-mpi-size, grid-p, grid-q > 0.");
    }
    if (cfg.grid_p * cfg.grid_q != cfg.expected_mpi_size) {
        throw std::runtime_error("Require grid-p * grid-q == expected-mpi-size.");
    }

    const int64_t l = cfg.k + cfg.oversample;
    const int64_t min_mn = std::min(cfg.m, cfg.n);
    if (l > min_mn) {
        throw std::runtime_error("Require k + oversample <= min(m, n). Currently l="
            + std::to_string(l) + ", min(m,n)=" + std::to_string(min_mn));
    }

    return cfg;
}

uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

float hash_to_unit_float(int64_t row, int64_t col, uint64_t seed)
{
    uint64_t key = static_cast<uint64_t>(row + 1) * 0x9e3779b97f4a7c15ULL;
    key ^= static_cast<uint64_t>(col + 1) * 0xbf58476d1ce4e5b9ULL;
    key ^= seed;
    uint64_t rnd = splitmix64(key);

    constexpr double inv_53 = 1.0 / static_cast<double>(1ULL << 53);
    double u = static_cast<double>(rnd >> 11) * inv_53;  // [0, 1)
    return static_cast<float>(2.0 * u - 1.0);            // [-1, 1)
}

void fill_random_matrix(slate::Matrix<float>& A, uint64_t seed)
{
    std::vector<int64_t> row_offset(A.mt() + 1, 0);
    std::vector<int64_t> col_offset(A.nt() + 1, 0);

    for (int64_t i = 0; i < A.mt(); ++i) {
        row_offset[i + 1] = row_offset[i] + A.tileMb(i);
    }
    for (int64_t j = 0; j < A.nt(); ++j) {
        col_offset[j + 1] = col_offset[j] + A.tileNb(j);
    }

    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (!A.tileIsLocal(i, j)) {
                continue;
            }

            A.tileGetForWriting(i, j, slate::HostNum, slate::LayoutConvert::ColMajor);
            auto tile = A(i, j, slate::HostNum);
            float* data = tile.data();
            const int64_t mb = tile.mb();
            const int64_t nb = tile.nb();
            const int64_t ld = tile.stride();
            const int64_t i0 = row_offset[i];
            const int64_t j0 = col_offset[j];

            for (int64_t jj = 0; jj < nb; ++jj) {
                for (int64_t ii = 0; ii < mb; ++ii) {
                    data[ii + jj * ld] = hash_to_unit_float(i0 + ii, j0 + jj, seed);
                }
            }
        }
    }
}

template <typename Func>
double timed_max_seconds(MPI_Comm comm, Func&& fn)
{
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();
    fn();
    MPI_Barrier(comm);
    double local = MPI_Wtime() - t0;
    double global_max = 0.0;
    MPI_Allreduce(&local, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    return global_max;
}

void clear_matrix(slate::Matrix<float>& A)
{
    A.releaseWorkspace();
    A.clear();
}

} // namespace

int main(int argc, char** argv)
{
    int mpi_initialized = 0;
    int mpi_rank = 0;
    int mpi_size = 0;

    try {
        int provided = 0;
        int ierr = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        if (ierr != MPI_SUCCESS) {
            std::cerr << "MPI_Init_thread failed." << std::endl;
            return 1;
        }
        mpi_initialized = 1;

        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        Config cfg = parse_args(argc, argv);
        const int64_t l = cfg.k + cfg.oversample;

        if (mpi_size != cfg.expected_mpi_size) {
            if (mpi_rank == 0) {
                std::cerr
                    << "ERROR: Expected mpi_size=" << cfg.expected_mpi_size
                    << " but got mpi_size=" << mpi_size << "\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        if (provided != MPI_THREAD_MULTIPLE) {
            if (mpi_rank == 0) {
                std::cerr << "ERROR: MPI library did not provide MPI_THREAD_MULTIPLE."
                          << " provided=" << provided << "\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        if (cfg.grid_p * cfg.grid_q != mpi_size) {
            if (mpi_rank == 0) {
                std::cerr << "ERROR: grid-p * grid-q does not match mpi_size." << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        if (blas::get_device_count() <= 0) {
            if (mpi_rank == 0) {
                std::cerr << "ERROR: No GPU device visible on this MPI rank, but Target::Devices"
                          << " is required for this baseline.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        slate::Options opts = {
            { slate::Option::Lookahead, 0 },
            { slate::Option::Target, slate::Target::Devices },
        };

        constexpr slate::GridOrder order = slate::GridOrder::Col;

        if (mpi_rank == 0) {
            std::cout
                << "SLATE RSVD Naive Baseline\n"
                << "  mpi_size: " << mpi_size << "\n"
                << "  process grid (p x q): " << cfg.grid_p << " x " << cfg.grid_q << "\n"
                << "  grid order: Col\n"
                << "  tile size nb: " << kTileSize << "\n"
                << "  target: Devices\n"
                << "  lookahead: 0\n"
                << "  A: " << cfg.m << " x " << cfg.n << "\n"
                << "  k: " << cfg.k << ", oversample: " << cfg.oversample
                << ", l: " << l << "\n";
        }

        slate::Matrix<float> A(cfg.m, cfg.n, kTileSize, kTileSize, order,
                               cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);
        slate::Matrix<float> Omega(cfg.n, l, kTileSize, kTileSize, order,
                                   cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);
        slate::Matrix<float> Y(cfg.m, l, kTileSize, kTileSize, order,
                               cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);
        slate::Matrix<float> Q(cfg.m, l, kTileSize, kTileSize, order,
                               cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);
        slate::Matrix<float> B(l, cfg.n, kTileSize, kTileSize, order,
                               cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);
        slate::Matrix<float> Uhat(l, l, kTileSize, kTileSize, order,
                                  cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);
        slate::Matrix<float> VT(l, cfg.n, kTileSize, kTileSize, order,
                                cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);
        slate::Matrix<float> U(cfg.m, l, kTileSize, kTileSize, order,
                               cfg.grid_p, cfg.grid_q, MPI_COMM_WORLD);

        // Keep origin tiles on host. Operations run on GPUs via Target::Devices.
        A.insertLocalTiles(slate::Target::Host);
        Omega.insertLocalTiles(slate::Target::Host);
        Y.insertLocalTiles(slate::Target::Host);
        Q.insertLocalTiles(slate::Target::Host);
        B.insertLocalTiles(slate::Target::Host);
        Uhat.insertLocalTiles(slate::Target::Host);
        VT.insertLocalTiles(slate::Target::Host);
        U.insertLocalTiles(slate::Target::Host);

        double t_fill = timed_max_seconds(MPI_COMM_WORLD, [&] {
            fill_random_matrix(A, cfg.seed + 101);
            fill_random_matrix(Omega, cfg.seed + 202);
            slate::set(0.0f, Y, opts);
            slate::set(0.0f, B, opts);
            slate::set(0.0f, Uhat, opts);
            slate::set(0.0f, VT, opts);
            slate::set(0.0f, U, opts);
        });

        // Step 4: Y = A * Omega
        double t_y = timed_max_seconds(MPI_COMM_WORLD, [&] {
            slate::multiply(1.0f, A, Omega, 0.0f, Y, opts);
        });

        // Step 5: explicit QR on Y.
        // geqrf stores reflectors in Y; then form explicit Q by applying implicit Q
        // to an identity matrix, equivalent to LAPACK's ungqr behavior.
        slate::TriangularFactors<float> T;
        double t_qr = timed_max_seconds(MPI_COMM_WORLD, [&] {
            slate::geqrf(Y, T, opts);
        });

        double t_ungqr = timed_max_seconds(MPI_COMM_WORLD, [&] {
            slate::set(0.0f, 1.0f, Q, opts);  // identity-like matrix
            slate::qr_multiply_by_q(slate::Side::Left, slate::Op::NoTrans, Y, T, Q, opts);
        });

        // Step 6: B = Q^T * A
        double t_b = timed_max_seconds(MPI_COMM_WORLD, [&] {
            auto QT = transpose(Q);
            slate::multiply(1.0f, QT, A, 0.0f, B, opts);
        });

        // Step 7: SVD of small matrix B.
        std::vector<float> sigma(static_cast<size_t>(std::min(B.m(), B.n())));
        double t_svd = timed_max_seconds(MPI_COMM_WORLD, [&] {
            // In current SLATE API this routine is named svd (LAPACK gesvd equivalent).
            slate::svd(B, sigma, Uhat, VT, opts);
        });

        // Step 8: U = Q * Uhat
        double t_u = timed_max_seconds(MPI_COMM_WORLD, [&] {
            slate::multiply(1.0f, Q, Uhat, 0.0f, U, opts);
        });

        const double t_total = t_y + t_qr + t_ungqr + t_b + t_svd + t_u;

        if (mpi_rank == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "Timing (max over ranks, seconds):\n"
                      << "  init/fill               : " << t_fill << "\n"
                      << "  Y = A*Omega             : " << t_y << "\n"
                      << "  geqrf(Y)                : " << t_qr << "\n"
                      << "  explicit Q (ungqr-like) : " << t_ungqr << "\n"
                      << "  B = Q^T*A               : " << t_b << "\n"
                      << "  svd(B)                  : " << t_svd << "\n"
                      << "  U = Q*Uhat              : " << t_u << "\n"
                      << "  total RSVD core         : " << t_total << "\n";

            std::cout << "Top-" << cfg.k << " singular values (descending): ";
            for (int64_t i = 0; i < std::min<int64_t>(cfg.k, 8); ++i) {
                int64_t idx = static_cast<int64_t>(sigma.size()) - 1 - i;
                std::cout << sigma[static_cast<size_t>(idx)] << (i + 1 == std::min<int64_t>(cfg.k, 8) ? '\n' : ' ');
            }
        }

        // Explicitly release matrix storage/workspace.
        clear_matrix(A);
        clear_matrix(Omega);
        clear_matrix(Y);
        clear_matrix(Q);
        clear_matrix(B);
        clear_matrix(Uhat);
        clear_matrix(VT);
        clear_matrix(U);

        MPI_Finalize();
        mpi_initialized = 0;
        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        if (mpi_initialized) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        return EXIT_FAILURE;
    }
}
