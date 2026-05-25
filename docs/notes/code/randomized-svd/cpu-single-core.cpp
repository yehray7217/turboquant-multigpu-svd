#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Options {
    int m = 256;
    int n = 128;
    int k = 16;
    int oversample = 4;
    int full_rank = 128;
    std::string decay_type = "exponential";
    double decay_param = 0.12;
    unsigned seed = 1234;
    double noise = 0.0;
    bool check_error = true;
};

using Matrix = std::vector<std::vector<double>>;

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --m <int>                 Number of rows of A. Default: 256\n"
        << "  --n <int>                 Number of cols of A. Default: 128\n"
        << "  --k <int>                 Target rank. Default: 16\n"
        << "  --oversample <int>        Oversampling p. l = k + p. Default: 4\n"
        << "  --full-rank <int>         Number of singular values in the test matrix. Default: 64\n"
        << "  --decay-type <string>          Singular value decay: exponential, polynomial, or step. Default: exponential\n"
        << "  --decay-param <float>     Decay parameter. alpha/p for exponential/polynomial, cutoff for step. Default: 0.12\n"
        << "  --noise <float>           Gaussian noise stddev added to A entries. Default: 0\n"
        << "  --seed <int>              RNG seed. Default: 1234\n"
        << "  --no-check-error          Skip reconstruction error.\n";
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
        else if (a == "--full-rank") opt.full_rank = std::stoi(need_value(a));
        else if (a == "--decay-type") opt.decay_type = need_value(a);
        else if (a == "--decay-param") opt.decay_param = std::stod(need_value(a));
        else if (a == "--noise") opt.noise = std::stod(need_value(a));
        else if (a == "--seed") opt.seed = static_cast<unsigned>(std::stoul(need_value(a)));
        else if (a == "--no-check-error") opt.check_error = false;
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + a);
        }
    }

    if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.oversample < 0 || opt.full_rank <= 0) {
        throw std::runtime_error("m, n, k, full-rank must be positive and oversample must be non-negative.");
    }
    if (opt.k + opt.oversample > std::min(opt.m, opt.n)) {
        throw std::runtime_error("Require k + oversample <= min(m, n).");
    }
    if (opt.decay_type != "exponential" && opt.decay_type != "polynomial" && opt.decay_type != "step") {
        throw std::runtime_error("decay must be one of: exponential, polynomial, step.");
    }
    if (opt.decay_type != "step" && opt.decay_param <= 0.0) {
        throw std::runtime_error("decay-param must be positive for exponential and polynomial decay.");
    }
    if (opt.noise < 0.0) {
        throw std::runtime_error("noise must be non-negative.");
    }
    opt.full_rank = std::min(opt.full_rank, std::min(opt.m, opt.n));
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

static Matrix zeros(int rows, int cols) {
    return Matrix(rows, std::vector<double>(cols, 0.0));
}

static Matrix random_matrix(int rows, int cols, unsigned seed, double scale) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, scale);
    Matrix A = zeros(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            A[i][j] = dist(gen);
        }
    }
    return A;
}

static Matrix transpose(const Matrix& A) {
    if (A.empty()) return {};
    Matrix AT = zeros(static_cast<int>(A[0].size()), static_cast<int>(A.size()));
    for (int i = 0; i < static_cast<int>(A.size()); ++i) {
        for (int j = 0; j < static_cast<int>(A[0].size()); ++j) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

static Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::runtime_error("matmul shape mismatch.");
    }
    const int m = static_cast<int>(A.size());
    const int p = static_cast<int>(A[0].size());
    const int n = static_cast<int>(B[0].size());
    Matrix C = zeros(m, n);

    for (int i = 0; i < m; ++i) {
        for (int kk = 0; kk < p; ++kk) {
            const double aik = A[i][kk];
            for (int j = 0; j < n; ++j) {
                C[i][j] += aik * B[kk][j];
            }
        }
    }
    return C;
}

static void modified_gram_schmidt_qr(const Matrix& A, Matrix& Q, Matrix& R) {
    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(A[0].size());
    Q = zeros(m, n);
    R = zeros(n, n);

    std::vector<double> v(m);
    for (int j = 0; j < n; ++j) {
        for (int row = 0; row < m; ++row) v[row] = A[row][j];

        for (int i = 0; i < j; ++i) {
            double dot = 0.0;
            for (int row = 0; row < m; ++row) dot += Q[row][i] * v[row];
            R[i][j] = dot;
            for (int row = 0; row < m; ++row) v[row] -= dot * Q[row][i];
        }

        double norm = 0.0;
        for (double x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm < 1e-12) {
            throw std::runtime_error("QR failed: columns are numerically rank deficient.");
        }

        R[j][j] = norm;
        for (int row = 0; row < m; ++row) Q[row][j] = v[row] / norm;
    }
}

static Matrix random_orthonormal_matrix(int rows, int cols, unsigned seed) {
    Matrix G = random_matrix(rows, cols, seed, 1.0);
    Matrix Q;
    Matrix R;
    modified_gram_schmidt_qr(G, Q, R);
    return Q;
}

static std::vector<double> make_singular_values(
    int full_rank, const std::string& decay_type, double decay_param) {
    std::vector<double> sigma(full_rank);
    for (int i = 0; i < full_rank; ++i) {
        if (decay_type == "exponential") {
            sigma[i] = std::exp(-decay_param * i);
        } else if (decay_type == "polynomial") {
            sigma[i] = std::pow(i + 1.0, -decay_param);
        } else if (decay_type == "step") {
            sigma[i] = (i < static_cast<int>(decay_param)) ? 1.0 : 0.01;
        } else {
            throw std::runtime_error("Unknown decay type: " + decay_type);
        }
    }
    return sigma;
}

static Matrix test_matrix_with_spectrum(
    int m, int n, int full_rank,
    const std::string& decay_type,
    double decay_param,
    double noise,
    unsigned seed) {
    Matrix U = random_orthonormal_matrix(m, full_rank, seed);
    Matrix V = random_orthonormal_matrix(n, full_rank, seed + 1);
    std::vector<double> sigma = make_singular_values(full_rank, decay_type, decay_param);

    Matrix A = zeros(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int r = 0; r < full_rank; ++r) {
                sum += U[i][r] * sigma[r] * V[j][r];
            }
            A[i][j] = sum;
        }
    }

    if (noise > 0.0) {
        std::mt19937 gen(seed + 2);
        std::normal_distribution<double> dist(0.0, noise);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                A[i][j] += dist(gen);
            }
        }
    }
    return A;
}

static void jacobi_eigen_symmetric(Matrix A, std::vector<double>& eigenvalues, Matrix& eigenvectors) {
    const int n = static_cast<int>(A.size());
    eigenvectors = zeros(n, n);
    for (int i = 0; i < n; ++i) eigenvectors[i][i] = 1.0;

    const int max_iters = 100 * n * n;
    const double eps = 1e-12;
    for (int iter = 0; iter < max_iters; ++iter) {
        int p = 0;
        int q = 1;
        double max_offdiag = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double value = std::abs(A[i][j]);
                if (value > max_offdiag) {
                    max_offdiag = value;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_offdiag < eps) break;

        const double app = A[p][p];
        const double aqq = A[q][q];
        const double apq = A[p][q];
        const double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(theta);
        const double s = std::sin(theta);

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            const double akp = A[k][p];
            const double akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[p][k] = A[k][p];
            A[k][q] = s * akp + c * akq;
            A[q][k] = A[k][q];
        }

        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = 0.0;
        A[q][p] = 0.0;

        for (int k = 0; k < n; ++k) {
            const double vkp = eigenvectors[k][p];
            const double vkq = eigenvectors[k][q];
            eigenvectors[k][p] = c * vkp - s * vkq;
            eigenvectors[k][q] = s * vkp + c * vkq;
        }
    }

    eigenvalues.assign(n, 0.0);
    for (int i = 0; i < n; ++i) eigenvalues[i] = A[i][i];
}

static void sort_eigenpairs_desc(std::vector<double>& eigenvalues, Matrix& eigenvectors) {
    const int n = static_cast<int>(eigenvalues.size());
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return eigenvalues[a] > eigenvalues[b];
    });

    std::vector<double> sorted_values(n);
    Matrix sorted_vectors = zeros(n, n);
    for (int new_col = 0; new_col < n; ++new_col) {
        const int old_col = order[new_col];
        sorted_values[new_col] = eigenvalues[old_col];
        for (int row = 0; row < n; ++row) {
            sorted_vectors[row][new_col] = eigenvectors[row][old_col];
        }
    }
    eigenvalues = std::move(sorted_values);
    eigenvectors = std::move(sorted_vectors);
}

static void svd_small_wide_matrix(
    const Matrix& B, int k, Matrix& U_tilde_k, std::vector<double>& S_k, Matrix& Vt_k) {
    const int l = static_cast<int>(B.size());
    const int n = static_cast<int>(B[0].size());

    // B is l x n with l << n. Diagonalizing B B^T is enough for this teaching code.
    Matrix C = matmul(B, transpose(B));
    std::vector<double> eigenvalues;
    Matrix U_tilde;
    jacobi_eigen_symmetric(C, eigenvalues, U_tilde);
    sort_eigenpairs_desc(eigenvalues, U_tilde);

    U_tilde_k = zeros(l, k);
    S_k.assign(k, 0.0);
    Vt_k = zeros(k, n);

    for (int r = 0; r < k; ++r) {
        const double sigma = std::sqrt(std::max(0.0, eigenvalues[r]));
        S_k[r] = sigma;
        for (int i = 0; i < l; ++i) U_tilde_k[i][r] = U_tilde[i][r];

        if (sigma > 1e-12) {
            for (int col = 0; col < n; ++col) {
                double sum = 0.0;
                for (int row = 0; row < l; ++row) {
                    sum += U_tilde[row][r] * B[row][col];
                }
                Vt_k[r][col] = sum / sigma;
            }
        }
    }
}

static double reconstruction_relative_error(
    const Matrix& A, const Matrix& U_k, const std::vector<double>& S_k, const Matrix& Vt_k) {
    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(A[0].size());
    const int k = static_cast<int>(S_k.size());

    long double err2 = 0.0L;
    long double norm2 = 0.0L;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            long double approx = 0.0L;
            for (int r = 0; r < k; ++r) {
                approx += static_cast<long double>(U_k[i][r]) * S_k[r] * Vt_k[r][j];
            }
            const long double diff = static_cast<long double>(A[i][j]) - approx;
            err2 += diff * diff;
            norm2 += static_cast<long double>(A[i][j]) * A[i][j];
        }
    }
    return std::sqrt(static_cast<double>(err2 / std::max(norm2, 1e-30L)));
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        const int m = opt.m;
        const int n = opt.n;
        const int k = opt.k;
        const int l = opt.k + opt.oversample;

        std::cout << "CPU single-core randomized SVD\n"
                  << "  m=" << m << " n=" << n << " k=" << k
                  << " oversample=" << opt.oversample << " l=" << l
                  << " seed=" << opt.seed << "\n"
                  << "  full_rank=" << opt.full_rank
                  << " decay=" << opt.decay_type
                  << " decay_param=" << opt.decay_param
                  << " noise=" << opt.noise << "\n";

        Timer timer;
        timer.tic();
        Matrix A = test_matrix_with_spectrum(
            m, n, opt.full_rank, opt.decay_type, opt.decay_param, opt.noise, opt.seed);
        Matrix Omega = random_matrix(n, l, opt.seed + 10, 1.0);
        const double t_init_ms = timer.toc_ms();

        // STEP 1: sample the column space, Y = A x Omega.
        timer.tic();
        Matrix Y = matmul(A, Omega);
        const double t_projection_ms = timer.toc_ms();

        // STEP 2: orthonormalize the sampled space, Y = Q x R.
        timer.tic();
        Matrix Q;
        Matrix R;
        modified_gram_schmidt_qr(Y, Q, R);
        const double t_qr_ms = timer.toc_ms();

        // STEP 3: project A into the small subspace, B = Q^T x A.
        timer.tic();
        Matrix B = matmul(transpose(Q), A);
        const double t_build_b_ms = timer.toc_ms();

        // STEP 4: compute SVD of small B, B = U_tilde x S x V^T.
        timer.tic();
        Matrix U_tilde_k;
        std::vector<double> S_k;
        Matrix Vt_k;
        svd_small_wide_matrix(B, k, U_tilde_k, S_k, Vt_k);
        const double t_svd_b_ms = timer.toc_ms();

        // STEP 5: map U_tilde back to the original row space, U_k = Q x U_tilde_k.
        timer.tic();
        Matrix U_k = matmul(Q, U_tilde_k);
        const double t_form_u_ms = timer.toc_ms();

        double rel_err = -1.0;
        double t_error_ms = 0.0;
        if (opt.check_error) {
            // STEP 6: optional correctness check, ||A - U_k S_k V_k^T||_F / ||A||_F.
            timer.tic();
            rel_err = reconstruction_relative_error(A, U_k, S_k, Vt_k);
            t_error_ms = timer.toc_ms();
        }

        std::cout << std::fixed << std::setprecision(4)
                  << "\nTimings (ms)\n"
                  << "  init_random_data      " << t_init_ms << "\n"
                  << "  projection_Y          " << t_projection_ms << "\n"
                  << "  qr_Y                  " << t_qr_ms << "\n"
                  << "  build_B               " << t_build_b_ms << "\n"
                  << "  svd_B                 " << t_svd_b_ms << "\n"
                  << "  form_U                " << t_form_u_ms << "\n";
        if (opt.check_error) {
            std::cout << "  reconstruction_error  " << t_error_ms << "\n";
        }

        std::cout << "\nLeading singular values\n  ";
        for (int i = 0; i < std::min(k, 10); ++i) {
            std::cout << std::setprecision(6) << S_k[i]
                      << (i + 1 == std::min(k, 10) ? "" : ", ");
        }
        std::cout << "\n";

        if (opt.check_error) {
            std::cout << std::setprecision(8)
                      << "\nRelative Frobenius reconstruction error\n"
                      << "  ||A - U_k S_k V_k^T||_F / ||A||_F = " << rel_err << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "!Error: " << e.what() << "\n";
        return 1;
    }
}
