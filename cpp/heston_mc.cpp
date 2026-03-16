// Heston Monte Carlo pricer for European SPX options.
//
// Variance: Andersen (2008) Quadratic-Exponential (QE) scheme, psi_c = 1.5.
// Log-forward: Andersen's central discretisation (gamma1 = gamma2 = 1/2), simulated
// under the T-forward measure so X_t = ln(F_t/F_0) needs no drift term from rates.
// One set of paths prices every option: the time grid hits each expiry exactly.
//
// Usage: heston_mc <options.csv> <out.csv> v0 kappa theta sigma rho [paths] [steps_per_year] [seed]
//   options.csv columns: id,T,F,DF,K,cp     (header row required)
//   out.csv columns:     id,price,stderr
//
// Build: g++ -O3 -march=native -fopenmp -std=c++17 heston_mc.cpp -o heston_mc
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

struct Option { long id; double T, F, DF, K; int cp; int exp_idx; };
struct Params { double v0, kappa, theta, sigma, rho; };

static std::vector<Option> read_options(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    std::string line; std::getline(in, line);  // header
    std::vector<Option> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line); std::string tok; Option o{};
        std::getline(ss, tok, ','); o.id = std::stol(tok);
        std::getline(ss, tok, ','); o.T = std::stod(tok);
        std::getline(ss, tok, ','); o.F = std::stod(tok);
        std::getline(ss, tok, ','); o.DF = std::stod(tok);
        std::getline(ss, tok, ','); o.K = std::stod(tok);
        std::getline(ss, tok, ','); o.cp = std::stoi(tok);
        out.push_back(o);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "usage: heston_mc options.csv out.csv v0 kappa theta sigma rho "
                     "[paths=200000] [steps_per_year=1460] [seed=42]\n";
        return 1;
    }
    std::vector<Option> opts = read_options(argv[1]);
    const Params p{std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5]),
                   std::stod(argv[6]), std::stod(argv[7])};
    const long n_paths = argc > 8 ? std::stol(argv[8]) : 200000;
    const double steps_per_year = argc > 9 ? std::stod(argv[9]) : 1460.0;
    const unsigned long long seed = argc > 10 ? std::stoull(argv[10]) : 42ULL;

    // distinct expiries -> index
    std::vector<double> exps;
    for (auto& o : opts) exps.push_back(o.T);
    std::sort(exps.begin(), exps.end());
    exps.erase(std::unique(exps.begin(), exps.end(),
                           [](double a, double b) { return std::fabs(a - b) < 1e-12; }), exps.end());
    for (auto& o : opts)
        o.exp_idx = int(std::lower_bound(exps.begin(), exps.end(), o.T - 1e-12) - exps.begin());
    std::vector<std::vector<int>> by_exp(exps.size());
    for (int i = 0; i < (int)opts.size(); ++i) by_exp[opts[i].exp_idx].push_back(i);

    // time grid that lands exactly on every expiry
    std::vector<double> dts; std::vector<int> stop_at;  // stop_at[j] = step index after which expiry j is reached
    double t_prev = 0.0;
    for (double Te : exps) {
        int n = std::max(1, int(std::ceil((Te - t_prev) * steps_per_year)));
        for (int s = 0; s < n; ++s) dts.push_back((Te - t_prev) / n);
        stop_at.push_back(int(dts.size()) - 1);
        t_prev = Te;
    }

    // per-step constants
    const double k = p.kappa, th = p.theta, sg = p.sigma, rho = p.rho;
    const double g1 = 0.5, g2 = 0.5, psi_c = 1.5;
    struct StepC { double e, c1, c2, K0, K1, K2, K3, K4; };
    std::vector<StepC> sc(dts.size());
    for (size_t i = 0; i < dts.size(); ++i) {
        double d = dts[i], e = std::exp(-k * d);
        sc[i].e = e;
        sc[i].c1 = sg * sg * e * (1 - e) / k;
        sc[i].c2 = th * sg * sg * (1 - e) * (1 - e) / (2 * k);
        sc[i].K0 = -rho * k * th / sg * d;
        sc[i].K1 = g1 * d * (k * rho / sg - 0.5) - rho / sg;
        sc[i].K2 = g2 * d * (k * rho / sg - 0.5) + rho / sg;
        sc[i].K3 = g1 * d * (1 - rho * rho);
        sc[i].K4 = g2 * d * (1 - rho * rho);
    }

    const size_t nO = opts.size();
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<double>> sum(n_threads, std::vector<double>(nO, 0.0)),
                                     sum2(n_threads, std::vector<double>(nO, 0.0));
    std::vector<std::vector<double>> mart(n_threads, std::vector<double>(exps.size(), 0.0));

    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        std::mt19937_64 rng(seed + 0x9E3779B97F4A7C15ULL * (tid + 1));
        std::normal_distribution<double> N01(0.0, 1.0);
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        auto& S = sum[tid]; auto& S2 = sum2[tid]; auto& M = mart[tid];

#pragma omp for schedule(static)
        for (long path = 0; path < n_paths; ++path) {
            double v = p.v0, x = 0.0;
            int j = 0;
            for (size_t i = 0; i < dts.size(); ++i) {
                const StepC& c = sc[i];
                // QE step for variance
                double m = th + (v - th) * c.e;
                double s2 = v * c.c1 + c.c2;
                double psi = s2 / (m * m);
                double vn;
                if (psi <= psi_c) {
                    double ip = 2.0 / psi;
                    double b2 = ip - 1.0 + std::sqrt(ip) * std::sqrt(ip - 1.0);
                    double a = m / (1.0 + b2);
                    double z = std::sqrt(b2) + N01(rng);
                    vn = a * z * z;
                } else {
                    double pp = (psi - 1.0) / (psi + 1.0);
                    double beta = (1.0 - pp) / m;
                    double u = U01(rng);
                    vn = (u <= pp) ? 0.0 : std::log((1.0 - pp) / (1.0 - u)) / beta;
                }
                // log-forward step
                x += c.K0 + c.K1 * v + c.K2 * vn + std::sqrt(std::max(0.0, c.K3 * v + c.K4 * vn)) * N01(rng);
                v = vn;
                if ((int)i == stop_at[j]) {
                    double ex = std::exp(x);
                    M[j] += ex;
                    for (int oi : by_exp[j]) {
                        const Option& o = opts[oi];
                        double ST = o.F * ex;
                        double pay = std::max(o.cp * (ST - o.K), 0.0);
                        S[oi] += pay; S2[oi] += pay * pay;
                    }
                    ++j;
                }
            }
        }
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::ofstream out(argv[2]);
    out << "id,price,stderr\n";
    out.precision(10);
    for (size_t i = 0; i < nO; ++i) {
        double s = 0, s2 = 0;
        for (int t = 0; t < n_threads; ++t) { s += sum[t][i]; s2 += sum2[t][i]; }
        double mean = s / n_paths;
        double var = std::max(0.0, s2 / n_paths - mean * mean);
        out << opts[i].id << "," << opts[i].DF * mean << "," << opts[i].DF * std::sqrt(var / n_paths) << "\n";
    }
    std::fprintf(stderr, "paths=%ld steps=%zu threads=%d time=%.2fs\n", n_paths, dts.size(), n_threads, secs);
    for (size_t j = 0; j < exps.size(); ++j) {
        double mj = 0; for (int t = 0; t < n_threads; ++t) mj += mart[t][j];
        std::fprintf(stderr, "  T=%.4f  E[F_T/F_0]-1 = %+.2e\n", exps[j], mj / n_paths - 1.0);
    }
    return 0;
}
