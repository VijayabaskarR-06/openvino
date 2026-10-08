// Host-side emulation of oneDNN's blocked GEMM launches with 2D quantization.
//
// The new helpers (quant_2d_a/quant_2d_b) are compiled verbatim from jit.cpp.
// The "old" logic is a transcription of the code being replaced. For every
// launch, the kernel's addressing of zero points and scales is emulated as
// gemmstone does it (base + offset_Xq + local index using ldXq) and compared
// against the address the same element has in the full, unblocked tensor.

#include <array>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "gemmstone/problem.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
#include "quant_helpers.inc"
quant_2d_t new_quant(const gemmstone::GEMMProblem &p, bool is_a, dim_t mn, dim_t k) {
    return is_a ? quant_2d_a(p, mn, k) : quant_2d_b(p, mn, k);
}
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

using dnnl::impl::dim_t;
static dim_t cdiv(dim_t a, dim_t b) { return (a + b - 1) / b; }
static dim_t crnd(dim_t a, dim_t b) { return cdiv(a, b) * b; }
using namespace gemmstone;

struct Case {
    std::string name;
    bool is_a;          // A side (m blocking) or B side (n blocking)
    dim_t MN, K;        // full m (A) or n (B), and k
    dim_t block_mn, block_k;
    int gMN, gK;
    bool mn_major;      // quantization parameters contiguous in m (A) / n (B)
    int zp_dims;        // -1 none, 0 scalar, 1 vector, 2 matrix
    bool scale2d;
    bool expect_fixed;  // false for combinations documented as unsupported
};

static GEMMProblem make_problem(const Case &c) {
    GEMMProblem p;
    p.Ta = p.Ta_ext = p.Tb = p.Tb_ext = Type::f16;
    p.Tao = p.Tbo = Type::u8;
    if (c.is_a) {
        p.aoPtrDims = c.zp_dims;
        if (c.zp_dims >= 0) p.aOffset = ABOffset::Calc;
        p.asPtrDims = c.scale2d ? 2 : -1;
        p.aqGroupM = c.gMN;
        p.aqGroupK = c.gK;
        auto l = c.mn_major ? MatrixLayout::N : MatrixLayout::T;
        p.AO.layout = p.A_scale.layout = p.Ag.layout = l;
    } else {
        p.boPtrDims = c.zp_dims;
        if (c.zp_dims >= 0) p.bOffset = ABOffset::Calc;
        p.bsPtrDims = c.scale2d ? 2 : -1;
        p.bqGroupN = c.gMN;
        p.bqGroupK = c.gK;
        auto l = c.mn_major ? MatrixLayout::T : MatrixLayout::N;
        p.BO.layout = p.B_scale.layout = p.Bg.layout = l;
    }
    return p;
}

struct Launch {
    dim_t Bmn, Bk, size_mn, size_k, off, ld;
    dim_t zp_off; // offset the kernel applies to the zero point pointer
};

// Host logic: produce the launches and the quantization arguments per launch.
static std::vector<Launch> plan(const Case &c, bool fixed) {
    auto p = make_problem(c);
    bool q2d = c.is_a ? p.quantized2DA() : p.quantized2DB();
    dim_t block_k = c.block_k;
    if (fixed && q2d) block_k = crnd(block_k, std::max(c.gK, 1));
    bool has_off_arg = c.zp_dims >= 1 || c.scale2d; // offset_Xq is an argument

    std::vector<Launch> out;
    for (dim_t Bk = 0; Bk < c.K; Bk += block_k)
        for (dim_t Bmn = 0; Bmn < c.MN; Bmn += c.block_mn) {
            Launch l;
            l.Bmn = Bmn;
            l.Bk = Bk;
            l.size_mn = std::min(c.block_mn, c.MN - Bmn);
            l.size_k = std::min(block_k, c.K - Bk);
            if (fixed) {
                auto q = dnnl::impl::gpu::intel::gemm::new_quant(p, c.is_a, c.MN, c.K);
                l.ld = q.ld;
                l.off = q.offset(c.zp_dims, Bmn, Bk);
                // Generator fix: offset_Xq no longer moves a scalar zero point.
                l.zp_off = (c.zp_dims >= 1) ? l.off : 0;
            } else {
                l.ld = c.mn_major ? cdiv(l.size_mn, c.gMN) : cdiv(c.K, c.gK);
                l.off = has_off_arg ? Bmn : 0;
                l.zp_off = has_off_arg ? l.off : 0;
            }
            out.push_back(l);
        }
    return out;
}

static dim_t addr2d(bool mn_major, dim_t i, dim_t kk, dim_t gMN, dim_t gK, dim_t ld) {
    dim_t a = i / gMN, b = kk / gK;
    return mn_major ? a + b * ld : a * ld + b;
}

// Returns the fraction of rows (of m for A, of n for B) that read at least one
// zero point or scale from the wrong address (out-of-bounds reads count too).
static double wrong_rows(const Case &c, bool fixed) {
    auto launches = plan(c, fixed);
    dim_t full_ld = c.mn_major ? cdiv(c.MN, c.gMN) : cdiv(c.K, c.gK);
    dim_t q_size = cdiv(c.MN, c.gMN) * cdiv(c.K, c.gK);
    std::vector<char> bad(c.MN, 0);
    for (auto &l : launches) {
        // Check every k position where either the local or the global group changes.
        std::vector<dim_t> ks;
        for (dim_t kk = 0; kk < l.size_k; kk++)
            if (kk % c.gK == 0 || (l.Bk + kk) % c.gK == 0) ks.push_back(kk);
        for (dim_t i = 0; i < l.size_mn; i++) {
            dim_t I = l.Bmn + i;
            if (bad[I]) continue;
            for (dim_t kk : ks) {
                dim_t KK = l.Bk + kk;
                dim_t truth = addr2d(c.mn_major, I, KK, c.gMN, c.gK, full_ld);
                if (c.scale2d) {
                    dim_t got = l.off + addr2d(c.mn_major, i, kk, c.gMN, c.gK, l.ld);
                    if (got != truth || got < 0 || got >= q_size) { bad[I] = 1; break; }
                }
                if (c.zp_dims == 2) {
                    dim_t got = l.zp_off + addr2d(c.mn_major, i, kk, c.gMN, c.gK, l.ld);
                    if (got != truth || got >= q_size) { bad[I] = 1; break; }
                } else if (c.zp_dims == 1) {
                    if (l.zp_off + i != I) { bad[I] = 1; break; }
                } else if (c.zp_dims == 0) {
                    if (l.zp_off != 0) { bad[I] = 1; break; }
                }
            }
        }
    }
    dim_t n_bad = 0;
    for (char b : bad) n_bad += b;
    return double(n_bad) / double(c.MN);
}

// Numerical replay of the issue's reproducer: y = x * dequant(W)^T, one row,
// u4 weights, f16-range scales, group 128, kernel m-blocking of 65536.
static void issue_replay(dim_t N, bool asym) {
    const dim_t K = 4096, G = 128, NG = K / G;
    Case c {"issue", true, N, K, 65536, 16777216, 1, int(G), true,
            asym ? 2 : -1, true, true};
    std::mt19937_64 rng(0);
    std::uniform_int_distribution<int> u4(0, 15);
    std::uniform_real_distribution<double> us(5e-4, 4.5e-3);
    std::normal_distribution<double> nx(0.0, 0.1);

    std::vector<double> x(K), S(N * NG), Z(N * NG, 0.0), xsum(NG, 0.0);
    for (auto &v : x) v = nx(rng);
    for (dim_t g = 0; g < NG; g++)
        for (dim_t t = 0; t < G; t++) xsum[g] += x[g * G + t];
    // Quantization parameters stored as [K/G][N] (N contiguous), as OV passes them.
    for (auto &v : S) v = us(rng);
    if (asym) for (auto &v : Z) v = u4(rng);

    // Per-row, per-group dot products of the raw u4 weights with x.
    std::vector<double> P(N * NG);
    for (dim_t i = 0; i < N; i++)
        for (dim_t g = 0; g < NG; g++) {
            double acc = 0;
            for (dim_t t = 0; t < G; t++) acc += u4(rng) * x[g * G + t];
            P[i * NG + g] = acc;
        }

    auto run = [&](bool fixed, std::vector<double> &y) {
        y.assign(N, 0.0);
        for (auto &l : plan(c, fixed))
            for (dim_t i = 0; i < l.size_mn; i++) {
                double acc = 0;
                for (dim_t g = 0; g < NG; g++) {
                    dim_t a = l.off + i + g * l.ld; // col-major, group_m = 1
                    double s = (a >= 0 && a < N * NG) ? S[a] : NAN;
                    double z = (a >= 0 && a < N * NG) ? Z[a] : NAN;
                    acc += s * (P[(l.Bmn + i) * NG + g] - z * xsum[g]);
                }
                y[l.Bmn + i] = acc;
            }
    };
    std::vector<double> ref(N), y_old, y_new;
    for (dim_t i = 0; i < N; i++) {
        double acc = 0;
        for (dim_t g = 0; g < NG; g++)
            acc += S[g * N + i] * (P[i * NG + g] - Z[g * N + i] * xsum[g]);
        ref[i] = acc;
    }
    run(false, y_old);
    run(true, y_new);
    double amax = 0;
    for (double v : ref) amax = std::max(amax, std::fabs(v));
    auto stats = [&](const std::vector<double> &y, double &mx, double &frac) {
        mx = 0;
        dim_t n_bad = 0;
        for (dim_t i = 0; i < N; i++) {
            double e = std::fabs(y[i] - ref[i]) / amax;
            if (!(e <= 1e-2)) n_bad++;
            if (!(e <= mx)) mx = std::isnan(e) ? INFINITY : e;
        }
        frac = double(n_bad) / double(N);
    };
    double mo, fo, mn, fn;
    stats(y_old, mo, fo);
    stats(y_new, mn, fn);
    std::printf("  N=%6lld %-4s  old: max err %.1e, wrong %3.0f%%   new: max err %.1e, wrong %3.0f%%\n",
            (long long)N, asym ? "asym" : "sym", mo, 100 * fo, mn, 100 * fn);
}

int main() {
    std::printf("== Replay of issue 38640 (1 row, K=4096, G=128, block_m=65536) ==\n");
    for (dim_t N : {65536, 65537, 128256, 151936, 248320}) {
        issue_replay(N, true);
        issue_replay(N, false);
    }

    std::vector<Case> cases;
    const int HUGE_G = 1 << 24;
    // A side, m blocked (weights decompression, the reported path and variants).
    for (bool mn_major : {true, false})
        for (int zp : {-1, 0, 2})
            for (dim_t M : {65536, 65537, 128256, 151936})
                for (int gK : {32, 128, 96})
                    for (dim_t K : {dim_t(1024), dim_t(4160)})
                        cases.push_back({"A", true, M, K, 65536, 16777216, 1, gK, mn_major, zp, true, true});
    // A side, smaller m blocks, per-tensor-in-m groups, and k blocking (Xe-LP default).
    for (bool mn_major : {true, false})
        for (int zp : {-1, 0, 2})
            for (int gM : {1, HUGE_G})
                for (dim_t block_k : {dim_t(1024), dim_t(2048), dim_t(16777216)})
                    for (int gK : {32, 96, 128})
                        cases.push_back({"A-kblk", true, 20000, 4800, 8192, block_k, gM, gK, mn_major, zp, true, true});
    // A side, per-row zero point vector with 2D scales: fixed where the shared
    // offset can represent both (m-contiguous scales, group_m = 1, no k blocking).
    for (dim_t M : {65537, 151936})
        cases.push_back({"A-zp1d", true, M, 4096, 65536, 16777216, 1, 128, true, 1, true, true});
    cases.push_back({"A-zp1d-rowmajor", true, 65537, 4096, 65536, 16777216, 1, 128, false, 1, true, false});
    cases.push_back({"A-zp1d-kblk", true, 20000, 4096, 8192, 1024, 1, 128, true, 1, true, false});
    // A side, no 2D quantization: zero point vector only (must be unchanged).
    cases.push_back({"A-zp1d-only", true, 65537, 4096, 65536, 16777216, 1, 128, true, 1, false, true});
    // B side, n blocked (dynamically quantized activations, > 16384 tokens).
    for (bool mn_major : {true, false})
        for (int zp : {-1, 0, 2})
            for (dim_t N : {16384, 16385, 40000})
                for (int gK : {32, 128})
                    cases.push_back({"B", false, N, 4096, 16384, 16777216, 1, gK, mn_major, zp, true, true});

    int n_fail = 0, n_old_bad = 0, n_known = 0;
    std::map<std::string, std::array<int, 3>> summary; // cases, broken before, broken after
    for (auto &c : cases) {
        double old_bad = wrong_rows(c, false), new_bad = wrong_rows(c, true);
        bool single = (c.MN <= c.block_mn) && (c.K <= c.block_k);
        if (old_bad > 0) n_old_bad++;
        if (c.expect_fixed) {
            static const char *zp_name[] = {"no zp", "scalar zp", "vector zp", "2D zp"};
            auto key = c.name + (c.is_a ? (c.mn_major ? " m-major" : " k-major") : (c.mn_major ? " n-major" : " k-major"))
                    + ", " + zp_name[c.zp_dims + 1] + (c.scale2d ? " + 2D scales" : "");
            auto &e = summary[key];
            e[0]++;
            e[1] += old_bad > 0;
            e[2] += new_bad > 0;
        }
        if (!c.expect_fixed) {
            n_known++;
            std::printf("[known] %-16s MN=%lld K=%lld gK=%d %s zp=%d: old %.4f%% new %.4f%% rows wrong\n",
                    c.name.c_str(), (long long)c.MN, (long long)c.K, c.gK,
                    c.mn_major ? "mn-major" : "k-major", c.zp_dims, 100 * old_bad, 100 * new_bad);
            continue;
        }
        if (new_bad > 0 || (single && old_bad > 0)) {
            n_fail++;
            std::printf("[FAIL]  %-16s MN=%lld K=%lld bmn=%lld bk=%lld gMN=%d gK=%d %s zp=%d: old %.2f%% new %.2f%%\n",
                    c.name.c_str(), (long long)c.MN, (long long)c.K, (long long)c.block_mn,
                    (long long)c.block_k, c.gMN, c.gK, c.mn_major ? "mn-major" : "k-major",
                    c.zp_dims, 100 * old_bad, 100 * new_bad);
        }
    }
    std::printf("\n%-44s %6s %14s %13s\n", "category", "cases", "broken before", "broken after");
    for (auto &e : summary)
        std::printf("%-44s %6d %14d %13d\n", e.first.c_str(), e.second[0], e.second[1], e.second[2]);
    std::printf("\n== Address checks: %zu cases, %d broken before the fix, %d failing after, %d known limitations ==\n",
            cases.size(), n_old_bad, n_fail, n_known);
    return n_fail ? 1 : 0;
}
