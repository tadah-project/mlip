#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/detail/cell_geometry.h>
#include <tadah/mlip/detail/structure_errors.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tadah {
namespace mlip {

namespace {

using detail::CellGeometry;
using detail::cell_geometry;
using detail::is_finite_bits;
constexpr double kEps = detail::kUnitRoundoff;

/* Largest number of image cells a search may span, prod_k (2 N_k + 1). A
 * cell needs more only when it is smaller than about cutoff/10 in every
 * direction (or about cutoff/28 in two): not a material, but a cell that
 * something degenerated, e.g. an unbounded relaxation step. Such a list
 * would hold thousands of entries per atom and stall every consumer
 * (M_Tadah_Base::fpredict is O(nn^2) through get_nn_iindex), so the
 * search fails fast instead; HPO turns the exception into a failed
 * candidate. Every cube with edge > cutoff/10 is accepted. */
constexpr double kMaxImageCells = 10000.0;

/* Positions more than 2^31 cells from the origin are refused: a double
 * then keeps fewer than 22 bits of the position within its cell, and the
 * integer shifts would approach the range where they stop being exact. */
constexpr double kMaxCells = 2147483648.0;  // 2^31

/* Validity of the first-order rounding bounds: the relative error of the
 * reduced cell's geometry, 16 eps (Q + G_U + 1), must stay below 2^-20. */
constexpr double kMaxConditioning = 536870912.0;  // 2^29

std::string fmt(double x) {
  std::ostringstream o;
  o.precision(10);
  o << x;
  return o.str();
}

void check_cutoff(double cutoff) {
  if (!is_finite_bits(cutoff) || !(cutoff > 0.0))
    throw std::invalid_argument(
        "NNFinder: the cutoff must be a finite number above 0, got " +
        fmt(cutoff));
}

/* The rows of st.cell, refused if non-finite or (numerically) singular. */
CellGeometry checked_cell(const Structure &st, double A[3][3]) {
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c) {
      A[k][c] = st.cell(k, c);
      if (!is_finite_bits(A[k][c]))
        throw std::invalid_argument(
            "NNFinder: cell entry (" + std::to_string(k) + ", " +
            std::to_string(c) + ") is not a finite number");
    }
  const CellGeometry g = cell_geometry(A);
  if (!(g.volume != 0.0) || !is_finite_bits(g.G) || !(g.G < kMaxConditioning))
    throw std::invalid_argument(
        "NNFinder: the cell is singular or numerically singular (volume " +
        fmt(g.volume) + " A^3, rows (" + fmt(A[0][0]) + ", " + fmt(A[0][1]) +
        ", " + fmt(A[0][2]) + "), (" + fmt(A[1][0]) + ", " + fmt(A[1][1]) +
        ", " + fmt(A[1][2]) + "), (" + fmt(A[2][0]) + ", " + fmt(A[2][1]) +
        ", " + fmt(A[2][2]) + "))");
  return g;
}

/* A reduced basis of the lattice: rows fl(U A) for an integer unimodular
 * U (so the same lattice). LLL (delta = 0.99) with exact integer
 * bookkeeping; the rows are recomputed from U after every step, so no
 * rounding accumulates. Any unimodular U gives an exact search; the
 * reduction only makes skewed descriptions of a benign lattice cheap. */
struct ReducedBasis {
  long long U[3][3];
  double a[3][3];
};

void rows_from_U(ReducedBasis &r, const double A[3][3], int k) {
  for (int c = 0; c < 3; ++c)
    r.a[k][c] = static_cast<double>(r.U[k][0]) * A[0][c] +
                static_cast<double>(r.U[k][1]) * A[1][c] +
                static_cast<double>(r.U[k][2]) * A[2][c];
}

ReducedBasis reduce_basis(const double A[3][3]) {
  ReducedBasis r{};
  for (int k = 0; k < 3; ++k) {
    for (int l = 0; l < 3; ++l) r.U[k][l] = (k == l) ? 1 : 0;
    rows_from_U(r, A, k);
  }
  // Entries of U stay below 2^40 and multipliers below 2^30, so every
  // product stays exact in double and in long long; a reduction that would
  // need more stops where it is (still a valid basis).
  constexpr long double kMaxMultiplier = 1073741824.0L;  // 2^30
  constexpr long long kMaxEntry = 1099511627776LL;       // 2^40
  long double bs[3][3], Bn[3], mu[3][3];
  auto gram_schmidt = [&]() {
    for (int i = 0; i < 3; ++i) {
      for (int c = 0; c < 3; ++c) bs[i][c] = r.a[i][c];
      for (int j = 0; j < i; ++j) {
        long double dot = 0;
        for (int c = 0; c < 3; ++c) dot += (long double)r.a[i][c] * bs[j][c];
        mu[i][j] = dot / Bn[j];
        for (int c = 0; c < 3; ++c) bs[i][c] -= mu[i][j] * bs[j][c];
      }
      Bn[i] = bs[i][0] * bs[i][0] + bs[i][1] * bs[i][1] + bs[i][2] * bs[i][2];
    }
  };
  gram_schmidt();
  int k = 1;
  for (int step = 0; k < 3 && step < 1000; ++step) {
    for (int j = k - 1; j >= 0; --j) {
      const long double q = std::round(mu[k][j]);
      if (q == 0) continue;
      if (!(std::fabs(q) < kMaxMultiplier)) return r;
      const long long qi = static_cast<long long>(q);
      long long nu[3];
      for (int l = 0; l < 3; ++l) {
        nu[l] = r.U[k][l] - qi * r.U[j][l];
        if (nu[l] > kMaxEntry || nu[l] < -kMaxEntry) return r;
      }
      for (int l = 0; l < 3; ++l) r.U[k][l] = nu[l];
      rows_from_U(r, A, k);
      gram_schmidt();
    }
    if (Bn[k] >= (0.99L - mu[k][k - 1] * mu[k][k - 1]) * Bn[k - 1]) {
      ++k;
    } else {
      for (int l = 0; l < 3; ++l) {
        std::swap(r.U[k][l], r.U[k - 1][l]);
        std::swap(r.a[k][l], r.a[k - 1][l]);
      }
      gram_schmidt();
      k = std::max(k - 1, 1);
    }
  }
  return r;
}

/* Everything the search needs about the lattice. */
struct SearchGeometry {
  CellGeometry orig;    // the cell as stored (distances, the guard band)
  ReducedBasis red;     // reduced basis
  CellGeometry g;       // geometry of fl(U A)
  double kappa;         // relative error bound of g's heights and fractions
};

SearchGeometry search_geometry(const Structure &st) {
  SearchGeometry s;
  double A[3][3];
  s.orig = checked_cell(st, A);
  s.red = reduce_basis(A);
  s.g = cell_geometry(s.red.a);
  // Rounding of fl(U A): row l is off by at most 3 eps L_l with
  // L_l = sum_m |U_lm| |a_m|; this moves a fractional coordinate by at
  // most 3 eps rho |b*_k| G_U and a height by a relative 6 eps G_U, with
  // G_U = sum_l |b*_l| L_l.
  double GU = 0.0;
  for (int l = 0; l < 3; ++l) {
    double L = 0.0;
    for (int m = 0; m < 3; ++m)
      L += std::fabs(static_cast<double>(s.red.U[l][m])) * s.orig.length[m];
    GU += s.g.recip[l] * L;
  }
  const double cond = s.g.Q + GU + 1.0;
  if (!(s.g.volume != 0.0) || !is_finite_bits(cond) ||
      !(cond < kMaxConditioning))
    throw std::invalid_argument(
        "NNFinder: the cell is numerically singular (volume " +
        fmt(s.orig.volume) + " A^3)");
  s.kappa = 16.0 * kEps * cond;
  return s;
}

/* Image bound of one axis for bins of n per cell: covers every pair
 * whose true distance is below Rg. In exact arithmetic the bin offset of
 * such a pair satisfies |Delta| < Rg n / h + 1. eta bounds the rounding
 * of the fractions (F per coordinate, 2 eps for the wrap and the bin
 * index) and of the height. */
double stencil_bound(double Rg, double n, double h, double F, double kappa) {
  const double Y = Rg * n / h;
  const double eta = 2.0 * n * (F + 2.0 * kEps) + 2.0 * kappa * Y;
  return std::floor(Y + eta) + 1.0;
}

long long floor_div(long long a, long long b) {
  long long d = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --d;
  return d;
}

long long mul_add_checked(long long acc, long long x, long long y) {
  long long p, s;
  if (__builtin_mul_overflow(x, y, &p) || __builtin_add_overflow(acc, p, &s))
    throw std::invalid_argument(
        "NNFinder: the image shifts of this cell and these positions do not "
        "fit in 64-bit integers");
  return s;
}

struct Entry {
  std::size_t j;
  long long s[3];
  bool operator<(const Entry &o) const {
    if (j != o.j) return j < o.j;
    if (s[0] != o.s[0]) return s[0] < o.s[0];
    if (s[1] != o.s[1]) return s[1] < o.s[1];
    return s[2] < o.s[2];
  }
};

std::shared_ptr<NNTable> build_table(const Structure &st, double R) {
  check_cutoff(R);
  const std::size_t natoms = st.natoms();
  auto tbl = std::make_shared<NNTable>();
  tbl->idx.resize(natoms);
  tbl->shift.resize(natoms);
  tbl->disp.resize(natoms);
  tbl->cutoff = R;
  if (natoms == 0) return tbl;

  const SearchGeometry geo = search_geometry(st);
  const CellGeometry &g = geo.g;
  const CellGeometry &g0 = geo.orig;

  // Positions: finite, and not absurdly far from the origin.
  double rho = 0.0;
  for (std::size_t i = 0; i < natoms; ++i) {
    const auto &r = st.atoms[i].position;
    for (int c = 0; c < 3; ++c)
      if (!is_finite_bits(r[c]))
        throw std::invalid_argument(
            "NNFinder: atom " + std::to_string(i + 1) + " of " +
            std::to_string(natoms) + " has a non-finite position");
    for (int k = 0; k < 3; ++k) {
      const double f = r[0] * g0.bstar[k][0] + r[1] * g0.bstar[k][1] +
                       r[2] * g0.bstar[k][2];
      if (!(std::fabs(f) < kMaxCells))
        throw std::invalid_argument(
            "NNFinder: atom " + std::to_string(i + 1) + " of " +
            std::to_string(natoms) +
            " lies more than 2^31 cells from the origin");
    }
    rho = std::max(rho, std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]));
  }

  // Guard band: delta bounds the rounding of any computed distance, so a
  // pair a consumer computes closer than R is truly closer than R + delta
  // and computes here closer than R + 2 delta. Pairs are kept below
  // Rinc = R + 2 delta, and the search covers true distances below
  // Rg = R + 3 delta, which contains every pair the test can keep: the
  // list is then exactly {pairs computed below Rinc}, however it is
  // searched.
  const double delta = detail::distance_rounding_bound(g0.G, R, rho);
  const double Rinc = R + 2.0 * delta;
  const double Rinc_sq = Rinc * Rinc;
  const double Rg = R + 3.0 * delta;
  tbl->guard = 2.0 * delta;

  // Per-axis rounding bound of a fractional coordinate.
  double F[3];
  for (int k = 0; k < 3; ++k) F[k] = geo.kappa * (rho * g.recip[k] + 1.0);

  // The image cap, from the image bound with one bin per cell.
  double N[3], total = 1.0;
  for (int k = 0; k < 3; ++k) {
    N[k] = stencil_bound(Rg, 1.0, g.height[k], F[k], geo.kappa);
    total *= 2.0 * N[k] + 1.0;
  }
  if (!is_finite_bits(total) || total > kMaxImageCells) {
    throw std::runtime_error(
        "NNFinder: the cell is too small for the cutoff " + fmt(R) +
        " A: the search would need (" + fmt(N[0]) + ", " + fmt(N[1]) + ", " +
        fmt(N[2]) + ") periodic images per direction, " + fmt(total) +
        " image cells, more than the limit of " + fmt(kMaxImageCells) +
        " (cell heights " + fmt(g.height[0]) + ", " + fmt(g.height[1]) +
        ", " + fmt(g.height[2]) +
        " A). Such a cell is degenerate, e.g. the result of an unbounded "
        "relaxation step");
  }

  // Cell list: bins at least Rg (1 + 2^-20) wide, so a thick axis keeps a
  // stencil of +-1 even when its height is an exact multiple of R; at most
  // about 2 natoms + 1 bins in all, so a vacuum box stays cheap. Any bin
  // counts are correct because the stencil is derived from them.
  const double limit = 2.0 * static_cast<double>(natoms) + 1.0;
  double nb[3];
  for (int k = 0; k < 3; ++k) {
    nb[k] = std::floor(g.height[k] / (Rg * (1.0 + 0x1p-20)));
    nb[k] = std::min(std::max(nb[k], 1.0), limit);
  }
  while (nb[0] * nb[1] * nb[2] > limit) {
    const int k = (nb[0] >= nb[1] && nb[0] >= nb[2]) ? 0
                  : (nb[1] >= nb[2])                 ? 1
                                                     : 2;
    nb[k] = std::max(1.0, std::floor(nb[k] / 2.0));
  }
  long long n[3], S[3];
  for (int k = 0; k < 3; ++k) {
    n[k] = static_cast<long long>(nb[k]);
    S[k] = static_cast<long long>(
        stencil_bound(Rg, nb[k], g.height[k], F[k], geo.kappa));
  }

  // Wrapped fractional coordinates in the reduced basis: f = k + w with
  // an integer k and w in [0, 1]; bin index floor(w n), clamped to n - 1
  // (w == 1 only when the true w is within 2^-53 of 1).
  std::vector<std::array<long long, 3>> kint(natoms);
  std::vector<std::array<long long, 3>> bin(natoms);
  for (std::size_t i = 0; i < natoms; ++i) {
    const auto &r = st.atoms[i].position;
    for (int k = 0; k < 3; ++k) {
      const double f = r[0] * g.bstar[k][0] + r[1] * g.bstar[k][1] +
                       r[2] * g.bstar[k][2];
      const double fl = std::floor(f);
      if (!(std::fabs(fl) < 0x1p62))
        throw std::invalid_argument(
            "NNFinder: atom " + std::to_string(i + 1) +
            " lies too far from the origin for this cell");
      kint[i][k] = static_cast<long long>(fl);
      const double w = f - fl;
      long long b = static_cast<long long>(std::floor(w * nb[k]));
      bin[i][k] = std::min(std::max(b, 0LL), n[k] - 1);
    }
  }
  const std::size_t nbins = static_cast<std::size_t>(n[0] * n[1] * n[2]);
  auto bin_id = [&](long long b0, long long b1, long long b2) {
    return static_cast<std::size_t>(b0 + n[0] * (b1 + n[1] * b2));
  };
  std::vector<std::size_t> start(nbins + 1, 0), members(natoms);
  for (std::size_t i = 0; i < natoms; ++i)
    ++start[bin_id(bin[i][0], bin[i][1], bin[i][2]) + 1];
  for (std::size_t b = 0; b < nbins; ++b) start[b + 1] += start[b];
  {
    std::vector<std::size_t> fill(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < natoms; ++i)
      members[fill[bin_id(bin[i][0], bin[i][1], bin[i][2])]++] = i;
  }

  const auto &A = g0.a;
  const auto &U = geo.red.U;
  std::vector<std::vector<Entry>> out(natoms);
  for (std::size_t p = 0; p < natoms; ++p) {
    const auto &rp = st.atoms[p].position;
    for (long long d0 = -S[0]; d0 <= S[0]; ++d0) {
      const long long B0 = bin[p][0] + d0;
      const long long m0 = floor_div(B0, n[0]);
      const long long q0 = B0 - m0 * n[0];
      for (long long d1 = -S[1]; d1 <= S[1]; ++d1) {
        const long long B1 = bin[p][1] + d1;
        const long long m1 = floor_div(B1, n[1]);
        const long long q1 = B1 - m1 * n[1];
        for (long long d2 = -S[2]; d2 <= S[2]; ++d2) {
          const long long B2 = bin[p][2] + d2;
          const long long m2 = floor_div(B2, n[2]);
          const long long q2 = B2 - m2 * n[2];
          const std::size_t b = bin_id(q0, q1, q2);
          for (std::size_t t = start[b]; t < start[b + 1]; ++t) {
            const std::size_t q = members[t];
            if (q < p) continue;
            // Image shift in the reduced basis, then in the stored one.
            const long long sr[3] = {m0 + kint[p][0] - kint[q][0],
                                     m1 + kint[p][1] - kint[q][1],
                                     m2 + kint[p][2] - kint[q][2]};
            if (q == p) {
              // (p, p, s) and (p, p, -s) are one pair: take s > 0.
              const bool positive =
                  sr[0] > 0 || (sr[0] == 0 && (sr[1] > 0 ||
                                               (sr[1] == 0 && sr[2] > 0)));
              if (!positive) continue;
            }
            long long s[3];
            for (int c = 0; c < 3; ++c) {
              long long acc = 0;
              for (int l = 0; l < 3; ++l) acc = mul_add_checked(acc, sr[l], U[l][c]);
              s[c] = acc;
            }
            // Distance as consumers compute it: r_p - (r_q + disp) with
            // disp = s0 a0 + s1 a1 + s2 a2 (populate_disp).
            const double s0 = static_cast<double>(s[0]);
            const double s1 = static_cast<double>(s[1]);
            const double s2 = static_cast<double>(s[2]);
            const auto &rq = st.atoms[q].position;
            double rsq = 0.0;
            for (int c = 0; c < 3; ++c) {
              const double D = s0 * A[0][c] + s1 * A[1][c] + s2 * A[2][c];
              const double del = rp[c] - (rq[c] + D);
              rsq += del * del;
            }
            if (rsq < Rinc_sq) {
              out[p].push_back({q, {s[0], s[1], s[2]}});
              out[q].push_back({p, {-s[0], -s[1], -s[2]}});
            }
          }
        }
      }
    }
  }

  for (std::size_t i = 0; i < natoms; ++i) {
    auto &e = out[i];
    std::sort(e.begin(), e.end());
    tbl->idx[i].resize(e.size());
    tbl->shift[i].resize(e.size());
    for (std::size_t k = 0; k < e.size(); ++k) {
      tbl->idx[i][k] = e[k].j;
      tbl->shift[i][k] = tadah::core::Vec3d(static_cast<double>(e[k].s[0]),
                                            static_cast<double>(e[k].s[1]),
                                            static_cast<double>(e[k].s[2]));
    }
    std::vector<Entry>().swap(e);
  }
  NNFinder().populate_disp(st, tbl);
  return tbl;
}

}  // namespace

double NNFinder::get_cutoff(const tadah::core::Context &context) const
{
  return context.get<double>("RCUTMAX");
}

bool NNFinder::check_box(const Structure &st, double cutoff)
{
  check_cutoff(cutoff);
  const SearchGeometry geo = search_geometry(st);
  for (int k = 0; k < 3; ++k)
    if (geo.g.height[k] < cutoff) return false;
  return true;
}

void NNFinder::populate_disp(const Structure &st,
                              std::shared_ptr<NNTable> tbl) const
{
  // Row-based cell layout matches existing nn_pos() math:
  //   disp = shift[0]*row(0) + shift[1]*row(1) + shift[2]*row(2)
  const double c00 = st.cell(0,0), c10 = st.cell(1,0), c20 = st.cell(2,0);
  const double c01 = st.cell(0,1), c11 = st.cell(1,1), c21 = st.cell(2,1);
  const double c02 = st.cell(0,2), c12 = st.cell(1,2), c22 = st.cell(2,2);

  const size_t natoms = tbl->idx.size();
  tbl->disp.assign(natoms, {});
  for (size_t i = 0; i < natoms; ++i) {
    const auto& sh_i = tbl->shift[i];
    auto& dp_i = tbl->disp[i];
    dp_i.resize(sh_i.size());
    for (size_t k = 0; k < sh_i.size(); ++k) {
      const double s0 = sh_i[k][0];
      const double s1 = sh_i[k][1];
      const double s2 = sh_i[k][2];
      dp_i[k][0] = s0*c00 + s1*c10 + s2*c20;
      dp_i[k][1] = s0*c01 + s1*c11 + s2*c21;
      dp_i[k][2] = s0*c02 + s1*c12 + s2*c22;
    }
  }
}

NNFinder::NNTablePtr NNFinder::build(const Structure &st, double cutoff) const
{
  return build_table(st, cutoff);
}

NNFinder::NNTablePtr NNFinder::build(const Structure &st,
                                     const core::Context &ctx) const
{
  return build(st, get_cutoff(ctx));
}

NNFinder::NNTablePtr NNFinder::build(Structure &st, double cutoff) const
{
  return build(static_cast<const Structure &>(st), cutoff);
}

NNFinder::NNTablePtr NNFinder::build(Structure &st,
                                     const core::Context &context) const
{
  return build(static_cast<const Structure &>(st), get_cutoff(context));
}

void NNFinder::calc(Structure &st, double cutoff) const
{
  st.nn = build(static_cast<const Structure &>(st), cutoff);
}

void NNFinder::calc(Structure &st, const core::Context &context) const
{
  calc(st, get_cutoff(context));
}

void NNFinder::calc(StructureDB &stdb, const core::Context &context) const
{
  // Read the cutoff once: a missing RCUTMAX is the context's error, not
  // any structure's.
  calc(stdb, get_cutoff(context));
}

void NNFinder::calc(StructureDB &stdb, double cutoff) const
{
  // An exception may not leave the OpenMP loop; keep the lowest-numbered
  // failure and rethrow it after the loop, naming the structure.
  const std::size_t n = stdb.size();
  std::vector<std::exception_ptr> errors(n);
#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic)
#endif
  for (std::size_t i = 0; i < n; i++) {
    try {
      calc(stdb(i), cutoff);
    } catch (...) {
      errors[i] = std::current_exception();
    }
  }
  for (std::size_t i = 0; i < n; ++i)
    if (errors[i])
      detail::rethrow_for_structure(errors[i], i, n, stdb(i).label,
                                    "NNFinder");
}

} // namespace mlip
} // namespace tadah
