// Independent brute-force oracle for neighbour lists (test support only).
//
// Shares no code with the neighbour search under test: it inverts the cell,
// computes fractional coordinates and distances itself in long double, and
// reads nothing from NNFinder except the table it checks. Conventions are
// those of Structure::cell: row k is the lattice vector a_k, a position is
// r = f0 a0 + f1 a1 + f2 a2, and a neighbour entry (j, s) of atom i stands
// for the image r_j + s0 a0 + s1 a1 + s2 a2 of atom j (s an integer triple
// relative to the stored, possibly unwrapped, positions).
//
// A pair (i, j, s) is classified by its exact-ish distance d (long double):
//   inside    d <  R - w
//   boundary  R - w <= d <= R + w
//   outside   d >  R + w
// with the window w = max(1e-9 R, 4 * guard_bound(st, R)), where
// guard_bound is the rounding bound of the design (16 eps (G+2)(R+2 rho)).
// A correct table holds every inside pair, and nothing outside.
//
// Not globbed by MLIP/tests/CMakeLists.txt (only *.cpp are); include it.

#ifndef TADAH_MLIP_TESTS_NN_ORACLE_H
#define TADAH_MLIP_TESTS_NN_ORACLE_H

#include <tadah/core/core_types.h>
#include <tadah/mlip/structure.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace nn_oracle {

// (i, j, s0, s1, s2): neighbour j of atom i at image shift s.
using Key = std::tuple<std::size_t, std::size_t, long, long, long>;
using KeySet = std::set<Key>;

struct Lattice {
  long double a[3][3];     // a[k][c]: component c of lattice vector a_k
  long double inv[3][3];   // inverse of the row matrix: f = r . inv
  long double h[3];        // heights V / |a_{k+1} x a_{k+2}|
  long double len[3];      // |a_k|
  long double recip[3];    // |b*_k| = 1 / h_k
  long double volume;      // |det|
};

inline Lattice lattice(const tadah::mlip::Structure &st) {
  Lattice L{};
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c) L.a[k][c] = st.cell(k, c);
  const auto &a = L.a;
  // Cofactors of the row matrix; inv = adj / det.
  long double cof[3][3];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) {
      const int r1 = (r + 1) % 3, r2 = (r + 2) % 3;
      const int c1 = (c + 1) % 3, c2 = (c + 2) % 3;
      cof[r][c] = a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1];
    }
  const long double det = a[0][0] * cof[0][0] + a[0][1] * cof[0][1] +
                          a[0][2] * cof[0][2];
  if (det == 0.0L) throw std::runtime_error("nn_oracle: singular cell");
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) L.inv[r][c] = cof[c][r] / det;
  L.volume = std::fabs(det);
  for (int k = 0; k < 3; ++k) {
    // |a_{k+1} x a_{k+2}| is the norm of cofactor row k.
    const long double cn = std::sqrt(cof[k][0] * cof[k][0] +
                                     cof[k][1] * cof[k][1] +
                                     cof[k][2] * cof[k][2]);
    L.h[k] = L.volume / cn;
    L.recip[k] = cn / L.volume;
    L.len[k] = std::sqrt(a[k][0] * a[k][0] + a[k][1] * a[k][1] +
                         a[k][2] * a[k][2]);
  }
  return L;
}

// Fractional coordinates f = r . inv of atom i.
inline std::array<long double, 3> frac(const Lattice &L,
                                       const tadah::mlip::Structure &st,
                                       std::size_t i) {
  std::array<long double, 3> f{};
  const auto &r = st.atoms[i].position;
  for (int k = 0; k < 3; ++k)
    f[k] = r[0] * L.inv[0][k] + r[1] * L.inv[1][k] + r[2] * L.inv[2][k];
  return f;
}

// Cartesian vector (r_j + s.A) - r_i in long double.
inline std::array<long double, 3> pair_vector(const Lattice &L,
                                              const tadah::mlip::Structure &st,
                                              std::size_t i, std::size_t j,
                                              long s0, long s1, long s2) {
  std::array<long double, 3> v{};
  const auto &ri = st.atoms[i].position;
  const auto &rj = st.atoms[j].position;
  for (int c = 0; c < 3; ++c)
    v[c] = (static_cast<long double>(rj[c]) - ri[c]) + s0 * L.a[0][c] +
           s1 * L.a[1][c] + s2 * L.a[2][c];
  return v;
}

inline long double norm(const std::array<long double, 3> &v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Largest |r_i| over the atoms.
inline long double max_position_norm(const tadah::mlip::Structure &st) {
  long double rho = 0;
  for (const auto &at : st.atoms) {
    const long double n = std::sqrt(
        static_cast<long double>(at.position[0]) * at.position[0] +
        static_cast<long double>(at.position[1]) * at.position[1] +
        static_cast<long double>(at.position[2]) * at.position[2]);
    rho = std::max(rho, n);
  }
  return rho;
}

// Rounding bound of the design's guard band, 16 eps (G + 2)(R + 2 rho) with
// G = sum_k |a_k| |b*_k| and eps = 2^-53. A correct table may hold pairs up
// to R plus (about) this much.
inline double guard_bound(const tadah::mlip::Structure &st, double R) {
  const Lattice L = lattice(st);
  long double G = 0;
  for (int k = 0; k < 3; ++k) G += L.len[k] * L.recip[k];
  const long double eps = std::ldexp(1.0L, -53);
  return static_cast<double>(16.0L * eps * (G + 2.0L) *
                             (R + 2.0L * max_position_norm(st)));
}

inline double window(const tadah::mlip::Structure &st, double R) {
  return std::max(1e-9 * R, 4.0 * guard_bound(st, R));
}

struct Reference {
  double R = 0;
  double w = 0;
  KeySet inside;     // d < R - w
  KeySet boundary;   // R - w <= d <= R + w
};

// Brute force over every ordered pair (i, j) and every image s whose
// distance can be below R + w. For the pair, u = s + f_j - f_i satisfies
// |u_k| <= d |b*_k| = d / h_k, so s_k lies within (R + w)/h_k + 1/2 of
// c_k = round(f_i - f_j); the range c_k +- (ceil((R + w)/h_k) + 2) covers it
// for any wrapping of the positions.
inline Reference reference(const tadah::mlip::Structure &st, double R) {
  Reference ref;
  ref.R = R;
  ref.w = window(st, R);
  const std::size_t n = st.natoms();
  if (n == 0) return ref;
  const Lattice L = lattice(st);
  std::vector<std::array<long double, 3>> f(n);
  for (std::size_t i = 0; i < n; ++i) f[i] = frac(L, st, i);
  long M[3];
  for (int k = 0; k < 3; ++k) {
    const long double m = std::ceil((R + ref.w) / L.h[k]) + 2.0L;
    if (!(m < 1e4L)) throw std::runtime_error("nn_oracle: image range too large");
    M[k] = static_cast<long>(m);
  }
  const long double lo = R - ref.w, hi = R + ref.w;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      long c[3];
      for (int k = 0; k < 3; ++k)
        c[k] = static_cast<long>(std::llround(f[i][k] - f[j][k]));
      for (long s0 = c[0] - M[0]; s0 <= c[0] + M[0]; ++s0)
        for (long s1 = c[1] - M[1]; s1 <= c[1] + M[1]; ++s1)
          for (long s2 = c[2] - M[2]; s2 <= c[2] + M[2]; ++s2) {
            if (i == j && s0 == 0 && s1 == 0 && s2 == 0) continue;
            const long double d = norm(pair_vector(L, st, i, j, s0, s1, s2));
            if (d < lo)
              ref.inside.emplace(i, j, s0, s1, s2);
            else if (d <= hi)
              ref.boundary.emplace(i, j, s0, s1, s2);
          }
    }
  return ref;
}

// The table of st.nn as a set of keys. Throws if a shift is not an integer
// (checked separately, with a message, by check_invariants).
inline KeySet table_set(const tadah::mlip::Structure &st) {
  KeySet out;
  if (!st.nn) throw std::runtime_error("nn_oracle: structure has no table");
  const auto &T = *st.nn;
  for (std::size_t i = 0; i < T.idx.size(); ++i)
    for (std::size_t k = 0; k < T.idx[i].size(); ++k) {
      const auto &s = T.shift[i][k];
      out.emplace(i, T.idx[i][k], std::lround(s[0]), std::lround(s[1]),
                  std::lround(s[2]));
    }
  return out;
}

inline std::string key_str(const Key &k) {
  std::ostringstream o;
  o << "(i=" << std::get<0>(k) << ", j=" << std::get<1>(k) << ", s=("
    << std::get<2>(k) << "," << std::get<3>(k) << "," << std::get<4>(k)
    << "))";
  return o.str();
}

struct Comparison {
  std::vector<Key> missing;   // inside pairs absent from the table
  std::vector<Key> extra;     // table pairs that are neither inside nor boundary
  std::size_t boundary = 0;   // pairs in the boundary window (either way OK)
  std::size_t table_size = 0;
  std::size_t inside_size = 0;

  bool exact() const { return missing.empty() && extra.empty(); }
  std::string summary(std::size_t max_items = 8) const {
    std::ostringstream o;
    o << "table " << table_size << " entries, oracle " << inside_size
      << " inside + " << boundary << " boundary; missing " << missing.size()
      << ", extra " << extra.size();
    std::size_t shown = 0;
    for (const auto &k : missing) {
      if (shown++ >= max_items) break;
      o << "\n  missing " << key_str(k);
    }
    shown = 0;
    for (const auto &k : extra) {
      if (shown++ >= max_items) break;
      o << "\n  extra " << key_str(k);
    }
    return o.str();
  }
};

// Compare st.nn with the brute force at cutoff R.
inline Comparison compare(const tadah::mlip::Structure &st, double R) {
  const Reference ref = reference(st, R);
  const KeySet tbl = table_set(st);
  Comparison c;
  c.boundary = ref.boundary.size();
  c.table_size = tbl.size();
  c.inside_size = ref.inside.size();
  for (const auto &k : ref.inside)
    if (!tbl.count(k)) c.missing.push_back(k);
  for (const auto &k : tbl)
    if (!ref.inside.count(k) && !ref.boundary.count(k)) c.extra.push_back(k);
  return c;
}

// Structural invariants of a correct table. Returns the violations found
// (empty when the table is sound); at most max_items of each kind.
inline std::vector<std::string> check_invariants(
    const tadah::mlip::Structure &st, std::size_t max_items = 8) {
  std::vector<std::string> bad;
  if (!st.nn) {
    bad.push_back("no table");
    return bad;
  }
  const auto &T = *st.nn;
  const std::size_t n = st.natoms();
  if (T.idx.size() != n || T.shift.size() != n || T.disp.size() != n) {
    bad.push_back("table size differs from the number of atoms");
    return bad;
  }
  const Lattice L = lattice(st);
  std::size_t n_int = 0, n_self = 0, n_dup = 0, n_order = 0, n_disp = 0,
              n_sym = 0, n_iidx = 0;
  const KeySet all = table_set(st);
  for (std::size_t i = 0; i < n; ++i) {
    if (T.shift[i].size() != T.idx[i].size() ||
        T.disp[i].size() != T.idx[i].size()) {
      bad.push_back("atom " + std::to_string(i) +
                    ": idx, shift and disp lengths differ");
      continue;
    }
    std::vector<std::array<long, 4>> keys;
    for (std::size_t k = 0; k < T.idx[i].size(); ++k) {
      const std::size_t j = T.idx[i][k];
      const auto &s = T.shift[i][k];
      bool integral = j < n;
      for (int c = 0; c < 3; ++c)
        integral = integral && std::isfinite(s[c]) && s[c] == std::round(s[c]);
      if (!integral) {
        if (n_int++ < max_items)
          bad.push_back("atom " + std::to_string(i) + " entry " +
                        std::to_string(k) +
                        ": neighbour index out of range or shift not an "
                        "integer triple");
        continue;
      }
      const long s0 = std::lround(s[0]), s1 = std::lround(s[1]),
                 s2 = std::lround(s[2]);
      if (j == i && s0 == 0 && s1 == 0 && s2 == 0 && n_self++ < max_items)
        bad.push_back("atom " + std::to_string(i) + ": itself at shift 0");
      keys.push_back({static_cast<long>(j), s0, s1, s2});
      // disp and nn_pos: nn_pos(i,k) - r_j == s.A
      const auto &dp = T.disp[i][k];
      const auto np = st.nn_pos(i, k);
      for (int c = 0; c < 3; ++c) {
        const long double sa = s0 * L.a[0][c] + s1 * L.a[1][c] +
                               s2 * L.a[2][c];
        const long double rj = st.atoms[j].position[c];
        const long double tol_disp = 1e-12L * (1.0L + std::fabs(sa));
        const long double tol_pos = tol_disp + 1e-12L * std::fabs(rj);
        const bool disp_bad = std::fabs(dp[c] - sa) > tol_disp;
        const bool pos_bad = std::fabs((np[c] - rj) - sa) > tol_pos;
        if ((disp_bad || pos_bad) && n_disp++ < max_items) {
          std::ostringstream o;
          o << "atom " << i << " entry " << k << ": disp/nn_pos differ from "
            << "s.A in component " << c;
          bad.push_back(o.str());
          break;
        }
      }
      // symmetry: (j, i, -s) present
      if (!all.count(Key(j, i, -s0, -s1, -s2)) && n_sym++ < max_items)
        bad.push_back("asymmetric: " + key_str(Key(i, j, s0, s1, s2)) +
                      " has no reverse entry");
      // get_nn_iindex finds the reverse entry
      if (all.count(Key(j, i, -s0, -s1, -s2))) {
        const std::size_t ii = st.get_nn_iindex(i, j, k);
        bool ok = ii < T.idx[j].size() && T.idx[j][ii] == i;
        if (ok) {
          const auto &rs = T.shift[j][ii];
          ok = std::lround(rs[0]) == -s0 && std::lround(rs[1]) == -s1 &&
               std::lround(rs[2]) == -s2;
        }
        if (!ok && n_iidx++ < max_items)
          bad.push_back("get_nn_iindex does not find the reverse of " +
                        key_str(Key(i, j, s0, s1, s2)));
      }
    }
    for (std::size_t k = 1; k < keys.size(); ++k) {
      if (keys[k] == keys[k - 1]) {
        if (n_dup++ < max_items)
          bad.push_back("atom " + std::to_string(i) + ": duplicate entry " +
                        std::to_string(k));
      } else if (keys[k] < keys[k - 1] && n_order++ < max_items) {
        bad.push_back("atom " + std::to_string(i) + ": entry " +
                      std::to_string(k) + " breaks the (j, s) order");
      }
    }
    // duplicates that are not adjacent (unsorted tables)
    std::vector<std::array<long, 4>> sorted = keys;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 1; k < sorted.size(); ++k)
      if (sorted[k] == sorted[k - 1] && n_dup++ < max_items)
        bad.push_back("atom " + std::to_string(i) + ": duplicate (j, s)");
  }
  return bad;
}

// Pairs a consumer uses at cutoff rc, computed as DescriptorsCalc does:
// delij = r_i - nn_pos(i, k), kept iff delij.delij < rc*rc.
inline KeySet consumer_filter(const tadah::mlip::Structure &st, double rc) {
  KeySet out;
  const auto &T = *st.nn;
  const double rc_sq = rc * rc;
  for (std::size_t i = 0; i < T.idx.size(); ++i) {
    const auto &ri = st.atoms[i].position;
    for (std::size_t k = 0; k < T.idx[i].size(); ++k) {
      const auto p = st.nn_pos(i, k);
      const double d0 = ri[0] - p[0], d1 = ri[1] - p[1], d2 = ri[2] - p[2];
      const double rsq = d0 * d0 + d1 * d1 + d2 * d2;
      if (rsq >= rc_sq) continue;
      const auto &s = T.shift[i][k];
      out.emplace(i, T.idx[i][k], std::lround(s[0]), std::lround(s[1]),
                  std::lround(s[2]));
    }
  }
  return out;
}

// Test-local supercell along the cell rows (independent of
// StructureTransformation::replicate): rows scaled by n_k, copy (x, y, z)
// of atom i at r_i + x a0 + y a1 + z a2, loop order x, y, z, then atoms.
// Energy, stress, forces and labels are not touched beyond copying the
// per-atom data; callers scale what they need.
inline tadah::mlip::Structure supercell(const tadah::mlip::Structure &st,
                                        int nx, int ny, int nz) {
  tadah::mlip::Structure out = st;
  out.atoms.clear();
  out.nn.reset();
  const int n[3] = {nx, ny, nz};
  for (int k = 0; k < 3; ++k)
    for (int c = 0; c < 3; ++c) out.cell(k, c) = st.cell(k, c) * n[k];
  for (int x = 0; x < nx; ++x)
    for (int y = 0; y < ny; ++y)
      for (int z = 0; z < nz; ++z)
        for (const auto &at : st.atoms) {
          auto copy = at;
          for (int c = 0; c < 3; ++c)
            copy.position[c] = at.position[c] + x * st.cell(0, c) +
                               y * st.cell(1, c) + z * st.cell(2, c);
          out.atoms.push_back(copy);
        }
  return out;
}

}  // namespace nn_oracle

#endif  // TADAH_MLIP_TESTS_NN_ORACLE_H
