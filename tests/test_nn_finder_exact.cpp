// NNFinder against an independent brute-force oracle (nn_oracle.h).
//
// The contract under test: for a non-singular cell (row k of Structure::cell
// is the lattice vector a_k), any finite positions, wrapped into the cell or
// not, and a cutoff R > 0, NNFinder::calc / NNFinder::build store for every
// atom i each (j, s) with |r_j + s0 a0 + s1 a1 + s2 a2 - r_i| < R (j == i
// when s != 0), and possibly pairs up to R plus a rounding guard band.
//   - The table is symmetric, (i -> j, s) <=> (j -> i, -s), holds no
//     duplicate (j, s), and each atom's entries are sorted by (j, s0, s1, s2).
//   - s is an integer triple relative to the STORED positions, which the
//     search never modifies; disp = s.A.
//   - The search reduces the lattice basis first, so a skewed description of
//     a benign lattice is searched like the reduced one.
//   - More than 10^4 image cells, prod_k (2 N_k + 1) with
//     N_k = floor(R / h_k) + 1 on the reduced heights h_k, is refused with
//     std::runtime_error ("image cells"); an invalid cutoff, position or cell
//     with std::invalid_argument; calc(StructureDB&) names the failing
//     structure.
//   - check_box(st, R) is true iff every reduced cell height is >= R.
//
// Every geometry avoids pairs at the cutoff itself (the oracle's boundary
// window is asserted empty) except the dedicated exact-cutoff case, so a
// table either equals the oracle's inside set or it is wrong.
//
// Supercells and wrapped copies are built here, not with production code.

#include "catch2/catch.hpp"
#include "nn_oracle.h"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using tadah::core::PeriodicTable;
using tadah::core::Vec3d;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

namespace {

// Uniform numbers from the raw std::mt19937 sequence, which the standard
// fixes; the <random> distributions are implementation-defined, so the
// cases would differ between standard libraries.
class Rng {
 public:
  explicit Rng(std::uint32_t seed) : g_(seed) {}
  double uniform() { return static_cast<double>(g_()) / 4294967296.0; }
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
  int integer(int lo, int hi) {  // inclusive
    const auto span = static_cast<std::uint32_t>(hi - lo + 1);
    return lo + static_cast<int>(static_cast<std::uint32_t>(g_()) % span);
  }

 private:
  std::mt19937 g_;
};

Structure make_cell(double a0x, double a0y, double a0z, double a1x, double a1y,
                    double a1z, double a2x, double a2y, double a2z) {
  Structure st;
  st.cell.load(a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z);
  return st;
}

void add_cart(Structure &st, double x, double y, double z) {
  st.add_atom(
      Atom(PeriodicTable::find_by_symbol("Ta"), x, y, z, 0.0, 0.0, 0.0));
}

// r = f0 a0 + f1 a1 + f2 a2 (rows of the cell).
void add_frac(Structure &st, double f0, double f1, double f2) {
  double r[3];
  for (int c = 0; c < 3; ++c)
    r[c] = f0 * st.cell(0, c) + f1 * st.cell(1, c) + f2 * st.cell(2, c);
  add_cart(st, r[0], r[1], r[2]);
}

// ---------------------------------------------------------------- geometries

// Conventional 2-atom bcc cube.
Structure bcc2(double a) {
  Structure st = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  add_cart(st, 0.0, 0.0, 0.0);
  add_cart(st, 0.5 * a, 0.5 * a, 0.5 * a);
  return st;
}

// 60-degree hexagonal surface cell, a = 9.4 (in-plane height 8.14 < 10),
// 24 A along c with 45 atoms at seeded positions between z = 2 and 12
// (vacuum 14 A > 10).
Structure hex_slab45() {
  const double a = 9.4;
  Structure st = make_cell(a, 0.0, 0.0, 0.5 * a, 0.5 * std::sqrt(3.0) * a,
                           0.0, 0.0, 0.0, 24.0);
  Rng rng(9401);
  for (int n = 0; n < 45; ++n) {
    const double f0 = rng.uniform(), f1 = rng.uniform();
    const double z = rng.uniform(2.0, 12.0);
    add_frac(st, f0, f1, z / 24.0);
  }
  return st;
}

// Rows (10,0,0), (9,4.5,0), (0,0,12): every edge (10, 10.06, 12) is above
// 7 but the heights along a0 and a1 are 4.47 and 4.5; 12 seeded atoms.
Structure tilted12() {
  Structure st = make_cell(10.0, 0.0, 0.0, 9.0, 4.5, 0.0, 0.0, 0.0, 12.0);
  Rng rng(7012);
  for (int n = 0; n < 12; ++n) {
    const double f0 = rng.uniform(), f1 = rng.uniform(), f2 = rng.uniform();
    add_frac(st, f0, f1, f2);
  }
  return st;
}

// Rows (4,0,0), (12,4,0), (0,0,4): a simple cubic 4 A lattice described with
// a2 = 3 a0 + (0,4,0); atoms inside the given cell.
Structure sheared(int natoms) {
  Structure st = make_cell(4.0, 0.0, 0.0, 12.0, 4.0, 0.0, 0.0, 0.0, 4.0);
  add_frac(st, 0.1, 0.2, 0.3);
  if (natoms > 1) add_frac(st, 0.6, 0.73, 0.81);
  return st;
}

// Rows (1,0,0), (40,1,0), (0,0,3): an orthorhombic 1 x 1 x 3 lattice (reduced
// heights 1, 1, 3) whose given basis has the height 0.025 along a0.
Structure skewed() {
  Structure st = make_cell(1.0, 0.0, 0.0, 40.0, 1.0, 0.0, 0.0, 0.0, 3.0);
  add_frac(st, 0.1, 0.2, 0.3);
  add_frac(st, 0.55, 0.8, 0.65);
  add_frac(st, 0.9, 0.45, 0.05);
  add_cart(st, 0.37, 0.61, 1.9);
  return st;
}

// 12 A cube (heights above R = 5) with 12 atoms, most outside the cell:
// fractional (2.3, 0.5, -1.7), -0.003 in each axis, 1.003, ...
Structure unwrapped_cube12() {
  Structure st = make_cell(12.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 12.0);
  add_frac(st, 2.3, 0.5, -1.7);
  add_frac(st, -0.003, 0.41, 0.23);
  add_frac(st, 0.12, -0.003, 0.97);
  add_frac(st, 0.64, 0.33, 0.49);
  add_frac(st, 0.93, 0.88, -0.003);
  add_frac(st, 1.62, -2.21, 0.77);
  add_frac(st, 0.27, 0.71, 1.003);
  add_frac(st, 0.35, 1.08, -0.9);
  add_frac(st, -1.2, 0.95, 0.52);
  add_frac(st, 0.58, 0.47, 0.61);
  add_frac(st, 3.04, 0.21, 0.08);
  add_frac(st, 0.76, -0.02, 2.44);
  return st;
}

// Thin bcc a = 3.41 (R = 10) with the atoms moved by lattice vectors:
// r0 = 2 a0 - 3 a2, r1 = (a/2)(1,1,1) + a0 - 5 a1.
Structure unwrapped_bcc() {
  const double a = 3.41;
  Structure st = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  add_cart(st, 2.0 * a, 0.0, -3.0 * a);
  add_cart(st, 0.5 * a + a, 0.5 * a - 5.0 * a, 0.5 * a);
  return st;
}

// Thin orthorhombic cell (3.1, 3.5, 4) at R = 5 with atoms at the wrapping
// edges: f0 = -1e-17, f0 = 1.0, f1 = f2 = 1.0, f2 = -1e-17. (Not 3 x 4:
// |a0 + a2| = 5 would put self-images exactly at R.)
Structure frac_edges() {
  Structure st = make_cell(3.1, 0.0, 0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 4.0);
  add_cart(st, -3.1e-17, 1.2, 0.7);
  add_cart(st, 3.1, 0.4, 2.9);
  add_cart(st, 1.4, 3.5, 4.0);
  add_cart(st, 2.2, 1.9, -4e-17);
  return st;
}

// Thin 3 A cube (R = 5) with two atoms 1e-7 A apart.
Structure close_pair() {
  Structure st = make_cell(3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0);
  add_cart(st, 1.0, 1.1, 1.2);
  add_cart(st, 1.0 + 1e-7, 1.1, 1.2);
  return st;
}

// Thick 10 A cube (R = 4.5), three atoms inside.
Structure thick_cube() {
  Structure st = make_cell(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0);
  add_frac(st, 0.1, 0.2, 0.3);
  add_frac(st, 0.45, 0.62, 0.18);
  add_frac(st, 0.83, 0.07, 0.71);
  return st;
}

// ------------------------------------------------------------------ checks

std::string join(const std::vector<std::string> &lines) {
  std::ostringstream o;
  for (const auto &l : lines) o << l << "\n";
  return o.str();
}

// How a table built at R departs from the contract, for the loop cases.
struct Verdict {
  bool threw = false;             // NNFinder refused a valid input
  bool set_differs = false;       // pairs missing from, or extra in, the table
  bool boundary_differs = false;  // the geometry has unplanned pairs at R
  bool invariants_broken = false;
  std::string text;
  bool ok() const {
    return !threw && !set_differs && !boundary_differs && !invariants_broken;
  }
};

// Build st's list at R and compare it with the oracle, whose boundary window
// must hold exactly expected_boundary pairs (0 unless the geometry has pairs
// at R by construction), and with the table invariants.
Verdict judge(Structure &st, double R, std::size_t expected_boundary = 0) {
  Verdict v;
  std::ostringstream o;
  try {
    NNFinder().calc(st, R);
  } catch (const std::exception &e) {
    v.threw = true;
    v.text = std::string("NNFinder threw: ") + e.what() + "\n";
    return v;
  }
  const nn_oracle::Comparison c = nn_oracle::compare(st, R);
  if (!c.exact()) {
    v.set_differs = true;
    o << c.summary(4) << "\n";
  }
  if (c.boundary != expected_boundary) {
    v.boundary_differs = true;
    o << "boundary window holds " << c.boundary << " pairs, expected "
      << expected_boundary << "\n";
  }
  const std::vector<std::string> bad = nn_oracle::check_invariants(st, 4);
  if (!bad.empty()) {
    v.invariants_broken = true;
    o << "invariants:\n" << join(bad);
  }
  v.text = o.str();
  return v;
}

// Counts of the verdicts of a loop, with the first few failures described.
class Tally {
 public:
  void add(const Verdict &v, const std::string &label) {
    ++run_;
    if (v.ok()) return;
    ++bad_;
    threw_ += v.threw;
    set_ += v.set_differs;
    boundary_ += v.boundary_differs;
    invariants_ += v.invariants_broken;
    if (bad_ <= 4) report_ << label << ":\n" << v.text;
  }
  std::size_t run() const { return run_; }
  std::size_t bad() const { return bad_; }
  std::string summary() const {
    std::ostringstream o;
    o << bad_ << " of " << run_ << " cases fail: " << set_
      << " differ from the oracle, " << invariants_
      << " break an invariant, " << threw_ << " threw, " << boundary_
      << " have unplanned pairs at R\n"
      << report_.str();
    return o.str();
  }

 private:
  std::size_t run_ = 0, bad_ = 0, threw_ = 0, set_ = 0, boundary_ = 0,
              invariants_ = 0;
  std::ostringstream report_;
};

// Single-structure form with one assertion per property.
void require_exact(Structure &st, double R) {
  NNFinder().calc(st, R);
  const nn_oracle::Comparison c = nn_oracle::compare(st, R);
  INFO(c.summary());
  REQUIRE(c.exact());
  REQUIRE(c.boundary == 0);
  const std::vector<std::string> bad = nn_oracle::check_invariants(st);
  INFO(join(bad));
  REQUIRE(bad.empty());
}

// The same with non-fatal assertions, where later checks stay informative.
void check_exact(Structure &st, double R) {
  NNFinder().calc(st, R);
  const nn_oracle::Comparison c = nn_oracle::compare(st, R);
  INFO(c.summary());
  CHECK(c.exact());
  CHECK(c.boundary == 0);
  const std::vector<std::string> bad = nn_oracle::check_invariants(st);
  INFO(join(bad));
  CHECK(bad.empty());
}

// Pairs in exactly one of two key sets, described; empty if the sets are
// equal.
std::string keyset_difference(const nn_oracle::KeySet &a, const char *a_name,
                              const nn_oracle::KeySet &b, const char *b_name) {
  std::vector<nn_oracle::Key> only_a, only_b;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::back_inserter(only_a));
  std::set_difference(b.begin(), b.end(), a.begin(), a.end(),
                      std::back_inserter(only_b));
  if (only_a.empty() && only_b.empty()) return "";
  std::ostringstream o;
  o << a_name << " " << a.size() << " pairs, " << b_name << " " << b.size()
    << ": " << only_a.size() << " only in " << a_name << ", " << only_b.size()
    << " only in " << b_name;
  for (std::size_t n = 0; n < only_a.size() && n < 3; ++n)
    o << "\n  only in " << a_name << " " << nn_oracle::key_str(only_a[n]);
  for (std::size_t n = 0; n < only_b.size() && n < 3; ++n)
    o << "\n  only in " << b_name << " " << nn_oracle::key_str(only_b[n]);
  return o.str();
}

using V3 = std::array<double, 3>;

// Neighbour vectors nn_pos(i, k) - r_i of atom i.
std::vector<V3> image_vectors(const Structure &st, std::size_t i) {
  std::vector<V3> out;
  const Vec3d &r = st.atoms[i].position;
  for (std::size_t k = 0; k < st.nn_size(i); ++k) {
    const Vec3d p = st.nn_pos(i, k);
    out.push_back({p[0] - r[0], p[1] - r[1], p[2] - r[2]});
  }
  return out;
}

// Multiset equality of two vector lists, component tolerance tol (distinct
// neighbour vectors of these cells are far more than tol apart, so a greedy
// match is exact). Returns a description of the difference, empty if equal.
std::string multiset_difference(const std::vector<V3> &a,
                                const std::vector<V3> &b, double tol) {
  std::ostringstream o;
  if (a.size() != b.size())
    o << "sizes " << a.size() << " and " << b.size() << "; ";
  std::vector<bool> used(b.size(), false);
  std::size_t unmatched = 0;
  for (const V3 &x : a) {
    bool found = false;
    for (std::size_t m = 0; m < b.size() && !found; ++m) {
      if (used[m]) continue;
      if (std::fabs(x[0] - b[m][0]) <= tol && std::fabs(x[1] - b[m][1]) <= tol &&
          std::fabs(x[2] - b[m][2]) <= tol) {
        used[m] = true;
        found = true;
      }
    }
    if (!found && unmatched++ < 3)
      o << "unmatched (" << x[0] << ", " << x[1] << ", " << x[2] << "); ";
  }
  const std::size_t left =
      static_cast<std::size_t>(std::count(used.begin(), used.end(), false));
  if (unmatched || left)
    o << unmatched << " vectors of the first list and " << left
      << " of the second have no partner";
  return o.str();
}

// A copy with every atom moved by an integer lattice vector into the home
// cell (floor of its fractional coordinates, in long double).
Structure wrapped_copy(const Structure &st) {
  Structure w = st;
  w.nn.reset();
  const nn_oracle::Lattice L = nn_oracle::lattice(st);
  for (std::size_t i = 0; i < st.natoms(); ++i) {
    const std::array<long double, 3> f = nn_oracle::frac(L, st, i);
    for (int c = 0; c < 3; ++c) {
      long double x = st.atoms[i].position[c];
      for (int k = 0; k < 3; ++k) x -= std::floor(f[k]) * L.a[k][c];
      w.atoms[i].position[c] = static_cast<double>(x);
    }
  }
  return w;
}

std::vector<V3> positions(const Structure &st) {
  std::vector<V3> out;
  for (const auto &at : st.atoms)
    out.push_back({at.position[0], at.position[1], at.position[2]});
  return out;
}

// Image cells prod_k (2 (floor(R / h_k) + 1) + 1) for the heights h.
double image_cells(const double h[3], double R) {
  double p = 1.0;
  for (int k = 0; k < 3; ++k) p *= 2.0 * (std::floor(R / h[k]) + 1.0) + 1.0;
  return p;
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

bool contains(const std::string &s, const std::string &part) {
  return s.find(part) != std::string::npos;
}

}  // namespace

// ------------------------------------------------------------------ cases

TEST_CASE("NNFinder exact: 2-atom bcc a=3.41 at R=10 has 180 neighbours per "
          "atom and zero neighbour-vector sums",
          "[nn][exact][bcc]") {
  PeriodicTable::initialize();
  // The corner -> body-centre vector (-2.5, 0.5, 0.5) a (8.86 A) needs the
  // image shift -3 along a0, its mirror (2.5, 0.5, 0.5) a only +2. A list
  // missing one side gives a net force on a perfect lattice.
  Structure st = bcc2(3.41);
  require_exact(st, 10.0);
  for (std::size_t i = 0; i < st.natoms(); ++i) {
    INFO("atom " << i);
    CHECK(st.nn_size(i) == 180);
    double sum[3] = {0.0, 0.0, 0.0};
    for (const V3 &v : image_vectors(st, i))
      for (int c = 0; c < 3; ++c) sum[c] += v[c];
    INFO("sum of neighbour vectors (" << sum[0] << ", " << sum[1] << ", "
                                      << sum[2] << ")");
    CHECK(std::fabs(sum[0]) <= 1e-9);
    CHECK(std::fabs(sum[1]) <= 1e-9);
    CHECK(std::fabs(sum[2]) <= 1e-9);
  }
}

TEST_CASE("NNFinder exact: bcc sweep of a from 2.50 to 4.00 and across "
          "integer and half-integer R/a at R=10",
          "[nn][exact][sweep]") {
  PeriodicTable::initialize();
  const double R = 10.0;
  std::vector<double> as;
  for (int k = 250; k <= 400; ++k) as.push_back(k / 100.0);
  for (double x : {2.5, 3.0, 3.5, 4.0})
    for (double d : {-1e-2, -1e-3, -1e-5, -1e-7, 1e-7, 1e-5, 1e-3, 1e-2})
      as.push_back(R / x * (1.0 + d));
  REQUIRE(as.size() == 151 + 32);

  Tally tally;
  for (double a : as) {
    Structure st = bcc2(a);
    // a = 2.50: R = 4a, so the six images (+-4, 0, 0), ... of each atom lie
    // at the cutoff itself; no other lattice constant here has such a pair.
    const std::size_t ties = (a == 2.5) ? 12 : 0;
    std::ostringstream label;
    label << "a = " << a;
    tally.add(judge(st, R, ties), label.str());
  }
  INFO(tally.summary());
  REQUIRE(tally.run() == as.size());
  REQUIRE(tally.bad() == 0);
}

TEST_CASE("NNFinder exact: 60-degree hexagonal slab a=9.4 with 45 atoms at "
          "R=10",
          "[nn][exact][hex]") {
  PeriodicTable::initialize();
  Structure st = hex_slab45();
  REQUIRE(st.natoms() == 45);
  require_exact(st, 10.0);
}

TEST_CASE("NNFinder exact: tilted cell with every edge above R and a height "
          "below R",
          "[nn][exact][tilted]") {
  PeriodicTable::initialize();
  Structure st = tilted12();
  CHECK_FALSE(NNFinder::check_box(st, 7.0));
  require_exact(st, 7.0);
}

TEST_CASE("NNFinder exact: sheared cell with rows 4 0 0 / 12 4 0 / 0 0 4 at "
          "R=10",
          "[nn][exact][sheared]") {
  PeriodicTable::initialize();
  SECTION("one atom") {
    Structure st = sheared(1);
    require_exact(st, 10.0);
  }
  SECTION("two atoms") {
    Structure st = sheared(2);
    require_exact(st, 10.0);
  }
}

TEST_CASE("NNFinder exact: a skewed description of a benign lattice builds "
          "and is exact",
          "[nn][exact][reduction]") {
  PeriodicTable::initialize();
  // Given-basis height along a0 is 0.025 A; the lattice is 1 x 1 x 3, so at
  // R = 3.5 the reduced search needs (2*4+1)^2 (2*2+1) = 405 image cells.
  Structure st = skewed();
  REQUIRE_NOTHROW(NNFinder().calc(st, 3.5));
  require_exact(st, 3.5);
}

TEST_CASE("NNFinder exact: unwrapped positions give the neighbours of the "
          "wrapped copy",
          "[nn][exact][unwrapped]") {
  PeriodicTable::initialize();
  Structure st;
  double R = 0.0;
  SECTION("12 A cube at R=5 with atoms outside the cell") {
    st = unwrapped_cube12();
    R = 5.0;
  }
  SECTION("thin bcc a=3.41 at R=10 moved by lattice vectors") {
    st = unwrapped_bcc();
    R = 10.0;
  }
  SECTION("thin cell at R=5 with fractional coordinates -1e-17 and 1.0") {
    st = frac_edges();
    R = 5.0;
  }
  REQUIRE(R > 0.0);
  const std::vector<V3> before = positions(st);
  {
    INFO("as given");
    check_exact(st, R);
  }
  // The search never moves an atom.
  const std::vector<V3> after = positions(st);
  for (std::size_t i = 0; i < before.size(); ++i)
    for (int c = 0; c < 3; ++c) {
      INFO("atom " << i << " component " << c);
      REQUIRE(after[i][c] == before[i][c]);
    }
  // The same periodic system with every atom moved into the home cell has,
  // atom by atom, the same neighbour vectors (a multiset: matched to 1e-9 A
  // per component; distinct vectors here are much further apart).
  Structure w = wrapped_copy(st);
  {
    INFO("wrapped copy");
    check_exact(w, R);
  }
  for (std::size_t i = 0; i < st.natoms(); ++i) {
    const std::string d =
        multiset_difference(image_vectors(st, i), image_vectors(w, i), 1e-9);
    INFO("atom " << i << ", as given vs wrapped: " << d);
    CHECK(d.empty());
  }
}

TEST_CASE("NNFinder exact: table invariants on every test cell and on two "
          "atoms 1e-7 A apart",
          "[nn][invariants]") {
  PeriodicTable::initialize();
  struct Case {
    const char *name;
    std::function<Structure()> make;
    double R;
  };
  const std::vector<Case> cases = {
      {"bcc a=3.41", [] { return bcc2(3.41); }, 10.0},
      {"bcc a=2.60", [] { return bcc2(2.60); }, 10.0},
      {"hexagonal slab", hex_slab45, 10.0},
      {"tilted cell", tilted12, 7.0},
      {"sheared cell 1 atom", [] { return sheared(1); }, 10.0},
      {"sheared cell 2 atoms", [] { return sheared(2); }, 10.0},
      {"skewed description", skewed, 3.5},
      {"unwrapped 12 A cube", unwrapped_cube12, 5.0},
      {"unwrapped thin bcc", unwrapped_bcc, 10.0},
      {"fractional -1e-17 and 1.0", frac_edges, 5.0},
      {"thick 10 A cube", thick_cube, 4.5},
      {"two atoms 1e-7 A apart", close_pair, 5.0},
  };
  std::size_t bad = 0;
  std::ostringstream report;
  for (const Case &c : cases) {
    Structure st = c.make();
    NNFinder().calc(st, c.R);
    const std::vector<std::string> v = nn_oracle::check_invariants(st, 3);
    if (!v.empty()) {
      ++bad;
      report << c.name << ":\n" << join(v);
    }
  }
  {
    INFO(bad << " of " << cases.size() << " cells break an invariant\n"
             << report.str());
    CHECK(bad == 0);
  }

  // The 1e-7 A pair is two atoms, not one: each has its own images.
  Structure st = close_pair();
  require_exact(st, 5.0);
}

TEST_CASE("NNFinder exact: 200 seeded random triclinic cells with unwrapped "
          "atoms",
          "[nn][exact][random]") {
  PeriodicTable::initialize();
  Tally tally;
  for (std::uint32_t seed = 1; seed <= 200; ++seed) {
    Rng rng(seed);
    // Lattice vectors of length 1.5-6 A in random directions (either
    // handedness), orthogonality defect Q = |a0||a1||a2| / V <= 5.
    Structure st;
    double h[3] = {0.0, 0.0, 0.0};
    for (int attempt = 0;; ++attempt) {
      REQUIRE(attempt < 1000);
      double rows[3][3];
      for (int k = 0; k < 3; ++k) {
        double v[3], n2 = 0.0;
        do {
          n2 = 0.0;
          for (int c = 0; c < 3; ++c) {
            v[c] = rng.uniform(-1.0, 1.0);
            n2 += v[c] * v[c];
          }
        } while (n2 > 1.0 || n2 < 0.04);
        const double len = rng.uniform(1.5, 6.0) / std::sqrt(n2);
        for (int c = 0; c < 3; ++c) rows[k][c] = v[c] * len;
      }
      st = make_cell(rows[0][0], rows[0][1], rows[0][2], rows[1][0],
                     rows[1][1], rows[1][2], rows[2][0], rows[2][1],
                     rows[2][2]);
      const nn_oracle::Lattice L = nn_oracle::lattice(st);
      const long double Q = L.len[0] * L.len[1] * L.len[2] / L.volume;
      if (Q > 5.0L) continue;
      for (int k = 0; k < 3; ++k) h[k] = static_cast<double>(L.h[k]);
      break;
    }
    // 1-6 atoms at random fractional coordinates plus integer offsets in
    // [-3, 3] along every row.
    const int natoms = rng.integer(1, 6);
    for (int n = 0; n < natoms; ++n) {
      double f[3];
      for (int k = 0; k < 3; ++k) {
        const double w = rng.uniform();
        const int offset = rng.integer(-3, 3);
        f[k] = w + offset;
      }
      add_frac(st, f[0], f[1], f[2]);
    }
    // R in [0.3 h_min, 3 h_max], lowered until the given basis needs at most
    // 2000 image cells (the cap is 10^4).
    const double hmin = std::min({h[0], h[1], h[2]});
    const double hmax = std::max({h[0], h[1], h[2]});
    double R = rng.uniform(0.3 * hmin, 3.0 * hmax);
    while (image_cells(h, R) > 2000.0 && R > 0.3 * hmin)
      R = std::max(0.3 * hmin, 0.95 * R);
    REQUIRE(image_cells(h, R) <= 2000.0);

    std::ostringstream label;
    label.precision(17);
    label << "seed " << seed << ", " << st.natoms() << " atoms, R = " << R
          << ", rows";
    for (int k = 0; k < 3; ++k)
      label << " (" << st.cell(k, 0) << " " << st.cell(k, 1) << " "
            << st.cell(k, 2) << ")";
    tally.add(judge(st, R), label.str());
  }
  INFO(tally.summary());
  REQUIRE(tally.run() == 200);
  REQUIRE(tally.bad() == 0);
}

TEST_CASE("NNFinder exact: a larger-cutoff list filtered to rc equals the rc "
          "list and the oracle",
          "[nn][filter]") {
  PeriodicTable::initialize();
  // HPO builds lists once at the largest cutoff and lets DescriptorsCalc
  // filter them at each candidate cutoff; that is only right if the filtered
  // list is the complete set at rc.
  struct Case {
    const char *name;
    std::function<Structure()> make;
    double rc;
  };
  const std::vector<Case> cases = {
      {"thin bcc a=3.41", [] { return bcc2(3.41); }, 10.0},
      {"tilted cell", tilted12, 7.0},
      {"sheared cell 2 atoms", [] { return sheared(2); }, 10.0},
      {"unwrapped 12 A cube", unwrapped_cube12, 5.0},
      {"unwrapped thin bcc", unwrapped_bcc, 10.0},
  };
  for (const Case &c : cases) {
    const Structure proto = c.make();
    const nn_oracle::Reference ref = nn_oracle::reference(proto, c.rc);
    Structure st_small = proto;
    NNFinder().calc(st_small, c.rc);
    const nn_oracle::KeySet small = nn_oracle::consumer_filter(st_small, c.rc);
    for (double R_large : {c.rc + 0.5, c.rc + 2.0, 2.0 * c.rc}) {
      INFO(c.name << ": rc = " << c.rc << ", list built at " << R_large);
      REQUIRE(ref.boundary.empty());
      Structure st_large = proto;
      NNFinder().calc(st_large, R_large);
      const nn_oracle::KeySet large =
          nn_oracle::consumer_filter(st_large, c.rc);
      const std::string large_vs_oracle =
          keyset_difference(large, "filtered list", ref.inside, "oracle");
      const std::string small_vs_oracle =
          keyset_difference(small, "rc list", ref.inside, "oracle");
      const std::string large_vs_small =
          keyset_difference(large, "filtered list", small, "rc list");
      {
        INFO(large_vs_oracle);
        CHECK(large_vs_oracle.empty());
      }
      {
        INFO(small_vs_oracle);
        CHECK(small_vs_oracle.empty());
      }
      {
        INFO(large_vs_small);
        CHECK(large_vs_small.empty());
      }
    }
  }
}

TEST_CASE("NNFinder exact: pairs at exactly the cutoff in simple cubic a=2.5 "
          "at R=5",
          "[nn][boundary]") {
  PeriodicTable::initialize();
  // Every coordinate is a multiple of 2.5, so the six images at 2a = R have
  // d*d == 25 exactly: the table may hold them (guard band), a consumer
  // testing d*d < R*R never uses them.
  Structure one = make_cell(2.5, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 2.5);
  add_cart(one, 0.0, 0.0, 0.0);
  Structure eight = nn_oracle::supercell(one, 2, 2, 2);
  REQUIRE(eight.natoms() == 8);
  for (Structure *st : {&one, &eight}) {
    INFO(st->natoms() << " atoms");
    NNFinder().calc(*st, 5.0);
    const nn_oracle::Comparison c = nn_oracle::compare(*st, 5.0);
    INFO(c.summary());
    CHECK(c.missing.empty());
    CHECK(c.extra.empty());
    CHECK(c.inside_size == 26 * st->natoms());
    CHECK(c.boundary == 6 * st->natoms());
    const std::vector<std::string> bad = nn_oracle::check_invariants(*st);
    INFO(join(bad));
    CHECK(bad.empty());
    const nn_oracle::KeySet used = nn_oracle::consumer_filter(*st, 5.0);
    CHECK(used.size() == 26 * st->natoms());
    for (std::size_t i = 0; i < st->natoms(); ++i) {
      INFO("atom " << i);
      CHECK(st->nn_size(i) >= 26);
      CHECK(st->nn_size(i) <= 32);
      std::size_t per_atom = 0;
      for (const auto &k : used) per_atom += std::get<0>(k) == i;
      CHECK(per_atom == 26);
    }
  }
}

TEST_CASE("NNFinder::check_box compares the reduced cell heights with the "
          "cutoff",
          "[nn][check_box]") {
  PeriodicTable::initialize();
  SECTION("tilted cell: edges 10 10.06 12 above R=7 but heights 4.47 4.5 12") {
    const Structure st = tilted12();
    CHECK_FALSE(NNFinder::check_box(st, 7.0));
    CHECK(NNFinder::check_box(st, 4.4));
  }
  SECTION("thin bcc and the hexagonal slab are thinner than R=10") {
    CHECK_FALSE(NNFinder::check_box(bcc2(3.41), 10.0));
    CHECK_FALSE(NNFinder::check_box(hex_slab45(), 10.0));
    CHECK(NNFinder::check_box(hex_slab45(), 8.0));
  }
  SECTION("orthogonal cells: true iff every edge is at least R") {
    const Structure cube =
        make_cell(5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0);
    CHECK(NNFinder::check_box(cube, 4.9));
    CHECK_FALSE(NNFinder::check_box(cube, 5.1));
    const Structure box =
        make_cell(4.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 9.0);
    CHECK(NNFinder::check_box(box, 3.9));
    CHECK_FALSE(NNFinder::check_box(box, 4.1));
    CHECK_FALSE(NNFinder::check_box(box, 8.9));
  }
  SECTION("a skewed description of a 5 A cube is judged by the cube") {
    // Rows (5,0,0), (20,5,0), (0,0,5): given-basis height 1.21 along a0.
    const Structure st =
        make_cell(5.0, 0.0, 0.0, 20.0, 5.0, 0.0, 0.0, 0.0, 5.0);
    CHECK(NNFinder::check_box(st, 4.5));
    CHECK_FALSE(NNFinder::check_box(st, 5.1));
  }
}

TEST_CASE("NNFinder exact: the image cap refuses R/a=10.5 and builds R/a=9.5",
          "[nn][shifts_cap]") {
  PeriodicTable::initialize();
  const double R = 5.0;
  SECTION("cube a = R/10.5 needs 23^3 image cells") {
    const double a = R / 10.5;
    Structure st = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    add_cart(st, 0.1 * a, 0.2 * a, 0.3 * a);
    CHECK_THROWS_AS(NNFinder().calc(st, R), std::runtime_error);
    std::string msg;
    try {
      NNFinder().calc(st, R);
    } catch (const std::runtime_error &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(contains(msg, "image cells"));
    // The refusal gives its example without "e.g." (before the fix: "Such
    // a cell is degenerate, e.g. the result of ...").
    const std::string tail = "). Such a cell is degenerate, such as the "
                             "result of an unbounded relaxation step";
    CHECK(msg.size() > tail.size());
    CHECK(msg.compare(msg.size() - tail.size(), tail.size(), tail) == 0);
  }
  SECTION("cube a = R/9.5 needs 21^3 image cells and is exact") {
    const double a = R / 9.5;
    Structure st = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    add_cart(st, 0.1 * a, 0.2 * a, 0.3 * a);
    add_cart(st, 0.6 * a, 0.45 * a, 0.85 * a);
    REQUIRE_NOTHROW(NNFinder().calc(st, R));
    require_exact(st, R);
  }
}

TEST_CASE("NNFinder exact: an invalid cutoff or position is refused",
          "[nn][invalid]") {
  PeriodicTable::initialize();
  Structure st = thick_cube();
  SECTION("cutoff 0") {
    CHECK_THROWS_AS(NNFinder().calc(st, 0.0), std::invalid_argument);
  }
  SECTION("negative cutoff") {
    CHECK_THROWS_AS(NNFinder().calc(st, -1.0), std::invalid_argument);
  }
  SECTION("NaN cutoff") {
    CHECK_THROWS_AS(NNFinder().calc(st, kNaN), std::invalid_argument);
  }
  SECTION("infinite cutoff") {
    CHECK_THROWS_AS(NNFinder().calc(st, kInf), std::invalid_argument);
  }
  SECTION("NaN cutoff through build on a const structure") {
    const Structure &cst = st;
    CHECK_THROWS_AS(NNFinder().build(cst, kNaN), std::invalid_argument);
  }
  SECTION("NaN position") {
    st.atoms[1].position[1] = kNaN;
    CHECK_THROWS_AS(NNFinder().calc(st, 4.5), std::invalid_argument);
  }
  SECTION("infinite position") {
    st.atoms[2].position[0] = -kInf;
    CHECK_THROWS_AS(NNFinder().calc(st, 4.5), std::invalid_argument);
  }
  SECTION("an atom more than 2^31 cells from the origin") {
    // 3e9 cells of 10 A along a0: a double keeps fewer than 22 bits of the
    // position within its cell there.
    st.atoms[1].position[0] = 3.0e10;
    CHECK_THROWS_AS(NNFinder().calc(st, 4.5), std::invalid_argument);
  }
}

TEST_CASE("NNFinder exact: a singular or non-finite cell is refused",
          "[nn][invalid][cell]") {
  PeriodicTable::initialize();
  Structure st;
  std::string what;
  SECTION("two equal rows") {
    what = "rows (4 0 0) (4 0 0) (0 0 4)";
    st = make_cell(4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0);
  }
  SECTION("a zero row") {
    what = "rows (4 0 0) (0 0 0) (0 0 4)";
    st = make_cell(4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0);
  }
  SECTION("a NaN entry") {
    what = "rows (4 0 0) (0 4 NaN) (0 0 4)";
    st = make_cell(4.0, 0.0, 0.0, 0.0, 4.0, kNaN, 0.0, 0.0, 4.0);
  }
  SECTION("an infinite entry") {
    what = "rows (4 0 0) (0 4 0) (inf 0 4)";
    st = make_cell(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, kInf, 0.0, 4.0);
  }
  SECTION("a numerically singular cell") {
    // Rows 4 A long whose second row leans 1e-9 A off the first: the
    // rounding of the cell's geometry is not bounded any more.
    what = "rows (4 0 0) (4 1e-9 0) (0 0 4)";
    st = make_cell(4.0, 0.0, 0.0, 4.0, 1e-9, 0.0, 0.0, 0.0, 4.0);
  }
  INFO(what);
  add_cart(st, 0.5, 0.7, 0.9);
  add_cart(st, 2.1, 0.3, 1.7);
  CHECK_THROWS_AS(NNFinder().calc(st, 3.0), std::invalid_argument);
  // check_box refuses the same cells instead of judging them.
  CHECK_THROWS_AS(NNFinder::check_box(st, 3.0), std::invalid_argument);
}

TEST_CASE("NNFinder exact: calc on a StructureDB names the failing structure",
          "[nn][stdb]") {
  PeriodicTable::initialize();
  StructureDB db;
  Structure s1 = bcc2(3.3);
  s1.label = "first bcc";
  Structure s2 = bcc2(3.3);
  s2.label = "second bcc with a NaN";
  s2.atoms[1].position[0] = kNaN;
  Structure s3 = thick_cube();
  s3.label = "third cube";
  db.add(s1);
  db.add(s2);
  db.add(s3);

  CHECK_THROWS_AS(NNFinder().calc(db, 5.0), std::invalid_argument);
  std::string msg;
  try {
    NNFinder().calc(db, 5.0);
  } catch (const std::invalid_argument &e) {
    msg = e.what();
  }
  INFO("message: " << msg);
  CHECK(contains(msg, "structure 2 of 3"));
  CHECK(contains(msg, "'second bcc with a NaN'"));
}

TEST_CASE("NNFinder exact: calc on a StructureDB rethrows the lowest failure "
          "in its own category",
          "[nn][stdb]") {
  PeriodicTable::initialize();
  const double R = 5.0;
  // A cube needing 23^3 image cells (std::runtime_error) and a NaN position
  // (std::invalid_argument).
  const double a = R / 10.5;
  Structure capped = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
  add_cart(capped, 0.1 * a, 0.2 * a, 0.3 * a);
  capped.label = "tiny cube";
  Structure nan_atom = thick_cube();
  nan_atom.atoms[0].position[2] = kNaN;
  nan_atom.label = "cube with a NaN";
  Structure fine = bcc2(3.3);
  fine.label = "fine bcc";

  SECTION("image cap in structure 2, NaN in structure 3") {
    StructureDB db;
    db.add(fine);
    db.add(capped);
    db.add(nan_atom);
    std::string msg, other;
    try {
      NNFinder().calc(db, R);
    } catch (const std::runtime_error &e) {
      msg = e.what();
    } catch (const std::exception &e) {
      other = e.what();
    }
    INFO("runtime_error: " << msg << "; other exception: " << other);
    CHECK(other.empty());
    CHECK(contains(msg, "structure 2 of 3"));
    CHECK(contains(msg, "'tiny cube'"));
  }
  SECTION("NaN in structure 1, image cap in structure 2") {
    StructureDB db;
    db.add(nan_atom);
    db.add(capped);
    db.add(fine);
    std::string msg, other;
    try {
      NNFinder().calc(db, R);
    } catch (const std::invalid_argument &e) {
      msg = e.what();
    } catch (const std::exception &e) {
      other = e.what();
    }
    INFO("invalid_argument: " << msg << "; other exception: " << other);
    CHECK(other.empty());
    CHECK(contains(msg, "structure 1 of 3"));
    CHECK(contains(msg, "'cube with a NaN'"));
  }
}

// The guard band: a consumer computes a pair distance as
// r_i - (r_j + s.A) with its own rounding, in both directions (i -> j and
// j -> i), while NNFinder tests each pair once. A pair a consumer computes
// strictly inside the cutoff must be listed even when the cutoff lies
// within a few units in the last place of that distance, and every pair
// whose exact (long double) distance is below the cutoff must be listed.
// The cutoffs here are set to consumer-computed pair distances and their
// neighbours 1-3 ulps away, on random triclinic cells with skewed stored
// bases and atoms up to 50 cells from the origin.
TEST_CASE("NNFinder exact: every pair a consumer computes inside a cutoff a "
          "few ulps from a pair distance is listed",
          "[nn][exact][guard]") {
  PeriodicTable::initialize();
  std::size_t cutoffs = 0, pairs_checked = 0;
  for (std::uint32_t seed = 1; seed <= 40; ++seed) {
    Rng rng(9000 + seed);
    // A near-orthogonal cell with 2.5-5 A edges, then a skewed description
    // of it (rows mixed with small integer multiples of each other).
    double A[3][3];
    for (int k = 0; k < 3; ++k)
      for (int c = 0; c < 3; ++c)
        A[k][c] = (k == c ? rng.uniform(2.5, 5.0) : 0.0) + rng.uniform(-0.8, 0.8);
    const int m1 = rng.integer(-3, 3), m2 = rng.integer(-3, 3);
    for (int c = 0; c < 3; ++c) {
      A[1][c] += m1 * A[0][c];
      A[2][c] += m2 * A[1][c];
    }
    Structure st = make_cell(A[0][0], A[0][1], A[0][2], A[1][0], A[1][1],
                             A[1][2], A[2][0], A[2][1], A[2][2]);
    const double spread = (seed % 2) ? 50.0 : 0.0;
    for (int i = 0; i < 3; ++i)
      add_frac(st, rng.uniform() + std::floor(rng.uniform(-spread, spread)),
               rng.uniform(), rng.uniform() + std::floor(rng.uniform(-spread, spread)));
    const auto L = nn_oracle::lattice(st);
    const std::size_t n = st.natoms();
    // Consumer-computed squared distance of (i, j, s), as DescriptorsCalc
    // computes it from Structure::nn_pos (disp = s0 a0 + s1 a1 + s2 a2).
    auto consumer_rsq = [&](std::size_t i, std::size_t j, long s0, long s1,
                            long s2) {
      double rsq = 0.0;
      for (int c = 0; c < 3; ++c) {
        const double D = static_cast<double>(s0) * st.cell(0, c) +
                         static_cast<double>(s1) * st.cell(1, c) +
                         static_cast<double>(s2) * st.cell(2, c);
        const double pos = st.atoms[j].position[c] + D;
        const double del = st.atoms[i].position[c] - pos;
        rsq += del * del;
      }
      return rsq;
    };
    // Brute force over the images that can be within R_max.
    const double R_max = 6.5;
    struct Pair { std::size_t i, j; long s[3]; double rsq; long double d; };
    std::vector<Pair> all;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        const auto fi = nn_oracle::frac(L, st, i), fj = nn_oracle::frac(L, st, j);
        long c0[3], M[3];
        for (int k = 0; k < 3; ++k) {
          c0[k] = static_cast<long>(std::llround(fi[k] - fj[k]));
          M[k] = static_cast<long>(std::ceil((R_max + 1e-6) / L.h[k])) + 2;
        }
        for (long s0 = c0[0] - M[0]; s0 <= c0[0] + M[0]; ++s0)
          for (long s1 = c0[1] - M[1]; s1 <= c0[1] + M[1]; ++s1)
            for (long s2 = c0[2] - M[2]; s2 <= c0[2] + M[2]; ++s2) {
              if (i == j && s0 == 0 && s1 == 0 && s2 == 0) continue;
              const long double d =
                  nn_oracle::norm(nn_oracle::pair_vector(L, st, i, j, s0, s1, s2));
              if (d > R_max) continue;
              all.push_back({i, j, {s0, s1, s2}, consumer_rsq(i, j, s0, s1, s2), d});
            }
      }
    // Cutoffs: consumer-computed distances between 2.5 and 6 A, and 1-3
    // ulps either side.
    std::vector<double> dist;
    for (const auto &p : all) {
      const double d = std::sqrt(p.rsq);
      if (d > 2.5 && d < 6.0) dist.push_back(d);
    }
    std::sort(dist.begin(), dist.end());
    dist.erase(std::unique(dist.begin(), dist.end()), dist.end());
    const std::size_t stride = std::max<std::size_t>(1, dist.size() / 5);
    for (std::size_t t = 0; t < dist.size(); t += stride) {
      for (int k = -3; k <= 3; ++k) {
        double R = dist[t];
        for (int u = 0; u < std::abs(k); ++u)
          R = std::nextafter(R, k > 0 ? 100.0 : 0.0);
        NNFinder().calc(st, R);
        ++cutoffs;
        const nn_oracle::KeySet table = nn_oracle::table_set(st);
        std::size_t missing = 0, missing_exact = 0;
        for (const auto &p : all) {
          const nn_oracle::Key key(p.i, p.j, p.s[0], p.s[1], p.s[2]);
          const bool listed = table.count(key) > 0;
          if (p.rsq < R * R) {
            ++pairs_checked;
            if (!listed) ++missing;
          }
          if (p.d < static_cast<long double>(R) && !listed) ++missing_exact;
        }
        INFO("seed " << seed << ", R = " << R << " (" << k
                     << " ulp from a pair distance): " << missing
                     << " consumer-inside pairs and " << missing_exact
                     << " exactly-inside pairs are not listed");
        REQUIRE(missing == 0);
        REQUIRE(missing_exact == 0);
      }
    }
  }
  INFO(cutoffs << " cutoffs, " << pairs_checked << " consumer-inside pairs");
  CHECK(cutoffs >= 40 * 4 * 7);
  CHECK(pairs_checked > 0);
}

// The image cap is a limit on degenerate cells, not a rounding bound: the
// same periodic system is accepted or refused whatever the stored positions
// of its atoms. A cube just thicker than cutoff/10 builds with its atom in
// the cell or a million cells away; a cube of exactly cutoff/10 is refused
// either way.
TEST_CASE("NNFinder exact: the image cap does not depend on where the atoms "
          "are stored",
          "[nn][shifts_cap]") {
  PeriodicTable::initialize();
  const double R = 5.0;
  for (const double offset : {0.0, 1.0e3, 1.0e6}) {
    INFO("atom offset " << offset << " cells along a0");
    {
      const double a = 0.5 * (1.0 + 1e-12);
      Structure st = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
      add_frac(st, 0.1 + offset, 0.2, 0.3);
      REQUIRE_NOTHROW(NNFinder().calc(st, R));
      // The images ten edges away lie 5e-12 A beyond R, inside the oracle's
      // boundary window: either answer is right for them.
      const nn_oracle::Comparison c = nn_oracle::compare(st, R);
      INFO(c.summary());
      CHECK(c.exact());
      const std::vector<std::string> bad = nn_oracle::check_invariants(st);
      INFO(join(bad));
      CHECK(bad.empty());
    }
    {
      const double a = 0.5;  // R / a = 10 exactly: 23^3 image cells
      Structure st = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
      add_frac(st, 0.1 + offset, 0.2, 0.3);
      CHECK_THROWS_AS(NNFinder().calc(st, R), std::runtime_error);
    }
  }
}

// Lengths outside about 1e-60 to 1e60 A are refused rather than computed:
// below it the squares in the cell geometry underflow (a 1e-81 A cube gave
// a list missing most of its images), above it they overflow. Inside it,
// scale does not matter.
TEST_CASE("NNFinder exact: cell vectors outside the supported length range "
          "are refused",
          "[nn][invalid][cell][scale]") {
  PeriodicTable::initialize();
  auto cube = [](double a) {
    Structure st = make_cell(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    add_frac(st, 0.3, 0.4, 0.5);
    add_frac(st, 0.8, 0.15, 0.65);
    return st;
  };
  SECTION("1e-81 A cube") {
    Structure st = cube(1e-81);
    std::string msg;
    try {
      NNFinder().calc(st, 5e-81);
    } catch (const std::invalid_argument &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(contains(msg, "range"));
    CHECK_THROWS_AS(NNFinder::check_box(st, 1e-79), std::invalid_argument);
  }
  SECTION("1e78 A cube") {
    Structure st = cube(1e78);
    std::string msg;
    try {
      NNFinder().calc(st, 5e78);
    } catch (const std::invalid_argument &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(contains(msg, "range"));
  }
  SECTION("1e-50 A and 1e50 A cubes are exact") {
    for (const double a : {1e-50, 1e50}) {
      INFO("edge " << a);
      Structure st = cube(a);
      REQUIRE_NOTHROW(NNFinder().calc(st, 2.5 * a));
      require_exact(st, 2.5 * a);
    }
  }
}

// The search covers the rounding of the fractional coordinates, which grows
// with the distance of the atoms from the origin and with the conditioning
// of the cell. For an atom far out in a cell near the conditioning limit
// that bound exceeds an image cell: the stored coordinates then say nothing
// about which image is which, and the input is refused (instead of a
// search over an unbounded number of cells).
TEST_CASE("NNFinder exact: positions whose rounding bound exceeds an image "
          "cell are refused",
          "[nn][invalid][rounding]") {
  PeriodicTable::initialize();
  // Reduced heights 1, 1e-4, 1 A (a1 leans 1e-4 A off a0): conditioning
  // about 2e4.
  Structure st = make_cell(1.0, 0.0, 0.0, 1.0, 1e-4, 0.0, 0.0, 0.0, 1.0);
  add_cart(st, 0.25, 0.3e-4, 0.5);
  const double R = 5.5e-4;  // 6 images along the thin direction
  SECTION("an atom inside the cell builds exactly") {
    REQUIRE_NOTHROW(NNFinder().calc(st, R));
    require_exact(st, R);
  }
  SECTION("an atom 3e7 A out is refused") {
    add_cart(st, 3.0e7 + 0.5, 0.7e-4, 0.25);
    std::string msg;
    try {
      NNFinder().calc(st, R);
    } catch (const std::invalid_argument &e) {
      msg = e.what();
    }
    INFO("message: " << msg);
    CHECK(contains(msg, "rounding bound"));
  }
}
