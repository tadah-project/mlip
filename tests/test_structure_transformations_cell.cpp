// test_structure_transformations_cell.cpp
//
// StructureTransformation on a genuinely triclinic cell (every entry
// non-zero, cell != cell^T). Row i of Structure::cell is the lattice vector
// a_i and r = f0*a0 + f1*a1 + f2*a2, so a linear map M of the atoms
// (r' = M r) must map every row the same way (a_i' = M a_i, i.e.
// cell' = cell * M^T): only then do the fractional coordinates stay put and
// the lattice stay consistent with the atoms. The old code used the column
// formula cell' = M * cell for rotate/shear, never reflected the cell in
// mirrorPlane/mirrorHkl, and replicate scaled only the diagonal and shifted
// the copies along the columns. test_structure_transformations.cpp only uses
// the cell 2*I, where rows and columns coincide.
//
// Expectations are hand-built in the test: fractional coordinates from the
// reciprocal rows b_i = (a_j x a_k)/det, and neighbour distances from a
// brute-force image loop over the rows (R = 3.35 A is below every height of
// the cell, so |n_k| <= 2 covers every image of atoms inside the cell; the
// nearest distances are 3.244 A inside and 3.428 A outside R).

#include "catch2/catch.hpp"

#include <tadah/core/core_types.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_transformations.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using tadah::core::Element;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::core::Vec3d;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureTransformation;

constexpr double kR = 3.35;  // brute-force neighbour radius (A)

Vec3d row_of(const Matrix3d &c, int i) {
  return Vec3d(c(i, 0), c(i, 1), c(i, 2));
}

// M * v, written out.
Vec3d mat_vec(const Matrix3d &m, const Vec3d &v) {
  return Vec3d(m(0, 0) * v[0] + m(0, 1) * v[1] + m(0, 2) * v[2],
               m(1, 0) * v[0] + m(1, 1) * v[1] + m(1, 2) * v[2],
               m(2, 0) * v[0] + m(2, 1) * v[1] + m(2, 2) * v[2]);
}

double hand_det(const Matrix3d &c) {
  return row_of(c, 0) * row_of(c, 1).cross(row_of(c, 2));
}

// r = f0*a0 + f1*a1 + f2*a2.
Vec3d cart(const Matrix3d &c, double f0, double f1, double f2) {
  return f0 * row_of(c, 0) + f1 * row_of(c, 1) + f2 * row_of(c, 2);
}

// f_i = b_i . r with b_i = (a_j x a_k) / det (a_i . b_j = delta_ij, any
// handedness).
Vec3d frac_of(const Matrix3d &c, const Vec3d &r) {
  const Vec3d a0 = row_of(c, 0), a1 = row_of(c, 1), a2 = row_of(c, 2);
  const double det = a0 * a1.cross(a2);
  return Vec3d((a1.cross(a2) * r) / det, (a2.cross(a0) * r) / det,
               (a0.cross(a1) * r) / det);
}

// Rows a0 = (4.2, 0.3, -0.2), a1 = (1.3, 3.9, 0.4), a2 = (-0.9, 1.1, 4.6):
// det = 70.61, heights 3.667, 3.563, 4.377 A.
Matrix3d triclinic_cell() {
  Matrix3d c;
  c.load(4.2, 0.3, -0.2,
         1.3, 3.9, 0.4,
         -0.9, 1.1, 4.6);
  return c;
}
constexpr double kV0 = 70.61;

const double kFrac[3][3] = {{0.10, 0.20, 0.30},
                            {0.55, 0.35, 0.80},
                            {0.80, 0.75, 0.15}};

// H, He, Li at kFrac, with distinct forces, a stress and an energy.
Structure triclinic_structure() {
  Structure st;
  st.label = "triclinic";
  st.cell = triclinic_cell();
  st.energy = -3.5;
  st.stress.load(0.11, 0.02, -0.03,
                 0.02, 0.07, 0.05,
                 -0.03, 0.05, -0.04);
  const char *sym[3] = {"H", "He", "Li"};
  for (int n = 0; n < 3; ++n) {
    const Vec3d r = cart(st.cell, kFrac[n][0], kFrac[n][1], kFrac[n][2]);
    st.add_atom(Atom(Element(sym[n]), r[0], r[1], r[2],
                     0.1 * (n + 1), -0.2 * (n + 1), 0.3 * (n + 1)));
  }
  return st;
}

// Sorted distances from atom i to every image r_j + n0*a0 + n1*a1 + n2*a2
// (|n_k| <= 2, not the atom itself) closer than kR, using the rows.
std::vector<double> image_distances(const Structure &st, std::size_t i) {
  std::vector<double> d;
  const Vec3d a0 = row_of(st.cell, 0), a1 = row_of(st.cell, 1),
              a2 = row_of(st.cell, 2);
  for (std::size_t j = 0; j < st.natoms(); ++j)
    for (int n0 = -2; n0 <= 2; ++n0)
      for (int n1 = -2; n1 <= 2; ++n1)
        for (int n2 = -2; n2 <= 2; ++n2) {
          if (i == j && n0 == 0 && n1 == 0 && n2 == 0) continue;
          const Vec3d t = double(n0) * a0 + double(n1) * a1 + double(n2) * a2;
          const double r = (st(j).position + t - st(i).position).norm();
          if (r < kR) d.push_back(r);
        }
  std::sort(d.begin(), d.end());
  return d;
}

void check_same_distances(const std::vector<double> &got,
                          const std::vector<double> &want) {
  REQUIRE(got.size() == want.size());
  for (std::size_t k = 0; k < got.size(); ++k) {
    INFO("distance " << k);
    CHECK(got[k] == Approx(want[k]).margin(1e-12));
  }
}

// Every row of st.cell is M a_i of the triclinic cell, every atom keeps its
// fractional coordinates, and |det| scales by |det M|.
void check_rows_and_fractions(const Structure &st, const Matrix3d &m,
                              double det_m) {
  const Matrix3d c0 = triclinic_cell();
  for (int i = 0; i < 3; ++i) {
    const Vec3d want = mat_vec(m, row_of(c0, i));
    for (int k = 0; k < 3; ++k) {
      INFO("cell(" << i << "," << k << ")");
      CHECK(st.cell(i, k) == Approx(want[k]).margin(1e-12));
    }
  }
  REQUIRE(st.natoms() == 3);
  for (std::size_t n = 0; n < 3; ++n) {
    const Vec3d f = frac_of(st.cell, st(n).position);
    for (int k = 0; k < 3; ++k) {
      INFO("atom " << n << " fractional " << k);
      CHECK(f[k] == Approx(kFrac[n][k]).margin(1e-12));
    }
  }
  CHECK(hand_det(st.cell) == Approx(det_m * kV0));
  CHECK(st.get_volume() == Approx(std::fabs(det_m) * kV0));
}

// A rigid map M (rotation or reflection) moves the forces with the atoms:
// F' = M F.
void check_forces_mapped(const Structure &st, const Structure &before,
                         const Matrix3d &m) {
  REQUIRE(st.natoms() == before.natoms());
  for (std::size_t n = 0; n < st.natoms(); ++n) {
    const Vec3d want = mat_vec(m, before(n).force);
    for (int k = 0; k < 3; ++k) {
      INFO("atom " << n << " force " << k);
      CHECK(st(n).force[k] == Approx(want[k]).margin(1e-12));
    }
  }
}

}  // namespace

TEST_CASE("StructureTransformation::rotate on a triclinic cell rotates every "
          "lattice row with the atoms",
          "[transform][cell_convention]") {
  // Rotation by 0.7 rad about (1,2,2)/3 (Rodrigues' formula written here).
  // The old cell' = R*cell left the atoms at other fractional coordinates,
  // and the forces were not rotated.
  PeriodicTable::initialize();
  const Vec3d u(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0);
  const double th = 0.7, c = std::cos(th), s = std::sin(th);
  Matrix3d R;
  R.load(c + (1 - c) * u[0] * u[0], (1 - c) * u[0] * u[1] - s * u[2],
         (1 - c) * u[0] * u[2] + s * u[1],
         (1 - c) * u[1] * u[0] + s * u[2], c + (1 - c) * u[1] * u[1],
         (1 - c) * u[1] * u[2] - s * u[0],
         (1 - c) * u[2] * u[0] - s * u[1], (1 - c) * u[2] * u[1] + s * u[0],
         c + (1 - c) * u[2] * u[2]);

  Structure st = triclinic_structure();
  const Structure before = st;
  StructureTransformation::rotate(st, R);

  for (std::size_t n = 0; n < 3; ++n) {
    const Vec3d want = mat_vec(R, before(n).position);
    for (int k = 0; k < 3; ++k)
      CHECK(st(n).position[k] == Approx(want[k]).margin(1e-12));
  }
  check_rows_and_fractions(st, R, 1.0);
  check_forces_mapped(st, before, R);
  for (std::size_t n = 0; n < 3; ++n) {
    INFO("atom " << n);
    check_same_distances(image_distances(st, n), image_distances(before, n));
  }
}

TEST_CASE("StructureTransformation::shear on a triclinic cell maps every "
          "lattice row with the atoms",
          "[transform][cell_convention]") {
  // A general non-symmetric deformation, det = 1.04487. The atoms follow
  // r' = S r; the rows must follow a_i' = S a_i (cell' = cell * S^T).
  PeriodicTable::initialize();
  Matrix3d S;
  S.load(1.05, 0.30, 0.00,
         0.00, 0.97, 0.20,
         0.10, 0.00, 1.02);
  const double det_s = 1.05 * (0.97 * 1.02 - 0.20 * 0.00) -
                       0.30 * (0.00 * 1.02 - 0.20 * 0.10) +
                       0.00 * (0.00 * 0.00 - 0.97 * 0.10);

  Structure st = triclinic_structure();
  const Structure before = st;
  StructureTransformation::shear(st, S);

  for (std::size_t n = 0; n < 3; ++n) {
    const Vec3d want = mat_vec(S, before(n).position);
    for (int k = 0; k < 3; ++k)
      CHECK(st(n).position[k] == Approx(want[k]).margin(1e-12));
  }
  check_rows_and_fractions(st, S, det_s);
}

TEST_CASE("StructureTransformation::mirrorPlane on a triclinic cell reflects "
          "the lattice with the atoms",
          "[transform][cell_convention]") {
  // M = I - 2 n n^T for n = (1,-2,0.5)/|n|. The cell is reflected too
  // (cell' = cell * M^T): the result is left-handed (det = -70.61), a valid
  // basis whose volume is still 70.61. The old code moved only the atoms,
  // not the lattice or the forces.
  PeriodicTable::initialize();
  const Vec3d n_raw(1.0, -2.0, 0.5);
  const Vec3d n = n_raw / n_raw.norm();
  Matrix3d M;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) M(i, j) = (i == j ? 1.0 : 0.0) - 2.0 * n[i] * n[j];

  Structure st = triclinic_structure();
  const Structure before = st;
  StructureTransformation::mirrorPlane(st, n_raw);

  for (std::size_t a = 0; a < 3; ++a) {
    const Vec3d want = mat_vec(M, before(a).position);
    for (int k = 0; k < 3; ++k)
      CHECK(st(a).position[k] == Approx(want[k]).margin(1e-12));
  }
  check_rows_and_fractions(st, M, -1.0);
  check_forces_mapped(st, before, M);
  for (std::size_t a = 0; a < 3; ++a) {
    INFO("atom " << a);
    check_same_distances(image_distances(st, a), image_distances(before, a));
  }
}

TEST_CASE("StructureTransformation::mirrorHkl on a triclinic cell reflects "
          "the lattice with the atoms",
          "[transform][cell_convention]") {
  // Plane (1,-1,2): normal g = b0 - b1 + 2 b2 with the reciprocal rows
  // b_i = (a_j x a_k)/det; M = I - 2 g g^T/|g|^2, cell' = cell * M^T.
  PeriodicTable::initialize();
  const Matrix3d c0 = triclinic_cell();
  const Vec3d a0 = row_of(c0, 0), a1 = row_of(c0, 1), a2 = row_of(c0, 2);
  const double det = a0 * a1.cross(a2);
  const Vec3d g = (1.0 / det) * a1.cross(a2) - (1.0 / det) * a2.cross(a0) +
                  (2.0 / det) * a0.cross(a1);
  const Vec3d n = g / g.norm();
  Matrix3d M;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) M(i, j) = (i == j ? 1.0 : 0.0) - 2.0 * n[i] * n[j];

  Structure st = triclinic_structure();
  const Structure before = st;
  StructureTransformation::mirrorHkl(st, 1, -1, 2);

  for (std::size_t a = 0; a < 3; ++a) {
    const Vec3d want = mat_vec(M, before(a).position);
    for (int k = 0; k < 3; ++k)
      CHECK(st(a).position[k] == Approx(want[k]).margin(1e-12));
  }
  check_rows_and_fractions(st, M, -1.0);
  check_forces_mapped(st, before, M);
  for (std::size_t a = 0; a < 3; ++a) {
    INFO("atom " << a);
    check_same_distances(image_distances(st, a), image_distances(before, a));
  }
}

TEST_CASE("StructureTransformation::replicate on a triclinic cell builds the "
          "supercell along the lattice rows",
          "[transform][cell_convention]") {
  // replicate(st, 2, 3, 2): row i is scaled by its own factor, copy
  // (ix, iy, iz) of atom a sits at r_a + ix*a0 + iy*a1 + iz*a2 (loop order
  // ix, iy, iz, then atoms), the energy is extensive (x12), the stress
  // (intensive) and the forces are copied, and the stale neighbour list is
  // dropped. The old code scaled only the diagonal, shifted along the
  // columns, kept E and kept st.nn.
  PeriodicTable::initialize();
  const int nx = 2, ny = 3, nz = 2;
  Structure st = triclinic_structure();
  NNFinder().calc(st, 2.5);
  REQUIRE(st.nn != nullptr);
  const Structure before = st;

  StructureTransformation::replicate(st, nx, ny, nz);

  const Matrix3d c0 = triclinic_cell();
  const double fac[3] = {double(nx), double(ny), double(nz)};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      INFO("cell(" << i << "," << k << ")");
      CHECK(st.cell(i, k) == Approx(fac[i] * c0(i, k)).margin(1e-12));
    }
  CHECK(st.get_volume() == Approx(12.0 * kV0));
  CHECK(st.energy == Approx(12.0 * before.energy));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) CHECK(st.stress(i, j) == before.stress(i, j));
  CHECK(st.nn == nullptr);

  REQUIRE(st.natoms() == 12 * before.natoms());
  std::size_t idx = 0;
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      for (int iz = 0; iz < nz; ++iz)
        for (std::size_t a = 0; a < before.natoms(); ++a, ++idx) {
          INFO("copy (" << ix << "," << iy << "," << iz << ") of atom " << a);
          const Vec3d want = before(a).position + double(ix) * row_of(c0, 0) +
                             double(iy) * row_of(c0, 1) +
                             double(iz) * row_of(c0, 2);
          for (int k = 0; k < 3; ++k) {
            CHECK(st(idx).position[k] == Approx(want[k]).margin(1e-12));
            CHECK(st(idx).force[k] == before(a).force[k]);
          }
          CHECK(st(idx).symbol == before(a).symbol);
          // Each copy sees exactly the neighbours of its original atom.
          check_same_distances(image_distances(st, idx),
                               image_distances(before, a));
        }
}

TEST_CASE("StructureTransformation::replicate refuses a factor below 1",
          "[transform][cell_convention]") {
  // A factor < 1 used to be clamped to 1 silently; it is an error, reported
  // before the structure is touched.
  PeriodicTable::initialize();
  Structure st = triclinic_structure();
  const Structure before = st;
  CHECK_THROWS_AS(StructureTransformation::replicate(st, 0, 1, 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(StructureTransformation::replicate(st, 1, -2, 1),
                  std::invalid_argument);
  CHECK_THROWS_AS(StructureTransformation::replicate(st, 1, 1, 0),
                  std::invalid_argument);
  CHECK(st.natoms() == before.natoms());
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) CHECK(st.cell(i, k) == before.cell(i, k));
  CHECK(st.energy == before.energy);
}

TEST_CASE("StructureTransformation::mirrorOrigin inverts the forces with the "
          "positions",
          "[transform][cell_convention]") {
  // Inversion through the origin, M = -I: r' = -r and F' = -F (it negated
  // only the positions). The lattice needs no change (-a_i span the same
  // lattice), so the cell and the fractional coordinates' lattice stay.
  PeriodicTable::initialize();
  Structure st = triclinic_structure();
  const Structure before = st;
  StructureTransformation::mirrorOrigin(st);
  Matrix3d minus_I;
  minus_I.load(-1.0, 0.0, 0.0,
               0.0, -1.0, 0.0,
               0.0, 0.0, -1.0);
  for (std::size_t n = 0; n < 3; ++n)
    for (int k = 0; k < 3; ++k)
      CHECK(st(n).position[k] == Approx(-before(n).position[k]).margin(1e-12));
  check_forces_mapped(st, before, minus_I);
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) CHECK(st.cell(i, k) == before.cell(i, k));
}
