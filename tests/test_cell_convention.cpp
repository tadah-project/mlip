// Tests that codify the row-as-lattice-vector convention for Structure::cell:
// row(i) == a_i (Angstrom), r = cell^T f, f = (cell^T)^{-1} r, and the image
// with integer shift n is displaced by n0*a1 + n1*a2 + n2*a3 (this file names
// the rows a1, a2, a3). Every reader, writer, transformation and the
// neighbour search must follow it. These tests use sheared cells, where
// cell != cell^T, so a transposed cell is observable; on a diagonal (or any
// symmetric) cell rows == columns and a transposition is silent.

#include "catch2/catch.hpp"

#include "tadah/core/element.h"
#include "tadah/core/periodic_table.h"
#include "tadah/core/vec3d.h"

#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_properties.h>
#include <tadah/mlip/structure_transformations.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using tadah::core::Element;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::core::Vec3d;
using namespace tadah::mlip;

namespace {
// Sheared cell:  a1 = (2.5, 0, 0)
//                a2 = (0.5, 2, 0)
//                a3 = (0,   0, 3)
// det = 15.
Structure make_sheared_structure() {
  Structure st;
  st.cell.load(2.5, 0.0, 0.0,
               0.5, 2.0, 0.0,
               0.0, 0.0, 3.0);
  return st;
}

// n0*a1 + n1*a2 + n2*a3 from the rows, written out.
Vec3d lattice_vector(const Matrix3d &cell, const Vec3d &n) {
  return Vec3d(n[0] * cell(0, 0) + n[1] * cell(1, 0) + n[2] * cell(2, 0),
               n[0] * cell(0, 1) + n[1] * cell(1, 1) + n[2] * cell(2, 1),
               n[0] * cell(0, 2) + n[1] * cell(1, 2) + n[2] * cell(2, 2));
}

// Neighbour entry (j, n0, n1, n2); the stored shift is an integer triple.
using NbrKey = std::array<long, 4>;
NbrKey nbr_key(std::size_t j, const Vec3d &n) {
  return {static_cast<long>(j), std::lround(n[0]), std::lround(n[1]),
          std::lround(n[2])};
}

struct CellConvTempDir {
  std::filesystem::path dir;
  CellConvTempDir() {
    static int counter = 0;
    dir = std::filesystem::temp_directory_path() /
          ("tadah_cellconv_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    std::filesystem::create_directories(dir);
  }
  ~CellConvTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};
}  // namespace

TEST_CASE("Cell convention: PT init", "[cell_convention]") {
  PeriodicTable::initialize();
}

TEST_CASE("cell.row(i) returns lattice vector a_i", "[cell_convention]") {
  Structure st = make_sheared_structure();

  Vec3d a1 = st.cell.row(0);
  Vec3d a2 = st.cell.row(1);
  Vec3d a3 = st.cell.row(2);
  REQUIRE(a1[0] == Approx(2.5));
  REQUIRE(a1[1] == Approx(0.0));
  REQUIRE(a1[2] == Approx(0.0));
  REQUIRE(a2[0] == Approx(0.5));
  REQUIRE(a2[1] == Approx(2.0));
  REQUIRE(a2[2] == Approx(0.0));
  REQUIRE(a3[0] == Approx(0.0));
  REQUIRE(a3[1] == Approx(0.0));
  REQUIRE(a3[2] == Approx(3.0));
}

TEST_CASE("Volume is scalar triple product of row-vectors", "[cell_convention]") {
  Structure st = make_sheared_structure();
  REQUIRE(st.get_volume() == Approx(15.0));
  REQUIRE(StructureProperties::getVolume(st) == Approx(15.0));
}

TEST_CASE("Fractional -> Cartesian = cell^T * frac", "[cell_convention]") {
  Structure st = make_sheared_structure();

  // frac (1, 0, 0) -> a1
  Vec3d c1 = st.cell.transpose() * Vec3d(1.0, 0.0, 0.0);
  REQUIRE(c1[0] == Approx(2.5));
  REQUIRE(c1[1] == Approx(0.0));
  REQUIRE(c1[2] == Approx(0.0));

  // frac (0, 1, 0) -> a2
  Vec3d c2 = st.cell.transpose() * Vec3d(0.0, 1.0, 0.0);
  REQUIRE(c2[0] == Approx(0.5));
  REQUIRE(c2[1] == Approx(2.0));
  REQUIRE(c2[2] == Approx(0.0));

  // frac (0.5, 0.5, 0.5) -> 0.5*(a1+a2+a3) = (1.5, 1.0, 1.5)
  Vec3d c3 = st.cell.transpose() * Vec3d(0.5, 0.5, 0.5);
  REQUIRE(c3[0] == Approx(1.5));
  REQUIRE(c3[1] == Approx(1.0));
  REQUIRE(c3[2] == Approx(1.5));
}

TEST_CASE("Cartesian -> Fractional via (cell^T)^{-1}", "[cell_convention]") {
  Structure st = make_sheared_structure();
  Matrix3d invCellT = st.cell.transpose().inverse();

  // (1.5, 1.0, 1.5) should map back to (0.5, 0.5, 0.5)
  Vec3d frac = invCellT * Vec3d(1.5, 1.0, 1.5);
  REQUIRE(frac[0] == Approx(0.5));
  REQUIRE(frac[1] == Approx(0.5));
  REQUIRE(frac[2] == Approx(0.5));

  // round-trip an arbitrary Cartesian point
  Vec3d cart_in(2.7, 1.3, 2.4);
  Vec3d frac_back = invCellT * cart_in;
  Vec3d cart_back = st.cell.transpose() * frac_back;
  REQUIRE(cart_back[0] == Approx(cart_in[0]));
  REQUIRE(cart_back[1] == Approx(cart_in[1]));
  REQUIRE(cart_back[2] == Approx(cart_in[2]));
}

TEST_CASE("NNFinder on the sheared cell stores each image as the lattice "
          "vector of its shift", "[cell_convention][nn_finder]") {
  // One atom at the origin, cutoff 3.2: its images closer than 3.2 are the
  // 8 lattice vectors +-a1, +-a2, +-(a1 - a2), +-a3 (next: |a1 + a2| = 3.61).
  // nn_pos(0, k) - r_0 must be n0*a1 + n1*a2 + n2*a3 of the stored shift n
  // (NNFinder::populate_disp). Under a transposed cell the shift (1, 0, 0)
  // would give (2.5, 0.5, 0).
  Structure st = make_sheared_structure();
  st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  NNFinder().calc(st, 3.2);

  struct Image { int n0, n1, n2; double x, y, z; };
  const std::vector<Image> expected = {
      { 1,  0,  0,  2.5,  0.0,  0.0}, {-1,  0,  0, -2.5,  0.0,  0.0},
      { 0,  1,  0,  0.5,  2.0,  0.0}, { 0, -1,  0, -0.5, -2.0,  0.0},
      { 1, -1,  0,  2.0, -2.0,  0.0}, {-1,  1,  0, -2.0,  2.0,  0.0},
      { 0,  0,  1,  0.0,  0.0,  3.0}, { 0,  0, -1,  0.0,  0.0, -3.0}};
  REQUIRE(st.nn_size(0) == expected.size());
  for (const Image &e : expected) {
    INFO("shift (" << e.n0 << "," << e.n1 << "," << e.n2 << ")");
    std::size_t hits = 0;
    for (std::size_t k = 0; k < st.nn_size(0); ++k) {
      const Vec3d &n = st.nn->shift[0][k];
      if (std::lround(n[0]) != e.n0 || std::lround(n[1]) != e.n1 ||
          std::lround(n[2]) != e.n2)
        continue;
      ++hits;
      REQUIRE(st.nn->idx[0][k] == 0);
      const Vec3d d = st.nn_pos(0, k) - st(0).position;
      CHECK(d[0] == Approx(e.x).margin(1e-12));
      CHECK(d[1] == Approx(e.y).margin(1e-12));
      CHECK(d[2] == Approx(e.z).margin(1e-12));
    }
    CHECK(hits == 1);
  }
}

TEST_CASE("NNFinder on a large sheared cell: the neighbour list equals a "
          "brute-force image search and every image is its shift's lattice "
          "vector", "[cell_convention][nn_finder]") {
  // Rows a1 = (9,0,0), a2 = (3.5,8.5,0), a3 = (-2.5,2,9.5): no row equals
  // its column. Seven atoms inside the cell, cutoff 4.5 below every height
  // (7.92, 8.32, 9.50), so |n_k| <= 1 finds every image; 18 neighbour
  // entries with shifts along a1, a2, a3 and their sums, e.g. (1, 1, 1).
  // Pair distances: <= 4.19 inside the cutoff, >= 4.77 outside it.
  Structure st;
  st.cell.load(9.0, 0.0, 0.0,
               3.5, 8.5, 0.0,
               -2.5, 2.0, 9.5);
  const double f[7][3] = {{0.05, 0.08, 0.06}, {0.92, 0.15, 0.30},
                          {0.40, 0.93, 0.07}, {0.60, 0.50, 0.94},
                          {0.25, 0.70, 0.55}, {0.88, 0.90, 0.85},
                          {0.50, 0.10, 0.60}};
  for (const auto &fi : f) {
    const Vec3d r = lattice_vector(st.cell, Vec3d(fi[0], fi[1], fi[2]));
    st.add_atom(Atom(Element("H"), r[0], r[1], r[2], 0.0, 0.0, 0.0));
  }
  const double rc = 4.5;
  NNFinder().calc(st, rc);

  std::size_t total = 0;
  bool along_a2 = false, along_a3 = false;
  for (std::size_t i = 0; i < st.natoms(); ++i) {
    INFO("atom " << i);
    // Brute force over the rows.
    std::vector<NbrKey> want;
    for (std::size_t j = 0; j < st.natoms(); ++j)
      for (int n0 = -1; n0 <= 1; ++n0)
        for (int n1 = -1; n1 <= 1; ++n1)
          for (int n2 = -1; n2 <= 1; ++n2) {
            if (i == j && n0 == 0 && n1 == 0 && n2 == 0) continue;
            const Vec3d n(n0, n1, n2);
            const Vec3d d =
                st(j).position + lattice_vector(st.cell, n) - st(i).position;
            if (d.norm() < rc) want.push_back(nbr_key(j, n));
          }
    std::vector<NbrKey> got;
    for (std::size_t k = 0; k < st.nn_size(i); ++k) {
      const std::size_t j = st.nn->idx[i][k];
      const Vec3d &n = st.nn->shift[i][k];
      got.push_back(nbr_key(j, n));
      const Vec3d d = st.nn_pos(i, k) - st(j).position;
      const Vec3d t = lattice_vector(st.cell, n);
      for (int c = 0; c < 3; ++c) CHECK(d[c] == Approx(t[c]).margin(1e-12));
      along_a2 = along_a2 || std::lround(n[1]) != 0;
      along_a3 = along_a3 || std::lround(n[2]) != 0;
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    CHECK(got == want);
    total += got.size();
  }
  CHECK(total == 18);
  CHECK(along_a2);  // the check is not limited to shifts along a1
  CHECK(along_a3);
}

TEST_CASE("remapToCell wraps fractional coords for sheared cell", "[cell_convention]") {
  Structure st = make_sheared_structure();
  // Put atom at frac (1.5, 1.5, 1.5): cart = 1.5*(a1+a2+a3) = (4.5, 3.0, 4.5)
  // After remap to [0,1) frac: (0.5, 0.5, 0.5) -> cart = (1.5, 1.0, 1.5)
  st.add_atom(Atom(Element("H"), 4.5, 3.0, 4.5, 0.0, 0.0, 0.0));
  StructureTransformation::remapToCell(st);
  REQUIRE(st(0).position[0] == Approx(1.5));
  REQUIRE(st(0).position[1] == Approx(1.0));
  REQUIRE(st(0).position[2] == Approx(1.5));
}

TEST_CASE("addAtom random-position lands inside the cell", "[cell_convention]") {
  Structure st = make_sheared_structure();
  Atom tpl(Element("H"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  // useRandomPosition=true: position is built from a uniform random fractional.
  StructureTransformation::addAtom(st, tpl, /*useRandomPosition=*/true,
                                   Vec3d(0.0, 0.0, 0.0), /*minDistance=*/0.0);
  // The new atom's fractional should round-trip to [0,1).
  Matrix3d invCellT = st.cell.transpose().inverse();
  Vec3d frac = invCellT * st(0).position;
  for (int i = 0; i < 3; ++i) {
    REQUIRE(frac[i] >= -1e-12);
    REQUIRE(frac[i] <= 1.0 + 1e-12);
  }
}

TEST_CASE("NNFinder on a sheared cell finds the image across the tilted a2 "
          "at the hand-computed distance", "[cell_convention][nn_finder]") {
  // Rows a1 = (10,0,0), a2 = (2,8,0), a3 = (0,0,9); every height (9.70, 8,
  // 9) exceeds the cutoff 3. Atom 0 at fractional (0.05, 0.05, 0.5) =
  // (0.6, 0.4, 4.5), atom 1 at (0.15, 0.95, 0.5) = (3.4, 7.6, 4.5). Their
  // only pair closer than 3 is atom 1 shifted by -a2:
  //   (3.4, 7.6, 4.5) - (2, 8, 0) - (0.6, 0.4, 4.5) = (0.8, -0.8, 0),
  // distance 0.8*sqrt(2) = 1.1314. Reading the cell as columns would put
  // that image at (2.8, -0.8, 0), distance 2.912.
  Structure st;
  st.cell.load(10.0, 0.0, 0.0,
                2.0, 8.0, 0.0,
                0.0, 0.0, 9.0);
  st.add_atom(Atom(Element("H"), 0.6, 0.4, 4.5, 0.0, 0.0, 0.0));
  st.add_atom(Atom(Element("H"), 3.4, 7.6, 4.5, 0.0, 0.0, 0.0));

  NNFinder finder;
  finder.calc(st, 3.0);

  struct Want { std::size_t i, j; int n1; double dx, dy; };
  for (const Want &w : {Want{0, 1, -1, 0.8, -0.8}, Want{1, 0, 1, -0.8, 0.8}}) {
    INFO("atom " << w.i);
    REQUIRE(st.nn_size(w.i) == 1);
    CHECK(st.nn->idx[w.i][0] == w.j);
    const Vec3d &n = st.nn->shift[w.i][0];
    CHECK(std::lround(n[0]) == 0);
    CHECK(std::lround(n[1]) == w.n1);
    CHECK(std::lround(n[2]) == 0);
    const Vec3d d = st.nn_pos(w.i, 0) - st(w.i).position;
    CHECK(d[0] == Approx(w.dx).margin(1e-12));
    CHECK(d[1] == Approx(w.dy).margin(1e-12));
    CHECK(d[2] == Approx(0.0).margin(1e-12));
    CHECK(d.norm() == Approx(0.8 * std::sqrt(2.0)));
  }
}

TEST_CASE("OUTCAR with sheared cell parses with row-as-lat-vec convention",
         "[cell_convention][outcar]") {
  StructureDB db;
  VaspOutcarReader reader(db, "./tests_data/sheared_outcars/OUTCAR_sheared");
  reader.parse_data();
  REQUIRE(db.size() >= 1);
  const Structure &st = db(0);

  // row 0 must be the printed first lattice vector a1 = (2.5, 0, 0)
  REQUIRE(st.cell(0,0) == Approx(2.5));
  REQUIRE(st.cell(0,1) == Approx(0.0));
  REQUIRE(st.cell(0,2) == Approx(0.0));
  // row 1 must be a2 = (0.5, 2.0, 0)
  REQUIRE(st.cell(1,0) == Approx(0.5));
  REQUIRE(st.cell(1,1) == Approx(2.0));
  REQUIRE(st.cell(1,2) == Approx(0.0));
  // row 2 must be a3 = (0, 0, 3.0)
  REQUIRE(st.cell(2,0) == Approx(0.0));
  REQUIRE(st.cell(2,1) == Approx(0.0));
  REQUIRE(st.cell(2,2) == Approx(3.0));
  REQUIRE(st.get_volume() == Approx(15.0));
}

TEST_CASE("OUTCAR -> tadahdb roundtrip preserves row convention",
         "[cell_convention][outcar][tadahdb]") {
  // The dataset file stores the cell row by row: its lines 4-6 are a1, a2,
  // a3 as printed in the OUTCAR, and reading the file back gives the same
  // rows. Both sides are checked against the hand values, so a transposed
  // cell in the writer or in the reader fails (comparing cell(i,j) with
  // cell(i,j) alone would not). A fresh directory avoids reading a stale
  // file: dump_to_file(path) appends.
  StructureDB db;
  VaspOutcarReader reader(db, "./tests_data/sheared_outcars/OUTCAR_sheared");
  reader.parse_data();
  REQUIRE(db.size() >= 1);
  const Structure st_in = db(0);
  const double a[3][3] = {{2.5, 0.0, 0.0}, {0.5, 2.0, 0.0}, {0.0, 0.0, 3.0}};

  CellConvTempDir tmp;
  const std::string file = (tmp.dir / "sheared.tadahdb").string();
  st_in.dump_to_file(file, /*prec=*/12);

  // Writer side: label, weights, energy/T, then the rows a1, a2, a3.
  {
    std::ifstream in(file);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    REQUIRE(lines.size() >= 6);
    for (int i = 0; i < 3; ++i) {
      INFO("file line " << 4 + i << ": " << lines[3 + i]);
      std::istringstream ls(lines[3 + i]);
      double x, y, z;
      REQUIRE(static_cast<bool>(ls >> x >> y >> z));
      CHECK(x == Approx(a[i][0]).margin(1e-12));
      CHECK(y == Approx(a[i][1]).margin(1e-12));
      CHECK(z == Approx(a[i][2]).margin(1e-12));
    }
  }

  // Reader side.
  Structure st_out;
  st_out.read(file);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("cell(" << i << "," << j << ")");
      REQUIRE(st_out.cell(i,j) == Approx(st_in.cell(i,j)));
      CHECK(st_out.cell(i,j) == Approx(a[i][j]).margin(1e-12));
    }
  REQUIRE(st_out.get_volume() == Approx(st_in.get_volume()));
  CHECK(st_out.get_volume() == Approx(15.0));
  REQUIRE(st_out.natoms() == st_in.natoms());
  for (std::size_t n = 0; n < st_in.natoms(); ++n)
    for (int k = 0; k < 3; ++k)
      CHECK(st_out(n).position[k] ==
            Approx(st_in(n).position[k]).margin(1e-12));
}
