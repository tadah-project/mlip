// Tests that codify the row-as-lattice-vector convention for Structure::cell.
// All distance/PBC code in MLIP assumes row(i) == a_i. These tests use a
// non-orthogonal (sheared) cell so that the convention is observable; on a
// diagonal cell rows == columns and the bug is silent.

#include "catch2/catch.hpp"

#include "tadah/core/element.h"
#include "tadah/core/periodic_table.h"
#include "tadah/core/vec3d.h"

#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/vasp_outcar_reader.h>
#include <tadah/mlip/fast_cell_frac.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_properties.h>
#include <tadah/mlip/structure_transformations.h>

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

TEST_CASE("Periodic shift formula returns lattice vectors", "[cell_convention]") {
  // Reproduces the formula at MLIP/src/nn_finder.cpp:63-65 and
  // structure.cpp:118-120. Under rows-as-lat-vecs:
  //   disp[k] = n1*cell(0,k) + n2*cell(1,k) + n3*cell(2,k) = (n1*a1 + n2*a2 + n3*a3)[k]
  Structure st = make_sheared_structure();
  const auto &cell = st.cell;
  auto shift_disp = [&](int n1, int n2, int n3) {
    Vec3d d;
    d[0] = n1*cell(0,0) + n2*cell(1,0) + n3*cell(2,0);
    d[1] = n1*cell(0,1) + n2*cell(1,1) + n3*cell(2,1);
    d[2] = n1*cell(0,2) + n2*cell(1,2) + n3*cell(2,2);
    return d;
  };

  Vec3d d100 = shift_disp(1, 0, 0);
  REQUIRE(d100[0] == Approx(2.5));   // a1.x
  REQUIRE(d100[1] == Approx(0.0));   // a1.y == 0 (was -1.99 under the bug)
  REQUIRE(d100[2] == Approx(0.0));

  Vec3d d010 = shift_disp(0, 1, 0);
  REQUIRE(d010[0] == Approx(0.5));   // a2
  REQUIRE(d010[1] == Approx(2.0));
  REQUIRE(d010[2] == Approx(0.0));

  Vec3d d001 = shift_disp(0, 0, 1);
  REQUIRE(d001[0] == Approx(0.0));
  REQUIRE(d001[1] == Approx(0.0));
  REQUIRE(d001[2] == Approx(3.0));
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

TEST_CASE("FastCellFrac on a sheared cell finds the right neighbour", "[cell_convention]") {
  // Two atoms a fractional-distance (1, 0, 0) apart in real space:
  // atom 0 at (0,0,0); atom 1 at a1 = (2.5, 0, 0). With a slightly enlarged
  // cell (so the closest periodic image is the home-cell partner), neighbour
  // distance must be 2.5 (the length of a1), not |row 0 of cell| under the
  // buggy convention.
  Structure st;
  // big cubic-ish but sheared cell so home-cell distance dominates the cutoff
  st.cell.load(10.0, 0.0, 0.0,
                2.0, 8.0, 0.0,
                0.0, 0.0, 9.0);
  st.add_atom(Atom(Element("H"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  st.add_atom(Atom(Element("H"), 2.5, 0.0, 0.0, 0.0, 0.0, 0.0));

  // neighbour-list cutoff above 2.5 but well below shortest a_i length.
  NNFinder finder;
  finder.calc(st, 3.0);

  // Atom 0 must see exactly one home-cell neighbour at distance 2.5.
  REQUIRE(st.nn_size(0) >= 1);
  bool found_25 = false;
  for (size_t n = 0; n < st.nn_size(0); ++n) {
    Vec3d p = st.nn_pos(0, n);
    double d = p.norm();
    if (std::abs(d - 2.5) < 1e-9) found_25 = true;
  }
  REQUIRE(found_25);
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
  // Read the sheared OUTCAR
  StructureDB db;
  VaspOutcarReader reader(db, "./tests_data/sheared_outcars/OUTCAR_sheared");
  reader.parse_data();
  const Structure st_in = db(0);

  // Save to a tadahdb-format file in the cwd, read back, compare
  const std::string tmp = "./tmp_sheared.tadahdb";
  st_in.dump_to_file(tmp, /*prec=*/12);
  Structure st_out;
  st_out.read(tmp);
  std::remove(tmp.c_str());

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      REQUIRE(st_out.cell(i,j) == Approx(st_in.cell(i,j)));
    }
  REQUIRE(st_out.get_volume() == Approx(st_in.get_volume()));
}
