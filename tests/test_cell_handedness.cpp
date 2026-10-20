// test_cell_handedness.cpp
//
// Row i of Structure::cell is the lattice vector a_i, and any handedness is
// valid: a reader keeps the cell as read. CASTEP accepts left-handed
// lattices; every frame of tests_data/valid_castep_castep/md_stress.castep
// has the rows (s,s,0), (s,0,s), (0,s,s) with det = -2 s^3 < 0, while CASTEP
// prints the volume +2 s^3. The volume of a structure is therefore
// |a0 . (a1 x a2)|, and everything derived from it (density, scaleToVolume,
// the stress design-matrix rows, the predicted stress) must be the same for
// a left-handed row order of a lattice as for a right-handed one. With a
// signed volume a left-handed cell flips the sign of every stress row and
// prediction and makes the density negative (or 0).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_types.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/castep_castep_reader.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_properties.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/trainer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::core::Vec3d;
using tadah::mlip::Atom;
using tadah::mlip::CastepCastepReader;
using tadah::mlip::NNFinder;
using tadah::mlip::StDescriptors;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::StructureProperties;
using tadah::mlip::StructureTransformation;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

// Lattice edge of the first md_stress.castep frame (A).
constexpr double kS = 4.9343;

// det(cell) = a0 . (a1 x a2), written out (independent of the code under test).
double hand_det(const Matrix3d &c) {
  return c(0, 0) * (c(1, 1) * c(2, 2) - c(1, 2) * c(2, 1)) -
         c(0, 1) * (c(1, 0) * c(2, 2) - c(1, 2) * c(2, 0)) +
         c(0, 2) * (c(1, 0) * c(2, 1) - c(1, 1) * c(2, 0));
}

// The md_stress.castep frame-0 rows, left-handed: det = -2 s^3.
Matrix3d left_handed_cell() {
  Matrix3d c;
  c.load(kS, kS, 0.0,
         kS, 0.0, kS,
         0.0, kS, kS);
  return c;
}

// The same lattice with rows 1 and 2 swapped: (a0, a2, a1), det = +2 s^3.
Matrix3d right_handed_cell() {
  Matrix3d c;
  c.load(kS, kS, 0.0,
         0.0, kS, kS,
         kS, 0.0, kS);
  return c;
}

// r = f0*a0 + f1*a1 + f2*a2 for the rows of the LEFT-handed cell.
Vec3d cart_left(double f0, double f1, double f2) {
  const Matrix3d c = left_handed_cell();
  return Vec3d(f0 * c(0, 0) + f1 * c(1, 0) + f2 * c(2, 0),
               f0 * c(0, 1) + f1 * c(1, 1) + f2 * c(2, 1),
               f0 * c(0, 2) + f1 * c(1, 2) + f2 * c(2, 2));
}

// Two iodine atoms at fractional (1/4,1/4,1/4) and (3/4,3/4,3/4): both
// row orders place them at the same Cartesian points.
Structure iodine_pair(const Matrix3d &cell) {
  Structure st;
  st.label = "left/right-handed I2";
  st.cell = cell;
  st.add_atom(Atom(Element("I"), 0.5 * kS, 0.5 * kS, 0.5 * kS, 0.0, 0.0, 0.0));
  st.add_atom(Atom(Element("I"), 1.5 * kS, 1.5 * kS, 1.5 * kS, 0.0, 0.0, 0.0));
  return st;
}

// Six H atoms inside the md_stress cell (fractional coordinates of the
// left-handed rows). Every height is 2 s/sqrt(3) = 5.698 A > rcut = 4.5, so
// the neighbour lists are complete for either row order; the pair distances
// are 2.03 ... 4.45 A inside the cutoff and >= 4.56 A outside it.
Structure h6(const Matrix3d &cell) {
  const double f[6][3] = {{0.10, 0.20, 0.30}, {0.45, 0.50, 0.25},
                          {0.70, 0.15, 0.80}, {0.30, 0.80, 0.60},
                          {0.85, 0.60, 0.10}, {0.55, 0.90, 0.95}};
  Structure st;
  st.label = "H6";
  st.cell = cell;
  for (const auto &fi : f) {
    const Vec3d r = cart_left(fi[0], fi[1], fi[2]);
    st.add_atom(Atom(Element("H"), r[0], r[1], r[2], 0.0, 0.0, 0.0));
  }
  return st;
}

// D2_BP pair model (BF_Linear) with FORCE and STRESS on, so the descriptor
// derivatives the stress rows need are computed.
Context make_h_stress_context(double rcut) {
  Context c;
  c.add("ATOM", "H");
  c.add("WATOM", 1);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_BP");
  c.add("TYPE2B", 2);
  c.add("TYPE2B", 2);
  c.add("TYPE2B", "H");
  c.add("TYPE2B", "H");
  c.add("TYPEMB", "DM_Dummy");
  c.add("TYPEMB", "H");
  c.add("TYPEMB", "H");
  c.add("CGRID2B", "2.5 3.5");
  c.add("SGRID2B", "1.0 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUT2B", rcut);
  c.add("RCUTMB", rcut);
  c.add("DIMER", "false");
  c.add("DIMER", 0);
  c.add("DIMER", "false");
  c.add("FORCE", "true");
  c.add("STRESS", "true");
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("VERBOSE", 0);
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  return c;
}

double max_abs_entry(const Matrix3d &m) {
  double v = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) v = std::max(v, std::fabs(m(i, j)));
  return v;
}

}  // namespace

TEST_CASE("Handedness: every left-handed md_stress.castep frame has the "
          "volume CASTEP prints",
          "[cell_handedness][castep]") {
  // The .castep reader keeps the cell as read (left-handed, det < 0); the
  // volume must be the one CASTEP prints ("Current cell volume"), i.e.
  // |det|, from both Structure::get_volume and StructureProperties.
  PeriodicTable::initialize();
  const std::string path = "./tests_data/valid_castep_castep/md_stress.castep";

  std::vector<double> printed;
  {
    std::ifstream in(path);
    REQUIRE(in.good());
    const char *key = "Current cell volume =";
    std::string line;
    while (std::getline(in, line)) {
      const auto p = line.find(key);
      if (p == std::string::npos) continue;
      std::istringstream ls(line.substr(p + std::strlen(key)));
      double v = 0.0;
      REQUIRE(static_cast<bool>(ls >> v));
      printed.push_back(v);
    }
  }
  REQUIRE(printed.size() == 4);  // one per MD frame
  CHECK(printed[0] == Approx(240.273928));

  StructureDB db;
  CastepCastepReader reader(db, path);
  reader.parse_data();
  REQUIRE(db.size() == printed.size());
  for (std::size_t k = 0; k < db.size(); ++k) {
    INFO("frame " << k);
    const Structure &st = db(k);
    CHECK(hand_det(st.cell) < 0.0);  // kept as read: left-handed
    CHECK(std::fabs(hand_det(st.cell)) == Approx(printed[k]).epsilon(1e-7));
    CHECK(st.get_volume() == Approx(printed[k]).epsilon(1e-7));
    CHECK(StructureProperties::getVolume(st) ==
          Approx(printed[k]).epsilon(1e-7));
  }
}

TEST_CASE("Handedness: the density is positive and the same for both row "
          "orders of one lattice",
          "[cell_handedness]") {
  // rho = m / |V| with |V| = 2 s^3 and m = 2 m_I (amu -> g, A^3 -> cm^3).
  // Structure::get_density was negative and StructureProperties::getDensity
  // returned 0 for the left-handed rows.
  PeriodicTable::initialize();
  const double mass_g = 2.0 * PeriodicTable::get_mass(53) * 1.66053906660e-24;
  const double volume_cm3 = 2.0 * kS * kS * kS * 1.0e-24;
  const double expected = mass_g / volume_cm3;

  const Structure left = iodine_pair(left_handed_cell());
  const Structure right = iodine_pair(right_handed_cell());
  REQUIRE(hand_det(left.cell) < 0.0);
  REQUIRE(hand_det(right.cell) > 0.0);

  CHECK(left.get_volume() == Approx(2.0 * kS * kS * kS));
  CHECK(right.get_volume() == Approx(2.0 * kS * kS * kS));
  CHECK(left.get_density() == Approx(expected));
  CHECK(right.get_density() == Approx(expected));
  CHECK(StructureProperties::getDensity(left) == Approx(expected));
  CHECK(StructureProperties::getDensity(right) == Approx(expected));
}

TEST_CASE("Handedness: scaleToVolume and scaleVolumeToDensity accept a "
          "left-handed cell and keep its row order",
          "[cell_handedness][transform]") {
  // scaleToVolume threw "Current volume is invalid (<= 0)" for any
  // left-handed cell. Scaling to twice the volume multiplies every row by
  // 2^(1/3) and keeps the rows in the order read (still left-handed).
  PeriodicTable::initialize();
  const double v0 = 2.0 * kS * kS * kS;
  const double f = std::cbrt(2.0);

  Structure st = iodine_pair(left_handed_cell());
  REQUIRE_NOTHROW(StructureTransformation::scaleToVolume(st, 2.0 * v0));
  const Matrix3d c0 = left_handed_cell();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("cell(" << i << "," << j << ")");
      CHECK(st.cell(i, j) == Approx(f * c0(i, j)).margin(1e-12));
    }
  CHECK(hand_det(st.cell) == Approx(-2.0 * v0));
  CHECK(st.get_volume() == Approx(2.0 * v0));
  CHECK(st.atoms[1].position[0] == Approx(1.5 * kS * f));

  // Doubling the density halves the volume back to v0.
  Structure st2 = iodine_pair(left_handed_cell());
  const double rho0 = 2.0 * PeriodicTable::get_mass(53) * 1.66053906660 / v0;
  REQUIRE_NOTHROW(StructureTransformation::scaleVolumeToDensity(st2, 2.0 * rho0));
  CHECK(st2.get_volume() == Approx(0.5 * v0));
  CHECK(hand_det(st2.cell) == Approx(-0.5 * v0));
}

TEST_CASE("Handedness: stress design-matrix rows and stress predictions do "
          "not depend on the handedness of the cell rows",
          "[cell_handedness][stress]") {
  // The same six H atoms in the same lattice, given once with the
  // left-handed rows (a0, a1, a2) of md_stress.castep and once with the
  // right-handed rows (a0, a2, a1). The DFT stress does not depend on the
  // basis, so neither may the model: before the fix V_inv = 1/det flipped
  // the sign of every stress row and prediction for the left-handed rows.
  PeriodicTable::initialize();
  const double rcut = 4.5;
  MLIPWorkspaceManager ws;

  std::size_t n_w = 0;
  {
    Context probe_ctx = make_h_stress_context(rcut);
    Trainer probe(probe_ctx, ws);
    n_w = probe.fb->get_phi_cols(probe.context);
  }
  REQUIRE(n_w > 0);
  Context ctx = make_h_stress_context(rcut);
  ctx.add("FIXINDEX", "1-" + std::to_string(n_w));
  for (std::size_t k = 0; k < n_w; ++k)
    ctx.add("FIXWEIGHT", 0.8 - 0.55 * static_cast<double>(k));
  Trainer tr(ctx, ws);
  tr.provision_fixed_weights();

  Structure left = h6(left_handed_cell());
  Structure right = h6(right_handed_cell());
  REQUIRE(hand_det(left.cell) < 0.0);
  REQUIRE(hand_det(right.cell) > 0.0);

  NNFinder nnf;
  nnf.calc(left, rcut);
  nnf.calc(right, rcut);
  for (std::size_t a = 0; a < left.natoms(); ++a) {
    INFO("atom " << a);
    REQUIRE(left.nn_size(a) == right.nn_size(a));  // same neighbour images
    REQUIRE(left.nn_size(a) > 0);
  }

  const StDescriptors d_left = tr.dc.calc(left);
  const StDescriptors d_right = tr.dc.calc(right);

  SECTION("stress design-matrix rows") {
    tadah::core::phi_type phi_left(6, n_w);
    tadah::core::phi_type phi_right(6, n_w);
    std::size_t row_left = 0;
    std::size_t row_right = 0;
    tr.fb->calc_phi_stress_rows(phi_left, row_left, left, d_left);
    tr.fb->calc_phi_stress_rows(phi_right, row_right, right, d_right);
    REQUIRE(row_left == 6);
    REQUIRE(row_right == 6);
    double scale = 0.0;
    for (std::size_t m = 0; m < 6; ++m)
      for (std::size_t d = 0; d < n_w; ++d)
        scale = std::max(scale, std::fabs(phi_right(m, d)));
    REQUIRE(scale > 1e-6);  // the rows are not trivially zero
    for (std::size_t m = 0; m < 6; ++m)
      for (std::size_t d = 0; d < n_w; ++d) {
        INFO("Phi(" << m << "," << d << ")");
        CHECK(phi_left(m, d) ==
              Approx(phi_right(m, d)).margin(1e-12 * scale));
      }
  }

  SECTION("predicted stress (spredict and stress_force_predict)") {
    const Matrix3d s_left = tr.model->spredict(d_left, left);
    const Matrix3d s_right = tr.model->spredict(d_right, right);
    const double scale = max_abs_entry(s_right);
    REQUIRE(scale > 1e-6);
    Structure p_left = left;
    Structure p_right = right;
    tr.model->stress_force_predict(d_left, p_left, left);
    tr.model->stress_force_predict(d_right, p_right, right);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        INFO("stress(" << i << "," << j << ")");
        CHECK(s_left(i, j) == Approx(s_right(i, j)).margin(1e-12 * scale));
        CHECK(p_left.stress(i, j) ==
              Approx(p_right.stress(i, j)).margin(1e-12 * scale));
        CHECK(p_right.stress(i, j) ==
              Approx(s_right(i, j)).margin(1e-12 * scale));
      }
  }
}
