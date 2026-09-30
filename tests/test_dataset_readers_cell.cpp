// Value tests for the structure and dataset readers on non-symmetric cells
// (AUDIT-cell-convention-2026-09-30: castep-5, castep-6, castep-7/docs-28,
// castep-8, data-10, vaspdb-14, vaspdb-15).
//
// Convention: row i of Structure::cell is the lattice vector a_i in
// Angstrom, and a fractional coordinate f maps to the Cartesian position
// r = cell^T f = f0 a0 + f1 a1 + f2 a2. Every fixture here has
// cell != cell^T, so a reader that stores the lattice as columns, or that
// converts with cell * f, fails these tests. On the symmetric and diagonal
// cells of the older fixtures both mistakes are invisible.
//
// Fixtures: ./tests_data/cell_convention/{castep,cell,geom,md}_* and
// ./tests_data/sheared_outcars/OUTCAR_sheared. Expected numbers were
// computed independently (python) and are written out below.

#include "catch2/catch.hpp"

#include "tadah/core/periodic_table.h"

#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/castep_castep_reader.h>
#include <tadah/mlip/dataset_readers/castep_geom_reader.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>
#include <tadah/mlip/dataset_readers/dataset_reader_selector.h>
#include <tadah/mlip/dataset_readers/vasp_vasprun_reader.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/castep_cell_reader.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using tadah::core::PeriodicTable;
using tadah::mlip::CastepCastepReader;
using tadah::mlip::CastepCellReader;
using tadah::mlip::CastepGeomReader;
using tadah::mlip::CastepMDReader;
using tadah::mlip::DatasetReader;
using tadah::mlip::DatasetReaderSelector;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::VaspVasprunReader;

namespace {

using V3 = std::array<double, 3>;
using Rows = std::array<V3, 3>;

const std::string kCellDir = "./tests_data/cell_convention/";
const std::string kShearedOutcar = "./tests_data/sheared_outcars/OUTCAR_sheared";

constexpr double kBohr = 0.529177210903;     // Angstrom per bohr (CODATA 2018)
constexpr double kHartree = 27.211386245988; // eV per hartree (CODATA 2018)
constexpr double kPi = 3.141592653589793;

// r = f0 a0 + f1 a1 + f2 a2, written out so that the expectation does not
// go through Matrix3d.
V3 rows_frac_to_cart(const Rows &a, const V3 &f) {
  V3 r{0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
    r[k] = f[0] * a[0][k] + f[1] * a[1][k] + f[2] * a[2][k];
  return r;
}

Rows rows_of(const Structure &st) {
  Rows a;
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) a[i][k] = st.cell(i, k);
  return a;
}

double triple(const V3 &u, const V3 &v, const V3 &w) {
  return u[0] * (v[1] * w[2] - v[2] * w[1]) -
         u[1] * (v[0] * w[2] - v[2] * w[0]) +
         u[2] * (v[0] * w[1] - v[1] * w[0]);
}

// Solves r = f0 a0 + f1 a1 + f2 a2 for f (Cramer's rule), with a_i taken
// from the rows of st.cell.
V3 frac_from_rows(const Structure &st, const V3 &r) {
  const Rows a = rows_of(st);
  const double det = triple(a[0], a[1], a[2]);
  return {triple(r, a[1], a[2]) / det, triple(a[0], r, a[2]) / det,
          triple(a[0], a[1], r) / det};
}

V3 position_of(const Structure &st, size_t k) {
  return {st.atoms[k].position[0], st.atoms[k].position[1],
          st.atoms[k].position[2]};
}

void check_rows(const Structure &st, const Rows &expected, double margin) {
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      INFO("cell(" << i << "," << k << ")");
      CHECK(st.cell(i, k) == Approx(expected[i][k]).margin(margin));
    }
}

void check_v3(const V3 &got, const V3 &expected, double margin) {
  for (int k = 0; k < 3; ++k) {
    INFO("component " << k);
    CHECK(got[k] == Approx(expected[k]).margin(margin));
  }
}

double vec_norm(const V3 &v) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

double angle_deg(const V3 &u, const V3 &v) {
  const double c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (vec_norm(u) * vec_norm(v));
  return std::acos(c) * 180.0 / kPi;
}

// Minimum distance between two atoms (images included) with the lattice
// taken from the rows of st.cell; brute force over -2..2 images.
double min_distance(const Structure &st) {
  const Rows a = rows_of(st);
  double dmin = std::numeric_limits<double>::max();
  for (size_t i = 0; i < st.natoms(); ++i)
    for (size_t j = 0; j < st.natoms(); ++j)
      for (int n0 = -2; n0 <= 2; ++n0)
        for (int n1 = -2; n1 <= 2; ++n1)
          for (int n2 = -2; n2 <= 2; ++n2) {
            if (i == j && n0 == 0 && n1 == 0 && n2 == 0) continue;
            V3 d;
            for (int k = 0; k < 3; ++k)
              d[k] = st.atoms[j].position[k] + n0 * a[0][k] + n1 * a[1][k] +
                     n2 * a[2][k] - st.atoms[i].position[k];
            dmin = std::min(dmin, vec_norm(d));
          }
  return dmin;
}

Structure read_cell_file(const std::string &path) {
  CastepCellReader reader;
  reader.read(path);
  return reader.getStructure();
}

// The reader must refuse the file with a std::runtime_error whose message
// contains `needle` (case-insensitive).
void check_cell_refused(const std::string &file, const std::string &needle) {
  INFO(file << " (expected message containing '" << needle << "')");
  CastepCellReader r1;
  CHECK_THROWS_AS(r1.read(kCellDir + file), std::runtime_error);
  CastepCellReader r2;
  CHECK_THROWS_WITH(r2.read(kCellDir + file),
                    Catch::Contains(needle, Catch::CaseSensitive::No));
}

// One frame of a .geom/.md fixture: the lattice rows and positions in
// Angstrom, the fractional coordinates the positions were built from, and
// the energy in hartree as printed.
struct Frame {
  double energy_ha;
  Rows cell;
  std::vector<std::string> symbols;
  std::vector<V3> frac;
  std::vector<V3> pos;
};

// Checks every structure the reader returned against the frame with the
// same energy.
void check_frames(const StructureDB &db, const std::vector<Frame> &frames) {
  REQUIRE(db.size() >= 1);
  for (size_t s = 0; s < db.size(); ++s) {
    const Structure &st = db(s);
    INFO("structure " << s << " energy " << st.energy);
    const Frame *fr = nullptr;
    for (const auto &f : frames)
      if (std::abs(st.energy - f.energy_ha * kHartree) < 1e-6) fr = &f;
    REQUIRE(fr != nullptr);
    check_rows(st, fr->cell, 1e-10);
    REQUIRE(st.natoms() == fr->pos.size());
    for (size_t k = 0; k < st.natoms(); ++k) {
      INFO("atom " << k);
      CHECK(st.atoms[k].symbol == fr->symbols[k]);
      check_v3(position_of(st, k), fr->pos[k], 1e-10);
      check_v3(frac_from_rows(st, position_of(st, k)), fr->frac[k], 1e-10);
    }
  }
}

}  // namespace

// --------------------------------------------------------------------
// CASTEP .cell (CastepCellReader)
// --------------------------------------------------------------------

// castep-5, castep-6: the LATTICE_CART unit line (bohr) is applied, and
// POSITIONS_FRAC, which comes before the lattice block, is converted with
// the final lattice (r = cell^T f). The unfixed reader skips the unit line
// (lattice 1.89x too small) and converts each fractional line with the
// still-zero cell (every atom at the origin).
TEST_CASE("CASTEP .cell reader: LATTICE_CART in bohr, POSITIONS_FRAC before the lattice",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  Structure st;
  REQUIRE_NOTHROW(st = read_cell_file(kCellDir + "cell_lattice_cart_bohr_frac_first.cell"));

  const Rows a = {{{6.0 * kBohr, 0.4 * kBohr, -0.2 * kBohr},
                   {2.0 * kBohr, 7.0 * kBohr, 0.3 * kBohr},
                   {-1.0 * kBohr, 1.5 * kBohr, 8.0 * kBohr}}};
  check_rows(st, a, 1e-12);
  CHECK(st.get_volume() == Approx(48.12743059192904).epsilon(1e-12));

  REQUIRE(st.natoms() == 2);
  CHECK(st.atoms[0].symbol == "Si");
  CHECK(st.atoms[0].Z == 14);
  CHECK(st.atoms[1].symbol == "O");
  CHECK(st.atoms[1].Z == 8);
  // Si (0.25, 0.5, 0.125) and O (0.75, 0.1, 0.6) as f0 a0 + f1 a1 + f2 a2.
  check_v3(position_of(st, 0), {1.2567958758946247, 2.0042586862951124, 0.5820949319933}, 1e-12);
  check_v3(position_of(st, 1), {2.1696265647022996, 1.0054367007157, 2.47654934702604}, 1e-12);
  check_v3(position_of(st, 0), rows_frac_to_cart(a, {0.25, 0.5, 0.125}), 1e-12);
  check_v3(position_of(st, 1), rows_frac_to_cart(a, {0.75, 0.1, 0.6}), 1e-12);
}

// castep-5: the LATTICE_ABC unit line (bohr) scales a, b and c. The cell
// follows CASTEP's documented LATTICE_ABC orientation (a along x, b in the
// xy plane, c completing a right-handed set). The unfixed reader reads
// 9, 10, 11 as Angstrom.
TEST_CASE("CASTEP .cell reader: LATTICE_ABC in bohr, CASTEP orientation",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  Structure st;
  REQUIRE_NOTHROW(st = read_cell_file(kCellDir + "cell_lattice_abc_bohr.cell"));

  const Rows a = {{{4.762594898126999, 0.0, 0.0},
                   {-1.3696114041592973, 5.111459346948248, 0.0},
                   {-0.5073291614752367, 0.9105158684400253, 5.7268690364338}}};
  check_rows(st, a, 1e-10);

  const Rows got = rows_of(st);
  CHECK(vec_norm(got[0]) == Approx(9.0 * kBohr).epsilon(1e-12));
  CHECK(vec_norm(got[1]) == Approx(10.0 * kBohr).epsilon(1e-12));
  CHECK(vec_norm(got[2]) == Approx(11.0 * kBohr).epsilon(1e-12));
  CHECK(angle_deg(got[1], got[2]) == Approx(80.0).epsilon(1e-12));   // alpha
  CHECK(angle_deg(got[0], got[2]) == Approx(95.0).epsilon(1e-12));   // beta
  CHECK(angle_deg(got[0], got[1]) == Approx(105.0).epsilon(1e-12));  // gamma
  CHECK(got[2][2] > 0.0);  // right-handed
  CHECK(st.get_volume() == Approx(139.41381290763775).epsilon(1e-12));

  REQUIRE(st.natoms() == 2);
  check_v3(position_of(st, 0), {0.050138460538269494, 1.2954466299216572, 1.7180607109301398}, 1e-10);
  check_v3(position_of(st, 1), {1.7088334152601445, 2.562949259605908, 4.867838680968729}, 1e-10);
}

// Regression pin (passes before the fix): LATTICE_ABC without a unit line
// is in Angstrom and uses the same orientation.
TEST_CASE("CASTEP .cell reader: LATTICE_ABC in the default unit",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  Structure st;
  REQUIRE_NOTHROW(st = read_cell_file(kCellDir + "cell_lattice_abc_ang.cell"));

  const Rows a = {{{4.5, 0.0, 0.0},
                   {-1.2940952255126041, 4.8296291314453415, 0.0},
                   {-0.4793565851121203, 0.8603128117387183, 5.411107015229681}}};
  check_rows(st, a, 1e-10);
  CHECK(st.get_volume() == Approx(117.6013803335468).epsilon(1e-12));

  REQUIRE(st.natoms() == 2);
  check_v3(position_of(st, 0), {0.04737397936384307, 1.2240196698106838, 1.6233321045689042}, 1e-10);
  check_v3(position_of(st, 1), {1.6146135737252862, 2.42163608598378, 4.599440962945229}, 1e-10);
}

// castep-5: the POSITIONS_ABS unit line (nm) scales the positions, and an
// explicit 'ang' lattice unit is accepted. The unfixed reader reads the
// positions as Angstrom (10x too small).
TEST_CASE("CASTEP .cell reader: POSITIONS_ABS in nm",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  Structure st;
  REQUIRE_NOTHROW(st = read_cell_file(kCellDir + "cell_positions_abs_nm.cell"));

  check_rows(st, {{{4.0, 0.0, 0.0}, {1.0, 4.5, 0.0}, {0.5, -0.8, 5.0}}}, 1e-12);
  REQUIRE(st.natoms() == 2);
  CHECK(st.atoms[0].symbol == "Si");
  CHECK(st.atoms[1].symbol == "O");
  check_v3(position_of(st, 0), {0.5, 1.2, 2.0}, 1e-12);
  check_v3(position_of(st, 1), {3.1, 2.8, 4.2}, 1e-12);
}

// castep-5: every CASTEP length unit, in any letter case, scales both
// LATTICE_CART and POSITIONS_ABS (ang = 1; bohr = a0 = 0.529177210903 A;
// nm = 10; cm = 1e8; m = 1e10; pm = 0.01; no unit line = ang). The same
// lattice and positions are written in each unit and must read back in
// Angstrom. The unfixed reader ignores every unit line.
TEST_CASE("CASTEP .cell reader: every length unit is applied, case-insensitively",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  const Rows lattice = {{{4.0, 0.3, -0.2}, {1.0, 4.5, 0.1}, {0.5, -0.8, 5.0}}};
  const std::vector<V3> positions = {{0.5, 1.2, 2.0}, {3.1, 2.8, 4.2}};
  const std::vector<std::pair<std::string, double>> units = {
      {"", 1.0},       {"ang", 1.0},    {"ANG", 1.0},  {"bohr", kBohr},
      {"Bohr", kBohr}, {"a0", kBohr},   {"A0", kBohr}, {"nm", 10.0},
      {"NM", 10.0},    {"cm", 1e8},     {"m", 1e10},   {"M", 1e10},
      {"pm", 1e-2},    {"Pm", 1e-2}};
  const std::string path = "./tmp_cell_convention_units.cell";

  for (const auto &u : units) {
    INFO("unit line '" << u.first << "'");
    std::ostringstream text;
    text << std::setprecision(17);
    text << "%BLOCK LATTICE_CART\n";
    if (!u.first.empty()) text << u.first << "\n";
    for (const auto &row : lattice)
      text << row[0] / u.second << " " << row[1] / u.second << " "
           << row[2] / u.second << "\n";
    text << "%ENDBLOCK LATTICE_CART\n\n%BLOCK POSITIONS_ABS\n";
    if (!u.first.empty()) text << u.first << "\n";
    for (const auto &p : positions)
      text << "Si " << p[0] / u.second << " " << p[1] / u.second << " "
           << p[2] / u.second << "\n";
    text << "%ENDBLOCK POSITIONS_ABS\n";
    std::remove(path.c_str());
    {
      std::ofstream out(path);
      REQUIRE(out.good());
      out << text.str();
    }

    Structure st;
    REQUIRE_NOTHROW(st = read_cell_file(path));
    std::remove(path.c_str());
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) {
        INFO("cell(" << i << "," << k << ")");
        CHECK(st.cell(i, k) == Approx(lattice[i][k]).epsilon(1e-12).margin(1e-12));
      }
    REQUIRE(st.natoms() == 2);
    for (size_t k = 0; k < 2; ++k)
      for (int c = 0; c < 3; ++c) {
        INFO("atom " << k << " component " << c);
        CHECK(st.atoms[k].position[c] ==
              Approx(positions[k][c]).epsilon(1e-12).margin(1e-12));
      }
  }
}

// castep-6: a file without a usable lattice is refused. The unfixed reader
// returns a zero cell (and, for POSITIONS_FRAC, every atom at the origin).
TEST_CASE("CASTEP .cell reader: a missing or incomplete lattice block is refused",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  check_cell_refused("cell_err_no_lattice.cell", "lattice");
  check_cell_refused("cell_err_abc_one_line.cell", "LATTICE_ABC");
}

// castep-6: LATTICE_CART with fewer than three vectors is refused; the
// unfixed reader leaves the third row zero.
TEST_CASE("CASTEP .cell reader: LATTICE_CART with two vectors is refused",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  check_cell_refused("cell_err_two_vectors.cell", "LATTICE_CART");
}

// castep-7/docs-28: a fourth LATTICE_CART vector is refused. The unfixed
// reader (lcCount < 4) writes it to cell(3, j): two in-bounds overwrites
// of row 0 and one write past the end of the 3x3 storage.
TEST_CASE("CASTEP .cell reader: LATTICE_CART with four vectors is refused",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  check_cell_refused("cell_err_four_vectors.cell", "LATTICE_CART");
}

// castep-6: a singular lattice (a3 = a1 + a2) is refused; the unfixed
// reader accepts it.
TEST_CASE("CASTEP .cell reader: a singular lattice is refused",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  check_cell_refused("cell_err_singular.cell", "singular");
}

// castep-5: an unknown unit line in LATTICE_CART, LATTICE_ABC or
// POSITIONS_ABS is refused, naming the unit; the unfixed reader skips it
// and reads the numbers as Angstrom.
TEST_CASE("CASTEP .cell reader: an unknown length unit is refused",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  check_cell_refused("cell_err_unknown_unit_cart.cell", "furlong");
  check_cell_refused("cell_err_unknown_unit_abc.cell", "league");
  check_cell_refused("cell_err_unknown_unit_abs.cell", "parsec");
}

// --------------------------------------------------------------------
// CASTEP .castep, .geom, .md and VASP vasprun.xml (DatasetReaders)
// --------------------------------------------------------------------

// castep-8, data-10 (regression pin, passes before the fix): the rows of
// 'Real Lattice(A)' are a_i (the reciprocal columns are ignored) and the
// fractional coordinates are converted with r = cell^T f.
TEST_CASE("CASTEP .castep reader: Real Lattice rows and fractional positions on a triclinic cell",
          "[cell_convention][castep]") {
  PeriodicTable::initialize();
  StructureDB db;
  CastepCastepReader reader(db, kCellDir + "castep_triclinic_spe.castep");
  reader.parse_data();
  REQUIRE(db.size() == 1);
  const Structure &st = db(0);

  const Rows a = {{{3.1, 0.2, -0.1}, {0.9, 3.4, 0.15}, {-0.6, 0.7, 4.2}}};
  check_rows(st, a, 1e-12);
  CHECK(st.get_volume() == Approx(42.9015).epsilon(1e-12));
  CHECK(st.energy == Approx(-312.345678901).epsilon(1e-12));

  REQUIRE(st.natoms() == 3);
  const std::vector<std::string> symbols = {"Si", "Si", "O"};
  const std::vector<V3> frac = {{0.1, 0.2, 0.3}, {0.6, 0.45, 0.8}, {0.35, 0.9, 0.15}};
  const std::vector<V3> pos = {{0.31, 0.91, 1.28}, {1.785, 2.21, 3.3675}, {1.805, 3.235, 0.73}};
  const std::vector<V3> force = {{0.01234, -0.02345, 0.03456},
                                 {-0.04567, 0.05678, -0.06789},
                                 {0.03333, -0.03333, 0.03333}};
  for (size_t k = 0; k < 3; ++k) {
    INFO("atom " << k);
    CHECK(st.atoms[k].symbol == symbols[k]);
    check_v3(position_of(st, k), pos[k], 1e-12);
    check_v3(position_of(st, k), rows_frac_to_cart(a, frac[k]), 1e-12);
    check_v3({st.atoms[k].force[0], st.atoms[k].force[1], st.atoms[k].force[2]},
             force[k], 1e-12);
  }
}

// castep-8, data-10 (regression pin, passes before the fix): '<-- h' lines
// are the lattice vectors a_i in bohr and '<-- R' lines Cartesian positions
// in bohr; both are converted to Angstrom. Each structure the reader returns
// is matched to its frame by energy, so the test does not depend on how
// many frames are returned (the header loop currently drops frame 0).
TEST_CASE("CASTEP .geom reader: lattice rows and positions on a triclinic cell",
          "[cell_convention][geom]") {
  PeriodicTable::initialize();
  const std::vector<Frame> frames = {
      {-1.2345678901234567e2,
       {{{4.0, 0.0, 0.0}, {1.2, 3.8, 0.0}, {-0.7, 0.9, 5.1}}},
       {"Ti", "Ti", "O"},
       {{0.05, 0.10, 0.15}, {0.55, 0.60, 0.65}, {0.30, 0.80, 0.40}},
       {{0.215, 0.515, 0.765}, {2.465, 2.865, 3.315}, {1.88, 3.4, 2.04}}},
      {-1.2346789012345678e2,
       {{{4.02, 0.05, -0.03}, {1.25, 3.79, 0.02}, {-0.68, 0.93, 5.08}}},
       {"Ti", "Ti", "O"},
       {{0.06, 0.11, 0.14}, {0.54, 0.61, 0.66}, {0.31, 0.79, 0.41}},
       {{0.2835, 0.5501, 0.7116}, {2.4845, 2.9527, 3.3488}, {1.9549, 3.3909, 2.0893}}}};
  StructureDB db;
  CastepGeomReader reader(db, kCellDir + "geom_triclinic.geom");
  reader.parse_data();
  check_frames(db, frames);
}

// castep-8, data-10 (regression pin, passes before the fix): as for .geom,
// for a CASTEP .md trajectory.
TEST_CASE("CASTEP .md reader: lattice rows and positions on a triclinic cell",
          "[cell_convention][md]") {
  PeriodicTable::initialize();
  const std::vector<Frame> frames = {
      {-2.1111111111111111e1,
       {{{5.0, 0.3, 0.0}, {-1.1, 4.6, 0.2}, {0.4, -0.8, 5.5}}},
       {"Si", "Si", "C"},
       {{0.12, 0.23, 0.34}, {0.62, 0.73, 0.84}, {0.40, 0.10, 0.90}},
       {{0.483, 0.822, 1.916}, {2.633, 2.872, 4.766}, {2.25, -0.14, 4.97}}},
      {-2.1222222222222222e1,
       {{{5.01, 0.31, 0.01}, {-1.12, 4.61, 0.19}, {0.41, -0.79, 5.52}}},
       {"Si", "Si", "C"},
       {{0.13, 0.22, 0.35}, {0.61, 0.74, 0.83}, {0.41, 0.09, 0.91}},
       {{0.5484, 0.778, 1.9751}, {2.5676, 2.9448, 4.7283}, {2.3264, -0.1769, 5.0444}}}};
  StructureDB db;
  CastepMDReader reader(db, kCellDir + "md_triclinic.md");
  reader.parse_data();
  check_frames(db, frames);
}

// vaspdb-14, data-10 (regression pin, passes before the fix): vasprun_1.xml
// has a triclinic Nb-Ti cell (max |C - C^T| = 3.67 A). The rows of the first
// calculation's cell are the XML basis vectors, atom 0 sits at cell^T f of
// its XML fractional coordinate, and the shortest distance is 2.442 A (a
// cell * f conversion with row images gives 1.154 A).
TEST_CASE("vasprun reader: triclinic basis rows, positions and minimum distance",
          "[cell_convention][vasprun]") {
  PeriodicTable::initialize();
  StructureDB db;
  VaspVasprunReader reader(db, "./tests_data/valid_vaspruns/vasprun_1.xml");
  reader.parse_data();
  REQUIRE(db.size() == 25);
  const Structure &st = db(0);

  const Rows basis = {{{10.60268882, -0.47236822, -0.25391071},
                       {-0.01373443, 10.88874999, 0.08792139},
                       {-1.69319726, -3.58548526, 7.46404261}}};
  check_rows(st, basis, 1e-12);
  CHECK(st.get_volume() == Approx(860.3944182851902).epsilon(1e-12));

  REQUIRE(st.natoms() == 49);
  CHECK(st.atoms[0].symbol == "Ti");
  const V3 f0 = {0.83265528, 0.66668361, 0.55079768};
  check_v3(position_of(st, 0), {7.886619186205921, 4.891154296351266, 3.9583730093605136}, 1e-10);
  check_v3(position_of(st, 0), rows_frac_to_cart(basis, f0), 1e-10);

  CHECK(min_distance(st) == Approx(2.44222907885528).margin(1e-6));
}

// vaspdb-15: the sheared OUTCAR fixture carries a 'POTCAR:' line, so
// DatasetReaderSelector (the path of 'tadah data convert --dft-file')
// recognises it, and its reciprocal vectors are those of its direct
// lattice (a_i . b_j = delta_ij, as VASP prints them). The hand-made CASTEP
// fixtures above are recognised by content as well, and reading through the
// selector gives the lattice rows.
TEST_CASE("DatasetReaderSelector: sheared OUTCAR and triclinic CASTEP fixtures",
          "[cell_convention][selector]") {
  PeriodicTable::initialize();
  CHECK(DatasetReaderSelector::determine_file_type_by_content(kShearedOutcar) == "VASP.OUTCAR");
  CHECK(DatasetReaderSelector::determine_file_type_by_content(
            kCellDir + "castep_triclinic_spe.castep") == "CASTEP.CASTEP");
  CHECK(DatasetReaderSelector::determine_file_type_by_content(
            kCellDir + "geom_triclinic.geom") == "CASTEP.GEOM");
  CHECK(DatasetReaderSelector::determine_file_type_by_content(
            kCellDir + "md_triclinic.md") == "CASTEP.MD");

  {
    StructureDB db;
    std::unique_ptr<DatasetReader> reader = DatasetReaderSelector::get_reader(kShearedOutcar, db);
    REQUIRE(reader != nullptr);
    reader->parse_data();
    REQUIRE(db.size() == 1);
    check_rows(db(0), {{{2.5, 0.0, 0.0}, {0.5, 2.0, 0.0}, {0.0, 0.0, 3.0}}}, 1e-12);
  }
  {
    StructureDB db;
    std::unique_ptr<DatasetReader> reader =
        DatasetReaderSelector::get_reader(kCellDir + "castep_triclinic_spe.castep", db);
    REQUIRE(reader != nullptr);
    reader->parse_data();
    REQUIRE(db.size() == 1);
    check_rows(db(0), {{{3.1, 0.2, -0.1}, {0.9, 3.4, 0.15}, {-0.6, 0.7, 4.2}}}, 1e-12);
  }

  // Direct (columns 1-3) and reciprocal (columns 4-6) vectors, row by row.
  std::ifstream in(kShearedOutcar);
  REQUIRE(in.good());
  std::string line;
  bool found = false;
  Rows direct{}, recip{};
  while (std::getline(in, line)) {
    if (line.find("direct lattice vectors") == std::string::npos) continue;
    for (int i = 0; i < 3; ++i) {
      REQUIRE(std::getline(in, line));
      std::istringstream iss(line);
      REQUIRE(static_cast<bool>(iss >> direct[i][0] >> direct[i][1] >> direct[i][2] >>
                                recip[i][0] >> recip[i][1] >> recip[i][2]));
    }
    found = true;
    break;
  }
  REQUIRE(found);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("a_" << i << " . b_" << j);
      const double dot = direct[i][0] * recip[j][0] + direct[i][1] * recip[j][1] +
                         direct[i][2] * recip[j][2];
      CHECK(dot == Approx(i == j ? 1.0 : 0.0).margin(1e-8));
    }
}

// CASTEP ignores everything after '!', '#' or ';' on a line, block headers
// and unit lines included. The reader skipped only whole comment lines: a
// unit line with a trailing comment was taken for data ("LATTICE_CART needs
// exactly three lattice vectors, found 4"), and a block header such as
// "%BLOCK POSITIONS_FRAC!sites" named no known block, so its atoms were
// dropped without a word (the v1-dev reader matched it by substring).
TEST_CASE("CASTEP .cell reader: trailing comments on block headers and unit "
          "lines are ignored",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  const std::string path = "./tmp_cell_convention_comments.cell";
  const std::string text =
      "%BLOCK LATTICE_CART   ! lattice\n"
      "bohr ! units of the lattice\n"
      "  6.0  0.4 -0.2\n"
      "  2.0  7.0  0.3  ; a1\n"
      " -1.0  1.5  8.0  # a2\n"
      "%ENDBLOCK LATTICE_CART\n"
      "%BLOCK POSITIONS_FRAC!sites\n"
      "Si 0.10 0.20 0.30 ! first\n"
      "O  0.55 0.35 0.85\n"
      "%ENDBLOCK POSITIONS_FRAC\n";
  std::remove(path.c_str());
  {
    std::ofstream out(path);
    REQUIRE(out.good());
    out << text;
  }
  Structure st;
  REQUIRE_NOTHROW(st = read_cell_file(path));
  std::remove(path.c_str());
  const Rows a = {{{6.0 * kBohr, 0.4 * kBohr, -0.2 * kBohr},
                   {2.0 * kBohr, 7.0 * kBohr, 0.3 * kBohr},
                   {-1.0 * kBohr, 1.5 * kBohr, 8.0 * kBohr}}};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      INFO("cell(" << i << "," << k << ")");
      CHECK(st.cell(i, k) == Approx(a[i][k]).epsilon(1e-12).margin(1e-12));
    }
  REQUIRE(st.atoms.size() == 2);
  const double f[2][3] = {{0.10, 0.20, 0.30}, {0.55, 0.35, 0.85}};
  for (int n = 0; n < 2; ++n)
    for (int k = 0; k < 3; ++k) {
      INFO("atom " << n << " component " << k);
      const double want = f[n][0] * a[0][k] + f[n][1] * a[1][k] + f[n][2] * a[2][k];
      CHECK(st.atoms[n].position[k] == Approx(want).epsilon(1e-12).margin(1e-12));
    }

  // A unit line with a trailing comment in LATTICE_ABC and POSITIONS_ABS.
  const std::string text2 =
      "%BLOCK LATTICE_ABC\n"
      "ang ; the default, written out\n"
      "4.0 5.0 6.0\n"
      "90 90 90\n"
      "%ENDBLOCK LATTICE_ABC\n"
      "%BLOCK POSITIONS_ABS\n"
      "nm # positions in nanometres\n"
      "Si 0.1 0.2 0.3\n"
      "%ENDBLOCK POSITIONS_ABS\n";
  {
    std::ofstream out(path);
    REQUIRE(out.good());
    out << text2;
  }
  REQUIRE_NOTHROW(st = read_cell_file(path));
  std::remove(path.c_str());
  CHECK(st.cell(0, 0) == Approx(4.0));
  CHECK(st.cell(1, 1) == Approx(5.0));
  CHECK(st.cell(2, 2) == Approx(6.0));
  REQUIRE(st.atoms.size() == 1);
  CHECK(st.atoms[0].position[0] == Approx(1.0));
  CHECK(st.atoms[0].position[1] == Approx(2.0));
  CHECK(st.atoms[0].position[2] == Approx(3.0));
}

// A .cell file without POSITIONS_FRAC or POSITIONS_ABS was read as a
// structure with no atoms; CASTEP (and ASE) refuse it.
TEST_CASE("CASTEP .cell reader: a file without a positions block is refused",
          "[cell_convention][castep_cell]") {
  PeriodicTable::initialize();
  const std::string path = "./tmp_cell_convention_nopos.cell";
  {
    std::ofstream out(path);
    REQUIRE(out.good());
    out << "%BLOCK LATTICE_CART\n4 0 0\n0 4 0\n0 0 4\n%ENDBLOCK LATTICE_CART\n";
  }
  CastepCellReader r1;
  CHECK_THROWS_AS(r1.read(path), std::runtime_error);
  CastepCellReader r2;
  CHECK_THROWS_WITH(r2.read(path),
                    Catch::Contains("POSITIONS", Catch::CaseSensitive::No));
  std::remove(path.c_str());
}
