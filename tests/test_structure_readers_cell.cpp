// Value tests for the structure readers (--structure / -s) against the cell
// convention: row i of Structure::cell is lattice vector a_i (Angstrom), atom
// positions are Cartesian, r = cell^T f and f = (cell^T)^{-1} r.
//
// Every fixture (tests_data/cell_convention/{cif,poscar}_*) has a
// non-orthogonal cell, so a transposed lattice, or atoms that disagree with
// the lattice, is observable; on a diagonal cell both conventions agree. The
// expected values are hand-computed and cross-checked with ASE's CIF and
// POSCAR readers; none is computed with the reader under test. Finding ids
// refer to AUDIT-cell-convention-2026-09-30.md.
//
// AFLOW (vaspdb-4) is covered here through AflowReader::parseAflowJson on
// hand-written entries whose fields are all JSON strings. The typed form
// (numbers and arrays) is covered in test_aflow_reader_typed.cpp, and the
// entry the service serves (Ta1_ICSD_76152, saved in tests_data/aflow/ on
// 2026-10-06, typed) in test_aflow_live_fixture.cpp.

#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_readers/aflow_reader.h>
#include <tadah/mlip/structure_readers/cif_reader.h>
#include <tadah/mlip/structure_readers/structure_reader_selector.h>
#include <tadah/mlip/structure_readers/vasp_poscar_reader.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using tadah::core::PeriodicTable;
using namespace tadah::mlip;

namespace {

namespace fs = std::filesystem;
using V3 = std::array<double, 3>;

const std::string kCellDir = "./tests_data/cell_convention/";
const double kPi = 3.14159265358979323846;

V3 cell_row(const Structure &st, int i) {
  return {st.cell(i, 0), st.cell(i, 1), st.cell(i, 2)};
}
V3 atom_pos(const Structure &st, size_t n) {
  return {st.atoms[n].position[0], st.atoms[n].position[1],
          st.atoms[n].position[2]};
}
double dot(const V3 &u, const V3 &v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}
V3 cross(const V3 &u, const V3 &v) {
  return {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
          u[0] * v[1] - u[1] * v[0]};
}
double norm(const V3 &u) { return std::sqrt(dot(u, u)); }
double angle_deg(const V3 &u, const V3 &v) {
  return std::acos(dot(u, v) / (norm(u) * norm(v))) * 180.0 / kPi;
}
double wrap_unit(double x) { return x - std::round(x); }

// f = (cell^T)^{-1} r under the rows convention, written with the reciprocal
// vectors: f_k = r . (a_{k+1} x a_{k+2}) / (a_0 . (a_1 x a_2)).
V3 frac_of(const Structure &st, const V3 &r) {
  const V3 a0 = cell_row(st, 0), a1 = cell_row(st, 1), a2 = cell_row(st, 2);
  const double V = dot(a0, cross(a1, a2));
  return {dot(r, cross(a1, a2)) / V, dot(r, cross(a2, a0)) / V,
          dot(r, cross(a0, a1)) / V};
}

// Shortest interatomic distance over the periodic images n in [-3,3]^3,
// taking the cell rows as the lattice vectors. Brute force, no NNFinder.
double min_image_distance(const Structure &st) {
  const V3 a0 = cell_row(st, 0), a1 = cell_row(st, 1), a2 = cell_row(st, 2);
  double best = std::numeric_limits<double>::max();
  for (size_t i = 0; i < st.atoms.size(); ++i) {
    for (size_t j = 0; j < st.atoms.size(); ++j) {
      for (int n0 = -3; n0 <= 3; ++n0) {
        for (int n1 = -3; n1 <= 3; ++n1) {
          for (int n2 = -3; n2 <= 3; ++n2) {
            if (i == j && n0 == 0 && n1 == 0 && n2 == 0) continue;
            const V3 ri = atom_pos(st, i), rj = atom_pos(st, j);
            V3 d;
            for (int k = 0; k < 3; ++k)
              d[k] = rj[k] + n0 * a0[k] + n1 * a1[k] + n2 * a2[k] - ri[k];
            best = std::min(best, norm(d));
          }
        }
      }
    }
  }
  return best;
}

// ---------------------------------------------------------------- CIF / COD

struct CifSite {
  std::string symbol;
  V3 frac;
};
struct CifExpect {
  double a, b, c, alpha, beta, gamma;  // Angstrom, degrees
  std::vector<CifSite> sites;          // the full unit-cell orbit, by hand
  double nn;                           // minimum interatomic distance
};

// The standard setting that gemmi (ITfC B) and ASE both use: a along x, b in
// the xy plane, c completing a right-handed set. Rows are a, b, c.
std::array<V3, 3> standard_rows(const CifExpect &e) {
  const double ca = std::cos(e.alpha * kPi / 180.0);
  const double cb = std::cos(e.beta * kPi / 180.0);
  const double cg = std::cos(e.gamma * kPi / 180.0);
  const double sg = std::sin(e.gamma * kPi / 180.0);
  const double cx = e.c * cb;
  const double cy = e.c * (ca - cb * cg) / sg;
  return {V3{e.a, 0.0, 0.0}, V3{e.b * cg, e.b * sg, 0.0},
          V3{cx, cy, std::sqrt(e.c * e.c - cx * cx - cy * cy)}};
}

void check_cif_structure(const Structure &st, const CifExpect &e) {
  REQUIRE(st.atoms.size() == e.sites.size());

  // Row lengths and inter-row angles are the CIF cell parameters, whatever
  // the orientation.
  const V3 r0 = cell_row(st, 0), r1 = cell_row(st, 1), r2 = cell_row(st, 2);
  CHECK(norm(r0) == Approx(e.a).margin(1e-9));
  CHECK(norm(r1) == Approx(e.b).margin(1e-9));
  CHECK(norm(r2) == Approx(e.c).margin(1e-9));
  CHECK(angle_deg(r1, r2) == Approx(e.alpha).margin(1e-8));
  CHECK(angle_deg(r0, r2) == Approx(e.beta).margin(1e-8));
  CHECK(angle_deg(r0, r1) == Approx(e.gamma).margin(1e-8));

  // Row i is lattice vector a_i in the frame the atoms were placed in.
  const auto want = standard_rows(e);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("cell(" << i << "," << j << ")");
      CHECK(st.cell(i, j) == Approx(want[i][j]).margin(1e-9));
    }

  // (cell^T)^{-1} r of every atom is one CIF site (mod 1), each used once,
  // with that site's element.
  std::vector<bool> used(e.sites.size(), false);
  for (size_t n = 0; n < st.atoms.size(); ++n) {
    const V3 f = frac_of(st, atom_pos(st, n));
    INFO("atom " << n << " (" << st.atoms[n].symbol << ") frac = " << f[0]
                 << " " << f[1] << " " << f[2]);
    bool matched = false;
    for (size_t s = 0; s < e.sites.size() && !matched; ++s) {
      if (used[s] || e.sites[s].symbol != st.atoms[n].symbol) continue;
      bool same = true;
      for (int k = 0; k < 3; ++k)
        same = same && std::abs(wrap_unit(f[k] - e.sites[s].frac[k])) < 1e-6;
      if (same) used[s] = matched = true;
    }
    CHECK(matched);
  }

  CHECK(min_image_distance(st) == Approx(e.nn).margin(1e-6));
}

// hcp Mg, P 6_3/m m c, Mg on Wyckoff 2c: (1/3, 2/3, 1/4), (2/3, 1/3, 3/4).
// Nearest neighbours: sqrt(a^2/3 + c^2/4) = 3.1971 A between the layers
// (6x) and a = 3.2094 A in-plane (6x). The transposed lattice (the reader
// before the vaspdb-1 fix) gives a spurious 2.779 A neighbour.
CifExpect hcp_mg() {
  const double a = 3.2094, c = 5.2108;
  return {a,
          a,
          c,
          90.0,
          90.0,
          120.0,
          {{"Mg", {1.0 / 3.0, 2.0 / 3.0, 0.25}},
           {"Mg", {2.0 / 3.0, 1.0 / 3.0, 0.75}}},
          std::sqrt(a * a / 3.0 + c * c / 4.0)};
}

// Synthetic triclinic P -1 cell with Si and O on general positions; the
// inversion adds (-x,-y,-z). The shortest distance, 2.0467539 A (O-O across
// the b boundary), was computed with ASE from the CIF; the transposed
// lattice gives 1.904 A.
CifExpect triclinic_sio() {
  return {4.2,
          5.1,
          6.3,
          81.5,
          96.2,
          103.7,
          {{"Si", {0.12, 0.23, 0.34}},
           {"Si", {-0.12, -0.23, -0.34}},
           {"O", {0.41, 0.07, 0.62}},
           {"O", {-0.41, -0.07, -0.62}}},
          2.0467539137};
}

std::string slurp(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in.is_open());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

Structure cif_by_read(const std::string &path) {
  CifReader r;
  r.read(path);
  return r.getStructure();
}
// CODReader::read hands the downloaded CIF text to parseCifContents; this is
// that path without the network.
Structure cif_by_contents(const std::string &path) {
  CifReader r;
  r.parseCifContents(slurp(path));
  return r.getStructure();
}
// The route of `tadah predict|data print|data write -s x.cif`.
Structure by_selector(const std::string &path) {
  auto r = StructureReaderSelector::getReader(path);
  r->read(path);
  return r->getStructure();
}

// ------------------------------------------------------------------- POSCAR

// Reference structure of every poscar_tri_* fixture, hand-computed and
// cross-checked with ase.io.vasp.read_vasp (VASP semantics). Lattice rows
// R = 1.3 L with L = (3.0 0.2 0.1 / 0.8 2.6 -0.3 / 0.5 0.7 3.4),
// det R = 58.224894; Ti, O, O at fractional (0.1 0.2 0.9), (0.25 0.5 0.75),
// (0.6 0.1 0.35), r = f0 R0 + f1 R1 + f2 R2.
const double kRefCell[3][3] = {
    {3.9, 0.26, 0.13}, {1.04, 3.38, -0.39}, {0.65, 0.91, 4.42}};
const double kRefPos[3][3] = {{1.183, 1.521, 3.913},
                              {1.9825, 2.4375, 3.1525},
                              {2.6715, 0.8125, 1.586}};
const char *const kRefSymbol[3] = {"Ti", "O", "O"};
const double kRefVolume = 58.224894;

void check_poscar_reference(const Structure &st) {
  REQUIRE(st.atoms.size() == 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("cell(" << i << "," << j << ")");
      CHECK(st.cell(i, j) == Approx(kRefCell[i][j]).epsilon(0).margin(1e-10));
    }
  for (size_t n = 0; n < 3; ++n) {
    INFO("atom " << n);
    CHECK(st.atoms[n].symbol == kRefSymbol[n]);
    for (int k = 0; k < 3; ++k) {
      INFO("component " << k);
      CHECK(st.atoms[n].position[k] ==
            Approx(kRefPos[n][k]).epsilon(0).margin(1e-10));
    }
  }
}

Structure read_poscar(const std::string &path) {
  VaspPoscarReader r;
  r.read(path);
  return r.getStructure();
}

struct PoscarTempDir {
  fs::path dir;
  PoscarTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_readers_cell_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~PoscarTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

// Line layout of the poscar_tri_* fixtures: 0 title, 1 scale, 2-4 lattice,
// 5 symbols, 6 counts, 7 coordinate mode, 8-10 atoms.
constexpr size_t kScaleLine = 1;
constexpr size_t kModeLine = 7;

// Writes a copy of `fixture` in which line `at` is replaced by `lines`. With
// `sd_flags`, " T T F" is appended to the atom lines (a Selective dynamics
// file). Returns the path of the copy.
std::string write_variant(const PoscarTempDir &tmp, const std::string &fixture,
                          size_t at, const std::vector<std::string> &lines,
                          bool sd_flags) {
  std::vector<std::string> src;
  {
    std::istringstream in(slurp(kCellDir + fixture));
    std::string l;
    while (std::getline(in, l)) src.push_back(l);
  }
  REQUIRE(src.size() == 11);
  static int counter = 0;
  const fs::path out_path =
      tmp.dir / ("POSCAR_variant_" + std::to_string(counter++));
  std::ofstream out(out_path);
  for (size_t i = 0; i < src.size(); ++i) {
    if (i == at) {
      for (const auto &l : lines) out << l << '\n';
    } else if (sd_flags && i > kModeLine) {
      out << src[i] << "  T T F\n";
    } else {
      out << src[i] << '\n';
    }
  }
  return out_path.string();
}

}  // namespace

// ============================================================ CIF / COD ====

// vaspdb-1 / docs-1: CifReader copied gemmi's orthogonalisation matrix, whose
// COLUMNS are a, b, c, into Structure::cell untransposed, so every
// non-orthogonal CIF (and COD entry, which goes through parseCifContents) got
// the transposed lattice while its atoms were correct Cartesian positions.
// Before the fix: row 1 = (0, 2.779, 0) instead of (-a/2, a sqrt(3)/2, 0),
// the atoms sit at fractional (0, 0.667, 0.25) / (0.5, 0.622, 0.75), and the
// nearest neighbour is 2.779 A.
TEST_CASE("CIF reader: hcp Mg (gamma 120) stores lattice vector a_i as cell "
          "row i",
          "[structure_readers][cell_convention][cif]") {
  PeriodicTable::initialize();
  const std::string path = kCellDir + "cif_hcp_mg.cif";
  const CifExpect e = hcp_mg();

  // Hand-written rows for the hexagonal case, on top of the general formula.
  SECTION("CifReader::read") {
    const Structure st = cif_by_read(path);
    CHECK(st.cell(1, 0) == Approx(-e.a / 2.0).margin(1e-9));
    CHECK(st.cell(1, 1) == Approx(e.a * std::sqrt(3.0) / 2.0).margin(1e-9));
    CHECK(st.cell(2, 2) == Approx(e.c).margin(1e-9));
    check_cif_structure(st, e);
  }
  SECTION("CifReader::parseCifContents (the CODReader path)") {
    check_cif_structure(cif_by_contents(path), e);
  }
  SECTION("StructureReaderSelector (-s x.cif)") {
    check_cif_structure(by_selector(path), e);
  }
}

// vaspdb-1 / docs-1, triclinic: all three angles differ from 90 deg, so every
// off-diagonal element of the lattice is pinned (row 2 carries c cos(beta)
// and the alpha term). Before the fix the rows are the columns of gemmi's
// matrix, with lengths 4.42 / 5.02 / 6.21 instead of 4.2 / 5.1 / 6.3.
TEST_CASE("CIF reader: triclinic P-1 cell stores lattice vector a_i as cell "
          "row i",
          "[structure_readers][cell_convention][cif]") {
  PeriodicTable::initialize();
  const std::string path = kCellDir + "cif_triclinic_sio.cif";
  const CifExpect e = triclinic_sio();

  SECTION("CifReader::read") { check_cif_structure(cif_by_read(path), e); }
  SECTION("CifReader::parseCifContents (the CODReader path)") {
    check_cif_structure(cif_by_contents(path), e);
  }
  SECTION("StructureReaderSelector (-s x.cif)") {
    check_cif_structure(by_selector(path), e);
  }
}

// =============================================================== POSCAR ====

// Control: Direct coordinates with a universal scale were already right. Pins
// that lattice line k (times the scale) is cell row k, and r = cell^T f. Also
// the fully written-out reference (scale 1, Cartesian).
TEST_CASE("POSCAR reader: lattice lines times the scale are the cell rows",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  SECTION("Direct, scale 1.3") {
    check_poscar_reference(read_poscar(kCellDir + "poscar_tri_direct_scale.vasp"));
  }
  SECTION("Cartesian, scale 1.0") {
    check_poscar_reference(read_poscar(kCellDir + "poscar_tri_cart_unit.vasp"));
  }
}

// vaspdb-6: VASP multiplies Cartesian positions by the scaling factor, as it
// does the lattice. Before the fix the positions were copied unscaled, so the
// atoms were 1/1.3 of where they belong in the (scaled) cell.
TEST_CASE("POSCAR reader: Cartesian positions are multiplied by the "
          "universal scale",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  check_poscar_reference(read_poscar(kCellDir + "poscar_tri_cart_scale.vasp"));
}

// vaspdb-5: a negative scale is the target cell volume, s = cbrt(V / |det L|)
// (= 1.3 here), applied to the lattice and, in Cartesian mode, the positions.
// Before the fix it was used as a plain multiplier: cell = -58.2 L, a huge
// left-handed cell with a negative get_volume().
TEST_CASE("POSCAR reader: a negative scale is the target cell volume",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  SECTION("Direct") {
    const Structure st =
        read_poscar(kCellDir + "poscar_tri_direct_negvol.vasp");
    check_poscar_reference(st);
    CHECK(st.get_volume() == Approx(kRefVolume).margin(1e-9));
  }
  SECTION("Cartesian") {
    const Structure st = read_poscar(kCellDir + "poscar_tri_cart_negvol.vasp");
    check_poscar_reference(st);
    CHECK(st.get_volume() == Approx(kRefVolume).margin(1e-9));
  }
}

// vaspdb-5 (and vaspdb-6 for Cartesian): three scale factors s_x s_y s_z
// multiply Cartesian component j of every lattice vector by s_j; in Cartesian
// mode component j of every position is multiplied by s_j too (VASP; ASE
// read_vasp: positions *= scale). Here s = (1.25, 1.3, 1.6). Before the fix
// only the first factor was read and used as a universal scale.
TEST_CASE("POSCAR reader: three scale factors scale each Cartesian component",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  SECTION("Direct") {
    check_poscar_reference(read_poscar(kCellDir + "poscar_tri_direct_3scale.vasp"));
  }
  SECTION("Cartesian") {
    check_poscar_reference(read_poscar(kCellDir + "poscar_tri_cart_3scale.vasp"));
  }
}

// vaspdb-11: VASP takes the coordinate mode from the first character of the
// line: C, c, K or k means Cartesian, anything else Direct. Before the fix the
// line was searched for "cart", so C, K and Kartesian were read as Direct and
// "Direct coordinates (not Cartesian)" as Cartesian. The Cartesian variants
// use the scale-1 fixture and the Direct ones the scale-1.3 Direct fixture,
// so only the mode decision can fail here.
TEST_CASE("POSCAR reader: the coordinate mode is the first character of its "
          "line",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  PoscarTempDir tmp;
  struct Case {
    const char *fixture;
    const char *mode_line;
  };
  const std::vector<Case> cases = {
      {"poscar_tri_cart_unit.vasp", "Cartesian"},
      {"poscar_tri_cart_unit.vasp", "cartesian"},
      {"poscar_tri_cart_unit.vasp", "C"},
      {"poscar_tri_cart_unit.vasp", "c"},
      {"poscar_tri_cart_unit.vasp", "K"},
      {"poscar_tri_cart_unit.vasp", "k"},
      {"poscar_tri_cart_unit.vasp", "Kartesian"},
      {"poscar_tri_direct_scale.vasp", "Direct"},
      {"poscar_tri_direct_scale.vasp", "direct"},
      {"poscar_tri_direct_scale.vasp", "D"},
      {"poscar_tri_direct_scale.vasp", "Direct coordinates (not Cartesian)"},
  };
  for (const auto &c : cases) {
    INFO("mode line '" << c.mode_line << "' in a copy of " << c.fixture);
    const std::string path =
        write_variant(tmp, c.fixture, kModeLine, {c.mode_line}, false);
    check_poscar_reference(read_poscar(path));
  }
}

// vaspdb-11: the optional Selective dynamics line is recognised by its first
// character, S or s; the mode line follows it. Before the fix only a line
// containing "selective" was recognised, so "Sel dyn" or "s" was taken as the
// mode line and the real mode line as the first atom (the read throws), and
// "K" after "Selective dynamics" was read as Direct.
TEST_CASE("POSCAR reader: Selective dynamics is recognised by a leading S",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  PoscarTempDir tmp;
  struct Case {
    const char *fixture;
    std::vector<std::string> lines;
  };
  const std::vector<Case> cases = {
      {"poscar_tri_direct_scale.vasp", {"Selective dynamics", "Direct"}},
      {"poscar_tri_cart_unit.vasp", {"Selective dynamics", "Cartesian"}},
      {"poscar_tri_direct_scale.vasp", {"Sel dyn", "Direct"}},
      {"poscar_tri_cart_unit.vasp", {"s", "Cartesian"}},
      {"poscar_tri_cart_unit.vasp", {"Selective dynamics", "K"}},
      {"poscar_tri_cart_scale.vasp", {"Sel dyn", "Kartesian"}},
  };
  for (const auto &c : cases) {
    INFO("lines '" << c.lines[0] << "' / '" << c.lines[1]
                   << "' in a copy of " << c.fixture);
    const std::string path =
        write_variant(tmp, c.fixture, kModeLine, c.lines, true);
    Structure st;
    try {
      st = read_poscar(path);
    } catch (const std::exception &ex) {
      FAIL_CHECK("VaspPoscarReader threw: " << ex.what());
      continue;
    }
    check_poscar_reference(st);
  }
}

// vaspdb-5: a scale line VASP cannot interpret is an error, not a silent
// fallback. Before the fix only the first token was read, "0" gave a zero
// cell, and an unparsable line silently became 1.0. The rules follow ASE's
// parse_poscar_scaling_factor: one number, or three positive numbers.
TEST_CASE("POSCAR reader: an invalid scale line is rejected",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  PoscarTempDir tmp;
  const std::vector<std::string> bad = {
      "0",                // zero universal scale: a degenerate cell
      "1.0 -1.0 1.0",     // three factors of mixed sign
      "-1.0 -1.0 -1.0",   // three factors must be positive
      "1.3 1.3",          // two factors: neither form
      "scale",            // no number at all
  };
  for (const auto &line : bad) {
    INFO("scale line '" << line << "'");
    const std::string path = write_variant(
        tmp, "poscar_tri_direct_scale.vasp", kScaleLine, {line}, false);
    CHECK_THROWS_AS(read_poscar(path), std::runtime_error);
  }
}

// A POSCAR without the element-symbol line (the VASP 4 layout) names no
// element; the reader filled the symbols with "X", which is no element, so
// it failed with "Unknown symbol: X". It now says what is missing ("...
// (the VASP 4 layout); an element-symbol line above the counts gives
// them."; before the fix that states it, "...; add the element-symbol line
// above the counts."). A symbol line and a count line of different lengths
// read past the symbols.
TEST_CASE("POSCAR reader: a missing element line or a symbol/count mismatch "
          "is refused",
          "[structure_readers][cell_convention][poscar]") {
  PeriodicTable::initialize();
  PoscarTempDir tmp;
  const size_t kSymbolLine = 5;
  {
    INFO("no element-symbol line");
    std::vector<std::string> src;
    {
      std::istringstream in(slurp(kCellDir + "poscar_tri_direct_scale.vasp"));
      std::string l;
      while (std::getline(in, l)) src.push_back(l);
    }
    const fs::path path = tmp.dir / "POSCAR_no_symbols";
    std::ofstream out(path);
    for (size_t i = 0; i < src.size(); ++i)
      if (i != kSymbolLine) out << src[i] << '\n';
    out.close();
    CHECK_THROWS_AS(read_poscar(path.string()), std::runtime_error);
    CHECK_THROWS_WITH(read_poscar(path.string()),
                      Catch::Contains("element", Catch::CaseSensitive::No));
    CHECK_THROWS_WITH(read_poscar(path.string()),
                      Catch::EndsWith("(the VASP 4 layout); an element-symbol "
                                      "line above the counts gives them."));
  }
  {
    INFO("two symbols, three counts");
    const std::string path = write_variant(
        tmp, "poscar_tri_direct_scale.vasp", kSymbolLine + 1, {"1 1 1"}, false);
    CHECK_THROWS_AS(read_poscar(path), std::runtime_error);
  }
}

// ------------------------------------------------------------------ AFLOW

// Rock-salt NaCl written the way the reader parses an AFLOW entry (every
// field a string): "geometry" is a, b, c, alpha, beta,
// gamma of the primitive fcc cell (a = 5.64/sqrt(2), 60 degrees),
// "positions_fractional" refer to that geometry, and "positions_cartesian"
// are in AFLOW's own frame (a1 = (0, a_c/2, a_c/2), ...), which is not the
// a-along-x frame the reader builds the cell in. Placing the Cartesian
// positions in the rebuilt cell put Na at fractional (0.010, 0.528, 0.866)
// with a 1.91 A Na-Cl distance (vaspdb-4). The rebuilt rows are
// (a,0,0), (a/2, a sqrt(3)/2, 0), (a/2, a/(2 sqrt 3), a sqrt(2/3)).
TEST_CASE("AFLOW reader: atoms are placed from positions_fractional in the "
          "cell built from geometry",
          "[structure_readers][cell_convention][aflow]") {
  PeriodicTable::initialize();
  const std::string json =
      R"({"auid":"aflow:test","compound":"ClNa","species":"Cl,Na",)"
      R"("composition":"1,1",)"
      R"("geometry":"3.98808224589,3.98808224589,3.98808224589,60,60,60",)"
      R"("positions_fractional":"0,0,0;0.5,0.5,0.5",)"
      R"("positions_cartesian":"0,0,0;2.82,2.82,2.82"})";
  AflowReader reader;
  reader.parseAflowJson(json);
  const Structure st = reader.getStructure();
  REQUIRE(st.atoms.size() == 2);
  CHECK(st.atoms[0].symbol == std::string("Cl"));
  CHECK(st.atoms[1].symbol == std::string("Na"));

  const double a = 3.98808224589;
  const V3 want_rows[3] = {{a, 0.0, 0.0},
                           {0.5 * a, 0.5 * std::sqrt(3.0) * a, 0.0},
                           {0.5 * a, a / (2.0 * std::sqrt(3.0)),
                            a * std::sqrt(2.0 / 3.0)}};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) {
      INFO("cell(" << i << "," << k << ")");
      CHECK(st.cell(i, k) == Approx(want_rows[i][k]).margin(1e-9));
    }
  const V3 f0 = frac_of(st, atom_pos(st, 0));
  const V3 f1 = frac_of(st, atom_pos(st, 1));
  for (int k = 0; k < 3; ++k) {
    CHECK(f0[k] == Approx(0.0).margin(1e-12));
    CHECK(f1[k] == Approx(0.5).margin(1e-12));
  }
  // Na-Cl nearest distance is half the conventional cube edge.
  CHECK(min_image_distance(st) == Approx(5.64 / 2.0).margin(1e-6));
}

TEST_CASE("AFLOW reader: an entry without geometry or positions_fractional "
          "is refused",
          "[structure_readers][cell_convention][aflow]") {
  // A missing "geometry" used to give a silent 1x1x1 A cube; a missing
  // position list gave a structure with no atoms.
  PeriodicTable::initialize();
  const std::string no_geometry =
      R"({"auid":"aflow:test","species":"Cl,Na","composition":"1,1",)"
      R"("positions_fractional":"0,0,0;0.5,0.5,0.5"})";
  const std::string no_positions =
      R"({"auid":"aflow:test","species":"Cl,Na","composition":"1,1",)"
      R"("geometry":"3.98808224589,3.98808224589,3.98808224589,60,60,60"})";
  AflowReader r1, r2;
  CHECK_THROWS_AS(r1.parseAflowJson(no_geometry), std::runtime_error);
  CHECK_THROWS_AS(r2.parseAflowJson(no_positions), std::runtime_error);
}

TEST_CASE("AFLOW reader: an entry whose counts, positions or angles do not "
          "fit is refused",
          "[structure_readers][cell_convention][aflow]") {
  // The reader claimed to label the atoms "X" when species, composition and
  // positions disagreed, but "X" is no element (it threw "Unknown symbol:
  // X"); two species with one composition count read past the counts, and
  // one species with two counts dropped the second site. A position with
  // two numbers became (f0, f1, 0), and angles no cell has were clamped to a
  // flat cell.
  PeriodicTable::initialize();
  const std::string geom =
      R"("geometry":"3.98808224589,3.98808224589,3.98808224589,60,60,60",)";
  const std::string pos = R"("positions_fractional":"0,0,0;0.5,0.5,0.5")";
  const std::vector<std::string> bad = {
      // sum of the composition differs from the number of positions
      R"({"auid":"t1","species":"Cl,Na","composition":"1,2",)" + geom + pos + "}",
      // two species, one composition count
      R"({"auid":"t2","species":"Cl,Na","composition":"2",)" + geom + pos + "}",
      // one species, two composition counts
      R"({"auid":"t3","species":"Na","composition":"1,1",)" + geom + pos + "}",
      // a position with two numbers
      R"({"auid":"t4","species":"Cl,Na","composition":"1,1",)" + geom +
          R"("positions_fractional":"0,0,0;0.5,0.5"})",
      // angles no cell has (gamma > alpha + beta)
      R"({"auid":"t5","species":"Cl,Na","composition":"1,1",)"
      R"("geometry":"4,4,4,60,60,150",)" + pos + "}",
  };
  for (const auto &json : bad) {
    INFO(json);
    AflowReader reader;
    CHECK_THROWS_AS(reader.parseAflowJson(json), std::runtime_error);
  }
}
