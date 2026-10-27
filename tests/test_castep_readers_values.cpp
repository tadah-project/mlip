// Value tests for the CASTEP readers and the CASTEP cell writer (stress
// convention and reader bugs).
//
// Stored stress convention: Cauchy stress, positive under tension, in eV/A^3,
// sigma = (1/V) dE/d(strain). CASTEP's '<-- S' block (.md/.geom) and the
// .castep Stress Tensor box already have that sign; the .md '<-- S' block also
// holds the kinetic stress -(1/V) sum_i m_i v_i (x) v_i, which the .md reader
// removes.
//
// Fixtures:
//  - ./tests_data/valid_castep_md/test1.md            (8 Si, 2 MD frames)
//  - ./tests_data/valid_castep_geom/{60,95}.geom       (H8Ra2, 12 and 8 frames)
//  - ./tests_data/valid_castep_castep/geom.castep      (H8Ra2 LBFGS run)
//  - ./tests_data/cell_convention/{md_triclinic.md,geom_triclinic.geom}
//  - ./tests_data/castep_readers/geom_lbfgs_revert.castep (new, synthetic):
//    geom.castep lines 1-605 (iteration 0 with its finite-basis SCFs, then
//    the LBFGS trial cell of iteration 1 and its SCF), followed by
//    'LBFGS: reverting to earlier configuration', a printout of the
//    iteration-0 cell and positions with no SCF after it, and
//    'LBFGS: finished iteration 1'. No real CASTEP output with a revert was
//    available; the layout follows the reader's own comment on reverts.
// Every other variant (tagged species labels, a one-frame .geom, truncated or
// malformed frames, a changed kinetic energy) is written by the test from
// these fixtures into ./tmp_castep_rv_* files, which are removed afterwards.
//
// Reference values were computed independently with numpy (and ASE 3.25 for
// the kinetic stress) from the fixture text, with the reader's CODATA 2018
// constants: 1 Ha = 27.211386245988 eV, 1 bohr = 0.529177210903 A, and
// m_u/m_e = 1822.888486209.

#include "catch2/catch.hpp"

#include "tadah/core/element.h"
#include "tadah/core/logger.h"
#include "tadah/core/periodic_table.h"

#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/castep_castep_reader.h>
#include <tadah/mlip/dataset_readers/castep_geom_reader.h>
#include <tadah/mlip/dataset_readers/castep_md_reader.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_readers/castep_cell_reader.h>
#include <tadah/mlip/structure_writers/castep_cell_writer.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using tadah::core::Element;
using tadah::core::Logger;
using tadah::core::LogLevel;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::CastepCastepReader;
using tadah::mlip::CastepCellReader;
using tadah::mlip::CastepCellWriter;
using tadah::mlip::CastepGeomReader;
using tadah::mlip::CastepMDReader;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

namespace {

constexpr double kHartree = 27.211386245988;  // eV per hartree (CODATA 2018)
constexpr double kGPa = 160.21766208;         // GPa per eV/A^3

const std::string kMdTest1 = "./tests_data/valid_castep_md/test1.md";
const std::string kGeom60 = "./tests_data/valid_castep_geom/60.geom";
const std::string kGeom95 = "./tests_data/valid_castep_geom/95.geom";
const std::string kGeomCastep = "./tests_data/valid_castep_castep/geom.castep";
const std::string kMdTriclinic = "./tests_data/cell_convention/md_triclinic.md";
const std::string kGeomTriclinic = "./tests_data/cell_convention/geom_triclinic.geom";
const std::string kRevert = "./tests_data/castep_readers/geom_lbfgs_revert.castep";

// test1.md frame energies (first '<-- E' column, hartree) and times (a.u.).
constexpr double kTest1E0 = -3.1438056609022318e1;
constexpr double kTest1E1 = -3.1439975473797670e1;

// A file in the working directory, removed on construction and destruction.
struct TempFile {
  std::string path;
  explicit TempFile(const std::string &name) : path("./" + name) {
    std::remove(path.c_str());
  }
  ~TempFile() { std::remove(path.c_str()); }
  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;
  std::string basename() const { return path.substr(2); }
  void write_lines(const std::vector<std::string> &lines) const {
    std::ofstream out(path);
    REQUIRE(out.good());
    for (const auto &l : lines) out << l << '\n';
  }
  void write_text(const std::string &text) const {
    std::ofstream out(path);
    REQUIRE(out.good());
    out << text;
  }
};

std::vector<std::string> read_lines(const std::string &path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  std::vector<std::string> lines;
  std::string l;
  while (std::getline(in, l)) lines.push_back(l);
  return lines;
}

bool ends_with_tag(const std::string &line, const std::string &tag) {
  std::string t = line;
  while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
  const std::string want = "<-- " + tag;
  return t.size() >= want.size() && t.compare(t.size() - want.size(), want.size(), want) == 0;
}

// Index of the n-th (1-based) line carrying `tag`, at or after `from`.
size_t nth_tagged(const std::vector<std::string> &lines, const std::string &tag,
                  size_t n, size_t from = 0) {
  size_t seen = 0;
  for (size_t i = from; i < lines.size(); ++i)
    if (ends_with_tag(lines[i], tag) && ++seen == n) return i;
  FAIL("fixture has fewer than " << n << " '<-- " << tag << "' lines");
  return 0;
}

// Replaces the whitespace-separated token `k` (0-based) of `line`, keeping
// the rest of the line (the '<-- X' tag included) as it is.
std::string replace_token(const std::string &line, size_t k, const std::string &with) {
  size_t pos = 0;
  for (size_t n = 0;; ++n) {
    pos = line.find_first_not_of(" \t", pos);
    REQUIRE(pos != std::string::npos);
    const size_t end = std::min(line.find_first_of(" \t", pos), line.size());
    if (n == k) return line.substr(0, pos) + with + line.substr(end);
    pos = end;
  }
}

// Replaces every occurrence of `from` by `to` in every line.
std::vector<std::string> substitute(std::vector<std::string> lines,
                                    const std::string &from, const std::string &to) {
  size_t count = 0;
  for (auto &l : lines)
    for (size_t p = l.find(from); p != std::string::npos; p = l.find(from, p + to.size())) {
      l.replace(p, from.size(), to);
      ++count;
    }
  REQUIRE(count > 0);
  return lines;
}

// `lines` without the lines that contain `what`; REQUIREs `count` such
// lines (0: any number above 0).
std::vector<std::string> drop_lines(const std::vector<std::string> &lines,
                                    const std::string &what, size_t count = 0) {
  std::vector<std::string> out;
  size_t dropped = 0;
  for (const auto &l : lines) {
    if (l.find(what) != std::string::npos) {
      ++dropped;
      continue;
    }
    out.push_back(l);
  }
  if (count) REQUIRE(dropped == count);
  REQUIRE(dropped > 0);
  return out;
}

// Two structures hold the same data, exactly: energy, T, weights, cell,
// stress and atoms.
void check_same_structure(const Structure &a, const Structure &b) {
  CHECK(a.energy == b.energy);
  CHECK(a.T == b.T);
  CHECK(a.eweight == b.eweight);
  CHECK(a.fweight == b.fweight);
  CHECK(a.sweight == b.sweight);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("(" << i << "," << j << ")");
      CHECK(a.cell(i, j) == b.cell(i, j));
      CHECK(a.stress(i, j) == b.stress(i, j));
    }
  REQUIRE(a.natoms() == b.natoms());
  for (size_t k = 0; k < a.natoms(); ++k) {
    INFO("atom " << k);
    CHECK(a.atoms[k].symbol == b.atoms[k].symbol);
    for (int c = 0; c < 3; ++c) {
      CHECK(a.atoms[k].position[c] == b.atoms[k].position[c]);
      CHECK(a.atoms[k].force[c] == b.atoms[k].force[c]);
    }
  }
}

// The lines [banner, closing] of the boxed block whose banner is
// lines[banner] (a ' ***... Title ...***' line): up to the next line, after
// the banner, that starts with ' ****'.
std::vector<std::string> block_at(const std::vector<std::string> &lines, size_t banner) {
  REQUIRE(banner < lines.size());
  for (size_t i = banner + 1; i < lines.size(); ++i)
    if (lines[i].rfind(" ****", 0) == 0)
      return std::vector<std::string>(lines.begin() + banner, lines.begin() + i + 1);
  FAIL("no closing line after line " << banner + 1);
  return {};
}

// `lines` without the block whose banner is lines[banner].
std::vector<std::string> erase_block(std::vector<std::string> lines, size_t banner) {
  const size_t n = block_at(lines, banner).size();
  lines.erase(lines.begin() + banner, lines.begin() + banner + n);
  return lines;
}

// The value after '=' in the energy line `line`.
double energy_after_equals(const std::string &line) {
  const auto p = line.find('=');
  REQUIRE(p != std::string::npos);
  return std::stod(line.substr(p + 1));
}

bool forces_are_zero(const Structure &st) {
  for (const auto &a : st.atoms)
    for (int c = 0; c < 3; ++c)
      if (a.force[c] != 0.0) return false;
  return true;
}

bool stress_is_zero(const Structure &st) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (st.stress(i, j) != 0.0) return false;
  return true;
}

template <class Reader>
void read_into(StructureDB &db, const std::string &path) {
  Reader reader(db, path);
  reader.parse_data();
}

// The std::runtime_error message of parsing `path`, or "" if parsing
// succeeded.
template <class Reader>
std::string parse_error(const std::string &path) {
  StructureDB db;
  try {
    read_into<Reader>(db, path);
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "";
}

// Parses `path` with the Logger at WARNING and std::cout (the Logger's
// console) and std::cerr captured; returns everything printed.
template <class Reader>
std::string parse_capturing(StructureDB &db, const std::string &path) {
  auto &log = Logger::instance();
  const LogLevel old_level = log.getLogLevel();
  log.setLogLevel(LogLevel::WARNING);
  std::ostringstream out, err;
  std::streambuf *old_out = std::cout.rdbuf(out.rdbuf());
  std::streambuf *old_err = std::cerr.rdbuf(err.rdbuf());
  try {
    read_into<Reader>(db, path);
  } catch (...) {
    std::cout.rdbuf(old_out);
    std::cerr.rdbuf(old_err);
    log.setLogLevel(old_level);
    throw;
  }
  std::cout.rdbuf(old_out);
  std::cerr.rdbuf(old_err);
  log.setLogLevel(old_level);
  return out.str() + err.str();
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

size_t count_ci(const std::string &text, const std::string &what) {
  const std::string t = lower(text), w = lower(what);
  size_t n = 0;
  for (size_t p = t.find(w); p != std::string::npos; p = t.find(w, p + w.size())) ++n;
  return n;
}

// True if `word` occurs in `text` with no letter or digit on either side.
bool contains_word(const std::string &text, const std::string &word) {
  for (size_t p = text.find(word); p != std::string::npos; p = text.find(word, p + 1)) {
    const bool left_ok = p == 0 || !std::isalnum(static_cast<unsigned char>(text[p - 1]));
    const size_t e = p + word.size();
    const bool right_ok = e >= text.size() || !std::isalnum(static_cast<unsigned char>(text[e]));
    if (left_ok && right_ok) return true;
  }
  return false;
}

// True if `msg` names `key` followed by the number n (case-insensitive,
// separators ' ', ':', '#', '=' allowed), e.g. "frame 6", "line: 154".
bool names_number(const std::string &msg, const std::string &key, long n) {
  const std::string m = lower(msg), k = lower(key);
  for (size_t p = m.find(k); p != std::string::npos; p = m.find(k, p + 1)) {
    size_t q = p + k.size();
    while (q < m.size() && (m[q] == ' ' || m[q] == ':' || m[q] == '#' || m[q] == '=')) ++q;
    size_t d = q;
    while (d < m.size() && std::isdigit(static_cast<unsigned char>(m[d]))) ++d;
    if (d > q && std::stol(m.substr(q, d - q)) == n) return true;
  }
  return false;
}

// The structure whose energy is e_ha hartree (within 1e-6 eV), or nullptr.
const Structure *by_energy(const StructureDB &db, double e_ha) {
  for (size_t s = 0; s < db.size(); ++s)
    if (std::abs(db(s).energy - e_ha * kHartree) < 1e-6) return &db(s);
  return nullptr;
}

void check_labels(const StructureDB &db, const std::string &file,
                  const std::vector<std::string> &endings) {
  CHECK(db.size() == endings.size());
  if (db.size() != endings.size() || db.size() == 0) return;
  for (size_t s = 0; s < db.size(); ++s) {
    INFO("structure " << s << " label '" << db(s).label << "'");
    CHECK_FALSE(db(s).label.empty());
    CHECK_THAT(db(s).label, Catch::EndsWith(endings[s]));
  }
  INFO("first label '" << db(0).label << "'");
  CHECK_THAT(db(0).label, Catch::Contains(file));
}

}  // namespace

// --------------------------------------------------------------------
// .md and .geom: the first frame
// --------------------------------------------------------------------

// Pins: CastepMDReader returns every frame of test1.md, the first included,
// each labelled with its time (time line x 2.418884326505e-5 ps: 0 and
// 82.682746688379041 a.u.), the first label naming the file; no label is
// empty. Energies are the first '<-- E' column of each frame.
// Before the fix: the header loop consumed frame 0's '<-- E' line without
// setting E_flag, so every later line of frame 0 was flagged as an error and
// the frame dropped silently: 1 structure, labelled 'Filename: ... | Time:
// 0.002000 ps'.
TEST_CASE("CASTEP .md reader: test1.md gives both frames each labelled",
          "[castep_readers][md][first_frame]") {
  PeriodicTable::initialize();
  StructureDB db;
  read_into<CastepMDReader>(db, kMdTest1);
  CHECK(db.size() == 2);
  REQUIRE(db.size() >= 1);
  CHECK(db(0).energy == Approx(kTest1E0 * kHartree).epsilon(1e-14));
  REQUIRE(db.size() == 2);
  CHECK(db(1).energy == Approx(kTest1E1 * kHartree).epsilon(1e-14));
  CHECK(db(0).natoms() == 8);
  CHECK(db(1).natoms() == 8);
  check_labels(db, kMdTest1, {"Time: 0.000000 ps", "Time: 0.002000 ps"});
}

// Pins: as above for the hand-made triclinic trajectory: 2 structures,
// labelled with times 0 and 41.341373336 a.u. = 0.001 ps.
// Before the fix: 1 structure (frame 0 dropped).
// The fixture's kinetic-energy column (third '<-- E' column, 0.1111 and
// 0.1222 Ha) is not 1/2 sum m v^2 of its velocities (7.2496480171486619e-3 Ha
// in both frames with Si 28.0855 and C 12.011; 2.8e-7 lower with C 12.0107),
// so the fixed reader's kinetic-energy check refuses it until that column is
// set to 7.2496480171486619E-003 (test_dataset_readers_cell.cpp reads the same
// file).
TEST_CASE("CASTEP .md reader: md_triclinic.md gives both frames each labelled",
          "[castep_readers][md][first_frame]") {
  PeriodicTable::initialize();
  StructureDB db;
  read_into<CastepMDReader>(db, kMdTriclinic);
  CHECK(db.size() == 2);
  REQUIRE(db.size() >= 1);
  CHECK(db(0).energy == Approx(-2.1111111111111111e1 * kHartree).epsilon(1e-14));
  check_labels(db, kMdTriclinic, {"Time: 0.000000 ps", "Time: 0.001000 ps"});
}

// Pins: CastepGeomReader returns every frame of each .geom file (2, 12 and
// 8 frames), the first included, labelled 'Step: <n>' with n the frame's
// step number (0 for the first), the first label naming the file; first and
// last energies are those of the file's first and last '<-- E' lines.
// Before the fix: 1, 11 and 7 structures (the first frame dropped).
TEST_CASE("CASTEP .geom reader: every frame is read with the first included",
          "[castep_readers][geom][first_frame]") {
  PeriodicTable::initialize();
  struct Case {
    std::string file;
    size_t frames;
    double e_first_ha, e_last_ha;
  };
  const std::vector<Case> cases = {
      {kGeomTriclinic, 2, -1.2345678901234567e2, -1.2346789012345678e2},
      {kGeom60, 12, -9.8996404913333208e1, -9.8976972393943711e1},
      {kGeom95, 8, -9.8862384287753613e1, -9.8843509724125283e1}};
  for (const auto &c : cases) {
    INFO(c.file);
    StructureDB db;
    read_into<CastepGeomReader>(db, c.file);
    CHECK(db.size() == c.frames);
    REQUIRE(db.size() >= 1);
    CHECK(db(0).energy == Approx(c.e_first_ha * kHartree).epsilon(1e-14));
    CHECK(db(db.size() - 1).energy == Approx(c.e_last_ha * kHartree).epsilon(1e-14));
    std::vector<std::string> endings;
    for (size_t k = 0; k < c.frames; ++k) endings.push_back("Step: " + std::to_string(k));
    check_labels(db, c.file, endings);
  }
}

// Pins: a .geom file with a single frame (60.geom's header and first frame)
// gives one structure with that frame's energy and a non-empty label.
// Before the fix: 0 structures and no error ('tadah data convert' exits 0
// with an empty output file).
TEST_CASE("CASTEP .geom reader: a one-frame file gives one structure",
          "[castep_readers][geom][first_frame]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines(kGeom60);
  const size_t second = nth_tagged(lines, "c", 2);
  // Keep the header, frame 0 and the blank line that ends it.
  lines.resize(second);
  TempFile tmp("tmp_castep_rv_one_frame.geom");
  tmp.write_lines(lines);

  StructureDB db;
  read_into<CastepGeomReader>(db, tmp.path);
  REQUIRE(db.size() == 1);
  CHECK(db(0).energy == Approx(-9.8996404913333208e1 * kHartree).epsilon(1e-14));
  CHECK(db(0).natoms() == 10);
  CHECK_FALSE(db(0).label.empty());
  CHECK_THAT(db(0).label, Catch::EndsWith("Step: 0"));
}

// --------------------------------------------------------------------
// .md kinetic stress and .geom stress
// --------------------------------------------------------------------

// Pins: the .md reader stores sigma = S + (1/V) sum_i m_i v_i (x) v_i, in
// eV/A^3 (tension positive): S is the '<-- S' block, v_i the raw '<-- V'
// velocities (no centre-of-mass removal), m_i the standard mass of the
// species (Si 28.0855 u, which reproduces the file's kinetic energy to
// 2.2e-10), V = |det h|. All six components of both frames of test1.md.
// The references come from numpy on the file text; ASE 3.25's
// read_castep_md (which subtracts atoms.get_kinetic_stress()) gives the same
// frame-0 tensor (the only frame it reads from this file) to 7e-8 relative
// with its Si mass set to 28.0855, the difference being ASE's older CASTEP
// unit table.
// The mass is CASTEP's default Si mass, 28.0855 u. CORE's PeriodicTable has
// 28.0850 u for Si (as ASE has), 1.78e-5 relative lower: the reader computes
// sum m v (x) v with the periodic-table masses and scales it so that its
// trace is twice the file's kinetic energy (the third '<-- E' column), which
// is exactly what CASTEP added. For one species that gives CASTEP's tensor
// whatever the mass table says.
// Tolerance 1e-9 relative: with the reader's CODATA 2018 constants the only
// freedom left is m_u/m_e (CODATA 2014/2018 or ASE differ by <= 5e-10, and
// K/sigma <= 0.28 for every component here) and rounding (1e-15). The
// tolerance is 1e3 x below the effect of Si 28.085 (2.2e-6 relative), and the
// alternatives are far off: no removal (frame 1 xx 0.019206, xy 0.007294),
// isotropic N k T/V (xx 0.021881, xy 0.007294), velocities relative to the
// centre of mass (80-98% of each entry of K here is centre-of-mass drift).
// Before the fix: frame 0 was not stored, and frame 1 held S minus N k_b T/V
// with T already in hartree (k_B applied twice), subtracted from all nine
// entries (Matrix3d -= double): xx 0.019206, xy 0.007294, i.e. S to 9e-9.
TEST_CASE("CASTEP .md reader: the kinetic stress is removed exactly",
          "[castep_readers][md][kinetic_stress]") {
  PeriodicTable::initialize();
  struct Expected {
    double e_ha;
    double xx, yy, zz, xy, xz, yz;
  };
  const std::vector<Expected> frames = {
      {kTest1E0, 0.012732017948783116, -0.037888902881186591, -0.047097428692507229,
       0.0089791883295869748, 0.016971366965646082, 0.030122260345921095},
      {kTest1E1, 0.021855833893013894, -0.026678582592068727, -0.035269232542746685,
       0.0095828480783205737, 0.01694614443622617, 0.029609838757005444}};
  StructureDB db;
  read_into<CastepMDReader>(db, kMdTest1);
  CHECK(db.size() == 2);
  for (size_t f = 0; f < frames.size(); ++f) {
    INFO("frame " << f);
    const Structure *st = by_energy(db, frames[f].e_ha);
    CHECK(st != nullptr);
    if (!st) continue;
    const auto &s = st->stress;
    const auto &e = frames[f];
    CHECK(s(0, 0) == Approx(e.xx).epsilon(1e-9));
    CHECK(s(1, 1) == Approx(e.yy).epsilon(1e-9));
    CHECK(s(2, 2) == Approx(e.zz).epsilon(1e-9));
    CHECK(s(0, 1) == Approx(e.xy).epsilon(1e-9));
    CHECK(s(0, 2) == Approx(e.xz).epsilon(1e-9));
    CHECK(s(1, 2) == Approx(e.yz).epsilon(1e-9));
    CHECK(s(1, 0) == Approx(e.xy).epsilon(1e-9));
    CHECK(s(2, 0) == Approx(e.xz).epsilon(1e-9));
    CHECK(s(2, 1) == Approx(e.yz).epsilon(1e-9));
    // The virial pressure -tr(sigma)/3 is 2.141 GPa in frame 1 (2.570 with
    // the kinetic term left in).
    if (f == 1) CHECK(-(s(0, 0) + s(1, 1) + s(2, 2)) / 3.0 * kGPa == Approx(2.141).margin(5e-4));
  }
}

// Pins: the .geom reader stores the '<-- S' block as read (x hartree/bohr^3
// -> eV/A^3), with no kinetic term, for the first and the last frame of
// 60.geom (all nine entries). Expected values: S x 27.211386245988 /
// 0.529177210903^3, computed in numpy; the reader does the same two-double
// product, so 1e-12 relative (margin 1e-30 for the exact zeros).
// Before the fix: the first frame (step 0) was not stored; the last frame
// already passed (a mutation that adds a kinetic or N k T/V term to .geom
// makes it fail).
TEST_CASE("CASTEP .geom reader: the stress is S as read",
          "[castep_readers][geom][stress]") {
  PeriodicTable::initialize();
  struct Expected {
    double e_ha;
    double s[3][3];
  };
  const std::vector<Expected> frames = {
      {-9.8996404913333208e1,
       {{3.1103982472401887e-02, 0.0, 3.8885490381955166e-20},
        {0.0, 3.1103982472401887e-02, 7.7770980763910332e-20},
        {3.8885490381955166e-20, 7.7770980763910332e-20, 3.1118885297262994e-02}}},
      {-9.8976972393943711e1,
       {{9.3997956454652353e-07, 0.0, 3.8885490381955166e-20},
        {0.0, 9.3997956462616094e-07, -2.7219843267368617e-19},
        {3.8885490381955166e-20, -2.7219843267368617e-19, -1.3787656427006211e-06}}}};
  StructureDB db;
  read_into<CastepGeomReader>(db, kGeom60);
  for (size_t f = 0; f < frames.size(); ++f) {
    INFO("frame energy " << frames[f].e_ha << " Ha");
    const Structure *st = nullptr;
    for (size_t k = 0; k < db.size(); ++k)
      if (std::abs(db(k).energy - frames[f].e_ha * kHartree) < 1e-9) st = &db(k);
    CHECK(st != nullptr);
    if (!st) continue;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        INFO("stress(" << i << "," << j << ")");
        CHECK(st->stress(i, j) == Approx(frames[f].s[i][j]).epsilon(1e-12).margin(1e-30));
      }
  }
}

// Pins: a .md file whose kinetic energy (third '<-- E' column) differs from
// 1/2 sum m v^2 with the periodic-table masses by more than 1e-3 relative
// (custom SPECIES_MASS: deuterium, 6Li, pure 28Si differ by 4e-3 or more) is
// refused with a std::runtime_error naming the species and the file. Mass
// tables of different codes differ by less (Si 28.0855 against 28.0850 is
// 1.8e-5), so such a file is read, its kinetic tensor scaled to the file's
// kinetic energy. Copies of test1.md with one frame's kinetic energy x 1.01
// (frame 1 or frame 2) or x 1.002 are refused; x (1 + 1e-4) is read.
// Before the fix: the velocities and the kinetic energy were never read; all
// copies were accepted.
TEST_CASE("CASTEP .md reader: velocities that contradict the kinetic energy are refused",
          "[castep_readers][md][kinetic_stress]") {
  PeriodicTable::initialize();
  const std::vector<std::string> base = read_lines(kMdTest1);
  struct Case {
    size_t frame;  // 1-based
    double factor;
  };
  for (const Case c : {Case{1, 1.01}, Case{2, 1.01}, Case{2, 1.002}}) {
    INFO("frame " << c.frame << ", kinetic energy x " << c.factor);
    std::vector<std::string> lines = base;
    const size_t i = nth_tagged(lines, "E", c.frame);
    std::istringstream iss(lines[i]);
    double e1, e2, ke;
    REQUIRE(static_cast<bool>(iss >> e1 >> e2 >> ke));
    std::ostringstream o;
    o << std::scientific << std::setprecision(16) << "                     " << e1
      << "   " << e2 << "   " << ke * c.factor << "  <-- E";
    lines[i] = o.str();
    TempFile tmp("tmp_castep_rv_kinetic.md");
    tmp.write_lines(lines);

    const std::string msg = parse_error<CastepMDReader>(tmp.path);
    INFO("message: '" << msg << "'");
    CHECK_FALSE(msg.empty());
    CHECK(contains_word(msg, "Si"));
    CHECK_THAT(msg, Catch::Contains(tmp.basename()));
  }
  SECTION("a kinetic energy 1e-4 relative off (another mass table) is read") {
    std::vector<std::string> lines = base;
    const size_t i = nth_tagged(lines, "E", 2);
    std::istringstream iss(lines[i]);
    double e1, e2, ke;
    REQUIRE(static_cast<bool>(iss >> e1 >> e2 >> ke));
    std::ostringstream o;
    o << std::scientific << std::setprecision(16) << "                     " << e1
      << "   " << e2 << "   " << ke * (1.0 + 1e-4) << "  <-- E";
    lines[i] = o.str();
    TempFile tmp("tmp_castep_rv_kinetic_ok.md");
    tmp.write_lines(lines);
    const std::string msg = parse_error<CastepMDReader>(tmp.path);
    INFO("message: '" << msg << "'");
    CHECK(msg.empty());
  }
}

// --------------------------------------------------------------------
// .md and .geom: malformed and truncated frames
// --------------------------------------------------------------------

// Pins: a malformed frame that is not the last one is an error
// (std::runtime_error) naming the file, the frame (1-based) and, for an
// unreadable number, the 1-based line:
//  - 60.geom with an x coordinate of frame 6's third '<-- R' line replaced
//    by NOT_A_NUMBER;
//  - test1.md with the same done to frame 1's third '<-- V' line (the
//    velocities now feed the stress);
//  - test1.md with frame 1's second '<-- h' line deleted (the frame is
//    named; which line reports it is left to the reader).
// Before the fix: the 'error' flag dropped such frames silently (and frame 1
// of a .md/.geom was dropped anyway), so every case parsed without an error.
TEST_CASE("CASTEP .md/.geom readers: a malformed interior frame is an error naming "
          "the file the frame and the line",
          "[castep_readers][md][geom][malformed]") {
  PeriodicTable::initialize();
  SECTION("unreadable position in .geom frame 6 of 12") {
    std::vector<std::string> lines = read_lines(kGeom60);
    const size_t c6 = nth_tagged(lines, "c", 6);
    const size_t r = nth_tagged(lines, "R", 3, c6);
    lines[r] = replace_token(lines[r], 2, "NOT_A_NUMBER");
    TempFile tmp("tmp_castep_rv_bad_position.geom");
    tmp.write_lines(lines);
    const std::string msg = parse_error<CastepGeomReader>(tmp.path);
    INFO("message: '" << msg << "' (expected frame 6, line " << r + 1 << ")");
    CHECK_FALSE(msg.empty());
    CHECK_THAT(msg, Catch::Contains(tmp.basename()));
    CHECK(names_number(msg, "frame", 6));
    CHECK(names_number(msg, "line", static_cast<long>(r + 1)));
  }
  SECTION("unreadable velocity in .md frame 1 of 2") {
    std::vector<std::string> lines = read_lines(kMdTest1);
    const size_t v = nth_tagged(lines, "V", 3);
    lines[v] = replace_token(lines[v], 2, "NOT_A_NUMBER");
    TempFile tmp("tmp_castep_rv_bad_velocity.md");
    tmp.write_lines(lines);
    const std::string msg = parse_error<CastepMDReader>(tmp.path);
    INFO("message: '" << msg << "' (expected frame 1, line " << v + 1 << ")");
    CHECK_FALSE(msg.empty());
    CHECK_THAT(msg, Catch::Contains(tmp.basename()));
    CHECK(names_number(msg, "frame", 1));
    CHECK(names_number(msg, "line", static_cast<long>(v + 1)));
  }
  SECTION("missing lattice line in .md frame 1 of 2") {
    std::vector<std::string> lines = read_lines(kMdTest1);
    const size_t h = nth_tagged(lines, "h", 2);
    lines.erase(lines.begin() + static_cast<std::ptrdiff_t>(h));
    TempFile tmp("tmp_castep_rv_missing_h.md");
    tmp.write_lines(lines);
    const std::string msg = parse_error<CastepMDReader>(tmp.path);
    INFO("message: '" << msg << "'");
    CHECK_FALSE(msg.empty());
    CHECK_THAT(msg, Catch::Contains(tmp.basename()));
    CHECK(names_number(msg, "frame", 1));
  }
}

// Pins: a file that ends inside its last frame (a run killed mid-write) is
// read up to the last complete frame, and the reader emits exactly one
// WARNING (Logger or std::cerr, case-insensitive), naming the file:
//  - test1.md without its last four '<-- F' lines -> 1 structure (frame 1);
//  - 60.geom without its last four '<-- F' lines -> 11 structures (frames
//    1-11), none with frame 12's energy.
// Before the fix: no warning at all, and 0 and 10 structures (the first frame
// dropped as well).
TEST_CASE("CASTEP .md/.geom readers: a truncated last frame is skipped with one WARNING",
          "[castep_readers][md][geom][truncated]") {
  PeriodicTable::initialize();
  SECTION(".md") {
    std::vector<std::string> lines = read_lines(kMdTest1);
    const size_t last_f = nth_tagged(lines, "F", 16);
    lines.resize(last_f - 3);  // drop the last four force lines
    TempFile tmp("tmp_castep_rv_truncated.md");
    tmp.write_lines(lines);
    StructureDB db;
    const std::string printed = parse_capturing<CastepMDReader>(db, tmp.path);
    INFO("printed: '" << printed << "'");
    CHECK(count_ci(printed, "warning") == 1);
    CHECK_THAT(printed, Catch::Contains(tmp.basename()));
    REQUIRE(db.size() == 1);
    CHECK(db(0).energy == Approx(kTest1E0 * kHartree).epsilon(1e-14));
  }
  SECTION(".geom") {
    std::vector<std::string> lines = read_lines(kGeom60);
    const size_t last_f = nth_tagged(lines, "F", 120);
    lines.resize(last_f - 3);
    TempFile tmp("tmp_castep_rv_truncated.geom");
    tmp.write_lines(lines);
    StructureDB db;
    const std::string printed = parse_capturing<CastepGeomReader>(db, tmp.path);
    INFO("printed: '" << printed << "'");
    CHECK(count_ci(printed, "warning") == 1);
    CHECK_THAT(printed, Catch::Contains(tmp.basename()));
    CHECK(db.size() == 11);
    REQUIRE(db.size() >= 1);
    CHECK(db(0).energy == Approx(-9.8996404913333208e1 * kHartree).epsilon(1e-14));
    CHECK(db(db.size() - 1).energy == Approx(-9.8976972562243105e1 * kHartree).epsilon(1e-14));
    for (size_t s = 0; s < db.size(); ++s)
      CHECK(std::abs(db(s).energy - (-9.8976972393943711e1 * kHartree)) > 1e-9);
  }
}

// Pins: an .md frame without a stress tensor ('<-- S' lines) is stored
// with zero stress and stress priority 0, so a STRESS fit ignores it; a frame
// with one keeps weight 1: test1.md gives 2 structures with sweight 1, and
// the same file without its '<-- S' lines 2 structures with sweight 0 (and
// the same energies).
// Before the fix: zero stress with sweight 1.
TEST_CASE("CASTEP .md reader: a frame without a stress tensor has stress priority 0",
          "[castep_readers][md][stress]") {
  PeriodicTable::initialize();
  StructureDB with, without;
  read_into<CastepMDReader>(with, kMdTest1);
  std::vector<std::string> lines = read_lines(kMdTest1);
  std::vector<std::string> kept;
  for (const auto &l : lines)
    if (!ends_with_tag(l, "S")) kept.push_back(l);
  REQUIRE(kept.size() == lines.size() - 6);
  TempFile tmp("tmp_castep_rv_no_s.md");
  tmp.write_lines(kept);
  read_into<CastepMDReader>(without, tmp.path);
  REQUIRE(with.size() == 2);
  REQUIRE(without.size() == 2);
  for (size_t k = 0; k < 2; ++k) {
    INFO("frame " << k + 1);
    CHECK(with(k).sweight == 1.0);
    CHECK_FALSE(stress_is_zero(with(k)));
    CHECK(without(k).sweight == 0.0);
    CHECK(stress_is_zero(without(k)));
    CHECK(without(k).energy == with(k).energy);
  }
}

// --------------------------------------------------------------------
// .castep
// --------------------------------------------------------------------

// Pins: in an (L)BFGS run with a revert, each SCF result is paired with the
// geometry printed just before that SCF, and the reverted geometry, which no
// SCF follows before 'finished iteration', is ignored:
//  - structure 1 is iteration 0: a = 2.9640232, c = 7.0990150 A, E =
//    -2690.182441105 eV (the last finite-basis SCF), stress xx -90.011421 GPa;
//  - structure 2 is the trial cell of iteration 1: a = 2.9541657, c =
//    7.0754048 A, E = -2689.827841473 eV, stress xx -93.305358 and zz
//    -93.879527 GPa, H1 at z = 0.318951 c, H1 force z = 0.00125 eV/A;
//  - no structure has the iteration-0 cell with the trial energy.
// Cell and position: 1e-12 (parsed decimals, one product); energy 1e-14
// relative; stress 1e-7 relative (the reader's GPa factor 0.00624150913 is
// 9e-9 from 1/160.21766208, and the check only has to tell -93.305358 from
// -90.011421).
// Before the fix: the reader matched 'reverting to earlier contexturation'
// (a corrupted 'configuration'), never skipped the revert block, and stored
// the reverted iteration-0 cell (a = 2.9640232) and positions, with zero
// forces (the revert's coordinate block reset them) and the trial SCF's
// energy -2689.827841473 eV: one geometry with two energies 0.355 eV apart.
// After that fix, until a later one: the label's positions range 'C: (636 :
// 645)' still named the reverted block.
TEST_CASE("CASTEP .castep reader: an LBFGS revert pairs each SCF with its own geometry",
          "[castep_readers][castep][revert]") {
  PeriodicTable::initialize();
  StructureDB db;
  read_into<CastepCastepReader>(db, kRevert);
  REQUIRE(db.size() == 2);

  const double p_conv = 1.0 / kGPa;
  {
    INFO("structure 1 (iteration 0)");
    const Structure &st = db(0);
    CHECK(st.cell(0, 0) == Approx(2.9640232).epsilon(1e-12));
    CHECK(st.cell(1, 1) == Approx(2.9640232).epsilon(1e-12));
    CHECK(st.cell(2, 2) == Approx(7.0990150).epsilon(1e-12));
    CHECK(st.energy == Approx(-2690.182441105).epsilon(1e-14));
    CHECK(st.stress(0, 0) == Approx(-90.011421 * p_conv).epsilon(1e-7));
  }
  {
    INFO("structure 2 (iteration 1, trial step)");
    const Structure &st = db(1);
    CHECK(st.cell(0, 0) == Approx(2.9541657).epsilon(1e-12));
    CHECK(st.cell(1, 1) == Approx(2.9541657).epsilon(1e-12));
    CHECK(st.cell(2, 2) == Approx(7.0754048).epsilon(1e-12));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        if (i != j) CHECK(st.cell(i, j) == Approx(0.0).margin(1e-12));
    CHECK(st.energy == Approx(-2689.827841473).epsilon(1e-14));
    CHECK(st.stress(0, 0) == Approx(-93.305358 * p_conv).epsilon(1e-7));
    CHECK(st.stress(2, 2) == Approx(-93.879527 * p_conv).epsilon(1e-7));
    REQUIRE(st.natoms() == 10);
    CHECK(st.atoms[0].symbol == "H");
    CHECK(st.atoms[0].position[2] == Approx(0.318951 * 7.0754048).epsilon(1e-12));
    CHECK(st.atoms[0].force[2] == Approx(0.00125).margin(1e-12));
    // The label names the positions stored, 534-543, not the reverted
    // block 636-645, and the energy line 561.
    CHECK_THAT(st.label, Catch::Contains("C: (534 : 543)"));
    CHECK_THAT(st.label, Catch::Contains("E: 561"));
  }
  for (size_t s = 0; s < db.size(); ++s) {
    INFO("structure " << s + 1);
    const bool iteration0_cell = std::abs(db(s).cell(0, 0) - 2.9640232) < 1e-6 &&
                                 std::abs(db(s).cell(2, 2) - 7.0990150) < 1e-6;
    const bool trial_energy = std::abs(db(s).energy - (-2689.827841473)) < 1e-6;
    CHECK_FALSE((iteration0_cell && trial_energy));
  }
}

// Pins: in an LBFGS geometry optimisation whose iterations go on with a
// line (or quadratic) minimisation, the structure of each iteration is the
// last SCF of that iteration on the geometry printed just before it -- the
// line-minimisation geometry, not the iteration's trial geometry. geom.castep
// (8 structures): energy, cell, H1 height and force, and stress xx of each
// are those of the image ASE 3.25 (castep-castep reader) gives the same
// energy; the label names the positions block and the energy line of that
// SCF (geom.castep lines). Cell 1e-12 (parsed decimals); energy 1e-14
// relative; stress 1e-7 relative (the reader's GPa factor).
// Before the fix (the first version of the revert fix, which took the
// geometry once per iteration): structures 2-6 paired the
// line-minimisation energy, forces and stress with the trial cell, e.g.
// structure 2 had a = 2.9541657 with the energy of a = 2.9499544.
TEST_CASE("CASTEP .castep reader: line and quad minimisation SCFs pair with their own geometry",
          "[castep_readers][castep][revert][line_minimisation]") {
  PeriodicTable::initialize();
  StructureDB db;
  read_into<CastepCastepReader>(db, kGeomCastep);
  REQUIRE(db.size() == 8);
  struct Expected {
    double e, a, c, h1z_frac, f1z, sxx_gpa;
    const char *positions, *energy_line;  // the label's C: and E: entries
  };
  const Expected expected[8] = {
      {-2690.182441105, 2.9640232, 7.0990150, 0.318951, -0.00001, -90.011421, "C: (295 : 304)", "E: 430"},
      {-2689.672649027, 2.9499544, 7.0653183, 0.318951, 0.00189, -94.737913, "C: (637 : 646)", "E: 663"},
      {-2689.670652110, 2.9485358, 7.0716787, 0.318952, -0.00964, -94.992185, "C: (863 : 872)", "E: 888"},
      {-2689.669050894, 2.9482042, 7.0729650, 0.318930, -0.01093, -95.041542, "C: (1190 : 1199)", "E: 1215"},
      {-2689.669242284, 2.9484328, 7.0718932, 0.318861, -0.00449, -94.979348, "C: (1416 : 1425)", "E: 1441"},
      {-2689.668548477, 2.9481184, 7.0732666, 0.318747, 0.00019, -95.000945, "C: (1645 : 1654)", "E: 1670"},
      {-2689.669041115, 2.9481118, 7.0733926, 0.318747, 0.00003, -94.999239, "C: (1769 : 1778)", "E: 1794"},
      {-2689.669012276, 2.9481080, 7.0734053, 0.318747, -0.00000, -94.999759, "C: (1892 : 1901)", "E: 1917"}};
  for (size_t k = 0; k < 8; ++k) {
    INFO("structure " << k + 1);
    const Structure &st = db(k);
    const Expected &x = expected[k];
    CHECK(st.energy == Approx(x.e).epsilon(1e-14));
    CHECK(st.cell(0, 0) == Approx(x.a).epsilon(1e-12));
    CHECK(st.cell(1, 1) == Approx(x.a).epsilon(1e-12));
    CHECK(st.cell(2, 2) == Approx(x.c).epsilon(1e-12));
    REQUIRE(st.natoms() == 10);
    CHECK(st.atoms[0].symbol == "H");
    CHECK(st.atoms[0].position[2] == Approx(x.h1z_frac * x.c).epsilon(1e-12));
    CHECK(st.atoms[0].force[2] == Approx(x.f1z).margin(1e-12));
    CHECK(st.stress(0, 0) == Approx(x.sxx_gpa / kGPa).epsilon(1e-7));
    CHECK_THAT(st.label, Catch::Contains(x.positions));
    CHECK_THAT(st.label, Catch::Contains(x.energy_line));
  }
}

// Pins: a fixed-cell geometry optimisation with the finite basis set
// correction on keeps its iteration 0. Mn.castep (30 Mn, 'stress
// calculation : on', but a fixed cell: no SCF prints a stress block, only
// the final configuration does) gives 16 structures, iterations 0-15. Structure 1 is the last finite-basis SCF
// on the initial geometry: E = -87430.87859822 eV (line 525), Mn 1 at
// fractional (0.135, 0.1356, 0.13561) (positions block lines 274-303),
// force (-0.11769, -0.0768, -0.10067) eV/A; structure 2 is the line
// minimisation of iteration 1: E = -87431.2322915 eV, Mn 1 at (0.133488,
// 0.133739, 0.133952), force (-0.04067, -0.04976, -0.0373). Values as ASE
// 3.25 reads them. Fractional positions 1e-9 (through the cell inverse).
// Before the fix: the finite-basis SCF was completed only by a stress block,
// which this file does not print, so iteration 0 was dropped silently: 15
// structures, the first one iteration 1.
TEST_CASE("CASTEP .castep reader: a fixed-cell geometry optimisation keeps its iteration 0",
          "[castep_readers][castep][fbsc]") {
  PeriodicTable::initialize();
  StructureDB db;
  read_into<CastepCastepReader>(db, "./tests_data/valid_castep_castep/Mn.castep");
  CHECK(db.size() == 16);
  REQUIRE(db.size() >= 2);
  struct Expected {
    double e;
    double frac[3], force[3];
  };
  const Expected expected[2] = {
      {-87430.87859822, {0.135, 0.1356, 0.13561}, {-0.11769, -0.0768, -0.10067}},
      {-87431.2322915, {0.133488, 0.133739, 0.133952}, {-0.04067, -0.04976, -0.0373}}};
  for (size_t k = 0; k < 2; ++k) {
    INFO("structure " << k + 1 << " label '" << db(k).label << "'");
    const Structure &st = db(k);
    CHECK(st.energy == Approx(expected[k].e).epsilon(1e-14));
    REQUIRE(st.natoms() == 30);
    const tadah::core::Matrix3d inv = st.cell.inverse();
    for (int c = 0; c < 3; ++c) {
      double f = 0;
      for (int r = 0; r < 3; ++r) f += st.atoms[0].position[r] * inv(r, c);
      CHECK(f == Approx(expected[k].frac[c]).margin(1e-9));
      CHECK(st.atoms[0].force[c] == Approx(expected[k].force[c]).margin(1e-12));
    }
  }
  CHECK_THAT(db(0).label, Catch::Contains("C: (274 : 303)"));
  CHECK_THAT(db(0).label, Catch::Contains("E: 525"));
}

// Pins: a fixed-occupancy (insulator) run prints 'Final energy =' and no
// 'Final free energy (E-TS)' line (as md_stress.castep does); with no
// smearing that energy is F, and it is stored. geom.castep rewritten in
// that form (each 'Final energy, E             =' line as 'Final energy =',
// its 'Final free energy (E-TS)' and 'NB est. 0K energy' lines removed)
// gives the same 8 structures with the 'Final energy, E' value of each
// structure's SCF as its energy (ASE 3.25 reads the same file the same way:
// energy = free_energy = 'Final energy').
// Before the fix: no energy line was read, so all 8 structures were stored
// with energy 0, silently.
TEST_CASE("CASTEP .castep reader: a fixed-occupancy run stores its 'Final energy'",
          "[castep_readers][castep][insulator]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines(kGeomCastep);
  lines = substitute(lines, "Final energy, E             =", "Final energy =");
  lines = drop_lines(lines, "Final free energy (E-TS)");
  lines = drop_lines(lines, "NB est. 0K energy");
  TempFile tmp("tmp_castep_rv_insulator.castep");
  tmp.write_lines(lines);
  StructureDB plain, db;
  read_into<CastepCastepReader>(plain, kGeomCastep);
  read_into<CastepCastepReader>(db, tmp.path);
  REQUIRE(plain.size() == 8);
  REQUIRE(db.size() == 8);
  const double e[8] = {-2690.147924655, -2689.637293811, -2689.635344489,
                       -2689.633758579, -2689.633972892, -2689.633335038,
                       -2689.633829093, -2689.633800316};
  for (size_t k = 0; k < 8; ++k) {
    INFO("structure " << k + 1);
    CHECK(db(k).energy == Approx(e[k]).epsilon(1e-14));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) CHECK(db(k).cell(i, j) == plain(k).cell(i, j));
    REQUIRE(db(k).natoms() == plain(k).natoms());
    CHECK(db(k).atoms[0].force[2] == plain(k).atoms[0].force[2]);
  }
}

// Pins: a .castep structure whose SCF has no energy line of its own is
// refused, naming the file: geom.castep without the 'Final free energy'
// line of iteration 1's line-minimisation SCF (line 663). The structure
// does not take the energy of the iteration's trial SCF, nor of an earlier
// structure.
// Before the fix: structure 2 was stored silently with the trial SCF's
// energy (-2689.827841473, line 561) on the line-minimisation geometry.
TEST_CASE("CASTEP .castep reader: a structure without an energy line of its own is refused",
          "[castep_readers][castep][energy]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines(kGeomCastep);
  REQUIRE(lines.at(662).find("Final free energy (E-TS)    =  -2689.672649027") == 0);
  lines.erase(lines.begin() + 662);
  TempFile tmp("tmp_castep_rv_no_energy.castep");
  tmp.write_lines(lines);
  const std::string what = parse_error<CastepCastepReader>(tmp.path);
  INFO("error: '" << what << "'");
  CHECK_THAT(what, Catch::Contains(tmp.basename()));
  CHECK_THAT(what, Catch::Contains("energy"));
}

// Pins: an MD run with the finite basis set correction keeps its step 0.
// The correction's SCFs run before ' Starting MD'; step 0 is completed at
// its MD Data 'Temperature:' line, with the MD Data potential energy and
// temperature, like every later step. md_no_stress.castep (NVT, 'stress
// calculation : on', no stress block) with 'finite basis set correction :
// automatic' and the line 'Calculating finite basis set correction with 3
// cut-off energies.' before ' Starting MD' gives exactly the structures of
// the unmodified file (steps 0-2), with 'stress calculation' on and off.
// (ASE 3.25's own test file md.castep has this layout.)
// Before the fix: with stress on, step 0 was dropped silently (2
// structures); with stress off it was completed at its forces block in the
// finite-basis state, with energy 0 and T = 0.
TEST_CASE("CASTEP .castep reader: an MD run with the finite basis correction keeps step 0",
          "[castep_readers][castep][fbsc][md]") {
  PeriodicTable::initialize();
  const std::string kMdNoStress = "./tests_data/valid_castep_castep/md_no_stress.castep";
  for (const char *stress : {"on", "off"}) {
    INFO("stress calculation " << stress);
    std::vector<std::string> lines = read_lines(kMdNoStress);
    lines = substitute(lines, "stress calculation                             : on",
                       std::string("stress calculation                             : ") + stress);
    TempFile ref("tmp_castep_rv_md_ref.castep");
    ref.write_lines(lines);
    lines = substitute(lines, "finite basis set correction                    : none",
                       "finite basis set correction                    : automatic");
    const auto start = std::find(lines.begin(), lines.end(), " Starting MD");
    REQUIRE(start != lines.end());
    lines.insert(start, "Calculating finite basis set correction with  3 cut-off energies.");
    TempFile tmp("tmp_castep_rv_md_fbsc.castep");
    tmp.write_lines(lines);
    StructureDB want, db;
    read_into<CastepCastepReader>(want, ref.path);
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(want.size() == 3);
    CHECK(want(0).T == Approx(400.0));
    REQUIRE(db.size() == want.size());
    for (size_t k = 0; k < db.size(); ++k) {
      INFO("structure " << k + 1);
      check_same_structure(db(k), want(k));
    }
  }
}

// Pins: a structure whose SCF prints no stress block is stored with zero
// stress and stress priority 0, so a STRESS fit ignores it; a structure with
// its own stress keeps weight 1. Each structure gets only its own SCF's
// results, and its label names only its own blocks ('S: (-1 : -1)' without
// one):
//  - md_no_stress.castep (NVT, 'stress calculation : on', no stress block):
//    3 structures, all sweight 0;
//  - geom.castep: 8 structures, all sweight 1;
//  - geom.castep without iteration 1's line-minimisation stress block
//    (lines 688-700, 'Symmetrised Stress Tensor'): structure 2 has zero
//    stress and sweight 0, not the trial SCF's stress (-93.305358 GPa);
//  - combined.castep: the three NVT structures (9-11) have sweight 0 and
//    'S: (-1 : -1)', not the GO run's final stress block (2047-2049).
// Before the fix: zero stress with sweight 1; structure 2 kept the trial
// SCF's stress; the NVT labels named lines 2047-2049.
TEST_CASE("CASTEP .castep reader: a structure without a stress block has stress priority 0",
          "[castep_readers][castep][stress]") {
  PeriodicTable::initialize();
  {
    INFO("md_no_stress.castep");
    StructureDB db;
    read_into<CastepCastepReader>(db, "./tests_data/valid_castep_castep/md_no_stress.castep");
    REQUIRE(db.size() == 3);
    for (size_t k = 0; k < db.size(); ++k) {
      INFO("structure " << k + 1);
      CHECK(db(k).sweight == 0.0);
      CHECK(stress_is_zero(db(k)));
      CHECK(db(k).eweight == 1.0);
      CHECK(db(k).fweight == 1.0);
    }
  }
  {
    INFO("geom.castep");
    StructureDB db;
    read_into<CastepCastepReader>(db, kGeomCastep);
    REQUIRE(db.size() == 8);
    for (size_t k = 0; k < db.size(); ++k) {
      INFO("structure " << k + 1);
      CHECK(db(k).sweight == 1.0);
      CHECK_FALSE(stress_is_zero(db(k)));
    }
  }
  {
    INFO("geom.castep without structure 2's stress block");
    std::vector<std::string> lines = read_lines(kGeomCastep);
    REQUIRE(lines.at(687).find("Symmetrised Stress Tensor") != std::string::npos);
    REQUIRE(lines.at(689).find("Cartesian components (GPa)") != std::string::npos);
    REQUIRE(lines.at(699).find("*****") != std::string::npos);
    lines.erase(lines.begin() + 687, lines.begin() + 700);
    TempFile tmp("tmp_castep_rv_no_stress2.castep");
    tmp.write_lines(lines);
    StructureDB db;
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == 8);
    CHECK(db(1).energy == Approx(-2689.672649027).epsilon(1e-14));
    CHECK(db(1).sweight == 0.0);
    CHECK(stress_is_zero(db(1)));
    CHECK_THAT(db(1).label, Catch::Contains("S: (-1 : -1)"));
    CHECK(db(2).sweight == 1.0);
  }
  {
    INFO("combined.castep");
    StructureDB db;
    read_into<CastepCastepReader>(db, "./tests_data/valid_castep_castep/combined.castep");
    REQUIRE(db.size() == 15);
    for (size_t k = 8; k < 11; ++k) {
      INFO("structure " << k + 1 << " label '" << db(k).label << "'");
      CHECK_THAT(db(k).label, Catch::Contains("MD NVT"));
      CHECK(db(k).sweight == 0.0);
      CHECK(stress_is_zero(db(k)));
      CHECK_THAT(db(k).label, Catch::Contains("S: (-1 : -1)"));
    }
    for (size_t k = 11; k < 15; ++k) {
      INFO("structure " << k + 1 << " label '" << db(k).label << "'");
      CHECK(db(k).sweight == 1.0);
      CHECK_FALSE(stress_is_zero(db(k)));
    }
  }
}

// Pins: the stress block CASTEP prints after '<algorithm>: Final
// Configuration' belongs to the final geometry, which is the last stored
// iteration's: Mn.castep (a fixed-cell LBFGS run with 'stress calculation :
// on' prints one stress block, at lines 4085-4095, for its final
// configuration) gives structure 16 that stress, xx -17.784980, xy
// -1.327631, zz -17.742115 GPa (as ASE 3.25 gives its last image), with
// stress weight 1 and the label's 'S: (4091 : 4093)'; structures 1-15 have
// no stress (sweight 0).
// Before the fix: the block was read after the last structure was stored
// and dropped; every structure had zero stress with weight 1.
TEST_CASE("CASTEP .castep reader: the final configuration's stress goes to the last iteration",
          "[castep_readers][castep][stress]") {
  PeriodicTable::initialize();
  StructureDB db;
  read_into<CastepCastepReader>(db, "./tests_data/valid_castep_castep/Mn.castep");
  REQUIRE(db.size() == 16);
  for (size_t k = 0; k < 15; ++k) {
    INFO("structure " << k + 1);
    CHECK(db(k).sweight == 0.0);
    CHECK(stress_is_zero(db(k)));
  }
  const Structure &last = db(15);
  INFO("label '" << last.label << "'");
  CHECK(last.sweight == 1.0);
  const double gpa[3][3] = {{-17.784980, -1.327631, -1.320542},
                            {-1.327631, -17.721855, -1.306493},
                            {-1.320542, -1.306493, -17.742115}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      INFO("(" << i << "," << j << ")");
      CHECK(last.stress(i, j) == Approx(gpa[i][j] / kGPa).epsilon(1e-7));
    }
  CHECK_THAT(last.label, Catch::Contains("S: (4091 : 4093)"));
}

// Pins: in a geometry optimisation with cell constraints CASTEP prints, after
// the 'Symmetrised Stress Tensor', a 'Constrained Stress Tensor' (the
// projection the optimiser uses), and with ionic constraints 'Constrained
// Symmetrised Forces' after the physical forces; the physical blocks are
// stored, the constrained ones skipped. geom.castep with such a stress block
// (diagonal -1, -2, -3 GPa) and such a forces block (every component 0.5)
// after iteration 1's line-minimisation blocks: structure 2 keeps stress xx
// -94.737913 GPa and H1 force z 0.00189 eV/A.
// Before the fix: every 'Cartesian components' block overwrote the
// previous one, so structure 2 stored the constrained blocks. (Real case:
// FLiBe flibe_geom.castep, 'Cell constraints are: 1 1 1 0 0 0'.)
TEST_CASE("CASTEP .castep reader: constrained stress and forces blocks are not stored",
          "[castep_readers][castep][constraints]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines(kGeomCastep);
  REQUIRE(lines.at(668).find("Symmetrised Forces") != std::string::npos);
  REQUIRE(lines.at(687).find("Symmetrised Stress Tensor") != std::string::npos);
  std::vector<std::string> stress = block_at(lines, 687);
  REQUIRE(stress.size() == 13);
  stress[0] = substitute({stress[0]}, "Symmetrised Stress Tensor", "Constrained Stress Tensor")[0];
  const char *diag[3] = {"-1.000000", "-2.000000", "-3.000000"};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      stress[6 + r] = replace_token(stress[6 + r], 2 + c, r == c ? diag[r] : "0.000000");
  std::vector<std::string> forces = block_at(lines, 668);
  forces[0] = " ************************** Constrained Symmetrised Forces **************************";
  for (size_t k = 6; k + 2 < forces.size(); ++k)
    for (int c = 0; c < 3; ++c) forces[k] = replace_token(forces[k], 3 + c, "0.50000");
  // After the stress block (lines 688-700), then the forces after the
  // physical forces block (lines 669-686).
  REQUIRE(forces.size() == 18);
  lines.insert(lines.begin() + 700, stress.begin(), stress.end());
  lines.insert(lines.begin() + 686, forces.begin(), forces.end());
  TempFile tmp("tmp_castep_rv_constrained.castep");
  tmp.write_lines(lines);
  StructureDB db;
  read_into<CastepCastepReader>(db, tmp.path);
  REQUIRE(db.size() == 8);
  CHECK(db(1).stress(0, 0) == Approx(-94.737913 / kGPa).epsilon(1e-7));
  CHECK(db(1).stress(2, 2) == Approx(-95.561163 / kGPa).epsilon(1e-7));
  CHECK(db(1).atoms[0].force[2] == Approx(0.00189).margin(1e-12));
  CHECK(db(1).atoms[0].force[0] == Approx(0.0).margin(1e-12));
}

// Pins: with a semi-empirical dispersion correction (DFT-D, SEDC) the
// forces, the stress and the optimiser's enthalpy include the correction, so
// the stored energy does too:
//  - fixed occupancy: 'Dispersion corrected final energy*' (the insulator
//    form of geom.castep, each SCF followed by that line at E - 1.234 eV);
//  - smearing: 'Dispersion corrected final free energy*' when printed (here
//    F - 2.5 eV), otherwise F plus the correction, F + (E_disp - E) (here
//    F - 1.234 eV when only the 'final energy*' line is printed).
// Before the fix: the uncorrected SCF energy was stored. (Real case: FLiBe
// 02c_geomopt_pbe_d3 flibe_geom.castep, D3.)
TEST_CASE("CASTEP .castep reader: a dispersion-corrected run stores the corrected energy",
          "[castep_readers][castep][sedc]") {
  PeriodicTable::initialize();
  StructureDB plain;
  read_into<CastepCastepReader>(plain, kGeomCastep);
  REQUIRE(plain.size() == 8);
  const std::vector<std::string> base = read_lines(kGeomCastep);
  SECTION("fixed occupancy") {
    std::vector<std::string> lines =
        substitute(base, "Final energy, E             =", "Final energy =");
    lines = drop_lines(lines, "Final free energy (E-TS)");
    lines = drop_lines(lines, "NB est. 0K energy");
    std::vector<std::string> out;
    for (const auto &l : lines) {
      out.push_back(l);
      if (l.rfind("Final energy =", 0) == 0) {
        std::ostringstream oss;
        oss << std::setprecision(15) << "Dispersion corrected final energy* =  "
            << energy_after_equals(l) - 1.234 << "     eV";
        out.push_back("(SEDC) Total Energy Correction : -0.123400E+01 eV");
        out.push_back(oss.str());
      }
    }
    TempFile tmp("tmp_castep_rv_sedc_ins.castep");
    tmp.write_lines(out);
    StructureDB db;
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == 8);
    const double e[8] = {-2690.147924655, -2689.637293811, -2689.635344489,
                         -2689.633758579, -2689.633972892, -2689.633335038,
                         -2689.633829093, -2689.633800316};
    for (size_t k = 0; k < 8; ++k) {
      INFO("structure " << k + 1);
      CHECK(db(k).energy == Approx(e[k] - 1.234).epsilon(1e-13));
    }
  }
  for (const bool free_form : {false, true}) {
    SECTION(free_form ? "smearing, free-energy form" : "smearing, energy form only") {
      std::vector<std::string> out;
      double E = 0.0, F = 0.0;
      for (const auto &l : base) {
        out.push_back(l);
        if (l.rfind("Final energy, E", 0) == 0) E = energy_after_equals(l);
        if (l.rfind("Final free energy (E-TS)", 0) == 0) F = energy_after_equals(l);
        if (l.rfind("NB est. 0K energy", 0) == 0) {
          std::ostringstream e1, f1;
          e1 << std::setprecision(15) << "Dispersion corrected final energy* =  "
             << E - 1.234 << "     eV";
          f1 << std::setprecision(15)
             << "Dispersion corrected final free energy* (Ecor-TS)  =  " << F - 2.5
             << "     eV";
          out.push_back("(SEDC) Total Energy Correction : -0.123400E+01 eV");
          out.push_back(e1.str());
          if (free_form) out.push_back(f1.str());
        }
      }
      TempFile tmp("tmp_castep_rv_sedc_smear.castep");
      tmp.write_lines(out);
      StructureDB db;
      read_into<CastepCastepReader>(db, tmp.path);
      REQUIRE(db.size() == 8);
      for (size_t k = 0; k < 8; ++k) {
        INFO("structure " << k + 1);
        CHECK(db(k).energy ==
              Approx(plain(k).energy - (free_form ? 2.5 : 1.234)).epsilon(1e-13));
      }
    }
  }
}

// Pins: a run's state does not leak into the next run of the same file.
//  - Run type comes from the run's 'type of calculation' line only: an SPE
//    file with the parameter line 'spectroscopy calculation : band
//    structure' still gives an SPE structure;
//  - a band-structure step without an SCF (a continuation: no energy) is
//    skipped with one WARNING, not stored and not refused: Gd0.castep
//    without its energy lines gives none, and after an SPE run the SPE's
//    structure is kept;
//  - geom.castep without any stress block (so its final configuration has
//    none) followed by a phonon run that prints a stress block for the same
//    geometry: the GO's last structure keeps no stress (sweight 0).
// Before the fix: the SPE was labelled BS; the BS step was refused, losing
// the SPE structure too; the phonon run's stress went to the GO.
TEST_CASE("CASTEP .castep reader: runs in one file do not share state",
          "[castep_readers][castep][runs]") {
  PeriodicTable::initialize();
  const std::string kSpe = "./tests_data/cell_convention/castep_triclinic_spe.castep";
  const std::string kGd0 = "./tests_data/valid_castep_castep/Gd0.castep";
  {
    INFO("spectroscopy parameter line in an SPE run");
    std::vector<std::string> lines = read_lines(kSpe);
    REQUIRE(lines.at(15).find("type of calculation") != std::string::npos);
    lines.insert(lines.begin() + 16,
                 " spectroscopy calculation                       : band structure");
    TempFile tmp("tmp_castep_rv_spe_spectro.castep");
    tmp.write_lines(lines);
    StructureDB db;
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == 1);
    CHECK_THAT(db(0).label, Catch::Contains("SPE"));
  }
  std::vector<std::string> bs = read_lines(kGd0);
  bs = drop_lines(bs, "Final energy, E", 1);
  bs = drop_lines(bs, "Final free energy (E-TS)", 1);
  bs = drop_lines(bs, "NB est. 0K energy", 1);
  {
    INFO("band structure without an SCF");
    TempFile tmp("tmp_castep_rv_bs_noscf.castep");
    tmp.write_lines(bs);
    StructureDB db;
    const std::string printed = parse_capturing<CastepCastepReader>(db, tmp.path);
    INFO("printed: '" << printed << "'");
    CHECK(db.size() == 0);
    CHECK(count_ci(printed, "warning") == 1);
    CHECK_THAT(printed, Catch::Contains(tmp.basename()));
  }
  {
    INFO("SPE followed by a band structure without an SCF");
    std::vector<std::string> lines = read_lines(kSpe);
    lines.insert(lines.end(), bs.begin(), bs.end());
    TempFile tmp("tmp_castep_rv_spe_bs.castep");
    tmp.write_lines(lines);
    StructureDB db;
    const std::string printed = parse_capturing<CastepCastepReader>(db, tmp.path);
    INFO("printed: '" << printed << "'");
    REQUIRE(db.size() == 1);
    CHECK(db(0).energy == Approx(-312.345678901).epsilon(1e-14));
    CHECK(count_ci(printed, "warning") == 1);
  }
  {
    INFO("GO without stress followed by a phonon run with stress");
    std::vector<std::string> go = read_lines(kGeomCastep);
    std::vector<std::string> stress_block;
    for (size_t i = 0; i < go.size();) {
      if (go[i].find("Stress Tensor") != std::string::npos) {
        if (stress_block.empty()) stress_block = block_at(go, i);
        go = erase_block(go, i);
      } else {
        ++i;
      }
    }
    REQUIRE(stress_block.size() == 13);
    go.push_back(" type of calculation                            : phonon calculation");
    go.push_back(" stress calculation                             : on");
    go.push_back("");
    go.insert(go.end(), stress_block.begin(), stress_block.end());
    TempFile tmp("tmp_castep_rv_go_phonon.castep");
    tmp.write_lines(go);
    StructureDB db;
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == 8);
    for (size_t k = 0; k < 8; ++k) {
      INFO("structure " << k + 1);
      CHECK(db(k).sweight == 0.0);
      CHECK(stress_is_zero(db(k)));
    }
  }
}

// Pins: a structure without a forces block of its own is stored with zero
// forces and force priority 0 (fweight), so that a FORCE fit ignores it; the
// forces printed for a geometry optimisation's final configuration go to the
// last iteration when that has the same geometry and none of its own; a
// variable-cell optimisation with 'stress calculation : off' still prints
// each iteration's stress, iteration 0 included, and it is stored.
//  - geom.castep without iteration 1's line-minimisation forces block (lines
//    669-686): structure 2 has fweight 0, the others 1;
//  - Mn.castep without iteration 15's forces block (lines 3935-3972): the
//    final configuration's forces block (4046-4083, the same numbers) gives
//    structure 16 its forces back, with fweight 1;
//  - geom.castep with 'stress calculation : off': structure 1 has its stress
//    (S: (475 : 477)) and sweight 1.
// Before the fix: zero forces with fweight 1; structure 16 kept zero forces;
// iteration 0 was completed at its forces block, before its stress.
TEST_CASE("CASTEP .castep reader: a structure without forces has force priority 0",
          "[castep_readers][castep][forces]") {
  PeriodicTable::initialize();
  {
    INFO("geom.castep without structure 2's forces");
    std::vector<std::string> lines = read_lines(kGeomCastep);
    REQUIRE(lines.at(668).find("Symmetrised Forces") != std::string::npos);
    lines = erase_block(lines, 668);
    TempFile tmp("tmp_castep_rv_no_forces2.castep");
    tmp.write_lines(lines);
    StructureDB db;
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == 8);
    for (size_t k = 0; k < 8; ++k) {
      INFO("structure " << k + 1);
      CHECK(db(k).fweight == (k == 1 ? 0.0 : 1.0));
    }
    CHECK(forces_are_zero(db(1)));
    CHECK_THAT(db(1).label, Catch::Contains("F: (-1 : -1)"));
  }
  {
    INFO("Mn.castep without iteration 15's forces");
    const std::string kMn = "./tests_data/valid_castep_castep/Mn.castep";
    StructureDB plain, db;
    read_into<CastepCastepReader>(plain, kMn);
    std::vector<std::string> lines = read_lines(kMn);
    REQUIRE(lines.at(3934).find("Forces") != std::string::npos);
    lines = erase_block(lines, 3934);
    TempFile tmp("tmp_castep_rv_mn_noforces.castep");
    tmp.write_lines(lines);
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == 16);
    CHECK(db(15).fweight == 1.0);
    REQUIRE(db(15).natoms() == plain(15).natoms());
    for (size_t a = 0; a < db(15).natoms(); ++a)
      for (int c = 0; c < 3; ++c) CHECK(db(15).atoms[a].force[c] == plain(15).atoms[a].force[c]);
  }
  {
    INFO("geom.castep with stress calculation off");
    const std::vector<std::string> lines =
        substitute(read_lines(kGeomCastep), "stress calculation                             : on",
                   "stress calculation                             : off");
    TempFile tmp("tmp_castep_rv_stress_off.castep");
    tmp.write_lines(lines);
    StructureDB db;
    read_into<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == 8);
    INFO("label '" << db(0).label << "'");
    CHECK(db(0).sweight == 1.0);
    CHECK_THAT(db(0).label, Catch::Contains("S: (475 : 477)"));
    CHECK(db(0).stress(0, 0) == Approx(-90.011421 / kGPa).epsilon(1e-7));
  }
}

// Pins: the stress block CASTEP prints after ' Finished MD' (a fixed-cell MD
// computes the stress once, for its last step) goes to the last step when the
// geometry is the same and it has no stress of its own: md_no_stress.castep
// with ' Finished MD' and a stress block (diagonal -94.737913, -94.737913,
// -95.561163 GPa) appended gives step 2 that stress with sweight 1; steps 0
// and 1 keep sweight 0.
// Before the fix: the block was dropped.
TEST_CASE("CASTEP .castep reader: an MD run's final stress goes to the last step",
          "[castep_readers][castep][md][stress]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines("./tests_data/valid_castep_castep/md_no_stress.castep");
  const std::vector<std::string> go = read_lines(kGeomCastep);
  REQUIRE(go.at(687).find("Symmetrised Stress Tensor") != std::string::npos);
  const std::vector<std::string> stress = block_at(go, 687);
  lines.push_back(" Finished MD");
  lines.push_back("");
  lines.insert(lines.end(), stress.begin(), stress.end());
  TempFile tmp("tmp_castep_rv_md_final.castep");
  tmp.write_lines(lines);
  StructureDB db;
  read_into<CastepCastepReader>(db, tmp.path);
  REQUIRE(db.size() == 3);
  CHECK(db(0).sweight == 0.0);
  CHECK(db(1).sweight == 0.0);
  CHECK(db(2).sweight == 1.0);
  CHECK(db(2).stress(0, 0) == Approx(-94.737913 / kGPa).epsilon(1e-7));
  CHECK(db(2).stress(2, 2) == Approx(-95.561163 / kGPa).epsilon(1e-7));
}

// Pins: only the physical forces and stress blocks are read: 'Forces',
// 'Symmetrised Forces', 'Unconstrained Forces', 'Stress Tensor' and
// 'Symmetrised Stress Tensor'. CASTEP at iprint >= 2 prints the per-term
// breakdown before the forces ('Ewald forces', 'Local potential forces',
// ...): castep_triclinic_spe.castep with 'stress calculation : off' and an
// 'Ewald forces' block (every component 9.0) before its 'Symmetrised Forces'
// gives the same forces as without it, with force weight 1.
// Before the fix: the first 'Cartesian components (eV/A)' block completed the
// single point, so the Ewald term was stored as the forces. (Real case:
// FLiBe md_lastframe_elf/seed.castep, 358 eV/A off.)
TEST_CASE("CASTEP .castep reader: per-term force breakdown blocks are not stored",
          "[castep_readers][castep][forces]") {
  PeriodicTable::initialize();
  const std::string kSpe = "./tests_data/cell_convention/castep_triclinic_spe.castep";
  std::vector<std::string> lines =
      substitute(read_lines(kSpe), "stress calculation                             : on",
                 "stress calculation                             : off");
  TempFile ref("tmp_castep_rv_spe_off.castep");
  ref.write_lines(lines);
  REQUIRE(lines.at(66).find("Symmetrised Forces") != std::string::npos);
  std::vector<std::string> ewald = block_at(lines, 66);
  ewald[0] = " ********************************** Ewald forces **********************************";
  for (size_t k = 6; k + 2 < ewald.size(); ++k)
    for (int c = 0; c < 3; ++c) ewald[k] = replace_token(ewald[k], 3 + c, "9.00000");
  lines.insert(lines.begin() + 66, ewald.begin(), ewald.end());
  lines.insert(lines.begin() + 66 + ewald.size(), "");
  TempFile tmp("tmp_castep_rv_ewald.castep");
  tmp.write_lines(lines);
  StructureDB want, db;
  read_into<CastepCastepReader>(want, ref.path);
  read_into<CastepCastepReader>(db, tmp.path);
  REQUIRE(want.size() == 1);
  REQUIRE(db.size() == 1);
  CHECK(db(0).fweight == 1.0);
  REQUIRE(db(0).natoms() == want(0).natoms());
  for (size_t a = 0; a < db(0).natoms(); ++a)
    for (int c = 0; c < 3; ++c) CHECK(db(0).atoms[a].force[c] == want(0).atoms[a].force[c]);
}

// Pins: an MD step ends at the next step boundary ('... finished MD
// iteration', 'Starting MD iteration', ' Finished MD', the next run, the end
// of the file), not at its MD Data 'Temperature:' line, so a stress tensor
// that CASTEP 20.11/23.1 prints after the MD Data (ab initio properties
// sampled every N steps) is the step's; the 'Finite Temperature Pressure
// Tensor' that follows is not read. md_no_stress.castep with a 'Stress
// Tensor' (diagonal -94.737913, -94.737913, -95.561163 GPa) and a 'Finite
// Temperature Pressure Tensor' after step 1's MD Data gives step 1 that
// stress with sweight 1; steps 0 and 2 keep sweight 0.
// Before the fix: the step was stored at 'Temperature:' and its stress block
// dropped. (Real case: CH4 ch4-phasei.castep, every sampled stress lost.)
TEST_CASE("CASTEP .castep reader: an MD step's stress printed after its MD Data is kept",
          "[castep_readers][castep][md][stress]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines("./tests_data/valid_castep_castep/md_no_stress.castep");
  REQUIRE(lines.at(1607).find("Temperature:") != std::string::npos);
  REQUIRE(lines.at(1612).find("finished MD iteration") != std::string::npos);
  const std::vector<std::string> go = read_lines(kGeomCastep);
  REQUIRE(go.at(687).find("Symmetrised Stress Tensor") != std::string::npos);
  std::vector<std::string> stress = block_at(go, 687);
  stress[0] = " ***************** Stress Tensor *****************";
  std::vector<std::string> ftp = block_at(go, 687);
  ftp[0] = " ****** Finite Temperature Pressure Tensor *******";
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) ftp[6 + r] = replace_token(ftp[6 + r], 2 + c, r == c ? "7.000000" : "0.000000");
  std::vector<std::string> insert = stress;
  insert.push_back("");
  insert.insert(insert.end(), ftp.begin(), ftp.end());
  insert.push_back("");
  lines.insert(lines.begin() + 1611, insert.begin(), insert.end());
  TempFile tmp("tmp_castep_rv_md_sampled.castep");
  tmp.write_lines(lines);
  StructureDB db;
  read_into<CastepCastepReader>(db, tmp.path);
  REQUIRE(db.size() == 3);
  CHECK(db(0).sweight == 0.0);
  CHECK(db(2).sweight == 0.0);
  INFO("label '" << db(1).label << "'");
  CHECK(db(1).sweight == 1.0);
  CHECK(db(1).stress(0, 0) == Approx(-94.737913 / kGPa).epsilon(1e-7));
  CHECK(db(1).stress(2, 2) == Approx(-95.561163 / kGPa).epsilon(1e-7));
  CHECK(db(1).T == Approx(563.548574));
}

// Pins: the last MD step ends at ' Finished MD' when the file has no
// '... finished MD iteration' line for it: md_no_stress.castep with a stress
// block (diagonal -94.737913, -94.737913, -95.561163 GPa) after step 2's MD
// Data, then ' Finished MD' and a second stress block (diagonal -1, -2, -3
// GPa), keeps the first for step 2: its own stress (the block printed after
// the run's end goes to the last step only when that has none).
// Before the fix: step 2 was stored at its Temperature line, so its own block
// was dropped and the second was stored as its stress. (With the step open
// until the end of the file instead, the second block replaces the first.)
TEST_CASE("CASTEP .castep reader: ' Finished MD' ends the last MD step",
          "[castep_readers][castep][md][stress]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines("./tests_data/valid_castep_castep/md_no_stress.castep");
  REQUIRE(lines.back().find("Temperature:") == std::string::npos);
  REQUIRE(lines.at(2248).find("Temperature:") != std::string::npos);
  const std::vector<std::string> go = read_lines(kGeomCastep);
  REQUIRE(go.at(687).find("Symmetrised Stress Tensor") != std::string::npos);
  const std::vector<std::string> own = block_at(go, 687);
  std::vector<std::string> after = block_at(go, 687);
  const char *diag[3] = {"-1.000000", "-2.000000", "-3.000000"};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      after[6 + r] = replace_token(after[6 + r], 2 + c, r == c ? diag[r] : "0.000000");
  lines.push_back("");
  lines.insert(lines.end(), own.begin(), own.end());
  lines.push_back("");
  lines.push_back(" Finished MD");
  lines.push_back("");
  lines.insert(lines.end(), after.begin(), after.end());
  TempFile tmp("tmp_castep_rv_md_finished.castep");
  tmp.write_lines(lines);
  StructureDB db;
  read_into<CastepCastepReader>(db, tmp.path);
  REQUIRE(db.size() == 3);
  INFO("label '" << db(2).label << "'");
  CHECK(db(2).sweight == 1.0);
  CHECK(db(2).stress(0, 0) == Approx(-94.737913 / kGPa).epsilon(1e-7));
  CHECK(db(2).stress(2, 2) == Approx(-95.561163 / kGPa).epsilon(1e-7));
  CHECK(db(2).T == Approx(962.274424));
}

// Pins: 'Unconstrained Forces', the physical forces a constrained
// optimisation prints before its 'Constrained Forces', are read: geom.castep
// with its iteration-1 'Symmetrised Forces' banner renamed 'Unconstrained
// Forces' and a 'Constrained Forces' block (every component 0.5) after it
// gives structure 2 force weight 1 and H1 force z 0.00189 eV/A.
// A pin of the list of physical blocks: without 'Unconstrained Forces' in it,
// neither block is read and the structure gets force priority 0.
TEST_CASE("CASTEP .castep reader: 'Unconstrained Forces' are the physical forces",
          "[castep_readers][castep][constraints]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = read_lines(kGeomCastep);
  REQUIRE(lines.at(668).find("Symmetrised Forces") != std::string::npos);
  std::vector<std::string> constrained = block_at(lines, 668);
  REQUIRE(constrained.size() == 18);
  constrained[0] = " ****************************** Constrained Forces ******************************";
  for (size_t k = 6; k + 2 < constrained.size(); ++k)
    for (int c = 0; c < 3; ++c) constrained[k] = replace_token(constrained[k], 3 + c, "0.50000");
  lines[668] = " ***************************** Unconstrained Forces *****************************";
  lines.insert(lines.begin() + 686, constrained.begin(), constrained.end());
  TempFile tmp("tmp_castep_rv_unconstrained.castep");
  tmp.write_lines(lines);
  StructureDB db;
  read_into<CastepCastepReader>(db, tmp.path);
  REQUIRE(db.size() == 8);
  CHECK(db(1).fweight == 1.0);
  CHECK(db(1).atoms[0].force[2] == Approx(0.00189).margin(1e-12));
  CHECK(db(1).atoms[0].force[0] == Approx(0.0).margin(1e-12));
}

// Pins: a run whose type is not single point energy, band structure,
// geometry optimization or molecular dynamics stores nothing, also when it
// does the finite basis set correction: castep_triclinic_spe.castep as a
// 'phonon calculation' with a 'Calculating finite basis set correction' line
// gives no structure.
// Before the fix: the correction's state stored its SCF as a 'CASTEP FBSC'
// structure.
TEST_CASE("CASTEP .castep reader: an unsupported run type stores nothing, FBSC included",
          "[castep_readers][castep][fbsc]") {
  PeriodicTable::initialize();
  std::vector<std::string> lines = substitute(
      read_lines("./tests_data/cell_convention/castep_triclinic_spe.castep"),
      "type of calculation                            : single point energy",
      "type of calculation                            : phonon calculation");
  lines.insert(lines.begin() + 17, "Calculating finite basis set correction with  3 cut-off energies.");
  TempFile tmp("tmp_castep_rv_phonon_fbsc.castep");
  tmp.write_lines(lines);
  StructureDB db;
  read_into<CastepCastepReader>(db, tmp.path);
  CHECK(db.size() == 0);
}

// Pins: atoms read from a .castep file carry their element, Z included:
// geom.castep (8 H and 2 Ra per structure) gives Z 1 and 88, and
// StructureDB::get_unique_elements() has both.
// Before the fix: only the symbol was copied, Z stayed 0, and the set of
// unique elements (ordered by Z) collapsed to one entry, 'H'.
TEST_CASE("CASTEP .castep reader: atoms carry their atomic number",
          "[castep_readers][castep][Z]") {
  PeriodicTable::initialize();
  StructureDB db;
  read_into<CastepCastepReader>(db, kGeomCastep);
  REQUIRE(db.size() >= 1);
  for (size_t s = 0; s < db.size(); ++s) {
    // Atoms per (symbol, Z) pair.
    size_t h_1 = 0, ra_88 = 0, other = 0;
    std::string seen;
    for (const Atom &a : db(s).atoms) {
      if (a.symbol == "H" && a.Z == 1) ++h_1;
      else if (a.symbol == "Ra" && a.Z == 88) ++ra_88;
      else ++other;
      seen += " " + a.symbol + "(Z " + std::to_string(a.Z) + ")";
    }
    INFO("structure " << s << ":" << seen);
    CHECK(h_1 == 8);
    CHECK(ra_88 == 2);
    CHECK(other == 0);
  }
  const std::set<Element> unique = db.get_unique_elements();
  REQUIRE(unique.size() == 2);
  CHECK(unique.begin()->Z == 1);
  CHECK(unique.begin()->symbol == "H");
  CHECK(unique.rbegin()->Z == 88);
  CHECK(unique.rbegin()->symbol == "Ra");
}

// --------------------------------------------------------------------
// Species labels with a CASTEP tag
// --------------------------------------------------------------------

// Pins: a CASTEP species label 'El:tag' is read as the element El (the tag
// is dropped), in all three readers:
//  - .cell: 'Ta:up' and 'Ta:dn' -> two Ta atoms (Z 73) at (0,0,0) and
//    (1.65,1.65,1.65) in a 3.3 A cube;
//  - .castep: geom.castep with Ra written 'Ra:1' -> the same structures,
//    symbols, Z, positions and energies as geom.castep;
//  - .md: test1.md with Si written 'Si:1' -> the same structures as test1.md,
//    stress included (the kinetic term uses the Si mass).
// Before the fix: each file was refused with 'Unknown symbol: Ta:up' /
// 'Ra:1' / 'Si:1'.
TEST_CASE("CASTEP readers: a species label with a tag is read as its element",
          "[castep_readers][species_tag]") {
  PeriodicTable::initialize();
  SECTION(".cell") {
    TempFile tmp("tmp_castep_rv_tagged.cell");
    tmp.write_text(
        "%BLOCK LATTICE_CART\nANG\n3.3 0 0\n0 3.3 0\n0 0 3.3\n%ENDBLOCK LATTICE_CART\n"
        "%BLOCK POSITIONS_FRAC\nTa:up 0.0 0.0 0.0\nTa:dn 0.5 0.5 0.5\n"
        "%ENDBLOCK POSITIONS_FRAC\n");
    CastepCellReader reader;
    REQUIRE_NOTHROW(reader.read(tmp.path));
    const Structure st = reader.getStructure();
    REQUIRE(st.natoms() == 2);
    for (size_t k = 0; k < 2; ++k) {
      INFO("atom " << k);
      CHECK(st.atoms[k].symbol == "Ta");
      CHECK(st.atoms[k].Z == 73);
      for (int c = 0; c < 3; ++c)
        CHECK(st.atoms[k].position[c] == Approx(k == 0 ? 0.0 : 1.65).margin(1e-12));
    }
  }
  SECTION(".castep") {
    TempFile tmp("tmp_castep_rv_tagged.castep");
    tmp.write_lines(substitute(read_lines(kGeomCastep), "x  Ra ", "x  Ra:1 "));
    StructureDB plain, tagged;
    read_into<CastepCastepReader>(plain, kGeomCastep);
    REQUIRE_NOTHROW(read_into<CastepCastepReader>(tagged, tmp.path));
    REQUIRE(tagged.size() == plain.size());
    for (size_t s = 0; s < plain.size(); ++s) {
      INFO("structure " << s);
      CHECK(tagged(s).energy == plain(s).energy);
      REQUIRE(tagged(s).natoms() == plain(s).natoms());
      for (size_t k = 0; k < plain(s).natoms(); ++k) {
        INFO("atom " << k);
        CHECK(tagged(s).atoms[k].symbol == plain(s).atoms[k].symbol);
        CHECK(tagged(s).atoms[k].Z == PeriodicTable::find_by_symbol(plain(s).atoms[k].symbol).Z);
        for (int c = 0; c < 3; ++c)
          CHECK(tagged(s).atoms[k].position[c] == plain(s).atoms[k].position[c]);
      }
    }
  }
  SECTION(".md") {
    TempFile tmp("tmp_castep_rv_tagged.md");
    tmp.write_lines(substitute(read_lines(kMdTest1), " Si ", " Si:1 "));
    StructureDB plain, tagged;
    read_into<CastepMDReader>(plain, kMdTest1);
    REQUIRE_NOTHROW(read_into<CastepMDReader>(tagged, tmp.path));
    REQUIRE(tagged.size() == plain.size());
    for (size_t s = 0; s < plain.size(); ++s) {
      INFO("structure " << s);
      CHECK(tagged(s).energy == plain(s).energy);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) CHECK(tagged(s).stress(i, j) == plain(s).stress(i, j));
      REQUIRE(tagged(s).natoms() == plain(s).natoms());
      for (size_t k = 0; k < plain(s).natoms(); ++k) {
        INFO("atom " << k);
        CHECK(tagged(s).atoms[k].symbol == "Si");
        CHECK(tagged(s).atoms[k].Z == 14);
        for (int c = 0; c < 3; ++c)
          CHECK(tagged(s).atoms[k].position[c] == plain(s).atoms[k].position[c]);
      }
    }
  }
}

// Pins: a tagged label whose part before ':' is not an element ('Qq:up' in
// a .cell, 'Zz:1' in a .castep, 'Xx:1' in a .md) is refused with a
// std::runtime_error naming the whole label.
// Passes before the fix (the whole label went to the periodic-table lookup,
// 'Unknown symbol: Xx:1'); a fix that strips the tag first and then reports
// only 'Xx', or maps an unknown prefix to Z = 0, makes it fail.
TEST_CASE("CASTEP readers: a tagged label with an unknown element is refused by name",
          "[castep_readers][species_tag]") {
  PeriodicTable::initialize();
  SECTION(".cell") {
    TempFile tmp("tmp_castep_rv_unknown.cell");
    tmp.write_text(
        "%BLOCK LATTICE_CART\nANG\n3.3 0 0\n0 3.3 0\n0 0 3.3\n%ENDBLOCK LATTICE_CART\n"
        "%BLOCK POSITIONS_FRAC\nQq:up 0.0 0.0 0.0\n%ENDBLOCK POSITIONS_FRAC\n");
    CastepCellReader r1;
    CHECK_THROWS_AS(r1.read(tmp.path), std::runtime_error);
    CastepCellReader r2;
    CHECK_THROWS_WITH(r2.read(tmp.path), Catch::Contains("Qq:up"));
  }
  SECTION(".castep") {
    TempFile tmp("tmp_castep_rv_unknown.castep");
    tmp.write_lines(substitute(read_lines(kGeomCastep), "x  Ra ", "x  Zz:1 "));
    const std::string msg = parse_error<CastepCastepReader>(tmp.path);
    INFO("message: '" << msg << "'");
    CHECK_THAT(msg, Catch::Contains("Zz:1"));
  }
  SECTION(".md") {
    TempFile tmp("tmp_castep_rv_unknown.md");
    tmp.write_lines(substitute(read_lines(kMdTest1), " Si ", " Xx:1 "));
    const std::string msg = parse_error<CastepMDReader>(tmp.path);
    INFO("message: '" << msg << "'");
    CHECK_THAT(msg, Catch::Contains("Xx:1"));
  }
}

// Pins: an error of the CASTEP .cell reader names the file, as
//   CastepCellReader: CASTEP cell file <path>: <what>
// for a species label that is no element ('Qq:up', the second atom: exactly
// "... <path>: CASTEP species label 'Qq:up' does not start with an element
// symbol"), a file without a positions block and an unknown length unit
// ('furlong'). A reader that read a good .cell (2 Ta atoms in a 3.3 A cube)
// keeps its structure (label, cell and atoms) after each refused file, and
// reading the good file again gives its 2 atoms, not 4.
// Before the fix: the messages began "CastepCellReader: " or "CASTEP species
// label" and named no file; the file refused at its second atom had already
// replaced the reader's cell (4 A) and appended its first atom (3 atoms), so
// the structure checks of every later refused file failed too, and reading
// the good file again appended its atoms (5).
TEST_CASE("CASTEP .cell reader: an error names the file, and a refused file "
          "leaves the structure",
          "[castep_readers][cell][species]") {
  PeriodicTable::initialize();
  const std::string cube4 =
      "%BLOCK LATTICE_CART\nANG\n4.0 0 0\n0 4.0 0\n0 0 4.0\n%ENDBLOCK LATTICE_CART\n";
  TempFile good("tmp_castep_rv_source_good.cell");
  good.write_text(
      "%BLOCK LATTICE_CART\nANG\n3.3 0 0\n0 3.3 0\n0 0 3.3\n%ENDBLOCK LATTICE_CART\n"
      "%BLOCK POSITIONS_FRAC\nTa 0.0 0.0 0.0\nTa 0.5 0.5 0.5\n"
      "%ENDBLOCK POSITIONS_FRAC\n");
  struct Bad {
    std::string name, body, what;
  };
  const std::vector<Bad> bad = {
      {"tmp_castep_rv_source_species.cell",
       cube4 + "%BLOCK POSITIONS_FRAC\nTa 0.0 0.0 0.0\nQq:up 0.5 0.5 0.5\n"
               "%ENDBLOCK POSITIONS_FRAC\n",
       "CASTEP species label 'Qq:up' does not start with an element symbol"},
      {"tmp_castep_rv_source_nopos.cell", cube4, "no positions block"},
      {"tmp_castep_rv_source_unit.cell",
       "%BLOCK LATTICE_CART\nfurlong\n4.0 0 0\n0 4.0 0\n0 0 4.0\n"
       "%ENDBLOCK LATTICE_CART\n%BLOCK POSITIONS_FRAC\nTa 0.0 0.0 0.0\n"
       "%ENDBLOCK POSITIONS_FRAC\n",
       "unknown length unit 'furlong'"}};
  CastepCellReader reader;
  REQUIRE_NOTHROW(reader.read(good.path));
  const Structure before = reader.getStructure();
  REQUIRE(before.natoms() == 2);
  for (const auto &b : bad) {
    TempFile tmp(b.name);
    tmp.write_text(b.body);
    INFO(b.name << ":\n" << b.body);
    std::string msg;
    try {
      reader.read(tmp.path);
    } catch (const std::runtime_error &e) {
      msg = e.what();
    }
    INFO("message: '" << msg << "'");
    const std::string head = "CastepCellReader: CASTEP cell file " + tmp.path + ": ";
    CHECK(msg.rfind(head, 0) == 0);
    CHECK_THAT(msg, Catch::Contains(b.what));
    if (b.name == "tmp_castep_rv_source_species.cell") CHECK(msg == head + b.what);
    const Structure after = reader.getStructure();
    CHECK(after.label == before.label);
    CHECK(after.natoms() == 2);
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) CHECK(after.cell(i, k) == before.cell(i, k));
  }
  REQUIRE_NOTHROW(reader.read(good.path));
  CHECK(reader.getStructure().natoms() == 2);
}

// --------------------------------------------------------------------
// CASTEP cell writer
// --------------------------------------------------------------------

// Pins: CastepCellWriter separates every field, so a Ta atom at
// (-1234.567890123, 1.65, 10000.25) in a 3.3 A cube reads back through
// CastepCellReader with the written coordinates: with the default precision
// (18) to 1e-14 relative (the decimal text round-trips the double), and with
// set_precision(5) (--numeric 5) to half a unit in the last written digit
// (5e-6 absolute for fixed notation, or 5e-5 relative for 5 significant
// digits). Every LATTICE_CART row has 3 fields and every POSITIONS_ABS row 4,
// and no line ends in whitespace (the '%ENDBLOCK POSITIONS_ABS' line had a
// trailing tab).
// Before the fix: fields were written with setw(p+6) and no separator, so
// the far atom's row fused into two tokens, 'Ta-1234.567890123000097446' and
// '1.64999999999999991110000.250000000000000000', CastepCellReader refused
// the file ("POSITIONS_ABS line needs a species and three coordinates"), and
// the '%ENDBLOCK POSITIONS_ABS' line ended in a tab.
TEST_CASE("CASTEP .cell writer: every field is separated so a far atom round-trips",
          "[castep_readers][castep_cell_writer]") {
  PeriodicTable::initialize();
  StructureDB db;
  {
    Structure st;
    st.label = "two Ta atoms, one unwrapped far outside the cell";
    st.energy = 0.0;
    st.cell.set_zero();
    st.cell(0, 0) = st.cell(1, 1) = st.cell(2, 2) = 3.3;
    st.stress.set_zero();
    st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("Ta"), -1234.567890123, 1.65, 10000.25));
    db.add(st);
  }
  const double far[3] = {-1234.567890123, 1.65, 10000.25};

  for (const int precision : {18, 5}) {
    INFO("precision " << precision);
    TempFile tmp("tmp_castep_rv_far.cell");
    CastepCellWriter writer(db);
    if (precision != 18) writer.set_precision(static_cast<size_t>(precision));
    writer.write_data(tmp.path, 0);

    // Field counts and trailing whitespace, from the text.
    const std::vector<std::string> lines = read_lines(tmp.path);
    std::string block;
    for (const auto &l : lines) {
      INFO("line '" << l << "'");
      CHECK((l.empty() || !std::isspace(static_cast<unsigned char>(l.back()))));
      std::istringstream iss(l);
      std::vector<std::string> toks;
      std::string t;
      while (iss >> t) toks.push_back(t);
      if (toks.empty() || toks[0][0] == '#') continue;
      if (lower(toks[0]) == "%block") {
        block = toks.size() > 1 ? toks[1] : "";
        continue;
      }
      if (lower(toks[0]) == "%endblock") {
        block.clear();
        continue;
      }
      if (lower(toks[0]) == "ang") continue;
      if (block == "LATTICE_CART") CHECK(toks.size() == 3);
      if (block == "POSITIONS_ABS") CHECK(toks.size() == 4);
    }

    CastepCellReader reader;
    REQUIRE_NOTHROW(reader.read(tmp.path));
    const Structure back = reader.getStructure();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        CHECK(back.cell(i, j) == Approx(i == j ? 3.3 : 0.0).margin(1e-12));
    REQUIRE(back.natoms() == 2);
    for (size_t k = 0; k < 2; ++k) {
      CHECK(back.atoms[k].symbol == "Ta");
      for (int c = 0; c < 3; ++c) {
        INFO("atom " << k << " component " << c);
        const double want = k == 0 ? 0.0 : far[c];
        const double tol = precision == 18
                               ? 1e-14 * std::abs(want)
                               : 0.5 * std::pow(10.0, -precision) +
                                     0.5 * std::pow(10.0, 1 - precision) * std::abs(want);
        CHECK(std::abs(back.atoms[k].position[c] - want) <= tol);
      }
    }
  }
}

// ---- every error of the .castep reader names the file and the line --------

namespace {

const std::string kMdStressCastep =
    "./tests_data/valid_castep_castep/md_stress.castep";

/* The message of the exception that parsing `path` with Reader throws (any
 * std::exception: a std::stoi failure included), or "no exception". */
template <class Reader>
std::string any_refusal(const std::string &path) {
  StructureDB db;
  std::ostringstream err;
  std::streambuf *old_err = std::cerr.rdbuf(err.rdbuf());
  std::string msg = "no exception";
  try {
    read_into<Reader>(db, path);
  } catch (const std::exception &e) {
    msg = e.what();
  }
  std::cerr.rdbuf(old_err);
  return msg;
}

/* `lines` with line `n` (1-based) holding `to` where it held `from`. */
std::vector<std::string> edited(std::vector<std::string> lines, size_t n,
                                const std::string &from, const std::string &to) {
  INFO("line " << n << ": '" << lines.at(n - 1) << "'");
  const auto p = lines.at(n - 1).find(from);
  REQUIRE(p != std::string::npos);
  lines[n - 1].replace(p, from.size(), to);
  return lines;
}

}  // namespace

/* Pins: the .castep reader refuses a value it cannot read, naming the file,
 * the line and what it reads there (geom.castep, and md_stress.castep for
 * the MD lines):
 *  - lattice vector 1 of a Unit Cell block 'abc' (line 271);
 *  - the number of ions 'x', or too large for any count (line 287);
 *  - the position of atom 1 'abc' (line 295), and the species 'Zz:1' (line
 *    303, which also names the label);
 *  - 'Final energy, E' and 'Final free energy (E-TS)' 'abc' (lines 393, 394);
 *  - the iteration number of 'LBFGS: finished iteration' and 'Starting LBFGS
 *    iteration' 'x' (lines 482, 495);
 *  - the MD 'Potential Energy:' and 'Temperature:' 'abc' (lines 505, 511);
 *  - a missing file: "CASTEP .castep <path>: cannot open the file".
 * Each message is "CASTEP .castep <path>: line <n>: ..." as listed below.
 * Before the fix: a 'Warning: ...' line on standard error with the line
 * number but not the file, and the file was read on (the cell partly
 * written, no atoms, an atom less, another energy line or none: no
 * exception); the MD potential energy as "... structure 1 (the MD step
 * ending at line ...) has no energy of its own ..."; the species as "CASTEP
 * species label 'Zz:1' does not start with an element symbol" (no file, no
 * line); the iteration numbers threw std::stoi's "stoi"; the missing file
 * "Could not open the file: <path>". */
TEST_CASE("CASTEP .castep reader: a value it cannot read is refused, naming "
          "the file and the line",
          "[castep_readers][source]") {
  PeriodicTable::initialize();
  struct Case {
    std::string name, fixture;
    size_t line;
    std::string from, to, what;
  };
  const std::vector<Case> cases = {
      {"cell", kGeomCastep, 271, "2.9640232    -0.0000000", "abc    -0.0000000",
       "lattice vector 1 of the Unit Cell block, 'abc    -0.0000000    "
       "-0.0000000        2.119816505   0.000000000   0.000000000', is not "
       "three numbers"},
      {"natoms", kGeomCastep, 287, "=   10", "=   x",
       "cannot read the number of ions from 'Total number of ions in cell =   "
       "x'"},
      {"natoms_huge", kGeomCastep, 287, "=   10",
       "=   99999999999999999999",
       "cannot read the number of ions from 'Total number of ions in cell =   "
       "99999999999999999999'"},
      {"position", kGeomCastep, 295, "-0.000000  -0.000000   0.318951",
       "abc  -0.000000   0.318951",
       "the position of atom 1, 'x  H            1        abc  -0.000000   "
       "0.318951   x', is not 'x <species> <number> u v w x'"},
      {"species", kGeomCastep, 303, "x  Ra ", "x  Zz:1 ",
       "CASTEP species label 'Zz:1' does not start with an element symbol"},
      {"energy", kGeomCastep, 393, "-2690.122546861", "abc",
       "cannot read the energy from 'Final energy, E             =  abc     "
       "eV'"},
      {"free_energy", kGeomCastep, 394, "-2690.157080108", "abc",
       "cannot read the energy from 'Final free energy (E-TS)    =  abc     "
       "eV'"},
      {"finished", kGeomCastep, 482, "iteration     0", "iteration     x",
       "cannot read the iteration number from 'LBFGS: finished iteration     "
       "x with enthalpy= -2.65320186E+003 eV'"},
      {"starting", kGeomCastep, 495, "iteration          1",
       "iteration          x",
       "cannot read the iteration number from 'Starting LBFGS iteration      "
       "    x ...'"},
      {"potential", kMdStressCastep, 505, "-2385.695732", "abc",
       "cannot read the potential energy from 'x   Potential Energy:  abc    "
       "               eV     x'"},
      {"temperature", kMdStressCastep, 511, "300.000000", "abc",
       "cannot read the temperature from 'x        Temperature:    abc       "
       "             K     x'"},
  };
  for (const Case &c : cases) {
    INFO(c.name);
    TempFile tmp("tmp_castep_source_" + c.name + ".castep");
    tmp.write_lines(edited(read_lines(c.fixture), c.line, c.from, c.to));
    CHECK(any_refusal<CastepCastepReader>(tmp.path) ==
          "CASTEP .castep " + tmp.path + ": line " + std::to_string(c.line) +
              ": " + c.what);
  }
  const std::string missing = "./tmp_castep_source_missing.castep";
  std::remove(missing.c_str());
  CHECK(any_refusal<CastepCastepReader>(missing) ==
        "CASTEP .castep " + missing + ": cannot open the file");
  CHECK(any_refusal<CastepMDReader>("./tmp_castep_source_missing.md") ==
        "CASTEP ./tmp_castep_source_missing.md: cannot open the file");
}

/* Pins: a forces or stress line of a .castep file that cannot be read (a
 * value CASTEP could not print, '*****') is no forces, or no stress, for that
 * structure: it is stored with force priority 0 (zero forces), or stress
 * weight 0 (zero stress), with one WARNING naming the file and the line
 * ("CASTEP .castep <path>: line 456: the forces line of atom 1, '...', cannot
 * be read; the structure is stored without forces"; "... line 475: the
 * stress line 1, '...', cannot be read; the structure is stored without
 * stress"); a file that ends inside a block warns naming the file, the block
 * and the line ("CASTEP .castep <path>: the file ends inside the forces
 * block at line 460").
 * Before the fix: the same weights, with 'Warning: Could not parse atomic
 * forces (line: 456).', 'Warning: Could not parse stress row (line: 475).'
 * and 'Warning: Truncated file reading forces at row 5 (line: 460).', which
 * name no file. */
TEST_CASE("CASTEP .castep reader: an unreadable forces or stress line gives "
          "that priority 0, with a warning naming the file",
          "[castep_readers][source]") {
  PeriodicTable::initialize();
  StructureDB good;
  read_into<CastepCastepReader>(good, kGeomCastep);
  REQUIRE(good.size() > 1);
  REQUIRE(good(0).fweight == 1.0);
  REQUIRE(good(0).sweight == 1.0);
  {
    TempFile tmp("tmp_castep_source_forces.castep");
    tmp.write_lines(edited(read_lines(kGeomCastep), 456, "0.00000   ", "*******   "));
    StructureDB db;
    const std::string printed = parse_capturing<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == good.size());
    CHECK(db(0).fweight == 0.0);
    CHECK(forces_are_zero(db(0)));
    CHECK(db(1).fweight == 1.0);
    CHECK_THAT(printed, Catch::Contains("CASTEP .castep " + tmp.path +
                                        ": line 456: the forces line of atom "
                                        "1, '"));
    CHECK_THAT(printed, Catch::Contains("', cannot be read; the structure is "
                                        "stored without forces"));
  }
  {
    TempFile tmp("tmp_castep_source_stress.castep");
    tmp.write_lines(edited(read_lines(kGeomCastep), 475, "-90.011421", "*********"));
    StructureDB db;
    const std::string printed = parse_capturing<CastepCastepReader>(db, tmp.path);
    REQUIRE(db.size() == good.size());
    CHECK(db(0).sweight == 0.0);
    CHECK(stress_is_zero(db(0)));
    CHECK(db(1).sweight == 1.0);
    CHECK_THAT(printed, Catch::Contains("CASTEP .castep " + tmp.path +
                                        ": line 475: the stress line 1, '"));
    CHECK_THAT(printed, Catch::Contains("', cannot be read; the structure is "
                                        "stored without stress"));
  }
  {
    TempFile tmp("tmp_castep_source_cut.castep");
    std::vector<std::string> l = read_lines(kGeomCastep);
    l.resize(460);
    tmp.write_lines(l);
    StructureDB db;
    const std::string printed = parse_capturing<CastepCastepReader>(db, tmp.path);
    CHECK_THAT(printed, Catch::Contains("CASTEP .castep " + tmp.path +
                                        ": the file ends inside the forces "
                                        "block at line 460"));
    CHECK_THAT(printed, !Catch::Contains("Truncated file"));
  }
}

/* Pins: a .castep file that ends inside a step outside its blocks (a run
 * stopped during the step's SCF, or between its blocks) keeps the steps
 * before and skips that step with one WARNING naming the file, the line,
 * the step and the line it starts at:
 *  - md_stress.castep cut after line 900 (MD step 2's forces and stress
 *    read, before its MD Data): steps 0 and 1 and "WARNING: CASTEP .castep
 *    <path>: the file ends at line 900 inside MD step 2, which starts at
 *    line 725 and was skipped; 2 structure(s) read."; cut after line 520
 *    (MD step 1 just started), step 0 and the warning for MD step 1 at line
 *    517;
 *  - geom.castep cut after line 700 (inside LBFGS iteration 1): iteration 0
 *    and the warning for "geometry optimisation iteration 1, which starts
 *    at line 495"; cut after line 440 (inside iteration 0, the finite basis
 *    set correction's SCFs): nothing and the warning for "geometry
 *    optimisation iteration 0, which starts at line 371";
 *  - Gd0.castep cut after line 5000 (its SCF energy read, before its
 *    forces): nothing and the warning for "the band-structure calculation,
 *    which starts at line 117".
 * A cut after a step's last line (md_stress after line 921, after
 * '... finished MD iteration 2'; geom after line 720, after 'finished
 * iteration 1') keeps the steps up to it without a warning, as the whole
 * files do. (Pins as well.)
 * Before the fix: the same structures, and no warning for any cut. */
TEST_CASE("CASTEP .castep reader: a file that ends inside a step outside its "
          "blocks warns, naming the file and the step",
          "[castep_readers][source][truncated]") {
  PeriodicTable::initialize();
  const std::string kGd0 = "./tests_data/valid_castep_castep/Gd0.castep";
  struct Cut {
    std::string file;
    size_t after;  // the last line kept, 0 for the whole file
    size_t stored;
    std::string step;  // the step warned about, empty for none
  };
  const Cut cuts[] = {
      {kMdStressCastep, 900, 2, "MD step 2, which starts at line 725"},
      {kMdStressCastep, 520, 1, "MD step 1, which starts at line 517"},
      {kMdStressCastep, 921, 3, ""},
      {kMdStressCastep, 0, 4, ""},
      {kGeomCastep, 700, 1,
       "geometry optimisation iteration 1, which starts at line 495"},
      {kGeomCastep, 440, 0,
       "geometry optimisation iteration 0, which starts at line 371"},
      {kGeomCastep, 720, 2, ""},
      {kGd0, 5000, 0,
       "the band-structure calculation, which starts at line 117"}};
  for (const Cut &c : cuts) {
    INFO(c.file << " cut after line " << c.after);
    std::vector<std::string> l = read_lines(c.file);
    if (c.after > 0) l.resize(c.after);
    TempFile tmp("tmp_castep_ends_step_" + std::to_string(c.after) + ".castep");
    tmp.write_lines(l);
    StructureDB db;
    const std::string printed = parse_capturing<CastepCastepReader>(db, tmp.path);
    INFO("printed: '" << printed << "'");
    CHECK(db.size() == c.stored);
    if (c.step.empty()) {
      CHECK(count_ci(printed, "warning") == 0);
    } else {
      CHECK(count_ci(printed, "warning") == 1);
      CHECK_THAT(printed,
                 Catch::Contains("WARNING: CASTEP .castep " + tmp.path +
                                 ": the file ends at line " +
                                 std::to_string(c.after) + " inside " + c.step +
                                 " and was skipped; " +
                                 std::to_string(c.stored) +
                                 " structure(s) read."));
    }
  }
}

// Pins: the output units of a .castep run (its 'output <quantity> unit :'
// header lines, which each run prints after its 'type of calculation'
// line) are those the reader reads -- length A, energy eV, force eV/A,
// pressure GPa -- or the file is refused, naming the file, the line, the
// quantity and the unit. Copies of md_stress.castep with, on the header
// line of the quantity (line 153, 158, 159 or 161):
//  - 'output length unit : Bohr' (the cells and positions printed in bohr);
//  - 'output energy unit : Ha';
//  - 'output force unit : Ha/Bohr' and its forces boxes headed
//    'Cartesian components (Ha/Bohr)';
//  - 'output pressure unit : kbar' and its stress boxes headed
//    'Cartesian components (kbar)'.
// Each refusal message has the file name, 'line <n>', '<quantity> unit',
// the unit in quotes and the unit read. combined.castep with 'output energy
// unit : Ha' in its second run only (line 2331) is refused naming that line:
// the units are each run's. Unmodified, md_stress.castep and combined.castep
// read as before (4 and 15 structures).
// Before the fix: no unit line was read. The bohr and Ha files were stored
// as if in A and eV (every structure the same as the unmodified file's); the
// Ha/Bohr file was stored with zero forces and force priority 0, the kbar
// file with zero stress and stress priority 0, and combined.castep's
// second run in eV: each parsed without an error.
TEST_CASE("CASTEP .castep reader: a run whose output units are not A, eV, "
          "eV/A and GPa is refused, naming the line and the unit",
          "[castep_readers][castep][units]") {
  PeriodicTable::initialize();
  const std::string kMdStress =
      "./tests_data/valid_castep_castep/md_stress.castep";
  const std::string kCombined =
      "./tests_data/valid_castep_castep/combined.castep";
  struct Unit {
    std::string quantity;  // as the header line names it
    std::string unit;      // the unit of the copy
    std::string read;      // the unit the reader reads
    std::string box_from, box_to;  // the boxes' header, if they name it
  };
  const std::vector<Unit> units = {
      {"length", "Bohr", "A", "", ""},
      {"energy", "Ha", "eV", "", ""},
      {"force", "Ha/Bohr", "eV/A", "Cartesian components (eV/A)",
       "Cartesian components (Ha/Bohr)"},
      {"pressure", "kbar", "GPa", "Cartesian components (GPa)",
       "Cartesian components (kbar)"}};
  // The index of the header line of `quantity` in `lines`, at or after
  // `from`: ' output <blanks> <quantity> unit <blanks> : <unit>'.
  auto header_line = [](const std::vector<std::string> &lines,
                        const std::string &quantity, size_t from) {
    for (size_t i = from; i < lines.size(); ++i) {
      std::istringstream ls(lines[i]);
      std::string w0, w1, w2;
      if (ls >> w0 >> w1 >> w2 && w0 == "output" && w1 == quantity &&
          w2 == "unit")
        return i;
    }
    FAIL("no 'output " << quantity << " unit' line after line " << from);
    return size_t(0);
  };
  {
    StructureDB db;
    read_into<CastepCastepReader>(db, kMdStress);
    CHECK(db.size() == 4);
    StructureDB c;
    read_into<CastepCastepReader>(c, kCombined);
    CHECK(c.size() == 15);
  }
  for (const Unit &u : units) {
    INFO("output " << u.quantity << " unit : " << u.unit);
    std::vector<std::string> lines = read_lines(kMdStress);
    const size_t at = header_line(lines, u.quantity, 0);
    const std::string old = lines[at];
    lines[at] = old.substr(0, old.find(':') + 1) + " " + u.unit;
    if (!u.box_from.empty()) lines = substitute(lines, u.box_from, u.box_to);
    TempFile tmp("tmp_castep_rv_unit_" + u.quantity + ".castep");
    tmp.write_lines(lines);
    const std::string what = parse_error<CastepCastepReader>(tmp.path);
    INFO("error: '" << what << "'");
    CHECK_THAT(what, Catch::Contains(tmp.basename()));
    CHECK_THAT(what, Catch::Contains("line " + std::to_string(at + 1) + ":"));
    CHECK_THAT(what, Catch::Contains(u.quantity + " unit"));
    CHECK_THAT(what, Catch::Contains("'" + u.unit + "'"));
    CHECK_THAT(what, Catch::Contains(" " + u.read));
  }
  {
    INFO("combined.castep, energy unit Ha in its second run");
    std::vector<std::string> lines = read_lines(kCombined);
    const size_t first = header_line(lines, "energy", 0);
    const size_t second = header_line(lines, "energy", first + 1);
    REQUIRE(second + 1 == 2331);
    lines[second] = lines[second].substr(0, lines[second].find(':') + 1) + " Ha";
    TempFile tmp("tmp_castep_rv_unit_run2.castep");
    tmp.write_lines(lines);
    const std::string what = parse_error<CastepCastepReader>(tmp.path);
    INFO("error: '" << what << "'");
    CHECK_THAT(what, Catch::Contains(tmp.basename() + ": line 2331:"));
    CHECK_THAT(what, Catch::Contains("energy unit"));
    CHECK_THAT(what, Catch::Contains("'Ha'"));
  }
}
