// Value tests for the CASTEP readers and the CASTEP cell writer (workstream D
// of the 1.6.0 known-issues session; RELEASE-1.6.0-FOLLOWUPS.md, "D: stress
// convention and reader bugs"; tmp/known-issues/report_readers.md C, D1-D6).
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
// .md and .geom: the first frame (report_readers.md D3 + D4)
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

// Pins: a .geom file with a single frame (60.geom's header and first frame,
// the file tmp/known-issues/wsD-readers/castep/one_frame.geom) gives one
// structure with that frame's energy and a non-empty label.
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
// .md kinetic stress and .geom stress (report_readers.md C)
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
// .md and .geom: malformed and truncated frames (report_readers.md DQ5)
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

// --------------------------------------------------------------------
// .castep (report_readers.md D1, D2)
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
// After that fix, until the review: the label's positions range 'C: (636 :
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
    // The label names the blocks stored: positions 534-543, not the
    // reverted block 636-645; energy line 561.
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
// Before the fix (a review finding on the first version of the revert fix,
// which took the geometry once per iteration): structures 2-6 paired the
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
// calculation : on', but a fixed cell, so no stress block is printed) gives
// 16 structures, iterations 0-15. Structure 1 is the last finite-basis SCF
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
// Species labels with a CASTEP tag (report_readers.md D6)
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

// --------------------------------------------------------------------
// CASTEP cell writer (report_readers.md D5)
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
