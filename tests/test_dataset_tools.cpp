// Tests for the data tools' building blocks (RELEASE-1.6.0-FOLLOWUPS.md,
// "Known-issues session ... decisions", D, "Data tools";
// tmp/known-issues/report_leftovers.md sections 1 and 2):
//
// - DatasetReaderSelector::get_reader, the factory behind 'tadah data
//   convert', refuses a file it does not recognise, or cannot open, with a
//   std::runtime_error that names the file (and, for an unrecognised file,
//   the recognised formats) instead of returning nullptr.
// - Structure::is_the_same, the test behind 'tadah data dedup', matches the
//   atoms independently of their order, each atom of the other structure at
//   most once, and periodically: the position difference is reduced to its
//   minimal image through the cell, and each Cartesian component of it
//   must lie within the threshold.
//
// Every expected value is written out by hand below.

#include "catch2/catch.hpp"

#include "tadah/core/periodic_table.h"

#include <tadah/mlip/atom.h>
#include <tadah/mlip/dataset_readers/dataset_reader_selector.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

using Catch::Matchers::Contains;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::DatasetReader;
using tadah::mlip::DatasetReaderSelector;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

namespace {

namespace fs = std::filesystem;

using V3 = std::array<double, 3>;
using Rows = std::array<V3, 3>;

/* A scratch directory, unique to this process and test case, removed on
 * exit. */
struct ScratchDir {
  fs::path dir;
  explicit ScratchDir(const std::string &stem)
      : dir(fs::temp_directory_path() /
            ("tadah_dataset_tools_" + stem + "_" + std::to_string(::getpid()))) {
    fs::remove_all(dir);
    fs::create_directories(dir);
  }
  ~ScratchDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string write(const std::string &name, const std::string &body) const {
    const fs::path p = dir / name;
    std::ofstream os(p, std::ios::binary);
    os << body;
    return p.string();
  }
  ScratchDir(const ScratchDir &) = delete;
  ScratchDir &operator=(const ScratchDir &) = delete;
};

/* What get_reader did with `path`: the message of the std::runtime_error it
 * threw, or nullopt and whether it returned a reader. */
struct SelectorOutcome {
  std::optional<std::string> error;
  bool returned_reader = false;
};

SelectorOutcome select(const std::string &path) {
  PeriodicTable::initialize();
  SelectorOutcome o;
  StructureDB db;
  try {
    std::unique_ptr<DatasetReader> r = DatasetReaderSelector::get_reader(path, db);
    o.returned_reader = (r != nullptr);
  } catch (const std::runtime_error &e) {
    o.error = std::string(e.what());
  }
  return o;
}

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

/* A structure with the given lattice rows and (symbol, Cartesian position)
 * atoms. */
Structure make(const Rows &cell,
               const std::vector<std::pair<std::string, V3>> &atoms) {
  PeriodicTable::initialize();
  Structure s;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) s.cell(i, j) = cell[i][j];
  for (const auto &a : atoms)
    s.add_atom(Atom(PeriodicTable::find_by_symbol(a.first), a.second[0],
                    a.second[1], a.second[2], 0.0, 0.0, 0.0));
  return s;
}

V3 add(const V3 &r, const V3 &d) { return {r[0] + d[0], r[1] + d[1], r[2] + d[2]}; }

/* n0 a0 + n1 a1 + n2 a2, written out. */
V3 lattice(const Rows &a, int n0, int n1, int n2) {
  V3 t{};
  for (int k = 0; k < 3; ++k) t[k] = n0 * a[0][k] + n1 * a[1][k] + n2 * a[2][k];
  return t;
}

/* Checks is_the_same both ways (the relation is symmetric) with the
 * threshold 1e-6 A of 'tadah data dedup --threshold 1e-6'. */
void check_same(const std::string &what, const Structure &a,
                const Structure &b, bool expected) {
  INFO(what << ": expected " << (expected ? "the same" : "not the same"));
  CHECK(a.is_the_same(b, 1e-6) == expected);
  CHECK(b.is_the_same(a, 1e-6) == expected);
}

const Rows kCube5 = {{{5.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 5.0}}};
/* A triclinic cell (cell != cell^T), so a minimal image taken with the
 * columns of the cell, or only along x, y, z, misses the lattice vectors. */
const Rows kTriclinic = {{{4.0, 0.0, 0.0}, {1.2, 3.8, 0.0}, {-0.7, 0.9, 5.1}}};

}  // namespace

// --------------------------------------------------------------------
// DatasetReaderSelector::get_reader
// --------------------------------------------------------------------

/* Pins: get_reader throws a std::runtime_error for a file it does not
 * recognise as DFT output (plain text, and a valid POSCAR, which is a
 * structure file read with -s), naming the file and the recognised formats
 * OUTCAR, vasprun.xml, .castep, .md and .geom. The recognised fixtures
 * OUTCAR_1 and vasprun_2.xml still get a reader (that part held before the
 * fix; it fails if the refusal were too eager).
 * Before the fix: get_reader printed 'Unknown type! Returning nullptr.' and
 * returned nullptr, which 'tadah data convert' dereferenced (SIGSEGV); here
 * REQUIRE(o.error) failed. */
TEST_CASE("DatasetReaderSelector: an unrecognised file is refused, naming it "
          "and the recognised formats",
          "[dataset_tools][selector]") {
  ScratchDir tmp("selector_unknown");
  const std::vector<std::pair<std::string, std::string>> inputs = {
      {"unknown.txt", "hello world\njust text\n"},
      {"POSCAR", "Fe bcc\n1.0\n2.87 0 0\n0 2.87 0\n0 0 2.87\nFe\n2\nDirect\n"
                 "0 0 0\n0.5 0.5 0.5\n"}};
  for (const auto &in : inputs) {
    const std::string path = tmp.write(in.first, in.second);
    INFO("input " << path);
    const SelectorOutcome o = select(path);
    INFO("returned a reader: " << o.returned_reader);
    REQUIRE(o.error);
    INFO("message: " << *o.error);
    CHECK_FALSE(o.returned_reader);
    CHECK_THAT(*o.error, Contains(path));
    for (const char *fmt : {"OUTCAR", "vasprun.xml", ".castep", ".md", ".geom"})
      CHECK_THAT(*o.error, Contains(fmt));
  }

  for (const std::string fixture : {"./tests_data/valid_outcars/OUTCAR_1",
                                    "./tests_data/valid_vaspruns/vasprun_2.xml"}) {
    INFO("fixture " << fixture);
    const SelectorOutcome o = select(fixture);
    CHECK_FALSE(o.error);
    CHECK(o.returned_reader);
  }
}

/* Pins: get_reader throws a std::runtime_error for a file that does not
 * exist, naming the file and saying it cannot be opened (the message holds
 * "open", in any case).
 * Before the fix: get_reader printed 'Could not open file: ...' and
 * 'Unknown type! Returning nullptr.' to stderr and returned nullptr; here
 * REQUIRE(o.error) failed. */
TEST_CASE("DatasetReaderSelector: a missing file is refused, naming it and "
          "saying it cannot be opened",
          "[dataset_tools][selector]") {
  ScratchDir tmp("selector_missing");
  const std::string path = (tmp.dir / "does_not_exist.xyz").string();
  REQUIRE_FALSE(fs::exists(path));
  const SelectorOutcome o = select(path);
  INFO("returned a reader: " << o.returned_reader);
  REQUIRE(o.error);
  INFO("message: " << *o.error);
  CHECK_FALSE(o.returned_reader);
  CHECK_THAT(*o.error, Contains(path));
  CHECK_THAT(lower(*o.error), Contains("open"));
}

// --------------------------------------------------------------------
// Structure::is_the_same
// --------------------------------------------------------------------

/* Pins: is_the_same does not depend on the order of the atoms. The NaCl
 * pair of report_leftovers.md section 2 with Cl first, and a triclinic
 * 4-atom cell listed in the order 2, 0, 3, 1, are the same structure as the
 * original; the identity order is the same too (it held before the fix).
 * Pins that held before the fix and must still hold (each fails if the
 * matching ignored species, positions or the cell): a permutation that
 * swaps the species of two sites, one atom moved by 1e-3 A, and the same
 * atoms in a different cell are not the same.
 * Before the fix: only the identity order matched (atom i of one structure
 * was compared with atoms j >= i of the other), so both permuted CHECKs
 * failed. */
TEST_CASE("Structure::is_the_same: the order of the atoms does not matter",
          "[dataset_tools][is_the_same]") {
  const Structure s1 = make(kCube5, {{"Na", {0, 0, 0}}, {"Cl", {2.5, 2.5, 2.5}}});
  const Structure s2 = make(kCube5, {{"Cl", {2.5, 2.5, 2.5}}, {"Na", {0, 0, 0}}});
  check_same("NaCl against itself", s1, s1, true);
  check_same("NaCl, Cl listed first", s1, s2, true);

  const std::vector<std::pair<std::string, V3>> atoms = {
      {"Ti", {0.215, 0.515, 0.765}},
      {"Ti", {2.465, 2.865, 3.315}},
      {"O", {1.88, 3.4, 2.04}},
      {"O", {3.1, 1.2, 0.4}}};
  const Structure t = make(kTriclinic, atoms);
  const Structure t_perm =
      make(kTriclinic, {atoms[2], atoms[0], atoms[3], atoms[1]});
  check_same("triclinic, atoms in the order 2 0 3 1", t, t_perm, true);

  /* Not the same: the species of sites 0 and 2 swapped. */
  const Structure t_swapped = make(
      kTriclinic, {{"O", atoms[0].second}, atoms[1], {"Ti", atoms[2].second}, atoms[3]});
  check_same("triclinic, species of sites 0 and 2 swapped", t, t_swapped, false);
  /* Not the same: atom 3 moved by 1e-3 A along y. */
  const Structure t_moved =
      make(kTriclinic, {atoms[0], atoms[1], atoms[2], {"O", add(atoms[3].second, {0, 1e-3, 0})}});
  check_same("triclinic, atom 3 moved by 1e-3 A", t, t_moved, false);
  /* Not the same: another cell. */
  check_same("the same atoms in another cell", t, make(kCube5, atoms), false);
}

/* Pins: is_the_same compares positions periodically. The difference of two
 * positions is reduced to its minimal image through the cell, and each of
 * its Cartesian components must be within the threshold (1e-6 A here):
 * - NaCl with Na stored at x + a0 = (5, 0, 0) is the same as with Na at 0;
 * - an atom at x = 4e-7 and the same atom at x = 5 - 4e-7 (8e-7 A apart
 *   across the cell face) are the same;
 * - in the triclinic cell, atoms shifted by a0 - a1 + a2 = (2.1, -2.9, 5.1)
 *   and by -a0 + 2 a1 (with the atoms also permuted) are the same;
 * - an image displaced by a further (0.9e-6, -0.9e-6, 0.9e-6) is the same,
 *   one displaced by a further 1.5e-6 along z is not (that held before the
 *   fix for the identity case; it fails if the threshold were applied to
 *   the norm, or not per component).
 * Before the fix: positions were compared without the cell, so every
 * periodic-image CHECK expecting true failed. */
TEST_CASE("Structure::is_the_same: atoms are compared through the periodic "
          "cell",
          "[dataset_tools][is_the_same]") {
  const Structure s1 = make(kCube5, {{"Na", {0, 0, 0}}, {"Cl", {2.5, 2.5, 2.5}}});
  const Structure s3 = make(kCube5, {{"Na", {5.0, 0, 0}}, {"Cl", {2.5, 2.5, 2.5}}});
  check_same("NaCl, Na stored at x + a0", s1, s3, true);

  const Structure face_a = make(kCube5, {{"Na", {4e-7, 1.0, 1.0}}, {"Cl", {2.5, 2.5, 2.5}}});
  const Structure face_b = make(kCube5, {{"Na", {5.0 - 4e-7, 1.0, 1.0}}, {"Cl", {2.5, 2.5, 2.5}}});
  check_same("x = 4e-7 against x = 5 - 4e-7", face_a, face_b, true);

  const std::vector<std::pair<std::string, V3>> atoms = {
      {"Ti", {0.215, 0.515, 0.765}},
      {"Ti", {2.465, 2.865, 3.315}},
      {"O", {1.88, 3.4, 2.04}},
      {"O", {3.1, 1.2, 0.4}}};
  const Structure t = make(kTriclinic, atoms);

  const V3 t1 = lattice(kTriclinic, 1, -1, 1);  // (2.1, -2.9, 5.1)
  CHECK(t1[0] == Approx(2.1).margin(1e-12));
  CHECK(t1[1] == Approx(-2.9).margin(1e-12));
  CHECK(t1[2] == Approx(5.1).margin(1e-12));
  const V3 t2 = lattice(kTriclinic, -1, 2, 0);  // (-1.6, 7.6, 0)
  const Structure t_images = make(
      kTriclinic, {{"Ti", add(atoms[0].second, t1)},
                   atoms[1],
                   {"O", add(atoms[2].second, t2)},
                   {"O", add(atoms[3].second, t1)}});
  check_same("triclinic, images by a0 - a1 + a2 and -a0 + 2 a1", t, t_images, true);

  const Structure t_images_perm = make(
      kTriclinic, {{"O", add(atoms[3].second, t1)},
                   {"Ti", add(atoms[1].second, t2)},
                   {"O", add(atoms[2].second, t2)},
                   {"Ti", add(atoms[0].second, t1)}});
  check_same("triclinic, images, permuted", t, t_images_perm, true);

  const Structure t_within = make(
      kTriclinic, {atoms[0], atoms[1], atoms[2],
                   {"O", add(add(atoms[3].second, t1), {0.9e-6, -0.9e-6, 0.9e-6})}});
  check_same("image + (0.9e-6, -0.9e-6, 0.9e-6)", t, t_within, true);

  const Structure t_outside = make(
      kTriclinic, {atoms[0], atoms[1], atoms[2],
                   {"O", add(add(atoms[3].second, t1), {0.0, 0.0, 1.5e-6})}});
  check_same("image + 1.5e-6 along z", t, t_outside, false);
  const Structure t_outside_identity = make(
      kTriclinic, {atoms[0], atoms[1], atoms[2],
                   {"O", add(atoms[3].second, {0.0, 0.0, 1.5e-6})}});
  check_same("1.5e-6 along z, no image", t, t_outside_identity, false);
}

/* Pins: each atom of the other structure is matched at most once.
 * - F1 = [Na, Na (coincident), Cl] and F2 = [Na, Na (coincident), K
 *   elsewhere] (report_leftovers.md section 2) are not the same, either
 *   way round.
 * - [Na at 0, Na at 0, Na at p] and [Na at 0, Na at p, Na at p] are not
 *   the same (that held before the fix; it fails if each atom were matched
 *   to any atom of the other structure, without the used-once rule).
 * - Two copies of F1 with the coincident pair listed in another order are
 *   the same.
 * Before the fix: one atom of the other structure could be counted for
 * several atoms, so F1 and F2 counted 3 matches for 3 atoms and were
 * reported the same (both directions): those CHECKs failed. */
TEST_CASE("Structure::is_the_same: each atom is matched at most once "
          "(coincident atoms)",
          "[dataset_tools][is_the_same]") {
  const Structure f1 = make(kCube5, {{"Na", {0, 0, 0}}, {"Na", {0, 0, 0}}, {"Cl", {2.5, 2.5, 2.5}}});
  const Structure f2 = make(kCube5, {{"Na", {0, 0, 0}}, {"Na", {0, 0, 0}}, {"K", {1.0, 1.0, 1.0}}});
  check_same("F1 [Na, Na, Cl] against F2 [Na, Na, K]", f1, f2, false);

  const V3 p = {1.5, 2.0, 2.5};
  const Structure two_at_0 = make(kCube5, {{"Na", {0, 0, 0}}, {"Na", {0, 0, 0}}, {"Na", p}});
  const Structure two_at_p = make(kCube5, {{"Na", {0, 0, 0}}, {"Na", p}, {"Na", p}});
  check_same("two Na at 0 against two Na at p", two_at_0, two_at_p, false);

  const Structure f1_reordered =
      make(kCube5, {{"Cl", {2.5, 2.5, 2.5}}, {"Na", {0, 0, 0}}, {"Na", {0, 0, 0}}});
  check_same("F1 against [Cl, Na, Na]", f1, f1_reordered, true);
}
