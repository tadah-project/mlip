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

#include "tadah_test_skip.hpp"
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
#include <limits>
#include <memory>
#include <optional>
#include <set>
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

/* Pins: is_the_same finds a one-to-one matching whenever one exists, so its
 * result does not depend on the order of the atoms, also with a threshold
 * comparable to the distance between same-element atoms. In a 10 A cube
 * with threshold 0.1 A, S1 = [H (0,0,0), H (0.15,0,0)] and
 * S2 = [H (0.08,0,0), H (-0.05,0,0)] are the same (0 <-> -0.05 at 0.05 A,
 * 0.15 <-> 0.08 at 0.07 A), with S2's atoms in either order and either way
 * round. A structure whose two atoms can only pair with one atom of the
 * other is still not the same.
 * Before the fix: atoms were matched greedily, each to the first unused
 * match: S1 against S2 paired 0 with 0.08 and then found no partner for
 * 0.15, so two of the four CHECKs (S1 vs S2 and S2 vs S1) failed. */
TEST_CASE("Structure::is_the_same: the matching does not depend on the atom "
          "order (threshold comparable to the atom spacing)",
          "[dataset_tools][is_the_same]") {
  const Rows cube10 = {{{10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}}};
  const double thr = 0.1;
  const Structure s1 = make(cube10, {{"H", {0, 0, 0}}, {"H", {0.15, 0, 0}}});
  const Structure s2 = make(cube10, {{"H", {0.08, 0, 0}}, {"H", {-0.05, 0, 0}}});
  const Structure s2_swapped = make(cube10, {{"H", {-0.05, 0, 0}}, {"H", {0.08, 0, 0}}});
  CHECK(s1.is_the_same(s2, thr));
  CHECK(s2.is_the_same(s1, thr));
  CHECK(s1.is_the_same(s2_swapped, thr));
  CHECK(s2_swapped.is_the_same(s1, thr));

  // H at 0 and at 0.02 can both pair only with the H at 0.05.
  const Structure t1 = make(cube10, {{"H", {0, 0, 0}}, {"H", {0.02, 0, 0}}, {"H", {5, 5, 5}}});
  const Structure t2 = make(cube10, {{"H", {0.05, 0, 0}}, {"H", {0.3, 0, 0}}, {"H", {5, 5, 5}}});
  CHECK_FALSE(t1.is_the_same(t2, thr));
  CHECK_FALSE(t2.is_the_same(t1, thr));
}

/* Pins: is_the_same compares positions through the cell, so it refuses a
 * structure whose cell is singular (zero, or with coplanar lattice
 * vectors) with a std::invalid_argument naming that structure's label,
 * whichever of the two it is (the structure called is checked first).
 * Before the fix: the inverse of the singular cell was NaN, no atom
 * matched, and is_the_same returned false: 'data dedup' kept exact
 * copies, silently. */
TEST_CASE("Structure::is_the_same: a singular cell is refused, naming the "
          "structure",
          "[dataset_tools][is_the_same]") {
  const std::vector<std::pair<std::string, V3>> atoms = {
      {"Ta", {0, 0, 0}}, {"Ta", {1.5, 1.5, 1.5}}};
  const Rows zero = {{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
  const Rows coplanar = {{{3, 0, 0}, {0, 3, 0}, {3, 3, 0}}};
  for (const auto &c : {std::make_pair(std::string("zero cell"), zero),
                        std::make_pair(std::string("coplanar cell"), coplanar)}) {
    INFO(c.first);
    Structure bad = make(c.second, atoms);
    bad.label = "singular-" + c.first;
    Structure copy = bad;
    copy.label = "copy";
    Structure good = make(kCube5, atoms);
    good.label = "good";
    for (const auto *pair : {&copy, &good}) {
      INFO("against " << pair->label);
      try {
        (void)bad.is_the_same(*pair, 1e-6);
        FAIL("no exception (singular cell as this)");
      } catch (const std::invalid_argument &e) {
        CHECK_THAT(e.what(), Catch::Contains(bad.label));
        CHECK_THAT(e.what(), Catch::Contains("singular"));
      }
      try {
        (void)pair->is_the_same(bad, 1e-6);
        FAIL("no exception (singular cell as the argument)");
      } catch (const std::invalid_argument &e) {
        // 'copy' is singular too, and is checked first.
        CHECK_THAT(e.what(), Catch::Contains(pair == &copy ? copy.label : bad.label));
        CHECK_THAT(e.what(), Catch::Contains("singular"));
      }
    }
  }
}

/* Pins: is_the_same finds an atom's partner through any lattice
 * translation, not only the one the rounded fractional coordinates of the
 * cell as written give, so the result does not depend on how the lattice is
 * described. The simple-cubic 3 A lattice written with the rows (3,0,0),
 * (6,3,0), (0,0,3) and with (3,0,0), (0,3,0), (0,0,3): S1 = [H (0,0,0),
 * H (1.5,1.5,1.5)] and S2 = [H (-0.54,0.495,0), H (1.5,1.5,1.5)] are the
 * same at threshold 0.55 A in both descriptions, either way round (the
 * displacement (-0.54, 0.495, 0) is within 0.55 with translation 0).
 * Before the fix: in the tilted description the rounded fractional
 * coordinate -0.51 gave the translation -a0 and a residual (2.46, 0.495, 0),
 * so S1 and S2 were not the same there. */
TEST_CASE("Structure::is_the_same: every lattice translation is tried, so the "
          "cell description does not matter",
          "[dataset_tools][is_the_same]") {
  const Rows tilted = {{{3.0, 0.0, 0.0}, {6.0, 3.0, 0.0}, {0.0, 0.0, 3.0}}};
  const Rows reduced = {{{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}}};
  for (const Rows *cell : {&tilted, &reduced}) {
    INFO((cell == &tilted ? "tilted rows" : "reduced rows"));
    const Structure s1 = make(*cell, {{"H", {0, 0, 0}}, {"H", {1.5, 1.5, 1.5}}});
    const Structure s2 = make(*cell, {{"H", {-0.54, 0.495, 0}}, {"H", {1.5, 1.5, 1.5}}});
    CHECK(s1.is_the_same(s2, 0.55));
    CHECK(s2.is_the_same(s1, 0.55));
    CHECK_FALSE(s1.is_the_same(s2, 0.5));
  }
}

/* Pins: is_the_same is symmetric when the two cells differ (by up to the
 * threshold): a translation of either cell counts. a = [Ta (0,0,0), Ta
 * (1.5,1.5,1.5)] in a 3 A cube, b = [Ta (6,0,0), Ta (1.5,1.5,1.5)] in a
 * 3.0000009 A cube, threshold 1e-6: 6 = 2 x 3 exactly through a's cell,
 * 1.8e-6 off through b's; the same either way round.
 * Before the fix: only the cell of the structure called was used, so
 * a.is_the_same(b) was true and b.is_the_same(a) false, and 'data dedup'
 * depended on the order of the files. */
TEST_CASE("Structure::is_the_same: the relation is symmetric when the cells "
          "differ within the threshold",
          "[dataset_tools][is_the_same]") {
  const Rows cube3 = {{{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}}};
  const double c = 3.0000009;
  const Rows cube3b = {{{c, 0.0, 0.0}, {0.0, c, 0.0}, {0.0, 0.0, c}}};
  const Structure a = make(cube3, {{"Ta", {0, 0, 0}}, {"Ta", {1.5, 1.5, 1.5}}});
  const Structure b = make(cube3b, {{"Ta", {6.0, 0, 0}}, {"Ta", {1.5, 1.5, 1.5}}});
  CHECK(a.is_the_same(b, 1e-6));
  CHECK(b.is_the_same(a, 1e-6));
}

/* Pins: a threshold at which two images of one position could both match
 * is refused, naming the structure and the threshold: the limit is
 * lambda/(2 sqrt 3), lambda the shortest lattice vector of either cell
 * (then at most one image lies within the threshold in every component).
 * A 10 A cube allows 2.88 A and refuses 2.9 A (the limit is 2.887 A); the
 * lattice with rows (4,0,0), (3.9,1,0), (0,0,4), whose shortest vector
 * (-0.1,1,0) is 1.005 A long, refuses 0.56 A (limit 0.290 A) and allows
 * 0.2 A.
 * Before the fix: any threshold was used, and in the second cell Ta (0,0,0)
 * against Ta (0.55,0.55,0) was 'not the same' at 0.56 A although the
 * displacement is within 0.56 A. */
TEST_CASE("Structure::is_the_same: a threshold too large for the lattice is "
          "refused",
          "[dataset_tools][is_the_same]") {
  const Rows cube10 = {{{10.0, 0.0, 0.0}, {0.0, 10.0, 0.0}, {0.0, 0.0, 10.0}}};
  Structure s = make(cube10, {{"Ta", {0, 0, 0}}, {"Ta", {5, 5, 5}}});
  s.label = "cube-10";
  CHECK(s.is_the_same(s, 2.88));
  try {
    (void)s.is_the_same(s, 2.9);
    FAIL("no exception at threshold 2.9");
  } catch (const std::invalid_argument &e) {
    CHECK_THAT(e.what(), Catch::Contains("cube-10"));
    CHECK_THAT(e.what(), Catch::Contains("2.9"));
    CHECK_THAT(e.what(), Catch::Contains("threshold"));
  }

  const Rows skew = {{{4.0, 0.0, 0.0}, {3.9, 1.0, 0.0}, {0.0, 0.0, 4.0}}};
  Structure t1 = make(skew, {{"Ta", {0, 0, 0}}});
  Structure t2 = make(skew, {{"Ta", {0.55, 0.55, 0}}});
  t1.label = "skew-1";
  t2.label = "skew-2";
  CHECK_THROWS_AS(t1.is_the_same(t2, 0.56), std::invalid_argument);
  CHECK_THROWS_AS(t2.is_the_same(t1, 0.56), std::invalid_argument);
  const Structure t3 = make(skew, {{"Ta", {0.15, 0.15, 0}}});
  CHECK(t1.is_the_same(t3, 0.2));
  CHECK_FALSE(t1.is_the_same(t2, 0.2));
}

// --------------------------------------------------------------------
// StructureDB::add (the dataset reader of every tool)
// --------------------------------------------------------------------

/* A two-structure dataset whose second structure has `bad` in place of
 * the text `good` of its block (one occurrence). */
std::string dataset_with(const std::string &good, const std::string &bad) {
  const std::string first =
      "S1\n1 1 1\n-11.5\n3.3 0 0\n0 3.3 0\n0 0 3.3\n0 0 0\n0 0 0\n0 0 0\n"
      "Ta 0 0 0 0 0 0\nTa 1.65 1.65 1.65 0 0 0\n\n";
  std::string second =
      "S2\n1 1 1\n-11.4\n3.31 0 0\n0 3.31 0\n0 0 3.31\n0.001 0 0\n0 0 0\n0 0 0\n"
      "Ta 0 0 0 0.1 0 0\nTa 1.655 1.655 1.655 -0.1 0 0\n\n";
  const auto p = second.find(good);
  REQUIRE(p != std::string::npos);
  REQUIRE(second.find(good, p + 1) == std::string::npos);
  second.replace(p, good.size(), bad);
  return first + second;
}

/* Pins: a dataset value that is not a finite number, or an atom line
 * that is not 'Element px py pz fx fy fz', is refused with a
 * std::runtime_error naming the file, the structure (1-based) and the bad
 * text: a cell entry 'nan', 'inf', '1e309' or 'abc'; a stress entry 'nan';
 * an energy 'nan'; an atom coordinate 'x'; an atom line with too few
 * fields; an element symbol 'Tantalum' (longer than any symbol). The good
 * file reads as 2 structures.
 * Before the fix: a cell entry that is not a number put the stream into a
 * failed state that never reaches end of file, so every tool reading the
 * dataset spun forever ('tadah data print', 'data dedup', training); a bad
 * atom field was read as garbage, and a symbol longer than two characters
 * overflowed a 3-byte buffer. */
TEST_CASE("StructureDB: a value that is not a finite number is refused, "
          "naming the file and the structure",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_values");
  {
    const std::string path = tmp.write("good.tadah", dataset_with("-11.4", "-11.4"));
    StructureDB db;
    REQUIRE_NOTHROW(db.add(path));
    CHECK(db.size() == 2);
  }
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"3.31 0 0\n0 3.31", "nan 0 0\n0 3.31"},
      {"3.31 0 0\n0 3.31", "inf 0 0\n0 3.31"},
      {"3.31 0 0\n0 3.31", "1e309 0 0\n0 3.31"},
      {"3.31 0 0\n0 3.31", "abc 0 0\n0 3.31"},
      {"0.001 0 0", "nan 0 0"},
      {"-11.4", "nan"},
      {"Ta 0 0 0 0.1", "Ta x 0 0 0.1"},
      {"Ta 0 0 0 0.1 0 0", "Ta 0 0 0 0.1"},
      {"Ta 0 0 0 0.1", "Tantalum 0 0 0 0.1"},
  };
  int n = 0;
  for (const auto &c : cases) {
    INFO("'" << c.first << "' -> '" << c.second << "'");
    const std::string path =
        tmp.write("bad" + std::to_string(++n) + ".tadah", dataset_with(c.first, c.second));
    StructureDB db;
    try {
      db.add(path);
      FAIL("no exception");
    } catch (const std::runtime_error &e) {
      INFO("error: " << e.what());
      CHECK_THAT(e.what(), Catch::Contains(path));
      CHECK_THAT(e.what(), Catch::Contains("structure 2"));
      const std::string token = c.second.substr(0, c.second.find_first_of(" \n"));
      if (token != "Ta") {
        CHECK_THAT(e.what(), Catch::Contains(token));
      }
    }
  }
}

/* Pins: a line of only whitespace separates structures like an empty one,
 * also in a file with Windows (CRLF) line endings: two structures separated
 * by a line of spaces, by two CRLF blank lines, or by a tab line all read as
 * the same 2 structures as the plain file.
 * Before the fix: only an empty line (or a lone '\r' inside a structure)
 * ended a structure; a whitespace line was read as the next structure's
 * label and the file was refused. */
TEST_CASE("StructureDB: whitespace-only lines and CRLF blank lines separate structures",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_blank");
  const std::string plain = dataset_with("-11.4", "-11.4");
  StructureDB want;
  want.add(tmp.write("plain.tadah", plain));
  REQUIRE(want.size() == 2);
  auto crlf = [](const std::string &t) {
    std::string out;
    for (char c : t) {
      if (c == '\n') out += '\r';
      out += c;
    }
    return out;
  };
  const std::string sep = "Ta 1.65 1.65 1.65 0 0 0\n\n";
  const auto p = plain.find(sep);
  REQUIRE(p != std::string::npos);
  const std::string head = plain.substr(0, p + sep.size() - 1);  // without the blank line
  const std::string tail = plain.substr(p + sep.size());
  const std::vector<std::pair<std::string, std::string>> variants = {
      {"spaces", head + "   \n" + tail},
      {"tab", head + "\t\n" + tail},
      {"CRLF, two blank lines", crlf(head + "\n\n" + tail)},
      {"CRLF, plain", crlf(plain)},
  };
  int n = 0;
  for (const auto &v : variants) {
    INFO(v.first);
    StructureDB db;
    REQUIRE_NOTHROW(db.add(tmp.write("v" + std::to_string(++n) + ".tadah", v.second)));
    REQUIRE(db.size() == 2);
    for (size_t k = 0; k < 2; ++k) {
      CHECK(db(k).energy == want(k).energy);
      CHECK(db(k).natoms() == want(k).natoms());
      CHECK(db(k).cell(0, 0) == want(k).cell(0, 0));
    }
  }
}

/* Pins: writing a dataset or a structure to a file that cannot be opened
 * throws std::runtime_error naming the file (a missing directory), for
 * StructureDB::dump_to_file and Structure::dump_to_file.
 * Before the fix: both printed 'Error: Could not open file for writing' and
 * returned, so every data command reported success. */
TEST_CASE("StructureDB/Structure::dump_to_file: an unwritable file is refused",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("dump_unwritable");
  StructureDB db;
  db.add(tmp.write("good.tadah", dataset_with("-11.4", "-11.4")));
  REQUIRE(db.size() == 2);
  const std::string path = (tmp.dir / "missing" / "out.tadah").string();
  try {
    db.dump_to_file(path, false);
    FAIL("no exception from StructureDB::dump_to_file");
  } catch (const std::runtime_error &e) {
    CHECK_THAT(e.what(), Catch::Contains(path));
  }
  try {
    db(0).dump_to_file(path);
    FAIL("no exception from Structure::dump_to_file");
  } catch (const std::runtime_error &e) {
    CHECK_THAT(e.what(), Catch::Contains(path));
  }
}

/* Pins: the scanners that read a dataset without loading it treat a
 * whitespace-only line, a CRLF blank line and extra blank lines as
 * Structure::read does: for the two-structure dataset written with a
 * separator line of spaces, with
 * CRLF line endings and with two blank lines,
 * StructureDB::find_unique_elements gives {Ta} (what tadah train and HPO
 * read first), StructureDB::count gives 2 structures and 4 atoms, and
 * StructureDB::add(file, 1) reads the second structure, S2.
 * Before the fix: find_unique_elements stopped only at an empty line, so a
 * separator of spaces or a CRLF blank line was read as an atom line
 * ('Unknown symbol: S2'); count and add(file, first) (through
 * Structure::next_structure, which also stopped at '\r') read a separator
 * of spaces as an atom line, merging the two structures, and an extra blank
 * line ended the count. */
TEST_CASE("StructureDB: the scanners accept whitespace-only, CRLF and extra blank lines",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_scan_blank");
  const std::string plain = dataset_with("-11.4", "-11.4");
  const std::string sep = "Ta 1.65 1.65 1.65 0 0 0\n\n";
  const auto p = plain.find(sep);
  REQUIRE(p != std::string::npos);
  const std::string head = plain.substr(0, p + sep.size() - 1);
  const std::string tail = plain.substr(p + sep.size());
  auto crlf = [](const std::string &t) {
    std::string out;
    for (char c : t) {
      if (c == '\n') out += '\r';
      out += c;
    }
    return out;
  };
  const std::vector<std::pair<std::string, std::string>> variants = {
      {"spaces", head + "   \n" + tail},
      {"CRLF", crlf(plain)},
      {"two blank lines", head + "\n\n" + tail}};
  int n = 0;
  for (const auto &v : variants) {
    INFO(v.first);
    const std::string path = tmp.write("s" + std::to_string(++n) + ".tadah", v.second);
    std::set<tadah::core::Element> elements;
    REQUIRE_NOTHROW(elements = StructureDB::find_unique_elements(path));
    CHECK(elements.size() == 1);
    CHECK(elements.count(PeriodicTable::find_by_symbol("Ta")) == 1);
    const auto counted = StructureDB::count(path);
    CHECK(counted.first == 2);
    CHECK(counted.second == 4);
    StructureDB second;
    second.add(path, 1, 1);
    REQUIRE(second.size() == 1);
    CHECK(second(0).label.rfind("S2", 0) == 0);
  }
}

/* Pins: StructureDB::add(file, first, N) returns the number of structures it
 * loaded, as documented: on the two-structure dataset, add(file, 1, 1) gives
 * 1, and add(file, 0, 5) (N larger than the dataset) gives 2, with the file
 * written without a trailing blank line and with two blank lines between the
 * structures.
 * Before the fix: it returned one less after a structure that ended the file
 * (0 and 1) and counted a blank line as a structure (3). */
TEST_CASE("StructureDB::add(file, first, N) returns the number of structures it loaded",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_add_count");
  const std::string plain = dataset_with("-11.4", "-11.4");
  const std::string sep = "Ta 1.65 1.65 1.65 0 0 0\n\n";
  const auto p = plain.find(sep);
  REQUIRE(p != std::string::npos);
  REQUIRE(plain.substr(plain.size() - 2) == "\n\n");
  const std::string no_trailing = tmp.write("no_trailing.tadah", plain.substr(0, plain.size() - 1));
  const std::string two_blank = tmp.write(
      "two_blank.tadah", plain.substr(0, p + sep.size()) + "\n" + plain.substr(p + sep.size()));
  for (const auto &path : {no_trailing, two_blank}) {
    INFO(path);
    StructureDB second;
    CHECK(second.add(path, 1, 1) == 1);
    CHECK(second.size() == 1);
    StructureDB all;
    CHECK(all.add(path, 0, 5) == 2);
    CHECK(all.size() == 2);
  }
}

/* Pins: StructureDB::add(file, first, N) with `first` at or past the end of
 * the file loads nothing and returns 0, as documented for N past the end: on
 * the two-structure dataset, add(file, 2, 1) and add(file, 10, 3).
 * Before the fix: passes as well (a pin: it went on skipping past the end,
 * one failed read per structure, and then read nothing). */
TEST_CASE("StructureDB::add(file, first, N) past the end of the file loads nothing",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_add_past_end");
  const std::string path = tmp.write("two.tadah", dataset_with("-11.4", "-11.4"));
  for (const std::size_t first : {std::size_t{2}, std::size_t{10}}) {
    INFO("first " << first);
    StructureDB db;
    CHECK(db.add(path, first, 3) == 0);
    CHECK(db.size() == 0);
  }
}

/* Pins: StructureDB::add(file, SIZE_MAX, 1) stops skipping at the end of
 * the file and returns 0 at once, loading nothing.
 * Before the fix: it skipped SIZE_MAX structures, one failed read at a time
 * past the end of the file, which does not end in practice (the test runs
 * until CTest's timeout); `tadah analysis descriptor -i 0` reached it with
 * first = 0 - 1 and hung. */
TEST_CASE("StructureDB::add(file, first, N) stops skipping at the end of the file",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_add_skip_end");
  const std::string path = tmp.write("two.tadah", dataset_with("-11.4", "-11.4"));
  StructureDB db;
  CHECK(db.add(path, std::numeric_limits<std::size_t>::max(), 1) == 0);
  CHECK(db.size() == 0);
}

/* Pins: StructureDB::add_selected(file, indices) adds the structures of the
 * file at the given 0-based indices, in the given order, reading the file
 * once: on the two-structure dataset, {1, 0} gives S2 then S1 and returns 2,
 * {1} gives S2; an index past the end ({0, 2}) is refused naming the file
 * and the number of structures it has, and adds nothing.
 * Before the fix: does not compile (there was no add_selected; tadah data
 * print and analysis descriptor called add(file, i) once per index, each
 * call re-reading the file from its start). */
TEST_CASE("StructureDB::add_selected(file, indices) adds those structures in order",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_add_selected");
  const std::string path = tmp.write("two.tadah", dataset_with("-11.4", "-11.4"));
  {
    StructureDB db;
    CHECK(db.add_selected(path, {1, 0}) == 2);
    REQUIRE(db.size() == 2);
    CHECK(db(0).label == "S2");
    CHECK(db(1).label == "S1");
  }
  {
    StructureDB db;
    CHECK(db.add_selected(path, {1}) == 1);
    REQUIRE(db.size() == 1);
    CHECK(db(0).label == "S2");
  }
  {
    StructureDB db;
    CHECK_THROWS_WITH(db.add_selected(path, {0, 2}),
                      Contains(path) && Contains("structure 3") &&
                          Contains("has 2 structures"));
    CHECK(db.size() == 0);
  }
}

/* A dataset entry labelled `label` with `natoms` Ta atoms (none: its blank
 * line comes straight after the stress lines), in a 3.3 A cubic cell. */
std::string ta_entry(const std::string &label, int natoms) {
  std::string e = label + "\n1 1 1\n-11.5\n3.3 0 0\n0 3.3 0\n0 0 3.3\n0 0 0\n0 0 0\n0 0 0\n";
  for (int k = 0; k < natoms; ++k)
    e += "Ta " + std::to_string(1.5 * k) + " 0 0 0 0 0\n";
  return e + "\n";
}

/* Pins: a structure without atoms (its blank line straight after the stress
 * lines) is one structure for the scanners that skip structures, as it is
 * for StructureDB::add(file):
 *  - S1 (2 atoms), E2 (no atoms), S3 (2 atoms): count gives 3 structures and
 *    4 atoms, add(file, 2, 1) returns 1 with S3, add(file, 1, 1) returns 1
 *    with E2, which has no atoms;
 *  - E1, S2 (2 atoms): count gives 2 structures and 2 atoms, add(file, 1, 1)
 *    returns 1 with S2;
 *  - S1, S2, E3 (2, 2 and no atoms), E3 ending the file without a blank
 *    line: count gives 3 structures and 4 atoms;
 *  in each, add(file) reads as many structures as count gives.
 * Before the fix: Structure::next_structure returned 0 both at the end of the
 * file and for a structure without atoms, so count stopped at the first
 * structure without atoms (1 structure and 2 atoms, 0 and 0, 2 and 4), and
 * add(file, 2, 1) on S1, E2, S3 and add(file, 1, 1) on E1, S2 stopped
 * skipping there and returned 0. */
TEST_CASE("StructureDB::count and add(file, first, N) count a structure without atoms",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_no_atoms");
  SECTION("S1, E2, S3") {
    const std::string path = tmp.write(
        "middle.tadah", ta_entry("S1", 2) + ta_entry("E2", 0) + ta_entry("S3", 2));
    const auto counted = StructureDB::count(path);
    CHECK(counted.first == 3);
    CHECK(counted.second == 4);
    StructureDB third;
    CHECK(third.add(path, 2, 1) == 1);
    REQUIRE(third.size() == 1);
    CHECK(third(0).label == "S3");
    StructureDB second;
    CHECK(second.add(path, 1, 1) == 1);
    REQUIRE(second.size() == 1);
    CHECK(second(0).label == "E2");
    CHECK(second(0).natoms() == 0);
    StructureDB all;
    all.add(path);
    CHECK(all.size() == 3);
  }
  SECTION("E1, S2") {
    const std::string path = tmp.write("first.tadah", ta_entry("E1", 0) + ta_entry("S2", 2));
    const auto counted = StructureDB::count(path);
    CHECK(counted.first == 2);
    CHECK(counted.second == 2);
    StructureDB second;
    CHECK(second.add(path, 1, 1) == 1);
    REQUIRE(second.size() == 1);
    CHECK(second(0).label == "S2");
    StructureDB all;
    all.add(path);
    CHECK(all.size() == 2);
  }
  SECTION("S1, S2, E3 ending the file") {
    std::string body = ta_entry("S1", 2) + ta_entry("S2", 2) + ta_entry("E3", 0);
    body.pop_back();  // E3 ends the file: no blank line after its stress lines
    const std::string path = tmp.write("last.tadah", body);
    const auto counted = StructureDB::count(path);
    CHECK(counted.first == 3);
    CHECK(counted.second == 4);
    StructureDB all;
    all.add(path);
    CHECK(all.size() == 3);
  }
}

/* Pins: StructureDB::add_selected(file, indices) reads only the structures it
 * adds and skips the others, as count does:
 *  - with S1's atom line 'Ta 0 0 0 abc 0 0' (not a number), {1} adds S2;
 *  - with S3's so (after the largest index), {1, 0} adds S2 and S1;
 *  - a selected structure that is not valid is refused naming the file, the
 *    structure (1-based) and the value: {0} on the first file, adding
 *    nothing;
 *  - S1, E2 (no atoms), S3: {2, 1} adds S3, then E2 with no atoms; {3} is
 *    refused as past the end of a file of 3 structures.
 * Before the fix: it read every structure from the first to the largest
 * index, keeping them all until it returned, so {1} was refused for S1's
 * 'abc' ("DBFILE ...: structure 1: ... 'abc' is not a finite number"). */
TEST_CASE("StructureDB::add_selected(file, indices) reads only the selected structures",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_add_selected_only");
  const std::string bad =
      "S1\n1 1 1\n-11.5\n3.3 0 0\n0 3.3 0\n0 0 3.3\n0 0 0\n0 0 0\n0 0 0\n"
      "Ta 0 0 0 abc 0 0\n\n";
  SECTION("S1 not valid") {
    const std::string path = tmp.write("bad_first.tadah", bad + ta_entry("S2", 2));
    StructureDB db;
    CHECK(db.add_selected(path, {1}) == 1);
    REQUIRE(db.size() == 1);
    CHECK(db(0).label == "S2");
    StructureDB refused;
    CHECK_THROWS_WITH(refused.add_selected(path, {0}),
                      Contains(path) && Contains("structure 1") && Contains("'abc'"));
    CHECK(refused.size() == 0);
  }
  SECTION("S3 not valid") {
    std::string third = bad;
    third.replace(0, 2, "S3");
    const std::string path =
        tmp.write("bad_last.tadah", ta_entry("S1", 2) + ta_entry("S2", 2) + third);
    StructureDB db;
    CHECK(db.add_selected(path, {1, 0}) == 2);
    REQUIRE(db.size() == 2);
    CHECK(db(0).label == "S2");
    CHECK(db(1).label == "S1");
  }
  SECTION("S1, E2, S3") {
    const std::string path = tmp.write(
        "no_atoms.tadah", ta_entry("S1", 2) + ta_entry("E2", 0) + ta_entry("S3", 2));
    StructureDB db;
    CHECK(db.add_selected(path, {2, 1}) == 2);
    REQUIRE(db.size() == 2);
    CHECK(db(0).label == "S3");
    CHECK(db(1).label == "E2");
    CHECK(db(1).natoms() == 0);
    StructureDB past;
    CHECK_THROWS_WITH(past.add_selected(path, {3}),
                      Contains(path) && Contains("structure 4") &&
                          Contains("has 3 structures"));
    CHECK(past.size() == 0);
  }
}

/* Pins: StructureDB::find_unique_elements (what tadah train and tadah hpo
 * read first) takes the whole first field of an atom line as the symbol, as
 * Structure::read does: a dataset with the atom line 'Tab 0 0 0 0.1 0 0' is
 * refused, naming 'Tab'.
 * Before the fix: it read two characters, so 'Tab' was taken as Ta. */
TEST_CASE("StructureDB::find_unique_elements reads the whole element field",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_scan_symbol");
  const std::string path =
      tmp.write("tab.tadah", dataset_with("Ta 0 0 0 0.1 0 0", "Tab 0 0 0 0.1 0 0"));
  CHECK_THROWS_WITH(StructureDB::find_unique_elements(path), Contains("Tab"));
}

/* Pins: a write that fails after the file opened (a full device: /dev/full
 * gives ENOSPC) is refused too, naming the file, for both dump_to_file (the
 * check after close() that 22158ff added had no test). Skipped where
 * /dev/full does not exist. */
TEST_CASE("StructureDB/Structure::dump_to_file: a failed write is refused",
          "[dataset_tools][structure_db]") {
  if (!fs::exists("/dev/full")) TADAH_SKIP("/dev/full does not exist here");
  PeriodicTable::initialize();
  ScratchDir tmp("dump_full");
  StructureDB db;
  db.add(tmp.write("good.tadah", dataset_with("-11.4", "-11.4")));
  REQUIRE(db.size() == 2);
  CHECK_THROWS_WITH(db.dump_to_file("/dev/full", false),
                    Catch::Contains("/dev/full") && Catch::Contains("write"));
  CHECK_THROWS_WITH(db(0).dump_to_file("/dev/full"),
                    Catch::Contains("/dev/full") && Catch::Contains("write"));
}

namespace {

/* The message f throws, or "no exception". */
template <class F> std::string thrown(F f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

/* A bcc Ta structure with 2 atoms whose cell and stress blocks are the
 * lines `blocks` (the format: 3 cell lines, then 3 stress lines). */
std::string ta_with_blocks(const std::string &label, const std::string &blocks) {
  return label + "\n1 1 1\n-11.5\n" + blocks +
         "Ta 0 0 0 0 0 0\nTa 1.65 1.65 1.65 0 0 0\n\n";
}

const char *const kBlocks33 = "3.3 0 0\n0 3.3 0\n0 0 3.3\n0 0 0\n0 0 0\n0 0 0\n";

}  // namespace

/* Pins: a dataset's cell and stress are six lines of exactly three
 * numbers (one lattice vector, then one stress-tensor row, per line, as
 * the Dataset Format of the docs and Structure::save write them), and every
 * reader of a dataset reads them so: StructureDB::add(file), count(file),
 * add(file, first, N) (skipping structures), add_selected and
 * find_unique_elements refuse a structure whose block has another layout,
 * naming the file, the structure (1-based, and its label), the block line
 * and its field count:
 *  - the cell on one line of 9 numbers ('cell line 1, ..., has 9 fields');
 *  - a stress line of 4 numbers ('stress line 3, '0 0 0 1', has 4 fields');
 *  - a blank line inside the cell block ('cell line 2, '', has 0 fields');
 *  - a file that ends inside the cell block ('the file ends inside the cell
 *    block, at line 3 of 3').
 * The 3+3 layout reads, and count() gives what add() reads (3 structures,
 * 6 atoms).
 * Before the fix: Structure::read took the 18 numbers whatever their lines
 * and dropped the rest of the last stress line, while next_structure (count,
 * add(file, first, N), add_selected) and find_unique_elements skipped 6
 * lines: no reader refused the three layouts. The one-line cell was read by
 * add() (3 structures, 6 atoms) but counted as 3 structures and 4 atoms
 * (S2's two atom lines skipped as stress lines); the 4-number stress line
 * was read with its 4th number dropped; the blank line inside the cell was
 * skipped by add() and counted as a cell line by the scanners (S2 counted
 * with 3 atoms, its last stress line taken for one, and find_unique_elements
 * refused it as 'Unknown symbol: 0'); the file ending inside the cell block
 * was refused by add() as "... the file ends inside the cell block" (no
 * line) and counted as 2 structures. */
TEST_CASE("StructureDB: the cell and the stress are three lines of three "
          "numbers, for every reader of a dataset",
          "[dataset_tools][structure_db]") {
  PeriodicTable::initialize();
  ScratchDir tmp("db_blocks");
  {
    const std::string path = tmp.write(
        "good.tadah", ta_with_blocks("S1", kBlocks33) +
                          ta_with_blocks("S2", kBlocks33) +
                          ta_with_blocks("S3", kBlocks33));
    StructureDB db;
    db.add(path);
    CHECK(db.size() == 3);
    CHECK(db.calc_natoms() == 6);
    const auto counted = StructureDB::count(path);
    CHECK(counted.first == 3);
    CHECK(counted.second == 6);
  }
  struct Case {
    std::string name, blocks, what;
  };
  const std::vector<Case> cases = {
      {"one_line_cell",
       "3.3 0 0 0 3.3 0 0 0 3.3\n0 0 0\n0 0 0\n0 0 0\n",
       "cell line 1, '3.3 0 0 0 3.3 0 0 0 3.3', has 9 fields: each cell line "
       "is one lattice vector, 3 numbers"},
      {"four_stress",
       "3.3 0 0\n0 3.3 0\n0 0 3.3\n0 0 0\n0 0 0\n0 0 0 1\n",
       "stress line 3, '0 0 0 1', has 4 fields: each stress line is one row "
       "of the stress tensor, 3 numbers"},
      {"blank_in_cell",
       "3.3 0 0\n\n0 3.3 0\n0 0 3.3\n0 0 0\n0 0 0\n0 0 0\n",
       "cell line 2, '', has 0 fields: each cell line is one lattice vector, "
       "3 numbers"},
  };
  for (const Case &c : cases) {
    INFO(c.name);
    const std::string path = tmp.write(
        c.name + ".tadah", ta_with_blocks("S1", kBlocks33) +
                               ta_with_blocks("S2", c.blocks) +
                               ta_with_blocks("S3", kBlocks33));
    const std::string want = "DBFILE " + path + ": structure 2: structure 'S2': " + c.what;
    StructureDB all;
    CHECK(thrown([&] { all.add(path); }) == want);
    CHECK(thrown([&] { (void)StructureDB::count(path); }) == want);
    StructureDB third;
    CHECK(thrown([&] { (void)third.add(path, 2, 1); }) == want);
    StructureDB selected;
    CHECK(thrown([&] { (void)selected.add_selected(path, {2}); }) == want);
    CHECK(thrown([&] { (void)StructureDB::find_unique_elements(path); }) == want);
  }
  {
    const std::string path = tmp.write(
        "ends.tadah", ta_with_blocks("S1", kBlocks33) +
                          "S2\n1 1 1\n-11.5\n3.3 0 0\n0 3.3 0\n");
    const std::string want =
        "DBFILE " + path +
        ": structure 2: structure 'S2': the file ends inside the cell block, "
        "at line 3 of 3";
    StructureDB all;
    CHECK(thrown([&] { all.add(path); }) == want);
    CHECK(thrown([&] { (void)StructureDB::count(path); }) == want);
    CHECK(thrown([&] { (void)StructureDB::find_unique_elements(path); }) == want);
  }
}
