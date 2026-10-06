// test_find_close_contacts.cpp
//
// find_close_contacts (data_stats_compute.h) lists every pair of atoms
// closer than a threshold: the CLOSE_CONTACT list of `tadah analysis
// dataset_stats`. test_dataset_stats_close_contact.cpp pins the report; this
// file pins the branches of the function itself:
//   - a pair through a periodic image carries the integer image shift of its
//     second atom in the stored basis (also for an atom stored outside the
//     cell), and the report prints it;
//   - in a cell thinner than the threshold an atom is close to its own image:
//     that pair is listed once, with the first non-zero component of the
//     shift positive, whichever lattice vector is short;
//   - a threshold beyond the cutoff of a structure's stored list is answered
//     in full from a list built for the scan, and the stored list is left as
//     it was (the same list, the same cutoff, the same entries);
//   - a threshold that is not a finite number greater than 0 (nan, +-inf, 0,
//     negative) is refused before any structure is scanned, by the function
//     and by the engine (CLOSE_CONTACT);
//   - a structure that cannot be scanned (a singular cell, a cell too small
//     for the threshold) is named, the lowest-numbered one:
//     "find_close_contacts: structure <i> of <N> ('<label>'): ", with the type
//     of the neighbour-list error kept.
//
// The geometries are built here and every expected pair, shift and distance
// is worked out by hand in the comment of its case. Values that are not
// finite are passed through a volatile, so that no compiler folds them into
// the call, and are never compared (the build may use -ffast-math).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/data_stats_compute.h>
#include <tadah/mlip/engines/analysis_dataset_stats_engine.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/nn_table.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::LogLevel;
using tadah::core::Logger;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::CloseContact;
using tadah::mlip::find_close_contacts;
using tadah::mlip::NNFinder;
using tadah::mlip::NNTablePtr;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Task;
using tadah::mlip::engines::AnalysisDatasetStatsEngine;
namespace fs = std::filesystem;

// A structure labelled `label` whose cell has the rows a0, a1 and a2.
Structure with_rows(const std::string &label, double a0x, double a0y,
                    double a0z, double a1x, double a1y, double a1z,
                    double a2x, double a2y, double a2z) {
  Structure st;
  st.label = label;
  st.energy = -10.0;
  st.cell.load(a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z);
  return st;
}

Structure cube(const std::string &label, double a) {
  return with_rows(label, a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
}

void add_ta(Structure &st, double x, double y, double z) {
  st.add_atom(Atom(PeriodicTable::find_by_symbol("Ta"), x, y, z, 0.0, 0.0,
                   0.0));
}

// A value the compiler cannot see through (read from a volatile).
double opaque(double x) {
  volatile double v = x;
  return v;
}

bool contains(const std::string &s, const std::string &part) {
  return s.find(part) != std::string::npos;
}

bool starts_with(const std::string &s, const std::string &prefix) {
  return s.compare(0, prefix.size(), prefix) == 0;
}

// What `f` throws: the type ("invalid_argument", "runtime_error" or
// "other exception") and the message; both empty when nothing is thrown.
struct Thrown {
  std::string type;
  std::string what;
};

template <class F>
Thrown thrown(F f) {
  try {
    f();
  } catch (const std::invalid_argument &e) {
    return {"invalid_argument", e.what()};
  } catch (const std::runtime_error &e) {
    return {"runtime_error", e.what()};
  } catch (const std::exception &e) {
    return {"other exception", e.what()};
  }
  return {"", ""};
}

// An expected close contact (0-based structure and atoms).
struct Want {
  std::size_t structure;
  std::size_t i;
  std::size_t j;
  std::array<long long, 3> shift;
  double r;
};

std::string describe(const std::vector<CloseContact> &cs) {
  std::ostringstream o;
  o << cs.size() << " contact(s)";
  o.precision(17);
  for (const auto &c : cs)
    o << "\n  structure " << c.structure << ": atoms " << c.i << " and "
      << c.j << ", shift (" << c.shift[0] << ", " << c.shift[1] << ", "
      << c.shift[2] << "), r " << c.r;
  return o.str();
}

// `got` is exactly `want`, in that order.
void check_contacts(const std::vector<CloseContact> &got,
                    const std::vector<Want> &want) {
  INFO(describe(got));
  REQUIRE(got.size() == want.size());
  for (std::size_t k = 0; k < want.size(); ++k) {
    INFO("contact " << k);
    CHECK(got[k].structure == want[k].structure);
    CHECK(got[k].i == want[k].i);
    CHECK(got[k].j == want[k].j);
    CHECK(got[k].shift[0] == want[k].shift[0]);
    CHECK(got[k].shift[1] == want[k].shift[1]);
    CHECK(got[k].shift[2] == want[k].shift[2]);
    CHECK(got[k].r == Approx(want[k].r).margin(1e-12));
  }
}

// A deep copy of a neighbour list (NNTable is held through a pointer to
// const, so a copy is what "the same entries" is compared against).
struct ListCopy {
  double cutoff = 0.0;
  std::vector<std::vector<std::size_t>> idx;
  std::vector<std::vector<std::array<double, 3>>> shift;
};

ListCopy copy_of(const NNTablePtr &t) {
  ListCopy c;
  c.cutoff = t->cutoff;
  c.idx = t->idx;
  c.shift.resize(t->shift.size());
  for (std::size_t i = 0; i < t->shift.size(); ++i)
    for (const auto &s : t->shift[i])
      c.shift[i].push_back({s[0], s[1], s[2]});
  return c;
}

bool same_list(const NNTablePtr &t, const ListCopy &c) {
  if (!t || t->cutoff != c.cutoff || t->idx != c.idx ||
      t->shift.size() != c.shift.size())
    return false;
  for (std::size_t i = 0; i < c.shift.size(); ++i) {
    if (t->shift[i].size() != c.shift[i].size()) return false;
    for (std::size_t k = 0; k < c.shift[i].size(); ++k)
      for (int a = 0; a < 3; ++a)
        if (t->shift[i][k][a] != c.shift[i][k][a]) return false;
  }
  return true;
}

struct CcTempDir {
  fs::path dir;
  CcTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_fcc_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~CcTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
};

// Runs `tadah analysis dataset_stats` on `db` (validate, then run, as the
// scheduler does) with RCUT2B 4 and CLOSE_CONTACT `cc`, at the CLI default
// verbosity (WARNING), with std::cout captured; returns what was printed.
// What the engine throws reaches the caller (std::cout and the log level
// restored).
std::string run_dataset_stats(const StructureDB &db, double cc,
                              const CcTempDir &tmp,
                              const std::string &outfile) {
  const std::string dbfile = tmp.path("contacts.tadahdb");
  db.dump_to_file(dbfile, /*append=*/false, 18);
  Context c;
  c.add("DBFILE", dbfile);
  c.add("RCUT2B", 4.0);  // RCUTMAX follows
  c.add("OUTFILE", outfile);
  c.add("CLOSE_CONTACT", cc);
  Task task("analysis", c);

  auto &log = Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(LogLevel::WARNING);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  int rc = -1;
  try {
    AnalysisDatasetStatsEngine engine;
    engine.validate(task);
    rc = engine.run(task);
  } catch (...) {
    std::cout.rdbuf(old_buf);
    log.setLogLevel(old_level);
    throw;
  }
  std::cout.rdbuf(old_buf);
  log.setLogLevel(old_level);
  REQUIRE(rc == EXIT_SUCCESS);
  return captured.str();
}

// The first line of `text` that contains `what`; empty when there is none.
std::string line_containing(const std::string &text, const std::string &what) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line))
    if (line.find(what) != std::string::npos) return line;
  return {};
}

// 3.3 A cubes with two or three Ta (0-based structures and atoms):
// 0 'image-x': (0.1, 0, 0) and (3.2, 0, 0). The image of atom 1 shifted by
//   -a0 is at (-0.1, 0, 0), 0.2 A from atom 0; in the cell they are 3.1 A
//   apart. Pair (0, 1), shift (-1, 0, 0), r 0.2.
// 1 'image-xyz': (0.1, 0.1, 0.1) and (3.2, 3.2, 3.2). The image of atom 1
//   shifted by -a0 - a1 - a2 is at (-0.1, -0.1, -0.1): pair (0, 1), shift
//   (-1, -1, -1), r sqrt(3 x 0.2^2) = sqrt(0.12) = 0.3464 A.
// 2 'outside-cell': (3.4, 0, 0) (outside the cell, as stored) and
//   (0.2, 0, 0). The image of atom 1 shifted by +a0 is at (3.5, 0, 0),
//   0.1 A from atom 0 as stored: pair (0, 1), shift (1, 0, 0), r 0.1.
// 3 'same-cell': bcc Ta (0, 0, 0), (1.65, 1.65, 1.65) and a third Ta at
//   (0.3, 0, 0): pair (0, 2), shift (0, 0, 0), r 0.3.
// No other pair is closer than 2.69 A (atoms 1 and 2 of 'same-cell'; counted
// by brute force over the images as well), so with threshold 0.5 these four
// are the whole list, in dataset order.
StructureDB image_db() {
  StructureDB db;
  Structure s0 = cube("image-x", 3.3);
  add_ta(s0, 0.1, 0.0, 0.0);
  add_ta(s0, 3.2, 0.0, 0.0);
  db.add(s0);
  Structure s1 = cube("image-xyz", 3.3);
  add_ta(s1, 0.1, 0.1, 0.1);
  add_ta(s1, 3.2, 3.2, 3.2);
  db.add(s1);
  Structure s2 = cube("outside-cell", 3.3);
  add_ta(s2, 3.4, 0.0, 0.0);
  add_ta(s2, 0.2, 0.0, 0.0);
  db.add(s2);
  Structure s3 = cube("same-cell", 3.3);
  add_ta(s3, 0.0, 0.0, 0.0);
  add_ta(s3, 1.65, 1.65, 1.65);
  add_ta(s3, 0.3, 0.0, 0.0);
  db.add(s3);
  return db;
}

// One Ta in a cell with one short lattice vector (0-based structures):
// 0 'thin-a0': rows (0.4, 0, 0), (0, 10, 0), (0, 0, 10), Ta at (0.1, 5, 5).
//   The atom's images along a0 lie 0.4, 0.8, 1.2 A away: shifts (k, 0, 0).
// 1 'thin-a1-a2': rows (10, 0, 0), (0, 3, 0), (0, 2.8, 0.3), Ta at
//   (1, 1, 0.1). No row is short, but a1 - a2 = (0, 0.2, -0.3) is
//   sqrt(0.13) = 0.3606 A long: images k (a1 - a2), shifts (0, k, -k), at
//   0.3606, 0.7211, 1.0817 A. Every other lattice vector is longer than
//   1 A: one with an a0 component is 10 A long or more, and n1 a1 + n2 a2
//   that is no multiple of a1 - a2 has |n2| >= 4 (1.2 A in z) or a y
//   component of at least 2 A.
// 2 'thin-a2': rows (10, 0, 0), (0, 10, 0), (0, 0, 0.42), Ta at (5, 5, 0.1).
//   Images along a2 at 0.42, 0.84, 1.26 A: shifts (0, 0, k).
// The neighbour list holds each such image as s and -s; the first non-zero
// component of the shift that is reported is in a different place in each
// structure (component 0, 1 and 2).
StructureDB thin_db() {
  StructureDB db;
  Structure s0 = with_rows("thin-a0", 0.4, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                           0.0, 10.0);
  add_ta(s0, 0.1, 5.0, 5.0);
  db.add(s0);
  Structure s1 = with_rows("thin-a1-a2", 10.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
                           2.8, 0.3);
  add_ta(s1, 1.0, 1.0, 0.1);
  db.add(s1);
  Structure s2 = with_rows("thin-a2", 10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                           0.0, 0.42);
  add_ta(s2, 5.0, 5.0, 0.1);
  db.add(s2);
  return db;
}

// bcc Ta (a = 3.3), first neighbours 2.858 A apart: no pair below 2.8 A.
Structure bcc_ta(const std::string &label) {
  Structure st = cube(label, 3.3);
  add_ta(st, 0.0, 0.0, 0.0);
  add_ta(st, 1.65, 1.65, 1.65);
  return st;
}

}  // namespace

/* Pins: a pair that is close only through a periodic image is listed once,
 * as (i, j) with i < j and the integer shift of atom j's image in the stored
 * basis: (-1, 0, 0) and (-1, -1, -1) for atoms stored in the cell, (1, 0, 0)
 * when atom i is stored outside the cell (the shift places the image
 * relative to the stored positions), and (0, 0, 0) for a pair in the cell;
 * the distance is the image distance (geometries and values in image_db()).
 * The report of `tadah analysis dataset_stats --close-contact 0.5` ends the
 * line of a pair through an image with ", image shift (<n0>, <n1>, <n2>)"
 * and prints no shift for a pair in the cell, as the engine documents.
 * Before the fix: passes; today's code does exactly this (the shifts and the
 * printed shift were not checked by any test). */
TEST_CASE("find_close_contacts gives a pair through a periodic image the "
          "image shift of its second atom, and the report prints it",
          "[close_contact][find_close_contacts][image]") {
  PeriodicTable::initialize();
  const StructureDB db = image_db();
  check_contacts(find_close_contacts(db, 0.5),
                 {{0, 0, 1, {{-1, 0, 0}}, 0.2},
                  {1, 0, 1, {{-1, -1, -1}}, std::sqrt(0.12)},
                  {2, 0, 1, {{1, 0, 0}}, 0.1},
                  {3, 0, 2, {{0, 0, 0}}, 0.3}});

  // The report, for the structures stored in the cell: 1-based structures
  // 1 'image-x', 2 'image-xyz', 3 'same-cell'.
  StructureDB in_cell;
  in_cell.add(db(0));
  in_cell.add(db(1));
  in_cell.add(db(3));
  CcTempDir tmp;
  const std::string log =
      run_dataset_stats(in_cell, 0.5, tmp, tmp.path("stats.tadah"));
  INFO(log);
  CHECK(contains(log, "Atom pairs closer than 0.5 A (CLOSE_CONTACT): 3"));
  const std::string x = line_containing(log, "structure 1 ('image-x'):");
  const std::string xyz = line_containing(log, "structure 2 ('image-xyz'):");
  const std::string same = line_containing(log, "structure 3 ('same-cell'):");
  CHECK(contains(x, "atoms 1 and 2 (Ta, Ta)"));
  CHECK(contains(x, ", image shift (-1, 0, 0)"));
  CHECK(contains(xyz, "atoms 1 and 2 (Ta, Ta)"));
  CHECK(contains(xyz, ", image shift (-1, -1, -1)"));
  CHECK(contains(same, "atoms 1 and 3 (Ta, Ta)"));
  CHECK_FALSE(same.empty());
  CHECK_FALSE(contains(same, "image shift"));
}

/* Pins: in a cell with a lattice vector shorter than the threshold, an atom
 * is close to its own images. Each such pair is listed once (the list holds
 * it as shift s and as -s), as (i, i) with the first non-zero component of s
 * positive, in the order NNFinder sorts the shifts: with threshold 0.5,
 * (1, 0, 0) at 0.4 A, (0, 1, -1) at sqrt(0.13) A (a1 - a2, a lattice vector
 * that is no row of the cell) and (0, 0, 1) at 0.42 A; with threshold 0.9
 * also the second images (2, 0, 0) at 0.8 A, (0, 2, -2) at 2 sqrt(0.13) A
 * and (0, 0, 2) at 0.84 A, each after the first image of its structure
 * (geometries in thin_db()).
 * Before the fix: passes; today's code does exactly this (the i == j branch
 * of find_close_contacts was not exercised by any test). */
TEST_CASE("find_close_contacts lists an atom and its own image in a thin "
          "cell once, with the first non-zero shift component positive",
          "[close_contact][find_close_contacts][thin_cell]") {
  PeriodicTable::initialize();
  const StructureDB db = thin_db();
  const double r1 = std::sqrt(0.13);
  check_contacts(find_close_contacts(db, 0.5),
                 {{0, 0, 0, {{1, 0, 0}}, 0.4},
                  {1, 0, 0, {{0, 1, -1}}, r1},
                  {2, 0, 0, {{0, 0, 1}}, 0.42}});
  check_contacts(find_close_contacts(db, 0.9),
                 {{0, 0, 0, {{1, 0, 0}}, 0.4},
                  {0, 0, 0, {{2, 0, 0}}, 0.8},
                  {1, 0, 0, {{0, 1, -1}}, r1},
                  {1, 0, 0, {{0, 2, -2}}, 2.0 * r1},
                  {2, 0, 0, {{0, 0, 1}}, 0.42},
                  {2, 0, 0, {{0, 0, 2}}, 0.84}});
}

/* Pins: the answer does not depend on the list a structure stores. bcc Ta
 * (a = 3.3) with a third Ta at (0.3, 0, 0), 0.3 A from atom 0, threshold
 * 0.5:
 * - with a stored list built for 0.25 A (which holds no pair at all), the
 *   pair (0, 2) at 0.3 A is still found, from a list built for the scan;
 * - with a stored list built for 4 A (which covers the threshold), the same
 *   single pair is found.
 * In both cases the structure keeps its list: the same NNTable object, with
 * the same cutoff and the same entries.
 * Before the fix: passes; today's code does exactly this (no test stored a
 * list below the threshold, nor checked that the stored list is kept). */
TEST_CASE("find_close_contacts answers a threshold beyond the cutoff of a "
          "stored list in full and leaves the stored list as it was",
          "[close_contact][find_close_contacts][stored_list]") {
  PeriodicTable::initialize();
  StructureDB db;
  Structure st = bcc_ta("close-0.3-Ta");
  add_ta(st, 0.3, 0.0, 0.0);
  db.add(st);
  double stored_cutoff = 0.0;
  SECTION("a stored list built for 0.25 A, below the 0.3 A pair") {
    stored_cutoff = 0.25;
  }
  SECTION("a stored list built for 4 A, beyond the threshold") {
    stored_cutoff = 4.0;
  }
  INFO("stored cutoff " << stored_cutoff);
  NNFinder().calc(db(0), stored_cutoff);
  const NNTablePtr before = db(0).nn;
  REQUIRE(before);
  const ListCopy saved = copy_of(before);
  if (stored_cutoff < 0.3)
    for (const auto &entries : saved.idx) REQUIRE(entries.empty());

  const StructureDB &cdb = db;
  check_contacts(find_close_contacts(cdb, 0.5),
                 {{0, 0, 2, {{0, 0, 0}}, 0.3}});

  CHECK(db(0).nn == before);
  CHECK(db(0).nn->cutoff == stored_cutoff);
  CHECK(same_list(db(0).nn, saved));
}

/* Pins: a threshold that is not a finite number greater than 0 (nan, +inf,
 * -inf, 0, -0.1) is refused with std::invalid_argument "find_close_contacts:
 * the threshold must be a finite distance greater than 0, got <value>",
 * before any structure is scanned: the same for a dataset without
 * structures, for one whose structures have no list, and for one whose
 * structures store a list built for 4 A. No structure is named.
 * Before the fix: passes in this workspace's build, where tadah.mlip and
 * test_mlip are compiled without -ffast-math. The check is
 * `!std::isfinite(threshold) || !(threshold > 0.0)`, and with -ffast-math
 * g++ folds std::isfinite to true (a standalone probe of that check, g++
 * 15.2 -O3 -ffast-math -march=native, accepts nan and +inf), so in such a
 * build nan and +inf get through: the empty dataset returns an empty list
 * and nothing is thrown, a structure without a list is refused by NNFinder
 * as "find_close_contacts: structure 1 of 1 ('bcc-Ta'): NNFinder: the
 * cutoff must be a finite number above 0, got inf", and what a structure
 * with a stored 4 A list gives for nan depends on how the comparisons with
 * nan are compiled. NNFinder tests the exponent bits
 * (detail::is_finite_bits) for this reason. */
TEST_CASE("find_close_contacts refuses a threshold that is not a finite "
          "number greater than 0 before scanning any structure",
          "[close_contact][find_close_contacts][invalid]") {
  PeriodicTable::initialize();
  StructureDB db;
  std::string kind;
  SECTION("a dataset without structures") { kind = "empty"; }
  SECTION("a structure without a neighbour list") {
    kind = "no list";
    db.add(bcc_ta("bcc-Ta"));
  }
  SECTION("a structure with a list built for 4 A") {
    kind = "stored list";
    db.add(bcc_ta("bcc-Ta"));
    NNFinder().calc(db(0), 4.0);
    REQUIRE(db(0).nn);
  }
  struct Bad {
    const char *name;
    double value;
  };
  const Bad bad[] = {{"nan", std::numeric_limits<double>::quiet_NaN()},
                     {"+inf", std::numeric_limits<double>::infinity()},
                     {"-inf", -std::numeric_limits<double>::infinity()},
                     {"0", 0.0},
                     {"-0.1", -0.1}};
  const StructureDB &cdb = db;
  for (const Bad &b : bad) {
    const Thrown t =
        thrown([&] { (void)find_close_contacts(cdb, opaque(b.value)); });
    INFO(kind << ", threshold " << b.name << ": " << t.type << ": "
              << t.what);
    CHECK(t.type == "invalid_argument");
    CHECK(contains(t.what, "find_close_contacts: the threshold must be a "
                           "finite distance greater than 0"));
    CHECK_FALSE(contains(t.what, "structure"));
  }
}

/* Pins: `tadah analysis dataset_stats` refuses a CLOSE_CONTACT of nan, inf
 * or -inf before anything is written (no statistics file), with a message
 * that names CLOSE_CONTACT and the value as given.
 * Before the fix: passes; the value never reaches the engine's own check
 * (`!std::isfinite(cc) || !(cc > 0.0)`, folded like find_close_contacts'
 * under -ffast-math): Context::get<double> reads the stored text with an
 * istringstream, which does not parse "nan", "inf" or "-inf", so validate()
 * throws std::runtime_error "Conversion failed for key=CLOSE_CONTACT,
 * value=\"nan\"" (and so on). */
TEST_CASE("analysis dataset_stats refuses a CLOSE_CONTACT of nan or inf "
          "before anything is written, naming the key",
          "[dataset_stats][close_contact][invalid]") {
  PeriodicTable::initialize();
  StructureDB db;
  db.add(bcc_ta("bcc-Ta"));
  struct Bad {
    const char *text;
    double value;
  };
  const Bad bad[] = {{"nan", std::numeric_limits<double>::quiet_NaN()},
                     {"inf", std::numeric_limits<double>::infinity()},
                     {"-inf", -std::numeric_limits<double>::infinity()}};
  for (const Bad &b : bad) {
    CcTempDir tmp;
    const std::string out = tmp.path("stats.tadah");
    const Thrown t = thrown(
        [&] { (void)run_dataset_stats(db, opaque(b.value), tmp, out); });
    INFO("CLOSE_CONTACT " << b.text << ": " << t.type << ": " << t.what);
    CHECK_FALSE(t.type.empty());
    CHECK(contains(t.what, "CLOSE_CONTACT"));
    CHECK(contains(t.what, b.text));
    CHECK_FALSE(fs::exists(out));
  }
}

/* Pins: a structure the neighbour search refuses is named, the
 * lowest-numbered one, and the error keeps its type:
 * - of bcc Ta, 'flat' and 'flat-too' (both with the coplanar rows
 *   (3.3, 0, 0), (0, 3.3, 0), (3.3, 3.3, 0), volume exactly 0), threshold
 *   0.5: std::invalid_argument beginning "find_close_contacts: structure 2
 *   of 3 ('flat'): " and saying the cell is singular; structure 3 is not
 *   named;
 * - of bcc Ta and 'tiny' (a 0.1 A cube), threshold 5: the search would need
 *   more than 10^4 image cells, std::runtime_error beginning
 *   "find_close_contacts: structure 2 of 2 ('tiny'): " and saying the cell
 *   is too small for the cutoff.
 * Before the fix: passes; today's code does exactly this (the prefix was
 * documented but not tested). */
TEST_CASE("find_close_contacts names the first structure it cannot scan, "
          "keeping the type of the error",
          "[close_contact][find_close_contacts][invalid]") {
  PeriodicTable::initialize();
  StructureDB db;
  db.add(bcc_ta("bcc-Ta"));
  SECTION("a singular cell") {
    for (const char *label : {"flat", "flat-too"}) {
      Structure flat = with_rows(label, 3.3, 0.0, 0.0, 0.0, 3.3, 0.0, 3.3,
                                 3.3, 0.0);
      add_ta(flat, 0.0, 0.0, 0.0);
      db.add(flat);
    }
    const Thrown t = thrown([&] { (void)find_close_contacts(db, 0.5); });
    INFO(t.type << ": " << t.what);
    CHECK(t.type == "invalid_argument");
    CHECK(starts_with(t.what,
                      "find_close_contacts: structure 2 of 3 ('flat'): "));
    CHECK(contains(t.what, "singular"));
    CHECK_FALSE(contains(t.what, "structure 3"));
    CHECK_FALSE(contains(t.what, "flat-too"));
  }
  SECTION("a cell too small for the threshold") {
    Structure tiny = cube("tiny", 0.1);
    add_ta(tiny, 0.0, 0.0, 0.0);
    db.add(tiny);
    const Thrown t = thrown([&] { (void)find_close_contacts(db, 5.0); });
    INFO(t.type << ": " << t.what);
    CHECK(t.type == "runtime_error");
    CHECK(starts_with(t.what,
                      "find_close_contacts: structure 2 of 2 ('tiny'): "));
    CHECK(contains(t.what, "too small for the cutoff"));
  }
}
