// test_dataset_stats_close_contact.cpp
//
// `tadah analysis dataset_stats` audits the pair distances of a dataset as
// the model sees them. Two atoms at one position (for example a disordered
// CIF site read as two atoms) are a pair at distance 0: it is counted, so
// the r_min written to the statistics file is 0. The CLOSE_CONTACT key
// (--close-contact <A>) lists every pair of atoms closer than A A, so that
// pairs which are legal but wreck a fit with a singular basis (LJ, MIE, ZBL
// at 0.1 A) can be found before training.
//
// Report contract pinned here (standard output), printed only when
// CLOSE_CONTACT is given, at every log level (it is part of the report the
// user asked for, not a warning):
//   - "Atom pairs closer than <A> A (CLOSE_CONTACT): <K>" once per run, A
//     as given, K the number of pairs listed (0 included);
//   - one line per pair, each unordered pair once (a pair through a
//     periodic image included), in dataset order:
//       "structure <s> ('<label>'): atoms <i> and <j> (<Si>, <Sj>),
//        distance <r> A"
//     on one line, with s, i and j 1-based (the INDEX of `tadah data
//     print`); anything after the distance (such as an image shift) is
//     free;
//   - a CLOSE_CONTACT that is not greater than 0 is refused before anything
//     is computed or written, naming the key;
//   - without CLOSE_CONTACT nothing of the above is printed, and r_min still
//     counts a pair at distance 0.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/core/stats/data_stats_io.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/analysis_dataset_stats_engine.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::LogLevel;
using tadah::core::Logger;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Task;
using tadah::mlip::engines::AnalysisDatasetStatsEngine;
namespace fs = std::filesystem;

struct DsCloseTempDir {
  fs::path dir;
  DsCloseTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_dsclose_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~DsCloseTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
};

// Ta atoms at the given Cartesian positions in a cubic cell of edge `a`.
Structure ta_cell(const std::string &label, double a,
                  const std::vector<std::vector<double>> &positions) {
  Structure st;
  st.label = label;
  st.energy = -10.0;
  st.cell.load(a, 0.0, 0.0,
               0.0, a, 0.0,
               0.0, 0.0, a);
  for (const auto &p : positions)
    st.add_atom(Atom(Element("Ta"), p[0], p[1], p[2], 0.0, 0.0, 0.0));
  return st;
}

// 1-based: 1 bcc Ta (a = 3.3, first neighbours at 2.858 A, no close pair);
// 2 the same with a third Ta at (0.3, 0, 0), 0.3 A from atom 1;
// 3 two Ta 3.1 A apart in the cell, 0.2 A apart through the periodic image
//   along a0 (the CIF "x = 0 and x = 1" route);
// 4 bcc Ta with a third Ta on top of atom 1 (distance 0).
// Within 4 A there is no other pair below 0.5 A (counted by brute force).
StructureDB close_contact_db() {
  StructureDB db;
  db.add(ta_cell("bcc-Ta", 3.3, {{0.0, 0.0, 0.0}, {1.65, 1.65, 1.65}}));
  db.add(ta_cell("close-0.3-Ta", 3.3,
                 {{0.0, 0.0, 0.0}, {1.65, 1.65, 1.65}, {0.3, 0.0, 0.0}}));
  db.add(ta_cell("image-0.2-Ta", 3.3, {{0.1, 0.0, 0.0}, {3.2, 0.0, 0.0}}));
  db.add(ta_cell("coincident-Ta", 3.3,
                 {{0.0, 0.0, 0.0}, {1.65, 1.65, 1.65}, {0.0, 0.0, 0.0}}));
  return db;
}

// Runs `tadah analysis dataset_stats` on `db` (validate, then run, as the
// scheduler does) with RCUT2B 4 and, when given, CLOSE_CONTACT, at the CLI
// default verbosity (WARNING), with std::cout (the Logger's console)
// captured; returns what was printed.
std::string run_dataset_stats(const StructureDB &db,
                              std::optional<double> close_contact,
                              const DsCloseTempDir &tmp,
                              const std::string &outfile) {
  const std::string dbfile = tmp.path("close.tadahdb");
  db.dump_to_file(dbfile, /*append=*/false, 18);
  Context c;
  c.add("DBFILE", dbfile);
  c.add("RCUT2B", 4.0);  // RCUTMAX follows
  c.add("OUTFILE", outfile);
  if (close_contact) c.add("CLOSE_CONTACT", *close_contact);
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

std::size_t count_of(const std::string &text, const std::string &what) {
  std::size_t n = 0;
  for (auto p = text.find(what); p != std::string::npos;
       p = text.find(what, p + what.size()))
    ++n;
  return n;
}

// The first line of `text` that contains `what`; empty when there is none.
std::string line_containing(const std::string &text, const std::string &what) {
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line))
    if (line.find(what) != std::string::npos) return line;
  return {};
}

// The number written after "distance " in `line`; nullopt when there is no
// such text or no number after it.
std::optional<double> distance_in(const std::string &line) {
  const std::string key = "distance ";
  const auto p = line.find(key);
  if (p == std::string::npos) return std::nullopt;
  const char *begin = line.c_str() + p + key.size();
  char *end = nullptr;
  const double v = std::strtod(begin, &end);
  if (end == begin) return std::nullopt;
  return v;
}

// The Ta-Ta pair statistics of the written statistics file.
tadah::core::stats::PerPairStats ta_ta_stats(const std::string &outfile) {
  std::ifstream in(outfile);
  REQUIRE(in.good());
  const auto stats = tadah::core::stats::load(in);
  REQUIRE(stats.per_pair.size() == 1);
  REQUIRE(stats.per_pair[0].elements.first == "Ta");
  REQUIRE(stats.per_pair[0].elements.second == "Ta");
  return stats.per_pair[0];
}

}  // namespace

/* Pins: with CLOSE_CONTACT 0.5 dataset_stats prints the count line and one
 * line per pair closer than 0.5 A, once per unordered pair, by structure
 * (1-based index and label), atoms (1-based, with symbols) and distance,
 * in dataset order: the 0.3 A pair of structure 2, the 0.2 A pair of
 * structure 3 that exists only through a periodic image (the neighbour
 * list holds it as 1 -> 2 and as 2 -> 1; it is one pair), and the pair at
 * distance 0 of structure 4. Structure 1 has no close pair and is not
 * named. The statistics file is still written, with r_min 0.
 * Before the fix: there was no CLOSE_CONTACT key; Context::add threw
 * "Key not allowed: CLOSE_CONTACT", so the run never started. */
TEST_CASE("analysis dataset_stats lists every pair closer than CLOSE_CONTACT "
          "once, by structure, atoms and distance",
          "[dataset_stats][close_contact]") {
  PeriodicTable::initialize();
  DsCloseTempDir tmp;
  const std::string out = tmp.path("stats.tadah");
  const std::string log =
      run_dataset_stats(close_contact_db(), 0.5, tmp, out);
  INFO(log);

  CHECK(count_of(log, "Atom pairs closer than 0.5 A (CLOSE_CONTACT): 3") == 1);

  struct Expected {
    const char *where;  // "structure <s> ('<label>'):"
    const char *atoms;  // "atoms <i> and <j>"
    double r;
  };
  const Expected expected[] = {
      {"structure 2 ('close-0.3-Ta'):", "atoms 1 and 3", 0.3},
      {"structure 3 ('image-0.2-Ta'):", "atoms 1 and 2", 0.2},
      {"structure 4 ('coincident-Ta'):", "atoms 1 and 3", 0.0},
  };
  for (const auto &e : expected) {
    INFO(e.where);
    CHECK(count_of(log, e.where) == 1);  // each pair once
    const std::string line = line_containing(log, e.where);
    REQUIRE_FALSE(line.empty());
    CHECK(line.find(e.atoms) != std::string::npos);
    CHECK(line.find("(Ta, Ta)") != std::string::npos);
    const auto r = distance_in(line);
    REQUIRE(r);
    CHECK(*r == Approx(e.r).margin(1e-6));
  }
  CHECK(log.find("('bcc-Ta')") == std::string::npos);
  CHECK(log.find("structure 1 (") == std::string::npos);
  // Dataset order.
  CHECK(log.find("structure 2 ('close-0.3-Ta')") <
        log.find("structure 3 ('image-0.2-Ta')"));
  CHECK(log.find("structure 3 ('image-0.2-Ta')") <
        log.find("structure 4 ('coincident-Ta')"));

  const auto tata = ta_ta_stats(out);
  CHECK(tata.r_min == 0.0);
}

/* Pins: CLOSE_CONTACT only adds the list. Without it nothing about close
 * contacts is printed, and the r_min written for the same dataset still
 * counts the pair at distance 0 of structure 4.
 * Before the fix: the run printed no list (that part held already), but
 * the pair loop skipped rij <= 0, so r_min was that of the image pair of
 * structure 3 (about 0.2 A), not 0. */
TEST_CASE("analysis dataset_stats without CLOSE_CONTACT lists no close "
          "contacts and writes r_min 0 for coincident atoms",
          "[dataset_stats][close_contact][coincident]") {
  PeriodicTable::initialize();
  DsCloseTempDir tmp;
  const std::string out = tmp.path("stats.tadah");
  const std::string log =
      run_dataset_stats(close_contact_db(), std::nullopt, tmp, out);
  INFO(log);

  CHECK(log.find("closer than") == std::string::npos);
  CHECK(log.find("CLOSE_CONTACT") == std::string::npos);
  CHECK(log.find("structure 2 ('close-0.3-Ta'):") == std::string::npos);
  CHECK(log.find("structure 3 ('image-0.2-Ta'):") == std::string::npos);
  CHECK(log.find("structure 4 ('coincident-Ta'):") == std::string::npos);

  const auto tata = ta_ta_stats(out);
  INFO("r_min " << tata.r_min);
  CHECK(tata.r_min == 0.0);
}

/* Pins: with CLOSE_CONTACT and no pair below it, the count line states 0
 * and no structure is listed.
 * Before the fix: Context::add threw "Key not allowed: CLOSE_CONTACT". */
TEST_CASE("analysis dataset_stats states that no pair is closer than "
          "CLOSE_CONTACT",
          "[dataset_stats][close_contact]") {
  PeriodicTable::initialize();
  DsCloseTempDir tmp;
  StructureDB db;
  db.add(ta_cell("bcc-Ta", 3.3, {{0.0, 0.0, 0.0}, {1.65, 1.65, 1.65}}));
  const std::string log =
      run_dataset_stats(db, 0.5, tmp, tmp.path("stats.tadah"));
  INFO(log);
  CHECK(count_of(log, "Atom pairs closer than 0.5 A (CLOSE_CONTACT): 0") == 1);
  CHECK(log.find("('bcc-Ta')") == std::string::npos);
}

/* Pins: a CLOSE_CONTACT that is not greater than 0 cannot list any pair
 * (no distance is below it), so it is refused, naming the key, before
 * anything is computed or written, instead of reporting 0 pairs.
 * Before the fix: Context::add threw "Key not allowed: CLOSE_CONTACT",
 * which does not say "greater than 0". */
TEST_CASE("analysis dataset_stats refuses a CLOSE_CONTACT that is not "
          "greater than 0",
          "[dataset_stats][close_contact][invalid]") {
  PeriodicTable::initialize();
  for (const double bad : {0.0, -0.1}) {
    INFO("CLOSE_CONTACT " << bad);
    DsCloseTempDir tmp;
    const std::string out = tmp.path("stats.tadah");
    CHECK_THROWS_WITH(run_dataset_stats(close_contact_db(), bad, tmp, out),
                      Catch::Matchers::Contains("CLOSE_CONTACT") &&
                          Catch::Matchers::Contains("greater than 0"));
    CHECK_FALSE(fs::exists(out));
  }
}
