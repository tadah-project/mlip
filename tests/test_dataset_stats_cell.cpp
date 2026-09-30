// test_dataset_stats_cell.cpp
//
// `tadah analysis dataset_stats` reports the structures whose stored cell is
// not symmetric (max |C - C^T| > 0). Datasets converted by older Tadah
// versions may hold a transposed cell (OUTCAR/CASTEP before 1.3.0-beta.1,
// CIF/COD non-orthogonal cells before 1.6.0); a transposed cell can only be
// told apart from a correct one when the cell is not symmetric, so the
// report lists exactly those structures and points to the data-migration
// note. It makes no guess about whether a listed cell is transposed.
//
// Report contract pinned here (standard output):
//   - "<K> of <N> structures have a non-symmetric cell" once per run, at
//     every log level: it is part of the report the user asked for, not a
//     warning (a non-symmetric cell is a valid cell);
//   - one line "structure <i>: <label>" per listed structure, where i is the
//     1-based position in the loaded dataset (the INDEX of `tadah data
//     print` / `tadah data write`), in dataset order;
//   - when K > 0, the text "migration note" (the pointer to the
//     CHANGELOG/docs note);
//   - the DataStats output file is still written and still loads
//     (tadah::core::stats::load) with every structure counted.

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

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::LogLevel;
using tadah::core::Logger;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Task;
using tadah::mlip::engines::AnalysisDatasetStatsEngine;
namespace fs = std::filesystem;

struct DsCellTempDir {
  fs::path dir;
  DsCellTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_dscell_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~DsCellTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
};

// Two Ta atoms at fractional (0,0,0) and (1/2,1/2,1/2) of `cell`.
Structure ta_pair(const std::string &label, const Matrix3d &cell) {
  Structure st;
  st.label = label;
  st.energy = -20.0;
  st.cell = cell;
  st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  double r[3];
  for (int k = 0; k < 3; ++k)
    r[k] = 0.5 * (cell(0, k) + cell(1, k) + cell(2, k));
  st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0.0, 0.0, 0.0));
  return st;
}

Matrix3d cubic() {  // symmetric (diagonal)
  Matrix3d c;
  c.load(3.3, 0.0, 0.0,
         0.0, 3.3, 0.0,
         0.0, 0.0, 3.3);
  return c;
}
Matrix3d fcc_primitive() {  // symmetric, not diagonal
  Matrix3d c;
  c.load(0.0, 2.1, 2.1,
         2.1, 0.0, 2.1,
         2.1, 2.1, 0.0);
  return c;
}
Matrix3d sheared() {  // a1 = (1.1, 3.3, 0): C(1,0) != C(0,1)
  Matrix3d c;
  c.load(3.3, 0.0, 0.0,
         1.1, 3.3, 0.0,
         0.0, 0.0, 3.3);
  return c;
}
Matrix3d hexagonal() {  // a1 = (-a/2, a*sqrt(3)/2, 0)
  const double a = 3.3;
  Matrix3d c;
  c.load(a, 0.0, 0.0,
         -0.5 * a, 0.5 * std::sqrt(3.0) * a, 0.0,
         0.0, 0.0, 3.6);
  return c;
}
Matrix3d tiny_asymmetry() {  // |C - C^T| = 1e-12: > 0, so it is listed
  Matrix3d c = cubic();
  c(0, 1) = 1e-12;
  return c;
}

// Runs `tadah analysis dataset_stats` on `db` at log level `level` with
// std::cout (the Logger's console) captured; returns the log.
std::string run_dataset_stats(const StructureDB &db, LogLevel level,
                              const DsCellTempDir &tmp,
                              const std::string &outfile) {
  const std::string dbfile = tmp.path("mixed.tadahdb");
  db.dump_to_file(dbfile, /*append=*/false, 18);
  Context c;
  c.add("DBFILE", dbfile);
  c.add("RCUT2B", 4.0);  // RCUTMAX follows
  c.add("OUTFILE", outfile);
  Task task("analysis", c);

  auto &log = Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(level);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  int rc = -1;
  try {
    rc = AnalysisDatasetStatsEngine().run(task);
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

}  // namespace

TEST_CASE("analysis dataset_stats lists exactly the structures whose cell is "
          "not symmetric",
          "[dataset_stats][cell_convention]") {
  // 1-based: 1 cubic (symmetric), 2 sheared, 3 fcc primitive (symmetric,
  // not diagonal: symmetry, not diagonality, decides), 4 hexagonal,
  // 5 cubic with a 1e-12 off-diagonal (the rule is > 0, no tolerance).
  PeriodicTable::initialize();
  DsCellTempDir tmp;
  StructureDB db;
  db.add(ta_pair("cubic-Ta", cubic()));
  db.add(ta_pair("sheared-Ta", sheared()));
  db.add(ta_pair("fccprim-Ta", fcc_primitive()));
  db.add(ta_pair("hex-Ta", hexagonal()));
  db.add(ta_pair("tinyasym-Ta", tiny_asymmetry()));

  const std::string out = tmp.path("stats.tadah");
  // WARNING is the CLI default verbosity: the report must show.
  const std::string log = run_dataset_stats(db, LogLevel::WARNING, tmp, out);
  INFO(log);

  CHECK(count_of(log, "3 of 5 structures have a non-symmetric cell") == 1);
  CHECK(count_of(log, "structure 2: sheared-Ta") == 1);
  CHECK(count_of(log, "structure 4: hex-Ta") == 1);
  CHECK(count_of(log, "structure 5: tinyasym-Ta") == 1);
  CHECK(log.find("cubic-Ta") == std::string::npos);
  CHECK(log.find("fccprim-Ta") == std::string::npos);
  CHECK(log.find("structure 1:") == std::string::npos);
  CHECK(log.find("structure 3:") == std::string::npos);
  CHECK(log.find("migration note") != std::string::npos);
  // Dataset order.
  CHECK(log.find("structure 2: sheared-Ta") <
        log.find("structure 4: hex-Ta"));
  CHECK(log.find("structure 4: hex-Ta") <
        log.find("structure 5: tinyasym-Ta"));

  // The statistics file is still written and still loads.
  std::ifstream in(out);
  REQUIRE(in.good());
  const auto stats = tadah::core::stats::load(in);
  CHECK(stats.n_structures == 5);
  CHECK(stats.n_atoms == 10);
}

TEST_CASE("analysis dataset_stats states that no cell is non-symmetric",
          "[dataset_stats][cell_convention]") {
  // Only symmetric cells: the count 0 is stated at the CLI default
  // verbosity, with no list and no pointer to the migration note.
  PeriodicTable::initialize();
  DsCellTempDir tmp;
  StructureDB db;
  db.add(ta_pair("cubic-Ta", cubic()));
  db.add(ta_pair("fccprim-Ta", fcc_primitive()));

  const std::string log =
      run_dataset_stats(db, LogLevel::WARNING, tmp, tmp.path("warn.tadah"));
  INFO(log);
  CHECK(count_of(log, "0 of 2 structures have a non-symmetric cell") == 1);
  CHECK(log.find("structure 1:") == std::string::npos);
  CHECK(log.find("structure 2:") == std::string::npos);
  CHECK(log.find("migration note") == std::string::npos);
}
