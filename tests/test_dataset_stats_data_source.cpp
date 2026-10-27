// test_dataset_stats_data_source.cpp
//
// Where `tadah analysis dataset_stats` takes its data and its cutoff from.
// The cutoff of the statistics is RCUTMAX, the largest cutoff of the
// configuration (-c): its RCUT2B and RCUTMB values and the cutoff RADIUS of
// each DESCRIPTOR block, as tadah train reads them. The data are the DBFILE
// and STRUCTURE lines of the configuration, or the datasets of --dbfile or
// the structures of --structure, which take the place of those lines: when
// the command line gives either, the configuration's DBFILE and STRUCTURE
// lines are not read (the configuration then gives the cutoff alone). The
// tasks are made by TaskFactory, as the command line makes them, so that
// each knows where its keys came from (Task::getKeyProvenance()).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/core/stats/data_stats_io.h>
#include <tadah/mlip/engines/analysis_dataset_stats_engine.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::core::stats::DataStats;
using tadah::mlip::Task;
using tadah::mlip::TaskFactory;
using tadah::mlip::engines::AnalysisDatasetStatsEngine;
using KV = std::unordered_map<std::string, std::vector<std::string>>;
namespace fs = std::filesystem;

struct DsSourceTempDir {
  fs::path dir;
  DsSourceTempDir() {
    static int counter = 0;
    dir = fs::temp_directory_path() /
          ("tadah_dssource_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
  }
  ~DsSourceTempDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
  std::string write(const std::string &name, const std::string &text) const {
    std::ofstream(path(name)) << text;
    return path(name);
  }
};

/* What TaskFactory gets from the command line: the blueprint task name and
 * the options, keyed by the blueprint's lower-case key names. */
struct OptionsInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  OptionsInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* TaskFactory stores the blueprint default VERBOSE 1 into the task's
 * context, which sets the global log level: put the level back. */
struct LogLevelGuard {
  tadah::core::LogLevel saved = tadah::core::Logger::instance().getLogLevel();
  ~LogLevelGuard() { tadah::core::Logger::instance().setLogLevel(saved); }
};

/* bcc-Ta: Ta at (0, 0, 0) and (2.5, 2.5, 2.5) in a 5 A cube, a dataset of
 * one structure. */
const char *const kTaDataset =
    "bcc-Ta\n1 1 1\n-1.0 0.0\n5 0 0\n0 5 0\n0 0 5\n0 0 0\n0 0 0\n0 0 0\n"
    "Ta 0 0 0 0 0 0\nTa 2.5 2.5 2.5 0 0 0\n\n";

/* A DM_Embed DESCRIPTOR block of the Ta-Ta pair whose density has the
 * cutoff RADIUS 6 A (the block of the Modular scalar embedding descriptor
 * page). */
const char *const kEmbedBlockRadius6 =
    "DESCRIPTOR ta DM_Embed\n"
    "  PAIRS Ta Ta\n"
    "  DENSITY\n"
    "    BASIS Blip5\n"
    "    FORM NONNEG\n"
    "    CENTRES 1.0 2.5 4.0\n"
    "    ETA 1.5 1.5 1.5\n"
    "    AMPLITUDES 0.4 0.2 0.1\n"
    "    CUTOFF\n"
    "      TYPE Cut_Poly\n"
    "      RADIUS 6.0\n"
    "    ENDCUTOFF\n"
    "  ENDDENSITY\n"
    "  EMBEDDING\n"
    "    BASIS Gaussian\n"
    "    CENTRES 0.0 0.5 1.0 1.5\n"
    "    WIDTHS 0.5 0.5 0.5 0.5\n"
    "    REFERENCE ZERO\n"
    "    ENVELOPE\n"
    "      TYPE Smoothstep2\n"
    "      CUTOFF 2.0\n"
    "    ENDENVELOPE\n"
    "  ENDEMBEDDING\n"
    "ENDDESCRIPTOR\n";

/* NaCl in a 4 x 5 x 6 A cell: Na at (0, 0, 0), Cl at (2, 2.5, 3) A. */
const char *const kNaClPoscar =
    "NaCl\n1.0\n4.0 0.0 0.0\n0.0 5.0 0.0\n0.0 0.0 6.0\nNa Cl\n1 1\nDirect\n"
    "0.0 0.0 0.0\n0.5 0.5 0.5\n";

/* Validates and runs the dataset_stats task (as the Scheduler does) with
 * std::cout captured; returns the statistics it wrote to `outfile`. */
DataStats run_task(Task &task, const std::string &outfile) {
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  int rc = -1;
  try {
    AnalysisDatasetStatsEngine engine;
    engine.validate(task);
    rc = engine.run(task);
  } catch (...) {
    std::cout.rdbuf(old_buf);
    throw;
  }
  std::cout.rdbuf(old_buf);
  REQUIRE(rc == EXIT_SUCCESS);
  std::ifstream in(outfile);
  REQUIRE(in.good());
  return tadah::core::stats::load(in);
}

/* The statistics of the element pair (a, b), a <= b; nullptr if none. */
const tadah::core::stats::PerPairStats *pair_of(const DataStats &s,
                                               const std::string &a,
                                               const std::string &b) {
  for (const auto &p : s.per_pair)
    if (p.elements.first == a && p.elements.second == b) return &p;
  return nullptr;
}

}  // namespace

/* Pins: with -c and --structure (CONFIG and STRUCTURE options, as TaskFactory
 * gets them from the command line) the statistics are those of the
 * structure of --structure alone, at the cutoff of the configuration: the
 * configuration's DBFILE line, which names a file that does not exist, is
 * not read. The NaCl cell at RCUT2B 4.5 gives elements Cl Na, one
 * structure of two atoms, and the Na-Na pair of the cell's 4 A images (2
 * ordered pairs at 4 A), which a cutoff below 4 A would not hold.
 * Before the fix: run() read the configuration's DBFILE as well and threw
 * "DBFILE does not exist: <dir>/absent.tadah". */
TEST_CASE("analysis dataset_stats with -c and --structure reads the "
          "structures in place of the DBFILE lines of the configuration",
          "[dataset_stats][data_source]") {
  PeriodicTable::initialize();
  LogLevelGuard guard;
  DsSourceTempDir tmp;
  const std::string poscar = tmp.write("POSCAR_NaCl", kNaClPoscar);
  const std::string cfg = tmp.write(
      "stats.cfg", "RCUT2B 4.5\nDBFILE " + tmp.path("absent.tadah") + "\n");
  const std::string out = tmp.path("stats.tadah");

  auto tasks = TaskFactory().createTasksFromInput(OptionsInput(
      "ANALYSIS.dataset_stats",
      KV{{"config", {cfg}}, {"structure", {poscar}}, {"outfile", {out}}}));
  REQUIRE(tasks.size() == 1);
  const DataStats s = run_task(tasks[0], out);

  CHECK(s.n_structures == 1);
  CHECK(s.n_atoms == 2);
  CHECK(s.elements == std::vector<std::string>{"Cl", "Na"});
  const auto *nana = pair_of(s, "Na", "Na");
  REQUIRE(nana != nullptr);
  CHECK(nana->n_pairs == 2);
  CHECK(nana->r_max == Approx(4.0).margin(1e-12));
}

/* Pins: with -c and --dbfile the statistics are those of the datasets of
 * --dbfile alone, at the cutoff of the configuration: the configuration's
 * DBFILE line is replaced by the option's, and its STRUCTURE line, which
 * names a file that does not exist, is not read. bcc-Ta at RCUT2B 5.5 gives
 * the 8 neighbours at 4.33 A and the 6 images at 5 A of each atom: 28
 * ordered Ta-Ta pairs, r_max 5 (at a cutoff of 4.5 A: 16 pairs).
 * Before the fix: run() read the configuration's STRUCTURE as well and threw
 * "the structure file <dir>/POSCAR_absent does not exist". */
TEST_CASE("analysis dataset_stats with -c and --dbfile reads the datasets "
          "in place of the DBFILE and STRUCTURE lines of the configuration",
          "[dataset_stats][data_source]") {
  PeriodicTable::initialize();
  LogLevelGuard guard;
  DsSourceTempDir tmp;
  const std::string ta = tmp.write("ta.tadah", kTaDataset);
  const std::string cfg = tmp.write(
      "stats.cfg", "RCUT2B 5.5\nDBFILE " + tmp.path("absent.tadah") +
                       "\nSTRUCTURE " + tmp.path("POSCAR_absent") + "\n");
  const std::string out = tmp.path("stats.tadah");

  auto tasks = TaskFactory().createTasksFromInput(OptionsInput(
      "ANALYSIS.dataset_stats",
      KV{{"config", {cfg}}, {"dbfile", {ta}}, {"outfile", {out}}}));
  REQUIRE(tasks.size() == 1);
  const DataStats s = run_task(tasks[0], out);

  CHECK(s.n_structures == 1);
  CHECK(s.n_atoms == 2);
  CHECK(s.elements == std::vector<std::string>{"Ta"});
  const auto *tata = pair_of(s, "Ta", "Ta");
  REQUIRE(tata != nullptr);
  CHECK(tata->n_pairs == 28);
  CHECK(tata->r_max == Approx(5.0).margin(1e-12));
}

/* Pins: a task without a record of where its keys came from (a Context a
 * program builds) reads every DBFILE and STRUCTURE value of its context:
 * bcc-Ta and the NaCl cell, two structures, elements Cl Na Ta.
 * Before the fix: passes as well (a pin of the engine's reading of a
 * context without KeyProvenance, which the fix leaves as it was). */
TEST_CASE("analysis dataset_stats without a key provenance reads every "
          "DBFILE and STRUCTURE value of its context",
          "[dataset_stats][data_source]") {
  PeriodicTable::initialize();
  DsSourceTempDir tmp;
  const std::string out = tmp.path("stats.tadah");
  Context c;
  c.add("DBFILE", tmp.write("ta.tadah", kTaDataset));
  c.add("STRUCTURE", tmp.write("POSCAR_NaCl", kNaClPoscar));
  c.add("RCUT2B", 4.5);
  c.add("OUTFILE", out);
  Task task("ANALYSIS.dataset_stats", c);
  REQUIRE(task.getKeyProvenance() == nullptr);

  const DataStats s = run_task(task, out);
  CHECK(s.n_structures == 2);
  CHECK(s.n_atoms == 4);
  CHECK(s.elements == std::vector<std::string>{"Cl", "Na", "Ta"});
}

/* Pins: a task whose configuration gives no cutoff (no RCUT2B or RCUTMB
 * line and no DESCRIPTOR block) is refused by validate(), before any data
 * are read: "analysis dataset_stats: no cutoff: the statistics take the
 * largest cutoff of a configuration given with -c (RCUT2B, RCUTMB or the
 * cutoff RADIUS of a DESCRIPTOR block), and no line gives one (--dbfile and
 * --structure give none)"; so is run() called without validate(), with
 * that message and not the missing DBFILE's.
 * Before the fix: validate() checked CLOSE_CONTACT alone and returned, and
 * run() read the data first and threw "DBFILE does not exist: <dir>/
 * absent.tadah".
 * Before the fix that takes the cutoff RADIUS of a DESCRIPTOR block: the
 * message named RCUT2B and RCUTMB alone, "... no cutoff: the statistics
 * take the largest RCUT2B or RCUTMB value of a configuration given with -c
 * as their cutoff, and no line gives one ...": both checks fail. */
TEST_CASE("analysis dataset_stats refuses a configuration without a cutoff "
          "before it reads the data",
          "[dataset_stats][data_source]") {
  PeriodicTable::initialize();
  LogLevelGuard guard;
  DsSourceTempDir tmp;
  const std::string cfg = tmp.write(
      "nocut.cfg", "DBFILE " + tmp.path("absent.tadah") + "\n");
  const std::string out = tmp.path("stats.tadah");
  auto tasks = TaskFactory().createTasksFromInput(OptionsInput(
      "ANALYSIS.dataset_stats", KV{{"config", {cfg}}, {"outfile", {out}}}));
  REQUIRE(tasks.size() == 1);

  const std::string message =
      "analysis dataset_stats: no cutoff: the statistics take the largest "
      "cutoff of a configuration given with -c (RCUT2B, RCUTMB or the cutoff "
      "RADIUS of a DESCRIPTOR block), and no line gives one (--dbfile and "
      "--structure give none)";
  AnalysisDatasetStatsEngine engine;
  CHECK_THROWS_WITH(engine.validate(tasks[0]), message);
  CHECK_THROWS_WITH(engine.run(tasks[0]), message);
  CHECK_FALSE(fs::exists(out));
}

/* Pins: the cutoff RADIUS of a DESCRIPTOR block is a cutoff of the
 * configuration, as tadah train reads it: with RCUT2B 4.5 and a DM_Embed
 * block of RADIUS 6, -c and --dbfile give the statistics of bcc-Ta at 6 A,
 * the 8 neighbours at 4.33 A and the 6 images at 5 A of each atom: 28
 * ordered Ta-Ta pairs, r_max 5. (At 4.5 A: 16 pairs, r_max 4.33.)
 * Before the fix: the cutoff was the largest RCUT2B or RCUTMB value alone,
 * 4.5 A: 16 pairs, r_max 4.33, exit status 0 and no message. */
TEST_CASE("analysis dataset_stats takes the cutoff RADIUS of a DESCRIPTOR "
          "block as a cutoff of the configuration",
          "[dataset_stats][data_source]") {
  PeriodicTable::initialize();
  LogLevelGuard guard;
  DsSourceTempDir tmp;
  const std::string ta = tmp.write("ta.tadah", kTaDataset);
  const std::string cfg = tmp.write(
      "embed.cfg", std::string("RCUT2B 4.5\n") + kEmbedBlockRadius6);
  const std::string out = tmp.path("stats.tadah");

  auto tasks = TaskFactory().createTasksFromInput(OptionsInput(
      "ANALYSIS.dataset_stats",
      KV{{"config", {cfg}}, {"dbfile", {ta}}, {"outfile", {out}}}));
  REQUIRE(tasks.size() == 1);
  const DataStats s = run_task(tasks[0], out);

  CHECK(s.n_structures == 1);
  const auto *tata = pair_of(s, "Ta", "Ta");
  REQUIRE(tata != nullptr);
  CHECK(tata->n_pairs == 28);
  CHECK(tata->r_max == Approx(5.0).margin(1e-12));
}

/* Pins: a configuration whose only cutoff is the RADIUS of a DESCRIPTOR
 * block (no RCUT2B or RCUTMB line) is accepted by validate() and run(): its
 * DBFILE line, bcc-Ta, gives the statistics at 6 A, 28 ordered Ta-Ta pairs,
 * r_max 5.
 * Before the fix: refused by validate(), "analysis dataset_stats: no
 * cutoff: ... and no line gives one ...", though a RADIUS line gives one. */
TEST_CASE("analysis dataset_stats accepts a configuration whose only cutoff "
          "is the RADIUS of a DESCRIPTOR block",
          "[dataset_stats][data_source]") {
  PeriodicTable::initialize();
  LogLevelGuard guard;
  DsSourceTempDir tmp;
  const std::string ta = tmp.write("ta.tadah", kTaDataset);
  const std::string cfg = tmp.write(
      "embed.cfg", "DBFILE " + ta + "\n" + kEmbedBlockRadius6);
  const std::string out = tmp.path("stats.tadah");

  auto tasks = TaskFactory().createTasksFromInput(OptionsInput(
      "ANALYSIS.dataset_stats", KV{{"config", {cfg}}, {"outfile", {out}}}));
  REQUIRE(tasks.size() == 1);
  const DataStats s = run_task(tasks[0], out);

  CHECK(s.n_structures == 1);
  const auto *tata = pair_of(s, "Ta", "Ta");
  REQUIRE(tata != nullptr);
  CHECK(tata->n_pairs == 28);
  CHECK(tata->r_max == Approx(5.0).margin(1e-12));
}

/* Pins: a task whose configuration gives a cutoff of 0 ("RCUT2B 0", or
 * "RCUTMB 0" without an RCUT2B line: RCUTMAX 0) is refused by validate(),
 * before any data are read: "analysis dataset_stats: the cutoff of the
 * statistics is the largest RCUT2B or RCUTMB value, 0, which is not above
 * 0"; so is run() called without validate(), with that message and not
 * that of the configuration's DBFILE, which names a file that does not
 * exist. No report is written.
 * Before the fix: validate() checked CLOSE_CONTACT alone and returned, and
 * run() read the data first and threw "DBFILE does not exist: <dir>/
 * absent.tadah", for either configuration. */
TEST_CASE("analysis dataset_stats refuses a cutoff of 0 before it reads "
          "the data",
          "[dataset_stats][data_source]") {
  PeriodicTable::initialize();
  LogLevelGuard guard;
  const std::string message =
      "analysis dataset_stats: the cutoff of the statistics is the largest "
      "RCUT2B or RCUTMB value, 0, which is not above 0";
  for (const std::string cutoff_line : {"RCUT2B 0", "RCUTMB 0"}) {
    INFO("configuration: " << cutoff_line);
    DsSourceTempDir tmp;
    const std::string cfg = tmp.write(
        "zero.cfg",
        cutoff_line + "\nDBFILE " + tmp.path("absent.tadah") + "\n");
    const std::string out = tmp.path("stats.tadah");
    auto tasks = TaskFactory().createTasksFromInput(OptionsInput(
        "ANALYSIS.dataset_stats", KV{{"config", {cfg}}, {"outfile", {out}}}));
    REQUIRE(tasks.size() == 1);

    AnalysisDatasetStatsEngine engine;
    CHECK_THROWS_WITH(engine.validate(tasks[0]), message);
    CHECK_THROWS_WITH(engine.run(tasks[0]), message);
    CHECK_FALSE(fs::exists(out));
  }
}
