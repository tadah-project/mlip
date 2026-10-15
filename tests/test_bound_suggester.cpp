// Catch2 v2 unit tests for the bound_suggester.
//
// Verifies:
//   * suggest_bounds_for() produces one OptimSuggestion per ParamMeta and
//     iterates vector entries when the seed contains them
//   * scalar params (arg_pos == -1) emit one suggestion
//   * vector params emit one suggestion per seed entry, with arg_pos
//     advancing 0..n-1
//   * lo, hi are populated from the BoundSuggester closure
//   * write_optim_block collapses vector entries to a single (1-N) line
//   * write_optim_block emits scalar entries as (1)
//   * `tadah analysis suggest_bounds` refuses a dataset with two atoms at
//     one position, naming the structure (NNFinder's refusal)

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/stats/data_stats.h>
#include <tadah/mlip/bound_suggester.h>
#include <tadah/models/descriptors/param_meta.h>

#include <sstream>
#include <string>

using namespace tadah::mlip;
using tadah::core::Context;
using tadah::core::stats::DataStats;
using tadah::core::stats::PerPairStats;
using tadah::models::DescriptorMeta;
using tadah::models::ParamMeta;
using tadah::models::SeedView;
using tadah::models::Scale;
using tadah::models::Criticality;
using tadah::models::PairScope;
using tadah::models::HardLimit;
using tadah::models::HealthVerdict;

static DataStats make_stub_data() {
  DataStats d;
  d.n_atoms = 10;
  d.n_structures = 1;
  d.elements = {"Ta"};
  PerPairStats pp;
  pp.elements = {"Ta", "Ta"};
  pp.n_pairs = 20;
  pp.r_min = 2.5;
  pp.r_max = 6.0;
  for (std::size_t i = 0; i < pp.r_percentiles.size(); ++i) {
    pp.r_percentiles[i] = 2.5 + 0.4 * static_cast<double>(i);
  }
  d.per_pair.push_back(pp);
  d.class_label = "metal_like";
  return d;
}

static DescriptorMeta make_test_meta() {
  DescriptorMeta dm;
  dm.label = "TEST";

  // Use real OPTIM keys; Context::add() rejects arbitrary names.
  ParamMeta scalar;
  scalar.key      = "RCUT2B";
  scalar.arg_pos  = -1;
  scalar.scale    = Scale::Distance;
  scalar.units    = "Å";
  scalar.role     = "scalar test param";
  scalar.crit     = Criticality::Critical;
  scalar.scope    = PairScope::Global;
  scalar.suggest  = [](const DataStats& d, const SeedView&, int) {
    auto& pp = d.per_pair[0];
    return std::make_pair(pp.r_min, pp.r_max);
  };
  dm.params.push_back(scalar);

  ParamMeta vec;
  vec.key      = "CGRID2B";
  vec.arg_pos  = 0;
  vec.scale    = Scale::Distance;
  vec.units    = "Å";
  vec.role     = "vector test param";
  vec.crit     = Criticality::Recommended;
  vec.scope    = PairScope::Global;
  vec.suggest  = [](const DataStats& d, const SeedView&, int /*ap*/) {
    auto& pp = d.per_pair[0];
    return std::make_pair(0.0, pp.r_max);
  };
  dm.params.push_back(vec);

  return dm;
}

TEST_CASE("bound_suggester: scalar param emits one suggestion", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  seed.add<double>("CGRID2B", 1.0);
  seed.add<double>("CGRID2B", 2.0);
  seed.add<double>("CGRID2B", 3.0);
  auto out = suggest_bounds_for(data, meta, seed);
  // 1 scalar + 3 vector entries
  REQUIRE(out.size() == 4);
  REQUIRE(out[0].key == "RCUT2B");
  REQUIRE(out[0].arg_pos == -1);
  REQUIRE(out[0].lo == Approx(2.5));
  REQUIRE(out[0].hi == Approx(6.0));
  REQUIRE(out[0].descriptor_label == "TEST");
}

TEST_CASE("bound_suggester: vector param iterates seed entries", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  for (int i = 0; i < 5; ++i) seed.add<double>("CGRID2B", static_cast<double>(i));
  auto out = suggest_bounds_for(data, meta, seed);
  // 1 scalar + 5 vector entries
  REQUIRE(out.size() == 6);
  for (int i = 0; i < 5; ++i) {
    auto& s = out[1 + static_cast<std::size_t>(i)];
    REQUIRE(s.key == "CGRID2B");
    REQUIRE(s.arg_pos == i);
    REQUIRE(s.lo == Approx(0.0));
    REQUIRE(s.hi == Approx(6.0));
  }
}

TEST_CASE("bound_suggester: vector param with empty seed emits one entry", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  // No CGRID2B entry — suggester should emit a single representative bound.
  auto out = suggest_bounds_for(data, meta, seed);
  REQUIRE(out.size() == 2);
  REQUIRE(out[1].key == "CGRID2B");
  REQUIRE(out[1].arg_pos == 0);
}

TEST_CASE("write_optim_block: scalar -> (1) ; vector -> (1-N)", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  for (int i = 0; i < 4; ++i) seed.add<double>("CGRID2B", static_cast<double>(i));
  auto out = suggest_bounds_for(data, meta, seed);
  std::ostringstream os;
  write_optim_block(out, os);
  const std::string txt = os.str();
  // Scalar RCUT2B should be "(1)"
  REQUIRE(txt.find("OPTIM RCUT2B (1) ") != std::string::npos);
  // Vector CGRID2B with 4 entries should collapse to "(1-4)"
  REQUIRE(txt.find("OPTIM CGRID2B (1-4) ") != std::string::npos);
}

TEST_CASE("print_suggestions: writes a header and one row per suggestion", "[bound_suggester]") {
  auto data = make_stub_data();
  auto meta = make_test_meta();
  Context seed;
  seed.add<double>("CGRID2B", 0.0);
  seed.add<double>("CGRID2B", 1.0);
  auto out = suggest_bounds_for(data, meta, seed);
  std::ostringstream os;
  print_suggestions(out, os);
  const std::string txt = os.str();
  REQUIRE(txt.find("# Suggested OPTIM bounds") != std::string::npos);
  REQUIRE(txt.find("TEST.RCUT2B") != std::string::npos);
  REQUIRE(txt.find("TEST.CGRID2B[0]") != std::string::npos);
  REQUIRE(txt.find("TEST.CGRID2B[1]") != std::string::npos);
  REQUIRE(txt.find("CRITICAL") != std::string::npos);
  REQUIRE(txt.find("recommended") != std::string::npos);
}

// ---------------------------------------------------------------------------
//  tadah analysis suggest_bounds on a dataset with two atoms at one position
// ---------------------------------------------------------------------------

#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/analysis_suggest_bounds_engine.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task.h>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace {

namespace sb_fs = std::filesystem;

struct SbTempDir {
  sb_fs::path dir;
  SbTempDir() {
    static int counter = 0;
    dir = sb_fs::temp_directory_path() /
          ("tadah_sbcoinc_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    sb_fs::create_directories(dir);
  }
  ~SbTempDir() {
    std::error_code ec;
    sb_fs::remove_all(dir, ec);
  }
  std::string path(const std::string &name) const {
    return (dir / name).string();
  }
};

/* bcc Ta (a = 3.3 A) labelled `label`; with `coincident` a third Ta on top
 * of atom 1, at (0, 0, 0). */
Structure sb_bcc_ta(const std::string &label, bool coincident) {
  tadah::core::PeriodicTable::initialize();
  const auto ta = tadah::core::PeriodicTable::find_by_symbol("Ta");
  Structure st;
  st.label = label;
  st.energy = -20.0;
  st.cell.load(3.3, 0.0, 0.0, 0.0, 3.3, 0.0, 0.0, 0.0, 3.3);
  st.add_atom(Atom(ta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  st.add_atom(Atom(ta, 1.65, 1.65, 1.65, 0.0, 0.0, 0.0));
  if (coincident) st.add_atom(Atom(ta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  return st;
}

/* Runs `tadah analysis suggest_bounds` (validate, then run, as the scheduler
 * does) on `db`, written to a dataset file, with RCUT2B 4 (no descriptor:
 * the dataset is scanned before any descriptor is looked at) and OUTFILE
 * `outfile`, at the CLI default verbosity (WARNING), with std::cout (the
 * suggestion table, and the Logger's console) captured. Returns the exit
 * code; what the engine throws reaches the caller, with std::cout and the
 * log level restored. */
int run_suggest_bounds(const StructureDB &db, const SbTempDir &tmp,
                       const std::string &outfile) {
  const std::string dbfile = tmp.path("data.tadahdb");
  db.dump_to_file(dbfile, /*append=*/false, 18);
  Context c;
  c.add("DBFILE", dbfile);
  c.add("RCUT2B", 4.0);  // RCUTMAX follows
  c.add("OUTFILE", outfile);
  Task task("analysis", c);

  auto &log = tadah::core::Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(tadah::core::LogLevel::WARNING);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  int rc = -1;
  try {
    engines::AnalysisSuggestBoundsEngine engine;
    engine.validate(task);
    rc = engine.run(task);
  } catch (...) {
    std::cout.rdbuf(old_buf);
    log.setLogLevel(old_level);
    throw;
  }
  std::cout.rdbuf(old_buf);
  log.setLogLevel(old_level);
  return rc;
}

/* The type and message of what @p f throws: {"", ""} when nothing is
 * thrown. */
template <class F>
std::pair<std::string, std::string> sb_thrown(F f) {
  try {
    f();
  } catch (const std::invalid_argument &e) {
    return {"invalid_argument", e.what()};
  } catch (const std::exception &e) {
    return {"other exception", e.what()};
  }
  return {"", ""};
}

}  // namespace

/* Pins: `tadah analysis suggest_bounds` refuses a dataset in which two atoms
 * are at the same position, as training, prediction and HPO do (NNFinder's
 * refusal), instead of suggesting bounds from statistics that count the
 * pair at distance 0: of a bcc Ta cell and 'coincident-Ta' (the same cell
 * with a third Ta on atom 1) it throws std::invalid_argument naming
 * "structure 2 of 2 ('coincident-Ta')", "atoms 1 and 3 (Ta, Ta)" and "same
 * position", and writes no OUTFILE. The bcc cell alone is accepted (exit
 * code 0, OUTFILE written), so the refusal is the dataset's. Only `tadah
 * analysis dataset_stats`, which audits the data, lists such a pair.
 * Before the fix: compute_data_stats builds its lists with
 * NNFinder::list_coincident(true), so it counts the pair at distance 0
 * (per_pair r_min 0) and throws nothing; the engine prints the suggestion
 * table and writes the OPTIM block to OUTFILE (headers only, no descriptor
 * being configured) and returns EXIT_SUCCESS, so the type CHECK fails
 * (nothing thrown, empty message), every message CHECK fails and OUTFILE
 * exists. */
TEST_CASE("analysis suggest_bounds refuses a dataset with two atoms at the "
          "same position, naming the structure",
          "[bound_suggester][suggest_bounds][coincident]") {
  tadah::core::PeriodicTable::initialize();
  {
    SbTempDir tmp;
    StructureDB clean;
    clean.add(sb_bcc_ta("bcc-Ta", false));
    const std::string out = tmp.path("clean.hpo");
    int rc = -1;
    REQUIRE_NOTHROW(rc = run_suggest_bounds(clean, tmp, out));
    CHECK(rc == EXIT_SUCCESS);
    CHECK(sb_fs::exists(out));
  }

  SbTempDir tmp;
  StructureDB db;
  db.add(sb_bcc_ta("bcc-Ta", false));
  db.add(sb_bcc_ta("coincident-Ta", true));
  const std::string out = tmp.path("suggested.hpo");
  const auto t = sb_thrown([&] { (void)run_suggest_bounds(db, tmp, out); });
  INFO(t.first << ": " << t.second);
  CHECK(t.first == "invalid_argument");
  CHECK(t.second.find("structure 2 of 2 ('coincident-Ta')") !=
        std::string::npos);
  CHECK(t.second.find("atoms 1 and 3 (Ta, Ta)") != std::string::npos);
  CHECK(t.second.find("same position") != std::string::npos);
  CHECK_FALSE(sb_fs::exists(out));
}
