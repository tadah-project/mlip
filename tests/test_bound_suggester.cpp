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

// ---------------------------------------------------------------------------
//  The suggestions read the seed: a Gaussian density centred at the origin
// ---------------------------------------------------------------------------

#include <tadah/core/registry.h>
#include <tadah/models/descriptors/d_base.h>

#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>

namespace {

/* Neighbour distances of bcc Ta: r_min 2.25 A, the first two shells at
 * 2.86 A (FWHM 0.7 A) and 3.30 A, r_max 5.6 A; no ρ statistics (as
 * `tadah analysis suggest_bounds` computes them). */
DataStats ta_shell_stats() {
  DataStats d;
  d.n_atoms = 100;
  d.n_structures = 10;
  d.elements = {"Ta"};
  PerPairStats pp;
  pp.elements = {"Ta", "Ta"};
  pp.n_pairs = 4000;
  pp.r_min = 2.25;
  pp.r_max = 5.6;
  pp.r_percentiles = {2.6, 2.8, 2.85, 2.9, 3.4, 4.7, 5.2, 5.4, 5.55,
                      4.0, 1.0};
  tadah::core::stats::KdePeak first, second;
  first.position = 2.86;
  first.density = 1.0;
  first.fwhm = 0.7;
  second.position = 3.30;
  second.density = 0.7;
  second.fwhm = 0.3;
  pp.r_peaks = {first, second};
  d.per_pair.push_back(pp);
  return d;
}

/* The Ta seed of the HPO tests, its many-body part: DM_mBlipEnv with the
 * radial GTO at `cgridmb`, SGRIDMB 0.2, RCUTMB 5.6. */
Context mblipenv_seed(double cgridmb) {
  Context c;
  c.add("TYPEMB", "DM_mBlipEnv 0 1 1 7 7 Ta Ta");
  c.add("RCUTENV", 5.0);
  c.add("RCUTMB", 5.6);
  c.add("SGRIDMB", 0.2);
  c.add("CGRIDMB", cgridmb);
  c.add("SEMBFUNC", "1 1 1 1 1 1 1");
  c.add("CEMBFUNC", "1 2 3 4 5 6 7");
  return c;
}

DescriptorMeta mblipenv_meta() {
  std::unique_ptr<tadah::models::D_Base> d(
      tadah::core::factory<tadah::models::D_Base>("DM_mBlipEnv"));
  auto meta = d->meta();
  REQUIRE(meta.has_value());
  return *meta;
}

const OptimSuggestion *row_of(const std::vector<OptimSuggestion> &rows,
                              const std::string &key) {
  for (const auto &s : rows)
    if (s.key == key) return &s;
  return nullptr;
}

}  // namespace

/* Pins: for a Gaussian density centred at the origin (DM_mBlipEnv, CGRIDMB
 * 0, SGRIDMB 0.2), suggest_bounds_for gives no CGRIDMB range (its width
 * sets its reach, not its centre): its CGRIDMB suggestion, when there is
 * one, has none and a rationale naming the Gaussian centred at the origin,
 * which the table prints as "no OPTIM line"; and the SGRIDMB range is the
 * width whose
 * exp(-SGRIDMB r^2) drops by a factor of 0.9 to 1/20 from the first
 * neighbour shell (2.86 A) to the second (3.30 A): ln(1/0.9) / (3.30^2 -
 * 2.86^2) = 0.0389 to ln(20) / (3.30^2 - 2.86^2) = 1.104, which holds the
 * seed's 0.2 and stays below ln(1e6) / r_min^2 = 2.73, where the Gaussian
 * is 1e-6 at r_min; its rationale names the Gaussian centred at the
 * origin. The OPTIM block has no CGRIDMB line.
 * Before the fix: the seed was not read: a CGRIDMB row 2.16 to 3.56 (the
 * first peak +- its FWHM), and SGRIDMB 0.51 to 18.4 (0.25/w^2 to 9/w^2 of
 * the shell-centred form), at whose upper end exp(-18.4 * 2.25^2) is 1e-41
 * at r_min. */
TEST_CASE("bound_suggester: a Gaussian density centred at the origin gets "
          "no CGRIDMB row and a width from the drop between the first two "
          "shells",
          "[bound_suggester][suggest_bounds][seed]") {
  const DataStats data = ta_shell_stats();
  const auto rows = suggest_bounds_for(data, mblipenv_meta(),
                                       mblipenv_seed(0.0));
  if (const OptimSuggestion *centre = row_of(rows, "CGRIDMB")) {
    CHECK(centre->lo == 0.0);
    CHECK(centre->hi == 0.0);
    CHECK(centre->rationale.find("centred at the origin") !=
          std::string::npos);
  }
  {
    std::ostringstream table;
    print_suggestions(rows, table);
    INFO(table.str());
    const auto at = table.str().find("DM_mBlipEnv.CGRIDMB");
    REQUIRE(at != std::string::npos);
    const std::string row =
        table.str().substr(at, table.str().find('\n', at) - at);
    CHECK(row.find("no OPTIM line") != std::string::npos);
  }
  const OptimSuggestion *width = row_of(rows, "SGRIDMB");
  REQUIRE(width != nullptr);
  const double gap = 3.30 * 3.30 - 2.86 * 2.86;
  CHECK(width->lo == Approx(std::log(1.0 / 0.9) / gap).epsilon(1e-12));
  CHECK(width->hi == Approx(std::log(20.0) / gap).epsilon(1e-12));
  CHECK(width->lo < 0.2);
  CHECK(width->hi > 0.2);
  CHECK(width->hi < std::log(1e6) / (2.25 * 2.25));
  CHECK(width->rationale.find("Gaussian") != std::string::npos);
  CHECK(width->rationale.find("centred at the origin") != std::string::npos);
  std::ostringstream block;
  write_optim_block(rows, block);
  INFO(block.str());
  CHECK(block.str().find("OPTIM CGRIDMB") == std::string::npos);
  CHECK(block.str().find("OPTIM SGRIDMB (1) ") != std::string::npos);
}

/* Pins: for a radial GTO centred on the first neighbour shell (CGRIDMB
 * 2.9), the CGRIDMB row is the first peak +- its FWHM with its lower end
 * at r_min: 2.25 (not 2.86 - 0.7 = 2.16) to 3.56, and its rationale and
 * that of SGRIDMB name the Gaussian centred on the first shell.
 * Before the fix: the CGRIDMB row began at 2.16, below every neighbour
 * distance, and the rationales gave only the scale and the role of the
 * key. */
TEST_CASE("bound_suggester: a shell-centred Gaussian density gets a CGRIDMB "
          "row from r_min",
          "[bound_suggester][suggest_bounds][seed]") {
  const auto rows = suggest_bounds_for(ta_shell_stats(), mblipenv_meta(),
                                       mblipenv_seed(2.9));
  const OptimSuggestion *centre = row_of(rows, "CGRIDMB");
  REQUIRE(centre != nullptr);
  CHECK(centre->lo == Approx(2.25).epsilon(1e-12));
  CHECK(centre->hi == Approx(2.86 + 0.7).epsilon(1e-12));
  CHECK(centre->rationale.find("Gaussian") != std::string::npos);
  CHECK(centre->rationale.find("first neighbour shell") != std::string::npos);
  const OptimSuggestion *width = row_of(rows, "SGRIDMB");
  REQUIRE(width != nullptr);
  CHECK(width->rationale.find("first neighbour shell") != std::string::npos);
}

/* Pins: without ρ statistics (`tadah analysis suggest_bounds` computes
 * none) the ranges of RCUTENV, CEMBFUNC and SEMBFUNC are fixed ones, not
 * derived from the data: each rationale says "fixed range", the table's
 * heading does not call every range data-driven, and the header of the
 * OPTIM block names the command that writes it, tadah analysis
 * suggest_bounds.
 * Before the fix: the rationales gave only the scale and the role, the
 * table was headed "(data-driven; review before use)", and the block said
 * "Auto-generated by `tadah hpo --suggest-bounds`", an option no command
 * has. */
TEST_CASE("bound_suggester: a fixed range is named as such, and the block "
          "names tadah analysis suggest_bounds",
          "[bound_suggester][suggest_bounds]") {
  const auto rows = suggest_bounds_for(ta_shell_stats(), mblipenv_meta(),
                                       mblipenv_seed(0.0));
  for (const char *key : {"RCUTENV", "CEMBFUNC", "SEMBFUNC"}) {
    INFO(key);
    const OptimSuggestion *r = row_of(rows, key);
    REQUIRE(r != nullptr);
    CHECK(r->rationale.find("fixed range") != std::string::npos);
  }
  const OptimSuggestion *rcut = row_of(rows, "RCUTMB");
  REQUIRE(rcut != nullptr);
  CHECK(rcut->rationale.find("fixed range") == std::string::npos);
  std::ostringstream table, block;
  print_suggestions(rows, table);
  write_optim_block(rows, block);
  INFO(table.str() << "\n" << block.str());
  CHECK(table.str().find("data-driven") == std::string::npos);
  CHECK(block.str().find("tadah analysis suggest_bounds") !=
        std::string::npos);
  CHECK(block.str().find("--suggest-bounds") == std::string::npos);
}

/* Pins: print_suggestions and write_optim_block leave the format of the
 * stream they write to as they found it (flags, precision, fill): written
 * to std::cout, as suggest_bounds does without --outfile, they change
 * nothing a later writer of std::cout (a kernel's settings line copies
 * its format) sees.
 * Before the fix: both left std::cout std::scientific, at precision 4
 * (print_suggestions) and 6 (write_optim_block). */
TEST_CASE("bound_suggester: the writers leave the format of std::cout as "
          "they found it",
          "[bound_suggester][suggest_bounds][stream_state]") {
  const auto rows = suggest_bounds_for(ta_shell_stats(), mblipenv_meta(),
                                       mblipenv_seed(0.0));
  std::ostringstream sink;
  std::streambuf *old = std::cout.rdbuf(sink.rdbuf());
  const auto flags = std::cout.flags();
  const auto precision = std::cout.precision();
  const auto fill = std::cout.fill();
  print_suggestions(rows, std::cout);
  const auto flags_print = std::cout.flags();
  const auto precision_print = std::cout.precision();
  write_optim_block(rows, std::cout);
  const auto flags_block = std::cout.flags();
  const auto precision_block = std::cout.precision();
  const auto fill_block = std::cout.fill();
  std::cout.flags(flags);
  std::cout.precision(precision);
  std::cout.fill(fill);
  std::cout.rdbuf(old);
  CHECK(flags_print == flags);
  CHECK(precision_print == precision);
  CHECK(flags_block == flags);
  CHECK(precision_block == precision);
  CHECK(fill_block == fill);
}

namespace {

/* Two bcc Ta cells (a = 3.3 A) in a dataset file of @p tmp. */
std::string bcc_ta_dataset(const SbTempDir &tmp) {
  StructureDB db;
  db.add(sb_bcc_ta("bcc-Ta-1", false));
  db.add(sb_bcc_ta("bcc-Ta-2", false));
  const std::string dbfile = tmp.path("bcc.tadahdb");
  db.dump_to_file(dbfile, /*append=*/false, 18);
  return dbfile;
}

}  // namespace

/* Pins: `tadah analysis suggest_bounds` reads a radial grid given by a
 * generator as the grid it makes: for TYPE2B D2_Blip 7 7 Ta Ta with
 * "CGRID2B LIN 7 0.1 3.0" and "SGRID2B LIN 7 1 1.2" (the Ta seed of the
 * HPO tests) the OPTIM block has "OPTIM CGRID2B (1-7)" and "OPTIM SGRID2B
 * (1-7)", one row per blip, and the run logs no ERROR.
 * Before the fix: the seed's tokens were read as numbers ("ERROR:
 * Context::convertStringTo -> Failed converting "LIN" for key=CGRID2B"),
 * the run went on and returned 0, and the block had "OPTIM CGRID2B (1-4)"
 * and "OPTIM SGRID2B (1-4)", one row per token of the line. */
TEST_CASE("analysis suggest_bounds reads a grid generator as the grid it "
          "makes",
          "[bound_suggester][suggest_bounds][seed]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip 7 7 Ta Ta");
  c.add("RCTYPE2B", "Cut_Poly");
  c.add("SGRID2B", "LIN 7 1 1.2");
  c.add("CGRID2B", "LIN 7 0.1 3.0");
  c.add("RCUT2B", 5.6);
  const std::string out = tmp.path("suggested.hpo");
  c.add("OUTFILE", out);
  Task task("analysis", c);

  auto &log = tadah::core::Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(tadah::core::LogLevel::WARNING);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  int rc = -1;
  std::string what;
  try {
    engines::AnalysisSuggestBoundsEngine engine;
    engine.validate(task);
    rc = engine.run(task);
  } catch (const std::exception &e) {
    what = e.what();
  }
  std::cout.rdbuf(old_buf);
  log.setLogLevel(old_level);
  std::ifstream in(out);
  const std::string block((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  INFO(what << "\n" << captured.str() << "\n" << block);
  CHECK(rc == EXIT_SUCCESS);
  CHECK(captured.str().find("ERROR") == std::string::npos);
  CHECK(block.find("OPTIM CGRID2B (1-7) ") != std::string::npos);
  CHECK(block.find("OPTIM SGRID2B (1-7) ") != std::string::npos);
}

/* Pins: the suggestions of a density read the seed entry by entry, and the
 * OPTIM block gives one line per run of entries with the same range: for a
 * DM_EAD seed with two Gaussian channels at the origin and two on the first
 * shell (CGRIDMB 0 0 2.9 2.9, SGRIDMB 0.2 0.2 1 1), the block has "OPTIM
 * CGRIDMB (3-4)", "OPTIM SGRIDMB (1-2)" and "OPTIM SGRIDMB (3-4)", and no
 * line of entries 1 to 4.
 * Before the fix: every entry got the shell-centred ranges, merged into
 * "OPTIM CGRIDMB (1-4)" and "OPTIM SGRIDMB (1-4)" (a block line took the
 * smallest low and the largest high end of all the entries of a key). */
TEST_CASE("write_optim_block gives one line per run of entries with the same "
          "range",
          "[bound_suggester][suggest_bounds][seed]") {
  std::unique_ptr<tadah::models::D_Base> d(
      tadah::core::factory<tadah::models::D_Base>("DM_EAD"));
  const auto meta = d->meta();
  REQUIRE(meta.has_value());
  Context seed;
  seed.add("CGRIDMB", "0 0 2.9 2.9");
  seed.add("SGRIDMB", "0.2 0.2 1 1");
  seed.add("RCUTMB", 5.6);
  const auto rows = suggest_bounds_for(ta_shell_stats(), *meta, seed);
  std::ostringstream block;
  write_optim_block(rows, block);
  INFO(block.str());
  CHECK(block.str().find("OPTIM CGRIDMB (3-4) ") != std::string::npos);
  CHECK(block.str().find("OPTIM SGRIDMB (1-2) ") != std::string::npos);
  CHECK(block.str().find("OPTIM SGRIDMB (3-4) ") != std::string::npos);
  CHECK(block.str().find("(1-4)") == std::string::npos);
}

/* Pins: a parameter at a fixed position of a key's values (ParamMeta
 * arg_pos > 0: the runtime position, the value at index arg_pos - 1, as
 * the screening parameters of D2_ZBL are values 2 to 5 of "TYPE2B D2_ZBL
 * a0 s0 p0 p1 Ta Ta") gets one suggestion, with the seed's value there,
 * and its OPTIM line counts the numeric values of the key as tadah hpo
 * does: for TYPE2B D2_ZBL 0.5292 0.88534 0.23 0.23 Ta Ta the four TYPE2B
 * rows have the seed values 0.5292, 0.88534, 0.23 and 0.23 and the block
 * has "OPTIM TYPE2B (1)" (0.265 to 0.795, a fixed range), "OPTIM TYPE2B
 * (2)" and "OPTIM TYPE2B (3-4)" (p0 and p1 share their range).
 * Before the fix: each such parameter was taken for a vector of the key's
 * every value: 7 rows each, the seed value read from "D2_ZBL" and the
 * element names (ERROR lines, values left out; with the seed read as
 * numbers, a throw), and one line "OPTIM TYPE2B (1-7)" spanning their
 * ranges. */
TEST_CASE("bound_suggester: a parameter at a fixed position of a key gets "
          "one row at its OPTIM index",
          "[bound_suggester][suggest_bounds][seed]") {
  std::unique_ptr<tadah::models::D_Base> d(
      tadah::core::factory<tadah::models::D_Base>("D2_ZBL"));
  const auto meta = d->meta();
  REQUIRE(meta.has_value());
  Context seed;
  seed.add("TYPE2B", "D2_ZBL 0.5292 0.88534 0.23 0.23 Ta Ta");
  seed.add("RCUT2B", 5.0);
  std::vector<OptimSuggestion> rows;
  std::string what;
  try {
    rows = suggest_bounds_for(ta_shell_stats(), *meta, seed);
  } catch (const std::exception &e) {
    what = e.what();
  }
  INFO(what);
  std::vector<double> seeds;
  for (const auto &r : rows)
    if (r.key == "TYPE2B") {
      REQUIRE(r.seed_value.has_value());
      seeds.push_back(*r.seed_value);
    }
  CHECK(seeds == std::vector<double>{0.5292, 0.88534, 0.23, 0.23});
  std::ostringstream block;
  write_optim_block(rows, block);
  INFO(block.str());
  CHECK(block.str().find("OPTIM TYPE2B (1) 2.650000e-01 7.950000e-01") !=
        std::string::npos);
  for (const char *line : {"OPTIM TYPE2B (2) ", "OPTIM TYPE2B (3-4) "})
    CHECK(block.str().find(line) != std::string::npos);
  CHECK(block.str().find("OPTIM TYPE2B (1-") == std::string::npos);
}
