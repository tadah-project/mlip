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

#include <algorithm>
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

/* Pins: the table has a header and one row per suggestion, each row
 * named by the descriptor, the key and the entry as an OPTIM line numbers
 * it, from 1 ("TEST.RCUT2B(1)", "TEST.CGRID2B(1)", "TEST.CGRID2B(2)"), as
 * the header says ("descriptor.key(i)").
 * Before the fix: the entries were numbered from 0 ("TEST.CGRID2B[0]",
 * "[1]"; the header "descriptor.key[arg_pos]") while the OPTIM block wrote
 * "OPTIM CGRID2B (1-2)". */
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
  INFO(txt);
  REQUIRE(txt.find("# Suggested OPTIM bounds") != std::string::npos);
  CHECK(txt.find("#   col 1: descriptor.key(i), i the entry as an OPTIM "
                 "line numbers it\n") != std::string::npos);
  CHECK(txt.find("TEST.RCUT2B(1) ") != std::string::npos);
  CHECK(txt.find("TEST.CGRID2B(1) ") != std::string::npos);
  CHECK(txt.find("TEST.CGRID2B(2) ") != std::string::npos);
  CHECK(txt.find("[0]") == std::string::npos);
  CHECK(txt.find("arg_pos") == std::string::npos);
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
 * 2.9), the CGRIDMB row is the seed's centre +- the FWHM of the shell
 * nearest to it (2.86 A, FWHM 0.7) with its lower end at r_min: 2.25 (not
 * 2.9 - 0.7 = 2.2) to 3.6, and its rationale and that of SGRIDMB name the
 * seed's centre and the nearest shell.
 * Before the fix: the row was the first peak +- its FWHM, 2.25 to 3.56,
 * whatever the seed's centre, its rationale naming the first neighbour
 * shell (and, before that, the row began at 2.16, below every neighbour
 * distance, and the rationales gave only the scale and the role of the
 * key). */
TEST_CASE("bound_suggester: a shell-centred Gaussian density gets a CGRIDMB "
          "row from r_min",
          "[bound_suggester][suggest_bounds][seed]") {
  const auto rows = suggest_bounds_for(ta_shell_stats(), mblipenv_meta(),
                                       mblipenv_seed(2.9));
  const OptimSuggestion *centre = row_of(rows, "CGRIDMB");
  REQUIRE(centre != nullptr);
  INFO(centre->rationale);
  CHECK(centre->lo == Approx(2.25).epsilon(1e-12));
  CHECK(centre->hi == Approx(2.9 + 0.7).epsilon(1e-12));
  CHECK(centre->rationale.find("Gaussian") != std::string::npos);
  CHECK(centre->rationale.find("seed's centre") != std::string::npos);
  CHECK(centre->rationale.find("nearest") != std::string::npos);
  const OptimSuggestion *width = row_of(rows, "SGRIDMB");
  REQUIRE(width != nullptr);
  CHECK(width->rationale.find("nearest") != std::string::npos);
}

/* Pins: without ρ statistics (`tadah analysis suggest_bounds` computes
 * none) the ranges of RCUTENV, CEMBFUNC and SEMBFUNC are fixed ones, not
 * derived from the data: each rationale says "fixed range", the table's
 * heading does not call every range data-driven, and the header of the
 * OPTIM block names the command that writes it, tadah analysis
 * suggest_bounds. The heading and the header say, impersonally, that the
 * ranges are suggestions, to be checked before use ("# Suggested OPTIM
 * bounds (each range is a suggestion, to be checked before use)"; "...
 * The ranges are suggestions, to be checked before tadah hpo runs.").
 * Before the fix: the rationales gave only the scale and the role, the
 * table was headed "(data-driven; review before use)", and the block said
 * "Auto-generated by `tadah hpo --suggest-bounds`", an option no command
 * has; then the heading said "(review before use)" and the block "Review
 * and edit before running tadah hpo.", instructions to the reader. */
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
  CHECK(table.str().find("# Suggested OPTIM bounds (each range is a "
                         "suggestion, to be checked before use)\n") == 0);
  CHECK(block.str().find("# Written by `tadah analysis suggest_bounds`, one "
                         "rationale\n# comment per line. The ranges are "
                         "suggestions, to be checked\n# before tadah hpo "
                         "runs.\n\n") == 0);
  for (const std::string &text : {table.str(), block.str()}) {
    CHECK(text.find("review") == std::string::npos);
    CHECK(text.find("Review") == std::string::npos);
  }
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

/* Pins: a parameter that is a value of a descriptor's TYPE2B line
 * (ParamMeta arg_pos > 0: the value at index arg_pos - 1 of the line, as
 * the screening parameters of D2_ZBL are values 2 to 5 of "TYPE2B D2_ZBL
 * a0 s0 p0 p1 Ta Ta") gets one row without an OPTIM line, since tadah hpo
 * searches no value of a TYPE2B line (it refuses OPTIM TYPE2B at its
 * line): for TYPE2B D2_ZBL 0.5292 0.88534 0.23 0.23 Ta Ta the four TYPE2B
 * rows have the seed values 0.5292, 0.88534, 0.23 and 0.23, none has a
 * range, and each rationale names TYPE2B and tadah hpo; the block has no
 * line that starts "OPTIM TYPE2B" (a comment may name OPTIM TYPE2B), a
 * comment "no OPTIM line" for each, and its RCUT2B line.
 * Before the fix: the block had "OPTIM TYPE2B (1) 2.650000e-01
 * 7.950000e-01", "OPTIM TYPE2B (2)" and "OPTIM TYPE2B (3-4)", lines that
 * tadah hpo refuses at startup (and, before that, each such parameter was
 * taken for a vector of the key's every value: 7 rows each, the seed value
 * read from "D2_ZBL" and the element names, ERROR lines logged for the
 * values left out, and one line "OPTIM TYPE2B (1-7)"). */
TEST_CASE("bound_suggester: a value of a TYPE2B line gets a row without an "
          "OPTIM line",
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
      INFO(r.rationale);
      REQUIRE(r.seed_value.has_value());
      seeds.push_back(*r.seed_value);
      CHECK(r.none);
      CHECK(r.rationale.find("TYPE2B") != std::string::npos);
      CHECK(r.rationale.find("tadah hpo") != std::string::npos);
    }
  CHECK(seeds == std::vector<double>{0.5292, 0.88534, 0.23, 0.23});
  std::ostringstream block;
  write_optim_block(rows, block);
  INFO(block.str());
  CHECK(block.str().find("\nOPTIM TYPE2B") == std::string::npos);
  std::size_t comments = 0;
  for (std::size_t at = block.str().find("# D2_ZBL :: TYPE2B");
       at != std::string::npos;
       at = block.str().find("# D2_ZBL :: TYPE2B", at + 1)) {
    const std::string line =
        block.str().substr(at, block.str().find('\n', at) - at);
    CHECK(line.find("no OPTIM line") != std::string::npos);
    ++comments;
  }
  CHECK(comments == 4u);
  CHECK(block.str().find("OPTIM RCUT2B (1) ") != std::string::npos);
}

namespace {

/* Runs `tadah analysis suggest_bounds` (validate, then run) on @p c, with
 * OUTFILE in @p tmp, std::cout captured at the CLI default verbosity
 * (WARNING): the exit code (-1 when the engine throws), what it throws,
 * the captured output and the OPTIM block written. */
struct SbRun {
  int rc = -1;
  std::string what, out, block;
};
SbRun run_suggest_bounds_ctx(Context c, const SbTempDir &tmp) {
  const std::string out = tmp.path("suggested.hpo");
  c.add("OUTFILE", out);
  Task task("analysis", c);
  auto &log = tadah::core::Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(tadah::core::LogLevel::WARNING);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  SbRun r;
  try {
    engines::AnalysisSuggestBoundsEngine engine;
    engine.validate(task);
    r.rc = engine.run(task);
  } catch (const std::exception &e) {
    r.what = e.what();
  }
  std::cout.rdbuf(old_buf);
  log.setLogLevel(old_level);
  r.out = captured.str();
  std::ifstream in(out);
  r.block.assign(std::istreambuf_iterator<char>(in),
                 std::istreambuf_iterator<char>());
  return r;
}

}  // namespace

/* Pins: `tadah analysis suggest_bounds` gives each descriptor of a joined
 * line the values of its own: for TYPE2B D2_mJoin of "D2_ZBL 0.5292
 * 0.88534 0.23 0.23 Ta Ta" and "D2_Blip 4 4 Ta Ta" (RCUT2B 2.0 5.0,
 * CGRID2B 2.5 3.0 3.5 4.0, SGRID2B 1 1 1 1) on two bcc Ta cells it exits 0
 * without an ERROR line, the block has no line that starts "OPTIM
 * TYPE2B", the cutoff of D2_ZBL is "OPTIM RCUT2B (1)" and that of D2_Blip
 * "OPTIM RCUT2B (2)", the values tadah hpo numbers them by, and the four
 * blips "OPTIM CGRID2B (1-4)".
 * Before the fix: the D2_ZBL parameters read the values of the whole
 * TYPE2B key, so their first seed value was "D2_ZBL": the run stopped
 * ("Conversion failed for key=TYPE2B, value="D2_ZBL" ..."); and every
 * descriptor's RCUT2B row was "(1)", the first value. */
TEST_CASE("analysis suggest_bounds gives each descriptor of a D2_mJoin line "
          "its own values and OPTIM indices",
          "[bound_suggester][suggest_bounds][join]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_mJoin");
  c.add("TYPE2B", "D2_ZBL 0.5292 0.88534 0.23 0.23 Ta Ta");
  c.add("TYPE2B", "D2_Blip 4 4 Ta Ta");
  c.add("RCTYPE2B", "Cut_Cos Cut_Cos");
  c.add("RCUT2B", "2.0 5.0");
  c.add("CGRID2B", "2.5 3.0 3.5 4.0");
  c.add("SGRID2B", "1 1 1 1");
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  CHECK(r.rc == EXIT_SUCCESS);
  CHECK(r.what.empty());
  CHECK(r.out.find("ERROR") == std::string::npos);
  CHECK(r.block.find("\nOPTIM TYPE2B") == std::string::npos);
  CHECK(r.block.find("OPTIM RCUT2B (1) ") != std::string::npos);
  CHECK(r.block.find("OPTIM RCUT2B (2) ") != std::string::npos);
  CHECK(r.block.find("OPTIM CGRID2B (1-4) ") != std::string::npos);
}

/* Pins: in a DM_mJoin line each descriptor's suggestions are for its own
 * channels, numbered as in the whole key: for TYPEMB DM_mJoin of "DM_EAD 0
 * 1 1 Ta Ta" (a Gaussian, CGRIDMB(1) = 0) and "DM_Blip 0 2 2 Ta Ta" (two
 * blips, CGRIDMB(2) = 3 and CGRIDMB(3) = 3.5), RCUTMB 5.6 5.6, the block
 * says "CGRIDMB (1): no OPTIM line" (the Gaussian at the origin), has a
 * CGRIDMB line of entry 2 and none of an index range from 1, and the
 * cutoff of DM_Blip is "OPTIM RCUTMB (2)".
 * Before the fix: each descriptor took every channel for its own: DM_Blip
 * gave "OPTIM CGRIDMB (1-3)" with a blip's range for the Gaussian, DM_EAD
 * a Gaussian's range for the two blips, and both cutoffs were "OPTIM
 * RCUTMB (1)". */
TEST_CASE("analysis suggest_bounds gives each descriptor of a DM_mJoin line "
          "its own channels",
          "[bound_suggester][suggest_bounds][join]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "false");
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_mJoin");
  c.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  c.add("TYPEMB", "DM_Blip 0 2 2 Ta Ta");
  c.add("RCTYPEMB", "Cut_Cos Cut_Cos");
  c.add("RCUTMB", "5.6 5.6");
  c.add("CGRIDMB", "0.0 3.0 3.5");
  c.add("SGRIDMB", "1.0 1.0 1.0");
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  CHECK(r.rc == EXIT_SUCCESS);
  CHECK(r.block.find("CGRIDMB (1): no OPTIM line") != std::string::npos);
  CHECK(r.block.find("OPTIM CGRIDMB (1") == std::string::npos);
  CHECK(r.block.find("OPTIM CGRIDMB (2") != std::string::npos);
  CHECK(r.block.find("OPTIM RCUTMB (2) ") != std::string::npos);
}

/* Pins: every range suggest_bounds_for gives holds the seed's value (tadah
 * hpo starts a search from the seed and refuses a starting point outside
 * its box): for D2_Blip with RCUT2B 5.6 on neighbour distances whose 99th
 * percentile is 5.55 A, the RCUT2B row is 3.465 (1.05 times the second
 * shell) to 5.6, its rationale saying that it was widened to the seed's
 * value; and every row of the Ta seed of the HPO tests (D2_Blip, CGRID2B
 * 0.1 to 3.0, SGRID2B 1 to 1.2, RCUT2B 5.6) holds its seed value.
 * Before the fix: the RCUT2B row ended at the 99th percentile of the
 * distances within the seed's cutoff, 5.55, below the seed's 5.6, and
 * tadah hpo refused the block with the seed ("RCUT2B = 5.6 lies outside
 * its OPTIM bounds"). */
TEST_CASE("bound_suggester: every range holds the seed's value",
          "[bound_suggester][suggest_bounds][seed]") {
  std::unique_ptr<tadah::models::D_Base> d(
      tadah::core::factory<tadah::models::D_Base>("D2_Blip"));
  const auto meta = d->meta();
  REQUIRE(meta.has_value());
  Context seed;
  seed.add("RCUT2B", 5.6);
  seed.add("CGRID2B", "0.1 0.583333 1.06667 1.55 2.03333 2.51667 3");
  seed.add("SGRID2B", "1 1.03333 1.06667 1.1 1.13333 1.16667 1.2");
  const auto rows = suggest_bounds_for(ta_shell_stats(), *meta, seed);
  const OptimSuggestion *rcut = row_of(rows, "RCUT2B");
  REQUIRE(rcut != nullptr);
  INFO(rcut->rationale);
  CHECK(rcut->lo == Approx(1.05 * 3.30).epsilon(1e-12));
  CHECK(rcut->hi == 5.6);
  CHECK(rcut->rationale.find("widened") != std::string::npos);
  CHECK(rcut->rationale.find("5.6") != std::string::npos);
  for (const auto &r : rows) {
    if (r.none || !r.seed_value) continue;
    INFO(r.key << "[" << r.arg_pos << "] = " << *r.seed_value << ": " << r.lo
                << " .. " << r.hi);
    CHECK(std::min(r.lo, r.hi) <= *r.seed_value);
    CHECK(std::max(r.lo, r.hi) >= *r.seed_value);
  }
}

/* Pins: the heading of the table says what a rationale gives: the form a
 * range assumes and the data it comes from, "fixed range" and why, or the
 * scale and the role of the parameter (a row whose suggester gives no
 * rationale: "scale=...; role=...").
 * Before the fix: the heading named the first two only, and the rows of
 * the cutoffs and of the 2-body grids, which give the scale and the role,
 * fitted neither. */
TEST_CASE("print_suggestions: the heading names every kind of rationale",
          "[bound_suggester][suggest_bounds]") {
  std::unique_ptr<tadah::models::D_Base> d(
      tadah::core::factory<tadah::models::D_Base>("D2_Blip"));
  const auto meta = d->meta();
  REQUIRE(meta.has_value());
  Context seed;
  seed.add("RCUT2B", 5.6);
  seed.add("CGRID2B", "1 2");
  seed.add("SGRID2B", "1 1");
  std::ostringstream table;
  print_suggestions(suggest_bounds_for(ta_shell_stats(), *meta, seed),
                    table);
  INFO(table.str());
  CHECK(table.str().find("scale=") != std::string::npos);
  CHECK(table.str().find("the scale and the role of the parameter") !=
        std::string::npos);
}

// ---------------------------------------------------------------------------
//  No range of zero width; cutoffs past the seed's; blips centred at the
//  origin; one row per key and index of a joined line
// ---------------------------------------------------------------------------

#include <set>
#include <vector>

namespace {

/* ta_shell_stats() without a pair of atoms within the reach of the
 * statistics, its atoms kept. */
DataStats no_pair_stats() {
  DataStats d = ta_shell_stats();
  d.per_pair.clear();
  return d;
}

/* The lines of @p block that start with @p prefix. */
std::vector<std::string> lines_starting(const std::string &block,
                                        const std::string &prefix) {
  std::vector<std::string> out;
  std::istringstream in(block);
  for (std::string line; std::getline(in, line);)
    if (line.compare(0, prefix.size(), prefix) == 0) out.push_back(line);
  return out;
}

/* The comment line before the line of @p block that starts with @p
 * prefix ("" when there is none). */
std::string comment_before(const std::string &block,
                           const std::string &prefix) {
  std::istringstream in(block);
  std::string last;
  for (std::string line; std::getline(in, line);) {
    if (line.compare(0, prefix.size(), prefix) == 0) return last;
    if (!line.empty()) last = line;
  }
  return "";
}

}  // namespace

/* Pins: no descriptor gets a range of zero width: for every registered
 * descriptor with metadata, suggest_bounds_for with an empty seed gives,
 * on the bcc Ta distances (ta_shell_stats()), rows that have no range
 * (none) or a range whose ends differ, and on data with atoms but no pair
 * within reach rows that have no range or a fixed one (no row comes from
 * distances that are not there); the OPTIM block of the second has no line
 * whose two bounds are equal.
 * Before the fix: the suggesters gave (0, 0) on data without a pair, and
 * DM_REAM's grids (CGRIDMB, SGRIDMB, AMPGRIDMB, CEMBFUNC, SEMBFUNC, KDEGMB,
 * KDEGEMB) gave (0, 0) on any data: "OPTIM CGRIDMB (1) 0.000000e+00
 * 0.000000e+00", a box tadah hpo refuses. */
TEST_CASE("bound_suggester: no descriptor gets a range of zero width",
          "[bound_suggester][suggest_bounds][zero_width]") {
  const DataStats data = ta_shell_stats();
  const DataStats none = no_pair_stats();
  std::size_t described = 0;
  for (const auto &label :
       tadah::core::Registry<tadah::models::D_Base>::names()) {
    std::unique_ptr<tadah::models::D_Base> d(
        tadah::core::factory<tadah::models::D_Base>(label));
    const auto meta = d->meta();
    if (!meta.has_value()) continue;
    ++described;
    CAPTURE(label);
    for (const auto &r : suggest_bounds_for(data, *meta, Context{})) {
      INFO(r.key << "(" << r.arg_pos + 1 << "): " << r.lo << " .. " << r.hi
                 << ": " << r.rationale);
      if (!r.none) CHECK(r.lo != r.hi);
    }
    const auto rows = suggest_bounds_for(none, *meta, Context{});
    for (const auto &r : rows) {
      INFO("no pair: " << r.key << "(" << r.arg_pos + 1 << "): " << r.lo
                       << " .. " << r.hi << ": " << r.rationale);
      CHECK((r.none || r.fixed));
    }
    std::ostringstream block;
    write_optim_block(rows, block);
    for (const auto &line : lines_starting(block.str(), "OPTIM ")) {
      std::istringstream in(line);
      std::string optim, key, index;
      double lo = 0.0, hi = 0.0;
      in >> optim >> key >> index >> lo >> hi;
      INFO(line);
      CHECK(lo != hi);
    }
  }
  CHECK(described > 20u);
}

/* Pins: for data with no pair of atoms within reach of the seed's
 * cutoffs (two bcc Ta cells, nearest neighbours 2.86 A; D2_Blip with
 * RCUT2B 2 and DM_EAD with RCUTMB 2, CGRIDMB 0), tadah analysis
 * suggest_bounds exits 0 and its block has no OPTIM line: each row is a
 * comment "no OPTIM line" saying that no pair of atoms is within reach.
 * Before the fix: the rows were (0, 0) ranges widened to hold the seed's
 * value: "OPTIM RCUT2B (1) 0.000000e+00 2.000000e+00", "OPTIM SGRID2B
 * (1) 0.000000e+00 1.000000e+00" (a width box from 0, which tadah hpo
 * refuses) and "OPTIM CGRIDMB (1) 0.000000e+00 0.000000e+00", of zero
 * width. */
TEST_CASE("analysis suggest_bounds writes no OPTIM line for data without a "
          "pair within reach",
          "[bound_suggester][suggest_bounds][zero_width]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "true");
  c.add("INITMB", "true");
  c.add("TYPE2B", "D2_Blip 2 2 Ta Ta");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", 2.0);
  c.add("CGRID2B", "1.0 1.5");
  c.add("SGRID2B", "1 1");
  c.add("TYPEMB", "DM_EAD 0 1 1 Ta Ta");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUTMB", 2.0);
  c.add("CGRIDMB", 0.0);
  c.add("SGRIDMB", 1.0);
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  CHECK(r.rc == EXIT_SUCCESS);
  CHECK(lines_starting(r.block, "OPTIM ").empty());
  std::size_t rows = 0;
  for (const auto &line : lines_starting(r.block, "# ")) {
    if (line.find("no OPTIM line") == std::string::npos) continue;
    ++rows;
    CHECK(line.find("no pair of atoms") != std::string::npos);
  }
  CHECK(rows >= 6u);
}

/* Pins: the range of a cutoff can reach past the seed's cutoff: the
 * neighbour statistics of tadah analysis suggest_bounds reach 1.25 times
 * the largest cutoff of the seed, so for D2_Blip with RCUT2B 5.6 on two
 * bcc Ta cells (a = 3.3 A: shells at 5.47, 5.72 and 6.6 A) the line
 * "OPTIM RCUT2B (1) lo hi" has hi above 5.6, and its comment names the
 * 99th percentile and the reach (1.25 times) and was not widened to hold
 * the seed.
 * Before the fix: the statistics stopped at the seed's cutoff, so the
 * range ended at the 99th percentile of the distances within 5.6 and was
 * widened to the seed's 5.6: "OPTIM RCUT2B (1) ... 5.600000e+00", its
 * comment "scale=Å; role=...; widened to hold the seed's value 5.6". */
TEST_CASE("analysis suggest_bounds suggests a cutoff range past the seed's "
          "cutoff",
          "[bound_suggester][suggest_bounds][cutoff]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_Blip 4 4 Ta Ta");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", 5.6);
  c.add("CGRID2B", "2.5 3.0 3.5 4.0");
  c.add("SGRID2B", "1 1 1 1");
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  REQUIRE(r.rc == EXIT_SUCCESS);
  const auto rcut = lines_starting(r.block, "OPTIM RCUT2B (1) ");
  REQUIRE(rcut.size() == 1u);
  std::istringstream in(rcut[0]);
  std::string optim, key, index;
  double lo = 0.0, hi = 0.0;
  in >> optim >> key >> index >> lo >> hi;
  CHECK(lo < 5.6);
  CHECK(hi > 5.6);
  const std::string why = comment_before(r.block, "OPTIM RCUT2B (1) ");
  CHECK(why.find("99th percentile") != std::string::npos);
  CHECK(why.find("1.25 times") != std::string::npos);
  CHECK(why.find("widened") == std::string::npos);
}

/* Pins: a blip density centred at the origin is searched by its width
 * alone: for DM_Blip with CGRIDMB 0 4.7 (SGRIDMB 0.8 1, RCUTMB 5.6) on two
 * bcc Ta cells, the block says "CGRIDMB (1): no OPTIM line", has an "OPTIM
 * CGRIDMB (2)" line and no OPTIM line of CGRIDMB entry 1.
 * Before the fix: the seed's centre was not read for a blip: "OPTIM
 * CGRIDMB (1) 0.000000e+00 ..." (the first shell's range widened to hold
 * 0). */
TEST_CASE("analysis suggest_bounds gives a blip density centred at the "
          "origin no CGRIDMB line",
          "[bound_suggester][suggest_bounds][seed]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "false");
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_Blip 0 2 2 Ta Ta");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUTMB", 5.6);
  c.add("CGRIDMB", "0 4.7");
  c.add("SGRIDMB", "0.8 1");
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  CHECK(r.rc == EXIT_SUCCESS);
  CHECK(r.block.find("CGRIDMB (1): no OPTIM line") != std::string::npos);
  CHECK(lines_starting(r.block, "OPTIM CGRIDMB (1").empty());
  CHECK(lines_starting(r.block, "OPTIM CGRIDMB (2) ").size() == 1u);
}

/* Pins: a key that the descriptors of a joined line share (not split
 * among them, RCUTENV of DM_mBlipEnv) gets one row: for DM_mJoin of two
 * DM_mBlipEnv on two bcc Ta cells the block has one "OPTIM RCUTENV (1)"
 * line, and each key and index one OPTIM line.
 * Before the fix: each descriptor gave its own RCUTENV row, both "OPTIM
 * RCUTENV (1)": tadah hpo refused the second ("the index is already
 * selected by the OPTIM RCUTENV line"). */
TEST_CASE("analysis suggest_bounds gives a key the descriptors of a joined "
          "line share one row",
          "[bound_suggester][suggest_bounds][join]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "false");
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_mJoin");
  c.add("TYPEMB", "DM_mBlipEnv 0 1 1 3 3 Ta Ta");
  c.add("TYPEMB", "DM_mBlipEnv 0 1 1 3 3 Ta Ta");
  c.add("RCUTENV", 5.0);
  c.add("RCTYPEMB", "Cut_Cos Cut_Cos");
  c.add("RCUTMB", "5.6 5.0");
  c.add("SGRIDMB", "0.2 0.3");
  c.add("CGRIDMB", "0.0 2.9");
  c.add("SEMBFUNC", "1 1 1 1 1 1");
  c.add("CEMBFUNC", "1 2 3 1 2 3");
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  CHECK(r.rc == EXIT_SUCCESS);
  CHECK(lines_starting(r.block, "OPTIM RCUTENV (1) ").size() == 1u);
  std::set<std::string> selected;
  for (const auto &line : lines_starting(r.block, "OPTIM ")) {
    std::istringstream in(line);
    std::string optim, key, index;
    in >> optim >> key >> index;
    INFO(line);
    CHECK(selected.insert(key + " " + index).second);
  }
}

/* Pins: the rows of the values of the TYPE2B lines of a joined line are
 * numbered as an OPTIM line numbers the values of the whole key (its
 * numbers, from 1): for D2_mJoin of "D2_MIE 12 6 Ta Ta" and "D2_ZBL 0.5292
 * 0.88534 0.23 0.27 Ta Ta" the block has the comments "# D2_MIE :: TYPE2B
 * (1)" and "(2)" and "# D2_ZBL :: TYPE2B (3)" to "(6)", each once, every
 * one "no OPTIM line", and no line that starts "OPTIM TYPE2B"; the table
 * has the rows D2_MIE.TYPE2B(1), (2) and D2_ZBL.TYPE2B(3) to (6).
 * Before the fix: each descriptor numbered the values of its own line:
 * D2_ZBL's rows were TYPE2B (1) to (4), so (1) and (2) named a value of
 * each line. */
TEST_CASE("analysis suggest_bounds numbers the values of the TYPE2B lines "
          "of a join as the whole key does",
          "[bound_suggester][suggest_bounds][join][arg_pos]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_mJoin");
  c.add("TYPE2B", "D2_MIE 12 6 Ta Ta");
  c.add("TYPE2B", "D2_ZBL 0.5292 0.88534 0.23 0.27 Ta Ta");
  c.add("RCTYPE2B", "Cut_Cos Cut_Cos");
  c.add("RCUT2B", "5.0 2.0");
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  CHECK(r.rc == EXIT_SUCCESS);
  CHECK(lines_starting(r.block, "OPTIM TYPE2B").empty());
  const std::pair<const char *, int> rows[] = {
      {"D2_MIE", 1}, {"D2_MIE", 2}, {"D2_ZBL", 3},
      {"D2_ZBL", 4}, {"D2_ZBL", 5}, {"D2_ZBL", 6}};
  for (const auto &[label, n] : rows) {
    CAPTURE(label, n);
    const std::string comment =
        "# " + std::string(label) + " :: TYPE2B (" + std::to_string(n) + ")";
    const auto found = lines_starting(r.block, comment);
    REQUIRE(found.size() == 1u);
    CHECK(found[0].find("no OPTIM line") != std::string::npos);
    CHECK(r.out.find(std::string(label) + ".TYPE2B(" + std::to_string(n) +
                     ")") != std::string::npos);
  }
  CHECK(lines_starting(r.block, "# D2_ZBL :: TYPE2B (1)").empty());
  CHECK(lines_starting(r.block, "# D2_ZBL :: TYPE2B (2)").empty());
}

/* Pins: the rows of tadah analysis suggest_bounds name a descriptor as the
 * configuration does: for TYPEMB DM_PS_SQ 2 2 2 Ta Ta the table has the
 * rows DM_PS_SQ.RCUTMB(1) and DM_PS_SQ.CGRIDMB(1), the block the comments
 * "# DM_PS_SQ ::", and neither names DM_Poly.
 * Before the fix: the rows took the label of the descriptor's metadata,
 * the class's ("DM_Poly.RCUTMB(1)", "# DM_Poly ::") for every DM_PS_*,
 * DM_GPS_*, DM_OW_* and DM_GOW_* descriptor (and "DM_mEAD" for DM_mRLR,
 * DM_mSQRT, ...). */
TEST_CASE("analysis suggest_bounds names a descriptor as the configuration "
          "does",
          "[bound_suggester][suggest_bounds]") {
  tadah::core::PeriodicTable::initialize();
  SbTempDir tmp;
  Context c;
  c.add("DBFILE", bcc_ta_dataset(tmp));
  c.add("INIT2B", "false");
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_PS_SQ 2 2 2 Ta Ta");
  c.add("RCTYPEMB", "Cut_Cos");
  c.add("RCUTMB", 5.6);
  c.add("CGRIDMB", "2.9 4.7");
  c.add("SGRIDMB", "1 1");
  const SbRun r = run_suggest_bounds_ctx(c, tmp);
  INFO(r.what << "\n" << r.out << "\n" << r.block);
  CHECK(r.rc == EXIT_SUCCESS);
  CHECK(r.out.find("DM_PS_SQ.RCUTMB(1)") != std::string::npos);
  CHECK(r.out.find("DM_PS_SQ.CGRIDMB(1)") != std::string::npos);
  CHECK(r.block.find("# DM_PS_SQ ::") != std::string::npos);
  CHECK(r.out.find("DM_Poly") == std::string::npos);
  CHECK(r.block.find("DM_Poly") == std::string::npos);
}
