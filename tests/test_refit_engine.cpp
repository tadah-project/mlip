// Full-engine integration tests for `tadah refit` (RefitEngine::run): a real
// end-to-end run on a synthesised single-element setfl, and the crystal gate
// that hard-errors on an unspecified / unsupported lattice. The EAM is written
// from smooth analytic phi/rho/F so the test needs no checked-in binary fixture.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/task.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/models/descriptors/dm/dm_ream.h>
#include <tadah/models/descriptors/dm/rhoform.h>
#include <tadah/models/eam/eam_setfl_io.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using tadah::core::Context;
using tadah::mlip::Task;
using tadah::mlip::engines::RefitEngine;
namespace fs = std::filesystem;

namespace {
struct PTFixture { PTFixture() { tadah::core::PeriodicTable::initialize(); } };

// A smooth, physically-sane single-element setfl: rho(r)=(1-r/rcut)^2,
// phi(r)=3(1-r/rcut)^3 (stored as z2r=phi*r), F(rho)=-2*sqrt(rho)+0.05*rho
// (+ F_shift: a constant that moves the table's F(0) off zero; rho_fn, when
// given, replaces the density table).
// rhomax is generous so the crystal stays in range even under compression.
void write_synth_eam(const std::string &path, const std::string &lat,
                     double a0, int nr = 400, int nrho = 400,
                     double F_shift = 0.0,
                     const std::function<double(double)> &rho_fn = nullptr) {
  const double rcut = 5.0;
  const double dr = rcut / (nr - 1);
  const double rhomax = 60.0;
  const double drho = rhomax / (nrho - 1);
  std::ofstream os(path);
  os << "# synthetic single-element EAM (refit integration test)\n"
     << "# comment 2\n# comment 3\n"
     << "1 Cu\n"
     << nrho << " " << drho << " " << nr << " " << dr << " " << rcut << "\n"
     << "29 63.55 " << a0 << " " << lat << "\n";
  os << std::setprecision(12);
  for (int i = 0; i < nrho; ++i) {
    const double rho = i * drho;
    os << (-2.0 * std::sqrt(rho) + 0.05 * rho + F_shift)
       << ((i % 5 == 4) ? "\n" : " ");
  }
  os << "\n";
  for (int i = 0; i < nr; ++i) {
    const double x = 1.0 - i * dr / rcut;
    os << (rho_fn ? rho_fn(i * dr) : x > 0 ? x * x : 0.0)
       << ((i % 5 == 4) ? "\n" : " ");
  }
  os << "\n";
  for (int i = 0; i < nr; ++i) {
    const double r = i * dr, x = 1.0 - r / rcut;
    const double phi = x > 0 ? 3.0 * x * x * x : 0.0;
    os << (phi * r) << ((i % 5 == 4) ? "\n" : " ");
  }
  os << "\n";
}

// refit writes several fixed-name files (refit_compare.dat, refit_curve_*.dat,
// ...) to the cwd; run each case in its own temp dir and restore cwd after.
struct ScopedTempDir {
  fs::path old_cwd, dir;
  ScopedTempDir() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_refit_it_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~ScopedTempDir() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

// What one refit run left in the current directory, plus its log.
struct RefitOutputs {
  std::string log;              ///< Logger output (INFO and above)
  std::string embf0;            ///< EMBF0 value in pot.tadah ("" if absent)
  std::string rhoform;          ///< RHOFORM value in pot.tadah ("" if absent)
  std::vector<double> amps;     ///< pot.tadah AMPGRIDMB
  std::vector<double> weights;  ///< pot.tadah WEIGHTS
  double setfl_F0 = 1.0;        ///< refit.eam F(rho = 0)
  int attempts = 0;             ///< dense escalation attempts (refit_opt.log)
  bool self_test_passed() const {
    return log.find("cohesive E/atom") != std::string::npos &&
           log.find("still above tolerance") == std::string::npos;
  }
};

// Runs `tadah refit` on `c` in the current directory with INFO logging on and
// std::cout (the Logger's console) captured, then reads the outputs back.
RefitOutputs run_refit(const Context &c) {
  RefitOutputs out;
  Task task("refit", c);
  auto &logger = tadah::core::Logger::instance();
  const auto old_level = logger.getLogLevel();
  logger.setLogLevel(tadah::core::LogLevel::INFO);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  int rc = -1;
  try {
    rc = RefitEngine().run(task);
  } catch (...) {
    std::cout.rdbuf(old_buf);
    logger.setLogLevel(old_level);
    throw;
  }
  std::cout.rdbuf(old_buf);
  logger.setLogLevel(old_level);
  REQUIRE(rc == EXIT_SUCCESS);
  out.log = captured.str();

  std::ifstream pot("pot.tadah");
  REQUIRE(pot.good());
  std::string line;
  while (std::getline(pot, line)) {
    std::istringstream ls(line);
    std::string key, tok;
    if (!(ls >> key)) continue;
    if (key == "EMBF0") ls >> out.embf0;
    if (key == "RHOFORM") ls >> out.rhoform;
    if (key == "WEIGHTS")
      while (ls >> tok) out.weights.push_back(std::stod(tok));
    if (key == "AMPGRIDMB")
      while (ls >> tok) out.amps.push_back(std::stod(tok));
  }
  tadah::models::EAMSetflIO setfl;
  setfl.read_setfl("refit.eam");
  out.setfl_F0 = setfl.frho().at(0);
  std::ifstream olog("refit_opt.log");
  while (std::getline(olog, line))
    if (line.find("] attempt ") != std::string::npos) ++out.attempts;
  return out;
}

// Dense, coarse (small-N) refit of `eam_file`. The tolerances are relaxed to
// meV scale so the smooth synthetic table passes robustly; a table
// reference error (0.37 eV in the shifted case) still fails them by far.
Context dense_refit_context(const std::string &eam_file,
                            const std::string &rhoform = "SIGNED") {
  Context c;
  c.add("EAM", eam_file);
  c.add("REFIT_MODE", std::string("dense"));
  c.add("REFIT_BASIS", std::string("Blip"));
  c.add("RHOFORM", rhoform);
  c.add("REFIT_LEVEL", std::string("coarse"));
  // The coarse synthetic fit ends at ~2 meV ambient / ~1.5 meV compressed in
  // BOTH references (measured); 5 / 20 meV still flags a raw F_tab left in
  // any zero-referenced comparison (a 370 meV/atom error on the shifted table).
  c.add("REFIT_TOL_AMBIENT", 5.0);
  c.add("REFIT_TOL_COMPRESSED", 20.0);
  c.add("REFIT_MAX_ESCALATIONS", 2);
  return c;
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine runs end-to-end (dense) and self-tests a valid EAM",
                 "[refit][engine][integration]") {
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);

  Context c;
  c.add("EAM", std::string("synth_fcc.eam.alloy"));
  c.add("REFIT_MODE", std::string("dense"));
  c.add("REFIT_BASIS", std::string("Blip"));
  c.add("RHOFORM", std::string("SIGNED"));
  c.add("REFIT_MAX_ESCALATIONS", 1);     // bound the dense escalation loop
  Task task("refit", c);

  RefitEngine engine;
  REQUIRE_NOTHROW(engine.validate(task));
  REQUIRE(engine.run(task) == EXIT_SUCCESS);

  // the headline outputs were written and are non-empty
  for (const char *f : {"pot.tadah", "refit.eam", "refit.hpotarget",
                        "refit_opt.log", "refit_compare.dat"}) {
    INFO("expected output: " << f);
    REQUIRE(fs::exists(f));
    REQUIRE(fs::file_size(f) > 0);
  }

  // the crystal self-test actually ran — i.e. the fcc lattice was recognised
  // (a skipped self-test would never reach the log). The dense path records the
  // per-attempt self-test line in refit_opt.log.
  std::ifstream log("refit_opt.log");
  const std::string body((std::istreambuf_iterator<char>(log)),
                         std::istreambuf_iterator<char>());
  REQUIRE(body.find("self-test") != std::string::npos);
}

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine honours NUMERIC from the refit configuration",
                 "[refit][engine][integration][numeric][lossless]") {
  // build_pot_context() starts from an empty Context, so the configuration's
  // NUMERIC never reached dumpPotFile: pot.tadah was always written at 18
  // digits, unlike train/hpo. NUMERIC is honoured now and never written.
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);

  Context c;
  c.add("EAM", std::string("synth_fcc.eam.alloy"));
  c.add("REFIT_MODE", std::string("dense"));
  c.add("REFIT_BASIS", std::string("Blip"));
  c.add("RHOFORM", std::string("SIGNED"));
  c.add("REFIT_MAX_ESCALATIONS", 0);
  c.add("NUMERIC", 12);
  Task task("refit", c);
  RefitEngine engine;
  REQUIRE(engine.run(task) == EXIT_SUCCESS);

  std::ifstream in("pot.tadah");
  REQUIRE(in.good());
  std::string line;
  bool saw_weights = false;
  std::size_t max_sig = 0;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    if (!(ls >> key)) continue;
    REQUIRE(key != "NUMERIC");
    if (key != "WEIGHTS") continue;
    saw_weights = true;
    std::string tok;
    while (ls >> tok) {
      // significant digits of the mantissa (leading zeros do not count)
      const std::string mant = tok.substr(0, tok.find_first_of("eE"));
      std::size_t n = 0;
      bool lead = true;
      for (char ch : mant) {
        if (ch < '0' || ch > '9') continue;
        if (lead && ch == '0') continue;
        lead = false;
        ++n;
      }
      max_sig = std::max(max_sig, n);
    }
  }
  REQUIRE(saw_weights);
  CHECK(max_sig <= 12);
  CHECK(max_sig >= 10);   // fitted weights are not short decimals
}

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine hard-errors on an unspecified lattice (NONE / a0=0)",
                 "[refit][engine][integration]") {
  ScopedTempDir tmp;
  write_synth_eam("synth_none.eam.fs", "NONE", 0.0);

  Context c;
  c.add("EAM", std::string("synth_none.eam.fs"));
  c.add("REFIT_MODE", std::string("dense"));
  c.add("REFIT_BASIS", std::string("Blip"));
  c.add("RHOFORM", std::string("SIGNED"));
  Task task("refit", c);

  RefitEngine engine;
  REQUIRE_NOTHROW(engine.validate(task));  // the EAM key itself is present
  REQUIRE_THROWS_WITH(engine.run(task), Catch::Contains("crystal usable"));
  // a failed gate must not have produced a potential
  REQUIRE_FALSE(fs::exists("pot.tadah"));
}

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine writes EMBF0 ZERO by default: refit.eam F(0) = 0 "
                 "and the self-test passes",
                 "[refit][engine][integration][embf0]") {
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
  const RefitOutputs out =
      run_refit(dense_refit_context("synth_fcc.eam.alloy"));
  INFO(out.log);
  CHECK(out.embf0 == "ZERO");
  CHECK(out.setfl_F0 == 0.0);   // the round-trip setfl starts at exactly 0
  CHECK(out.attempts >= 1);
  CHECK(out.self_test_passed());
  // the table's F(0) (0 here) is reported, and the cohesive line is relabelled
  CHECK(out.log.find("EMBF0 ZERO: table F(0) = 0 eV removed") !=
        std::string::npos);
  CHECK(out.log.find("relative to the isolated atom") != std::string::npos);
  // the hpotarget recipe points to the per-atom reference energy keys
  std::ifstream hpo("refit.hpotarget");
  const std::string body((std::istreambuf_iterator<char>(hpo)),
                         std::istreambuf_iterator<char>());
  CHECK(body.find("ESHIFT_ATOM") != std::string::npos);
}

TEST_CASE_METHOD(PTFixture, "RefitEngine honours EMBF0 FREE",
                 "[refit][engine][integration][embf0]") {
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
  Context c = dense_refit_context("synth_fcc.eam.alloy");
  c.add("EMBF0", std::string("FREE"));
  const RefitOutputs out = run_refit(c);
  INFO(out.log);
  CHECK(out.embf0 == "FREE");
  CHECK(out.self_test_passed());
  CHECK(out.log.find("table F(0)") == std::string::npos);
  CHECK(out.log.find("relative to the isolated atom") == std::string::npos);
  std::ifstream hpo("refit.hpotarget");
  const std::string body((std::istreambuf_iterator<char>(hpo)),
                         std::istreambuf_iterator<char>());
  CHECK(body.find("EMBF0 ZERO") == std::string::npos);
}

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine EMBF0 ZERO: a constant shift of the table F "
                 "leaves the fit and the escalation unchanged",
                 "[refit][engine][integration][embf0]") {
  // Every absolute use of the table's F (self-test, attribution, dumps, fit
  // target) must be zero-referenced: one raw F_tab left anywhere shows up as
  // a 370 meV/atom self-test error on the shifted table and escalates.
  RefitOutputs base, shifted;
  {
    ScopedTempDir tmp;
    write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
    base = run_refit(dense_refit_context("synth_fcc.eam.alloy"));
  }
  {
    ScopedTempDir tmp;
    write_synth_eam("synth_fcc_shift.eam.alloy", "fcc", 3.6, 400, 400, 0.37);
    Context c = dense_refit_context("synth_fcc_shift.eam.alloy");
    // the F(0) pin does not exist under ZERO: explicitly disabling it only
    // warns and changes nothing
    c.add("REFIT_CONSTRAIN_F0", std::string("false"));
    shifted = run_refit(c);
  }
  INFO(shifted.log);
  CHECK(base.self_test_passed());
  CHECK(shifted.self_test_passed());
  CHECK(shifted.attempts == base.attempts);
  CHECK(shifted.embf0 == "ZERO");
  CHECK(shifted.setfl_F0 == 0.0);
  CHECK(shifted.log.find("EMBF0 ZERO: table F(0) = 0.37 eV removed") !=
        std::string::npos);
  CHECK(shifted.log.find("REFIT_CONSTRAIN_F0 false has no effect") !=
        std::string::npos);
  // same weights up to the 12-digit rounding of the two synthetic tables
  REQUIRE(shifted.weights.size() == base.weights.size());
  REQUIRE_FALSE(base.weights.empty());
  double wmax = 0.0, dmax = 0.0;
  for (std::size_t k = 0; k < base.weights.size(); ++k) {
    wmax = std::max(wmax, std::fabs(base.weights[k]));
    dmax = std::max(dmax, std::fabs(shifted.weights[k] - base.weights[k]));
  }
  CHECK(dmax <= 1e-7 * wmax);
}

// ---------------------------------------------------------------------------
//  RHOFORM (FEAT-02): the refit configuration declares the density form
// ---------------------------------------------------------------------------
namespace {
std::string slurp(const std::string &f) {
  std::ifstream in(f);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

// The (lo, hi) of every `[# ]OPTIM AMPGRIDMB (i) lo hi` line of a file.
std::vector<std::pair<double, double>> ampgrid_bounds(const std::string &f) {
  std::vector<std::pair<double, double>> out;
  std::ifstream in(f);
  for (std::string line; std::getline(in, line);) {
    const auto p = line.find("OPTIM AMPGRIDMB (");
    if (p == std::string::npos) continue;
    std::istringstream ls(line.substr(line.find(')', p) + 1));
    double lo, hi;
    ls >> lo >> hi;
    out.emplace_back(lo, hi);
  }
  return out;
}

// Runs one refit expecting it to throw; returns the message and what the
// Logger printed (INFO and above) before it.
std::pair<std::string, std::string> refit_error(const Context &c) {
  Task task("refit", c);
  auto &logger = tadah::core::Logger::instance();
  const auto old_level = logger.getLogLevel();
  logger.setLogLevel(tadah::core::LogLevel::INFO);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  std::string what;
  try {
    RefitEngine().run(task);
  } catch (const std::exception &e) {
    what = e.what();
  }
  std::cout.rdbuf(old_buf);
  logger.setLogLevel(old_level);
  return {what, captured.str()};
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine: RHOFORM is mandatory in the refit configuration",
                 "[refit][engine][rhoform]") {
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
  Context c = dense_refit_context("synth_fcc.eam.alloy");
  c.remove("RHOFORM");
  Task task("refit", c);
  RefitEngine engine;
  REQUIRE_THROWS_AS(engine.validate(task), tadah::models::DensityFormError);
  REQUIRE_THROWS_WITH(engine.validate(task),
                      Catch::Contains("tadah refit configuration: DM_REAM: "
                                      "RHOFORM is required: SIGNED"));
  REQUIRE_THROWS_AS(engine.run(task), tadah::models::DensityFormError);
  REQUIRE_FALSE(fs::exists("pot.tadah"));
  // an invalid value is refused the same way
  c.add("RHOFORM", std::string("POSITIVE"));
  REQUIRE_THROWS_WITH(RefitEngine().validate(Task("refit", c)),
                      Catch::Contains("RHOFORM 'POSITIVE' is not a density "
                                      "form"));
}

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine fits, self-tests and writes every RHOFORM "
                 "(dense)",
                 "[refit][engine][integration][rhoform]") {
  for (const char *form : {"SIGNED", "NONNEG", "MONOTONE", "monotone"}) {
    INFO("RHOFORM " << form);
    ScopedTempDir tmp;
    write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
    Context c = dense_refit_context("synth_fcc.eam.alloy", form);
    // measured: SIGNED passes after 2 escalations, MONOTONE after 3 (its
    // steps need more density functions near the cutoff of this table)
    c.remove("REFIT_MAX_ESCALATIONS");
    c.add("REFIT_MAX_ESCALATIONS", 4);
    const RefitOutputs out = run_refit(c);
    const tadah::models::RhoForm f = tadah::models::parse_rhoform(form);
    const std::string name = tadah::models::rhoform_name(f);
    INFO(out.log);
    CHECK(out.rhoform == name);
    CHECK(out.self_test_passed());
    CHECK(out.log.find("tadah refit: RHOFORM " + name + ": ") !=
          std::string::npos);
    // SIGNED names the release that introduced the forms
    if (f == tadah::models::RhoForm::Signed)
      CHECK(out.log.find("with signed amplitudes (the density before tadah "
                         "1.6.0)") != std::string::npos);
    CHECK(slurp("refit_opt.log").find("rhoform=" + name) !=
          std::string::npos);
    REQUIRE_FALSE(out.amps.empty());
    // the written potential loads in its form
    Context pot = tadah::core::CoreUserConfigParser().parse("pot.tadah");
    pot.add("VERBOSE", 0);
    tadah::models::DM_REAM dm(pot);
    CHECK(dm.density().form == f);
    // the hpotarget: non-negative AMPGRIDMB boxes under the positive forms,
    // recorded in the parsed header line
    const std::string hpo = slurp("refit.hpotarget");
    const bool positive = tadah::models::rhoform_is_positive(f);
    CHECK((hpo.find(tadah::mlip::engines::rhoform_bounds_line(
               tadah::models::RhoForm::NonNeg)) != std::string::npos) ==
          (f == tadah::models::RhoForm::NonNeg));
    CHECK((hpo.find("# RHOFORM ") != std::string::npos) == positive);
    const auto bounds = ampgrid_bounds("refit.hpotarget");
    REQUIRE(bounds.size() == out.amps.size());
    if (positive) {
      for (double a : out.amps) REQUIRE(a >= 0.0);
      for (const auto &b : bounds) REQUIRE(b.first >= 0.0);
    }
  }
}

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine RHOFORM: table pre-checks stop a refit the form "
                 "cannot pass",
                 "[refit][engine][rhoform]") {
  SECTION("NONNEG and a negative table") {
    ScopedTempDir tmp;
    write_synth_eam("neg.eam.alloy", "fcc", 3.6, 400, 400, 0.0, [](double r) {
      const double x = 1.0 - r / 5.0;
      return x * x - 0.02;
    });
    const auto [what, logged] =
        refit_error(dense_refit_context("neg.eam.alloy", "NONNEG"));
    CHECK(what.find("RHOFORM NONNEG cannot fit this table") !=
          std::string::npos);
    CHECK(what.find("< 0 at r = 4.29") != std::string::npos);
    CHECK_FALSE(fs::exists("pot.tadah"));
    // SIGNED fits it
    const RefitOutputs ok = run_refit(dense_refit_context("neg.eam.alloy"));
    CHECK(ok.rhoform == "SIGNED");
  }
  SECTION("MONOTONE and a rising table") {
    ScopedTempDir tmp;
    write_synth_eam("rise.eam.alloy", "fcc", 3.6, 400, 400, 0.0,
                    [](double r) {
                      const double x = 1.0 - r / 5.0, g = (r - 3.5) / 0.2;
                      return x * x + 0.05 * std::exp(-g * g);
                    });
    const auto [what, logged] =
        refit_error(dense_refit_context("rise.eam.alloy", "MONOTONE"));
    CHECK(what.find("RHOFORM MONOTONE cannot fit this table: the tabulated "
                    "density rises") != std::string::npos);
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
}

TEST_CASE_METHOD(PTFixture,
                 "RefitEngine RHOFORM x REFIT_BASIS x REFIT_MODE",
                 "[refit][engine][rhoform]") {
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
  auto config = [](const char *form, const char *basis, const char *mode) {
    Context c;
    c.add("EAM", std::string("synth_fcc.eam.alloy"));
    c.add("RHOFORM", std::string(form));
    if (basis) c.add("REFIT_BASIS", std::string(basis));
    if (mode) c.add("REFIT_MODE", std::string(mode));
    return c;
  };
  SECTION("MONOTONE is not defined for the knot families") {
    for (const char *b : {"Knot5", "Knot3", "KnotMix"}) {
      INFO(b);
      const auto [what, logged] = refit_error(config("MONOTONE", b, "dense"));
      CHECK(what.find(std::string("RHOFORM MONOTONE is not defined for "
                                  "REFIT_BASIS ") + b) != std::string::npos);
      CHECK(what.find("use a bump basis (Blip, Blip5) or RHOFORM NONNEG") !=
            std::string::npos);
    }
  }
  SECTION("compact mode accepts the positive forms with a bump basis") {
    // The HPO engine fits them in the compact fitter (HPO test_compact_*);
    // this MLIP-only RefitEngine has no compact hook, so it falls back to
    // the dense path, in the declared form.
    for (const char *form : {"NONNEG", "MONOTONE"}) {
      INFO(form);
      ScopedTempDir sub;
      write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
      Context c = dense_refit_context("synth_fcc.eam.alloy", form);
      c.remove("REFIT_MODE");
      c.add("REFIT_MODE", std::string("compact"));
      const RefitOutputs out = run_refit(c);
      INFO(out.log);
      CHECK(out.rhoform == form);
      CHECK(out.log.find("falling back to dense fitting") !=
            std::string::npos);
      // a bump basis: no knot convex-lock HINT or WARNING
      CHECK(out.log.find("RHOFORM NONNEG with") == std::string::npos);
      REQUIRE_FALSE(out.amps.empty());
      for (double a : out.amps) CHECK(a >= 0.0);
    }
  }
  SECTION("compact mode: MONOTONE names a defaulted knot basis") {
    // REFIT_MODE compact defaults to Knot5, for which MONOTONE is not
    // defined: the message says the basis is the default, with the remedy
    for (const char *mode : {static_cast<const char *>(nullptr), "compact"}) {
      INFO((mode ? mode : "(default mode)"));
      const auto [what, logged] =
          refit_error(config("MONOTONE", nullptr, mode));
      CHECK(what.find("RHOFORM MONOTONE is not defined for REFIT_BASIS Knot5 "
                      "(the REFIT_MODE compact default, a one-sided knot "
                      "family): use a bump basis (Blip, Blip5) or RHOFORM "
                      "NONNEG") != std::string::npos);
      CHECK_FALSE(fs::exists("pot.tadah"));
    }
    // an explicit knot basis in compact mode: the same refusal, no default
    const auto [what, logged] =
        refit_error(config("MONOTONE", "Knot3", nullptr));
    CHECK(what.find("REFIT_BASIS Knot3 (the one-sided knot family)") !=
          std::string::npos);
  }
  SECTION("compact mode: NONNEG with a knot family: the convex-lock HINT "
          "for a chosen basis, a WARNING for the default") {
    // Explicit Knot5: the HINT, then (MLIP-only build, no compact fitter)
    // the knot families' refusal of the dense path.
    const auto [what, logged] =
        refit_error(config("NONNEG", "Knot5", nullptr));
    CHECK(logged.find("WARNING: HINT: RHOFORM NONNEG with REFIT_BASIS Knot5: "
                      "a knot density with every amplitude >= 0 is "
                      "non-increasing and convex") != std::string::npos);
    CHECK(logged.find("compact default") == std::string::npos);
    CHECK(what.find("REFIT_BASIS Knot5 requires REFIT_MODE compact on an HPO "
                    "(Ceres) build") != std::string::npos);
    // The defaulted Knot5 (no REFIT_BASIS): a WARNING, not a HINT, naming
    // REFIT_BASIS Blip5 as the remedy (a concave-core table can fail the
    // self-test in the non-negative knot density); without the compact
    // fitter the refit then falls back to a dense Blip fit, in NONNEG.
    ScopedTempDir sub;
    write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
    Context c = dense_refit_context("synth_fcc.eam.alloy", "NONNEG");
    c.remove("REFIT_MODE");
    c.remove("REFIT_BASIS");
    const RefitOutputs out = run_refit(c);
    INFO(out.log);
    CHECK(out.log.find("WARNING: tadah refit: RHOFORM NONNEG with the "
                       "REFIT_MODE compact default basis Knot5: a knot "
                       "density with every amplitude >= 0 is non-increasing "
                       "and convex") != std::string::npos);
    CHECK(out.log.find("Set REFIT_BASIS Blip5 (a non-negative bump density "
                       "without the convex lock), or use RHOFORM SIGNED.") !=
          std::string::npos);
    CHECK(out.log.find("HINT: RHOFORM NONNEG") == std::string::npos);
    CHECK(out.rhoform == "NONNEG");
    // REFIT_MODE compact given explicitly, basis still defaulted: the same
    c.add("REFIT_MODE", std::string("compact"));
    const RefitOutputs out2 = run_refit(c);
    CHECK(out2.log.find("WARNING: tadah refit: RHOFORM NONNEG with the "
                        "REFIT_MODE compact default basis Knot5") !=
          std::string::npos);
  }
  SECTION("NONNEG with a knot family in dense mode: the knot refusal, no "
          "HINT") {
    const auto [what, logged] =
        refit_error(config("NONNEG", "Knot5", "dense"));
    CHECK(what.find("REFIT_BASIS Knot5 requires REFIT_MODE compact") !=
          std::string::npos);
    CHECK(logged.find("HINT: RHOFORM NONNEG") == std::string::npos);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: the configuration numbers are read strictly, "
                 "with their ranges",
                 "[refit][strict_numbers]") {
  /* Pins: RefitEngine::validate(), and run() before it reads the EAM file,
   * read every refit configuration number with the number grammar of the
   * hpotarget file (core::parse_decimal_normal) or as decimal digits, and
   * refuse one out of its range naming the key and the token as written:
   * the tolerances, REFIT_COMPACT_SHARE, OUTPUT_GRID_SCALE, OPTIM_PCT and
   * REFIT_REACH_MARGIN > 0, REFIT_REACH_SMIN and REFIT_CURVE_RELAX in
   * (0, 1] (in either REFIT_OBJECTIVE), REFIT_R_LO_FRAC in (0, 1),
   * REFIT_CURVE_DERIV_WEIGHT >= 0, the compact counts from 1 (JITTERS from
   * 0 to 2147483646, its candidates jitters + 1 an int), REFIT_COMPACT_SEED
   * from 0 to 4294967295, N2B, NRHO_BASIS and NF_BASIS from 1 to 1024 (the
   * escalation cap), REFIT_MAX_ESCALATIONS from 0, REFIT_COMPACT_NMAX one
   * or three integers from 1; a number below the smallest normal double
   * says it is too small, and a scalar key takes one value. The values at
   * the ends of the ranges are accepted.
   * Before the fix: validate() read none of them (no exception), and run()
   * read them with Context::get (istringstream) unchecked, so the first
   * message checked is not thrown and run() fails reading the absent EAM
   * file instead. */
  const std::string i32 = "2147483647";
  struct Case {
    std::string key;
    std::vector<std::string> tokens;
    std::string message;
  };
  const std::string tiny =
      " (too small: its magnitude is below the smallest normal double, "
      "2.2250738585072014e-308)";
  const std::vector<Case> cases = {
      {"REFIT_COMPACT_NMAX", {"+3"},
       "REFIT_COMPACT_NMAX must be an integer from 1 to " + i32 + ", got '+3'"},
      {"REFIT_COMPACT_NMAX", {"5", "0", "5"},
       "REFIT_COMPACT_NMAX must be an integer from 1 to " + i32 + ", got '0'"},
      {"REFIT_COMPACT_NMAX", {"5", "6"},
       "REFIT_COMPACT_NMAX takes 1 value (all functions) or 3 (phi rho F), "
       "got 2"},
      {"REFIT_COMPACT_NMAX", {"2147483648"},
       "REFIT_COMPACT_NMAX must be an integer from 1 to " + i32 +
           ", got '2147483648'"},
      {"REFIT_TOL_AMBIENT", {"0"},
       "REFIT_TOL_AMBIENT must be a positive number, got '0'"},
      {"REFIT_TOL_AMBIENT", {"-0.1"},
       "REFIT_TOL_AMBIENT must be a positive number, got '-0.1'"},
      {"REFIT_TOL_AMBIENT", {"1e-400"},
       "REFIT_TOL_AMBIENT must be a positive number, got '1e-400'" + tiny},
      {"REFIT_TOL_AMBIENT", {"0x1p-3"},
       "REFIT_TOL_AMBIENT must be a positive number, got '0x1p-3'"},
      {"REFIT_TOL_AMBIENT", {"1", "2"},
       "REFIT_TOL_AMBIENT takes one value, got 2"},
      {"REFIT_TOL_COMPRESSED", {"0"},
       "REFIT_TOL_COMPRESSED must be a positive number, got '0'"},
      {"OUTPUT_GRID_SCALE", {"0"},
       "OUTPUT_GRID_SCALE must be a positive number, got '0'"},
      {"OPTIM_PCT", {"-5"}, "OPTIM_PCT must be a positive number, got '-5'"},
      {"REFIT_REACH_SMIN", {"2"},
       "REFIT_REACH_SMIN must be a number in (0, 1], got '2'"},
      {"REFIT_REACH_SMIN", {"-1"},
       "REFIT_REACH_SMIN must be a number in (0, 1], got '-1'"},
      {"REFIT_REACH_MARGIN", {"0"},
       "REFIT_REACH_MARGIN must be a positive number, got '0'"},
      {"REFIT_R_LO_FRAC", {"2"},
       "REFIT_R_LO_FRAC must be a number in (0, 1), got '2'"},
      {"REFIT_R_LO_FRAC", {"1"},
       "REFIT_R_LO_FRAC must be a number in (0, 1), got '1'"},
      {"REFIT_COMPACT_STALL", {"0"},
       "REFIT_COMPACT_STALL must be an integer from 1 to " + i32 +
           ", got '0'"},
      {"REFIT_COMPACT_OPT_EVERY", {"0"},
       "REFIT_COMPACT_OPT_EVERY must be an integer from 1 to " + i32 +
           ", got '0'"},
      {"REFIT_COMPACT_INSERT_TOPK", {"0"},
       "REFIT_COMPACT_INSERT_TOPK must be an integer from 1 to " + i32 +
           ", got '0'"},
      {"REFIT_COMPACT_SAMPLES", {"0"},
       "REFIT_COMPACT_SAMPLES must be an integer from 1 to " + i32 +
           ", got '0'"},
      {"REFIT_COMPACT_JITTERS", {"2147483647"},
       "REFIT_COMPACT_JITTERS must be an integer from 0 to 2147483646, got "
       "'2147483647'"},
      {"REFIT_COMPACT_JITTERS", {"-1"},
       "REFIT_COMPACT_JITTERS must be an integer from 0 to 2147483646, got "
       "'-1'"},
      {"REFIT_COMPACT_SEED", {"4294967296"},
       "REFIT_COMPACT_SEED must be an integer from 0 to 4294967295, got "
       "'4294967296'"},
      {"REFIT_COMPACT_SHARE", {"-1"},
       "REFIT_COMPACT_SHARE must be a positive number, got '-1'"},
      {"REFIT_CURVE_RELAX", {"0"},
       "REFIT_CURVE_RELAX must be a number in (0, 1], got '0'"},
      {"REFIT_CURVE_DERIV_WEIGHT", {"-1"},
       "REFIT_CURVE_DERIV_WEIGHT must be a number >= 0, got '-1'"},
      {"N2B", {"0"}, "N2B must be an integer from 1 to 1024, got '0'"},
      {"NRHO_BASIS", {"1025"},
       "NRHO_BASIS must be an integer from 1 to 1024, got '1025'"},
      {"NF_BASIS", {"5x"}, "NF_BASIS must be an integer from 1 to 1024, got "
                           "'5x'"},
      {"REFIT_MAX_ESCALATIONS", {"-1"},
       "REFIT_MAX_ESCALATIONS must be an integer from 0 to " + i32 +
           ", got '-1'"}};
  const auto with = [](const std::string &key,
                       const std::vector<std::string> &tokens) {
    Context c = dense_refit_context("absent.eam.alloy");
    c.remove(key);
    for (const auto &t : tokens) c.add(key, t);
    return c;
  };
  ScopedTempDir tmp;
  for (const auto &cs : cases) {
    INFO(cs.key << " " << cs.tokens.front()
                << (cs.tokens.size() > 1 ? " ..." : ""));
    const Context c = with(cs.key, cs.tokens);
    Task task("refit", c);
    RefitEngine engine;
    CHECK_THROWS_WITH(engine.validate(task), "tadah refit: " + cs.message);
    // run() refuses it before it reads the (absent) EAM file
    CHECK_THROWS_WITH(engine.run(task), "tadah refit: " + cs.message);
  }
  // REFIT_CURVE_RELAX under REFIT_OBJECTIVE energy too (it was checked in
  // curve mode only)
  {
    Context c = with("REFIT_CURVE_RELAX", {"1.5"});
    c.add("REFIT_OBJECTIVE", std::string("energy"));
    Task task("refit", c);
    CHECK_THROWS_WITH(RefitEngine().validate(task),
                      "tadah refit: REFIT_CURVE_RELAX must be a number in "
                      "(0, 1], got '1.5'");
  }
  // The ends of the ranges are accepted.
  for (const auto &[key, tok] :
       std::vector<std::pair<std::string, std::string>>{
           {"REFIT_COMPACT_SEED", "3000000000"},
           {"REFIT_COMPACT_SEED", "4294967295"},
           {"REFIT_COMPACT_SEED", "0"},
           {"REFIT_COMPACT_JITTERS", "0"},
           {"REFIT_COMPACT_JITTERS", "2147483646"},
           {"REFIT_COMPACT_NMAX", i32},
           {"REFIT_MAX_ESCALATIONS", "0"},
           {"REFIT_CURVE_DERIV_WEIGHT", "0"},
           {"REFIT_CURVE_RELAX", "1"},
           {"REFIT_REACH_SMIN", "1"},
           {"N2B", "1024"},
           {"REFIT_TOL_AMBIENT", ".5"},
           {"REFIT_R_LO_FRAC", "0.15"}}) {
    INFO(key << " " << tok);
    Task task("refit", with(key, {tok}));
    CHECK_NOTHROW(RefitEngine().validate(task));
  }
}
