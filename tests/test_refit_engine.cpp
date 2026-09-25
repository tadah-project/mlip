// Full-engine integration tests for `tadah refit` (RefitEngine::run): a real
// end-to-end run on a synthesised single-element setfl, and the crystal gate
// that hard-errors on an unspecified / unsupported lattice. The EAM is written
// from smooth analytic phi/rho/F so the test needs no checked-in binary fixture.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/models/eam/eam_setfl_io.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
// (+ F_shift: a constant that moves the table's F(0) off zero).
// rhomax is generous so the crystal stays in range even under compression.
void write_synth_eam(const std::string &path, const std::string &lat,
                     double a0, int nr = 400, int nrho = 400,
                     double F_shift = 0.0) {
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
    os << (x > 0 ? x * x : 0.0) << ((i % 5 == 4) ? "\n" : " ");
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
    if (key == "WEIGHTS")
      while (ls >> tok) out.weights.push_back(std::stod(tok));
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
Context dense_refit_context(const std::string &eam_file) {
  Context c;
  c.add("EAM", eam_file);
  c.add("REFIT_MODE", std::string("dense"));
  c.add("REFIT_BASIS", std::string("Blip"));
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
