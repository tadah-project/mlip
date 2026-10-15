// Full-engine integration tests for `tadah refit` (RefitEngine::run): a real
// end-to-end run on a synthesised single-element setfl, and the crystal gate
// that hard-errors on an unspecified / unsupported lattice. The EAM is written
// from smooth analytic phi/rho/F so the test needs no checked-in binary fixture.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/key_provenance.h>
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
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
// (+ F_shift: a constant that moves the table's F(0) off zero; rho_fn and
// phi_fn, when given, replace the density and the pair function).
// rhomax is generous so the crystal stays in range even under compression.
void write_synth_eam(const std::string &path, const std::string &lat,
                     double a0, int nr = 400, int nrho = 400,
                     double F_shift = 0.0,
                     const std::function<double(double)> &rho_fn = nullptr,
                     const std::function<double(double)> &phi_fn = nullptr) {
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
    const double phi = phi_fn ? phi_fn(r) : x > 0 ? 3.0 * x * x * x : 0.0;
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

// The REFIT_MODE compact counterpart of dense_refit_context: a bump basis,
// the same tolerances and none of the dense mode's keys (REFIT_LEVEL,
// REFIT_MAX_ESCALATIONS, ...), which REFIT_MODE compact refuses.
Context compact_refit_context(const std::string &eam_file,
                              const std::string &rhoform = "SIGNED") {
  Context c;
  c.add("EAM", eam_file);
  c.add("REFIT_MODE", std::string("compact"));
  c.add("REFIT_BASIS", std::string("Blip5"));
  c.add("RHOFORM", rhoform);
  c.add("REFIT_TOL_AMBIENT", 5.0);
  c.add("REFIT_TOL_COMPRESSED", 20.0);
  return c;
}

// The refusal of REFIT_MODE compact by this engine, which has no compact
// fitter (HPO's engine has it); `defaulted`: the configuration has no
// REFIT_MODE key.
std::string no_compact_fitter(bool defaulted = false) {
  return std::string("tadah refit: REFIT_MODE compact") +
         (defaulted ? " (the default)" : "") +
         " needs the compact fitter, which this tadah was built without (it "
         "needs TADAH_ENABLE_NFP and TADAH_ENABLE_CERES); set REFIT_MODE "
         "dense";
}

// The mode rule's refusal of `what` (a key, "the <KEY> block" or
// "REFIT_OBJECTIVE curve") under REFIT_MODE `mode` (dense | compact);
// `defaulted`: the configuration has no REFIT_MODE key.
std::string no_effect_in_mode(const std::string &what, const std::string &mode,
                              bool defaulted = false) {
  const std::string other = mode == "dense" ? "compact" : "dense";
  return "tadah refit: " + what + " has no effect under REFIT_MODE " + mode +
         (defaulted ? " (the default)" : "") + ": only REFIT_MODE " + other +
         " reads it; remove " + what + ", or set REFIT_MODE " + other;
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
    // (REFIT_CONSTRAIN_F0, which the F(0) pin of EMBF0 FREE reads, is
    // refused under ZERO: "REFIT_CONSTRAIN_F0, REFIT_LEVEL, REFIT_PLACEMENT
    // and REFIT_BASIS are read as written or refused" below; this case
    // set it false and checked its WARNING until 2026-10-12)
    Context c = dense_refit_context("synth_fcc_shift.eam.alloy");
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

/* Pins: the crystal gate of RefitEngine::run() refuses, before any fitting,
 * a lattice constant whose crystal the table cannot describe: the
 * synthetic fcc table whose element line gives a0 = 1e-323 (a subnormal,
 * which the setfl reader reads; dr = 0.0125313 A), whose nearest
 * neighbours are 5e-324 A apart at the default REFIT_REACH_SMIN 0.75, is
 * refused with "tadah refit: the lattice constant of the EAM header, a0 =
 * 1e-323 A, puts the nearest neighbours of the fcc crystal 5e-324 A apart
 * at the strongest compression the refit tests (a/a0 = REFIT_REACH_SMIN
 * 0.75), less than the step of the table's radial grid, dr = 0.0125313 A:
 * ... set the equilibrium lattice constant, in Angstrom." and writes no
 * pot.tadah; with REFIT_REACH_SMIN 0.1 the message says 0 A and 0.1.
 * Before the fix: the gate asked for a known lattice and a0 > 0 only, so
 * the dense refit ran (its crystal sums enumerated no neighbour: ceil(rcut
 * / a) does not fit an int) and wrote pot.tadah: no exception. */
TEST_CASE_METHOD(PTFixture,
                 "RefitEngine refuses a lattice constant whose crystal's "
                 "nearest neighbours at REFIT_REACH_SMIN are less than the "
                 "table's dr apart",
                 "[refit][engine][integration]") {
  const std::string tail =
      " less than the step of the table's radial grid, dr = 0.0125313 A: "
      "the table describes no pair distance of that crystal. Edit the "
      "element line of the setfl file (`<Z> <mass> <a0> <lattice>`): set "
      "the equilibrium lattice constant, in Angstrom.";
  for (const char *smin : {"", "0.1"}) {
    INFO("REFIT_REACH_SMIN '" << smin << "'");
    ScopedTempDir tmp;
    write_synth_eam("synth_tiny.eam.alloy", "fcc", 1e-323);
    Context c = dense_refit_context("synth_tiny.eam.alloy");
    if (*smin) c.add("REFIT_REACH_SMIN", std::stod(smin));
    const auto [what, logged] = refit_error(c);
    INFO(logged);
    CHECK(what ==
          std::string("tadah refit: the lattice constant of the EAM header, "
                      "a0 = 1e-323 A, puts the nearest neighbours of the fcc "
                      "crystal ") +
              (*smin ? "0" : "5e-324") +
              " A apart at the strongest compression the refit tests (a/a0 "
              "= REFIT_REACH_SMIN " +
              (*smin ? "0.1" : "0.75") + ")," + tail);
    CHECK_FALSE(fs::exists("pot.tadah"));
  }
}

/* Pins: the crystal gate of RefitEngine::run() refuses, before any fitting,
 * a lattice constant that puts the crystal's nearest neighbours at or
 * beyond the table's cutoff, where the crystal has no pair: the synthetic
 * fcc table (rcut 5 A) whose element line gives a0 = 361 (copper's lattice
 * constant in picometres) is refused with exactly "tadah refit: the lattice
 * constant of the EAM header, a0 = 361 A, puts the nearest neighbours of
 * the fcc crystal 255.26554800834364 A apart, at or beyond the table's
 * cutoff, rcut = 5 A: the crystal has no pair within the cutoff. Edit the
 * element line of the setfl file (`<Z> <mass> <a0> <lattice>`): set the
 * equilibrium lattice constant, in Angstrom (a value in picometres is 100
 * times the value in Angstrom)." and writes no pot.tadah.
 * Before the fix: the gate took no cutoff, so the dense refit ran: its
 * reachable pair distances began at 191.4 A, its self-test compared
 * crystals without neighbours (0.000 meV/atom) and it wrote pot.tadah: no
 * exception. */
TEST_CASE_METHOD(PTFixture,
                 "RefitEngine refuses a lattice constant whose crystal's "
                 "nearest neighbours are at or beyond the table's cutoff",
                 "[refit][engine][integration]") {
  ScopedTempDir tmp;
  write_synth_eam("synth_pm.eam.alloy", "fcc", 361.0);
  const auto [what, logged] =
      refit_error(dense_refit_context("synth_pm.eam.alloy"));
  INFO(logged);
  CHECK(what ==
        "tadah refit: the lattice constant of the EAM header, a0 = 361 A, "
        "puts the nearest neighbours of the fcc crystal 255.26554800834364 A "
        "apart, at or beyond the table's cutoff, rcut = 5 A: the crystal has "
        "no pair within the cutoff. Edit the element line of the setfl file "
        "(`<Z> <mass> <a0> <lattice>`): set the equilibrium lattice constant, "
        "in Angstrom (a value in picometres is 100 times the value in "
        "Angstrom).");
  CHECK_FALSE(fs::exists("pot.tadah"));
}

namespace {

/* An engine with a compact fitter whose hook returns without a fit: what
 * RefitEngine::run() does when an engine breaks the hook's contract (fill
 * the fit and its self-test, or throw). */
struct CompactFitterWithoutFit : RefitEngine {
protected:
  bool has_compact_fitter() const override { return true; }
  void compact_fit(const tadah::core::Context &,
                   const tadah::models::EAMSetflIO &,
                   tadah::models::EAMRefitConfig &,
                   std::unique_ptr<tadah::models::EAMRefit> &,
                   SelfTest &) override {}
};

}  // namespace

/* Pins: RefitEngine::compact_fit returns nothing -- the compact fitter fits
 * the table or throws -- and run() has no dense fallback: a hook that
 * returns without a fit (CompactFitterWithoutFit) makes run() of REFIT_MODE
 * compact throw std::logic_error "tadah refit: internal error: the compact
 * fitter returned without a fit (RefitEngine::compact_fit fills the fit
 * and its self-test, or throws); this is a bug, please report it", with the
 * explicit REFIT_BASIS Blip5 and with the defaulted basis (Knot5), writes
 * no pot.tadah and logs no "falls back" WARNING.
 * Before the fix: the hook returned bool, and a false return ("the fitter
 * declined this table") fitted the table in dense mode, with the WARNING
 * "tadah refit: defaulted basis falls back Knot5 -> Blip for the dense
 * path (set REFIT_BASIS explicitly to control this)." for the defaulted
 * basis, and wrote pot.tadah; this test, written for the hook without a
 * return value, does not compile (tmp/before_fix_rg_mlip.txt holds that
 * build and a run of the same engine returning false). */
TEST_CASE_METHOD(PTFixture,
                 "RefitEngine: a compact fitter that returns without a fit "
                 "is an internal error, not a fall back to the dense fit",
                 "[refit][engine][compact_hook]") {
  for (const bool defaulted_basis : {false, true}) {
    INFO((defaulted_basis ? "the default basis" : "REFIT_BASIS Blip5"));
    ScopedTempDir tmp;
    write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
    Context c = compact_refit_context("synth_fcc.eam.alloy");
    if (defaulted_basis) c.remove("REFIT_BASIS");
    Task task("refit", c);
    auto &logger = tadah::core::Logger::instance();
    const auto old_level = logger.getLogLevel();
    logger.setLogLevel(tadah::core::LogLevel::INFO);
    std::ostringstream captured;
    std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
    std::string what;
    bool logic_error = false;
    try {
      CompactFitterWithoutFit().run(task);
    } catch (const std::logic_error &e) {
      logic_error = true;
      what = e.what();
    } catch (const std::exception &e) {
      what = e.what();
    }
    std::cout.rdbuf(old_buf);
    logger.setLogLevel(old_level);
    INFO("log:\n" << captured.str());
    CHECK(logic_error);
    CHECK(what ==
          "tadah refit: internal error: the compact fitter returned without "
          "a fit (RefitEngine::compact_fit fills the fit and its self-test, "
          "or throws); this is a bug, please report it");
    CHECK_FALSE(fs::exists("pot.tadah"));
    CHECK(captured.str().find("falls back") == std::string::npos);
  }
}

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
  // The REFIT_MODE compact cases of this matrix (MONOTONE with the
  // defaulted Knot5, the NONNEG x knot-family HINT and WARNING, the
  // positive forms with a bump basis) run through HPO's engine, which has
  // the compact fitter: HPO test_compact_refit_rhoform.cpp. This engine has
  // none and refuses REFIT_MODE compact before the table is read (below,
  // "REFIT_MODE compact without a compact fitter"); they ran here through
  // the fallback to the dense fit that the refusal replaced.
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
   * REFIT_REACH_MARGIN > 0, REFIT_REACH_SMIN from 0.1 to 1 (below 0.1 the
   * compressed-lattice neighbour lists of the self-test and the E-V scans
   * grow as smin^-3: a/a0 0.01 took a minute, 1e-6 did not finish; the
   * reachability probe lists neighbours out to rcut/0.1 only; it was
   * (0, 1], the range of the first fix), REFIT_CURVE_RELAX in (0, 1],
   * REFIT_R_LO_FRAC in (0, 1),
   * REFIT_CURVE_DERIV_WEIGHT >= 0, the compact counts from 1 (JITTERS from
   * 0 to 2147483646, its candidates jitters + 1 an int), REFIT_COMPACT_SEED
   * from 0 to 4294967295, N2B, NRHO_BASIS and NF_BASIS from 1 to 1024 (the
   * escalation cap), REFIT_MAX_ESCALATIONS from 0, REFIT_COMPACT_NMAX one
   * or three integers from 1 to 2147483647 whose largest gives a curve
   * sample grid of 4 x NMAX points per function that fits an int (so at
   * most 536870911; refused naming the grid); a number below the smallest
   * normal double says it is too small. (A scalar key holds one value:
   * Context::add overwrites it.) The values at the ends of the ranges are
   * accepted. A key that only REFIT_MODE compact reads (REFIT_COMPACT_*,
   * REFIT_CURVE_*) is set in a REFIT_MODE compact configuration, since
   * REFIT_MODE dense refuses it (the mode rule, before the numbers), a
   * REFIT_CURVE_* key under REFIT_OBJECTIVE curve, since energy refuses it;
   * once its numbers pass, this engine, which has no compact fitter,
   * refuses REFIT_MODE compact. Under REFIT_OBJECTIVE energy REFIT_CURVE_RELAX
   * 1.5 is refused by the rule, before its value is read.
   * Before the fix: validate() read none of them (no exception), and run()
   * read them with Context::get (istringstream) unchecked, so the first
   * message checked is not thrown and run() fails reading the absent EAM
   * file instead. Before the REFIT_REACH_SMIN floor: its range was (0, 1],
   * so its messages said "a number in (0, 1]" and 0.05 and 1e-6 were
   * accepted. Before the NMAX grid check moved into the reader: the compact
   * fit checked the sample grid after the EAM file was read (compact mode
   * only), so validate() accepted NMAX 536870912 and 2147483647 (an end of
   * the accepted range here then). Before the mode rule: the compact keys
   * were set in the dense configuration, which ignored them, and
   * validate() accepted REFIT_MODE compact without a compact fitter (run()
   * fell back to the dense fit), so for the ends of the compact keys'
   * ranges the refusal of REFIT_MODE compact is not thrown. */
  const std::string i32 = "2147483647";
  struct Case {
    std::string key;
    std::vector<std::string> tokens;
    std::string message;
  };
  const std::string tiny =
      " (too small: its magnitude is below the smallest normal double, "
      "2.2250738585072014e-308)";
  const auto grid = [](const std::string &nmax, const std::string &points) {
    return "REFIT_COMPACT_NMAX " + nmax + " needs a curve sample grid of 4 "
           "x " + nmax + " = " + points + " points per function "
           "(REFIT_COMPACT_SAMPLES is raised to at least 4 x "
           "REFIT_COMPACT_NMAX), above 2147483647; lower REFIT_COMPACT_NMAX";
  };
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
      {"REFIT_COMPACT_NMAX", {"536870912"}, grid("536870912", "2147483648")},
      {"REFIT_COMPACT_NMAX", {"5", "536870912", "5"},
       grid("536870912", "2147483648")},
      {"REFIT_COMPACT_NMAX", {i32}, grid(i32, "8589934588")},
      {"REFIT_TOL_AMBIENT", {"0"},
       "REFIT_TOL_AMBIENT must be a positive number, got '0'"},
      {"REFIT_TOL_AMBIENT", {"-0.1"},
       "REFIT_TOL_AMBIENT must be a positive number, got '-0.1'"},
      {"REFIT_TOL_AMBIENT", {"1e-400"},
       "REFIT_TOL_AMBIENT must be a positive number, got '1e-400'" + tiny},
      {"REFIT_TOL_AMBIENT", {"0x1p-3"},
       "REFIT_TOL_AMBIENT must be a positive number, got '0x1p-3'"},
      {"REFIT_TOL_COMPRESSED", {"0"},
       "REFIT_TOL_COMPRESSED must be a positive number, got '0'"},
      {"OUTPUT_GRID_SCALE", {"0"},
       "OUTPUT_GRID_SCALE must be a positive number, got '0'"},
      {"OPTIM_PCT", {"-5"}, "OPTIM_PCT must be a positive number, got '-5'"},
      {"REFIT_REACH_SMIN", {"2"},
       "REFIT_REACH_SMIN must be a number from 0.1 to 1, got '2'"},
      {"REFIT_REACH_SMIN", {"-1"},
       "REFIT_REACH_SMIN must be a number from 0.1 to 1, got '-1'"},
      {"REFIT_REACH_SMIN", {"0.05"},
       "REFIT_REACH_SMIN must be a number from 0.1 to 1, got '0.05'"},
      {"REFIT_REACH_SMIN", {"1e-6"},
       "REFIT_REACH_SMIN must be a number from 0.1 to 1, got '1e-6'"},
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
  // A key of the compact mode only is read in a REFIT_MODE compact
  // configuration (REFIT_MODE dense refuses it: the mode rule), the others
  // in a dense one.
  const auto compact_key = [](const std::string &key) {
    return key.rfind("REFIT_COMPACT_", 0) == 0 ||
           key.rfind("REFIT_CURVE_", 0) == 0;
  };
  const auto with = [&](const std::string &key,
                        const std::vector<std::string> &tokens) {
    Context c = compact_key(key) ? compact_refit_context("absent.eam.alloy")
                                 : dense_refit_context("absent.eam.alloy");
    // a curve knob is read under REFIT_OBJECTIVE curve only (the rule
    // refuses it under energy, before its value is read)
    if (key.rfind("REFIT_CURVE_", 0) == 0)
      c.add("REFIT_OBJECTIVE", std::string("curve"));
    // REFIT_COMPACT_OPT_EVERY and REFIT_COMPACT_INSERT_TOPK are read by the
    // knot families only (the rule refuses them under compact_refit_context's
    // Blip5, before their value is read; until it did, these cases ran on
    // Blip5)
    if (key == "REFIT_COMPACT_OPT_EVERY" || key == "REFIT_COMPACT_INSERT_TOPK") {
      c.remove("REFIT_BASIS");
      c.add("REFIT_BASIS", std::string("Knot5"));
    }
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
  // REFIT_CURVE_RELAX under REFIT_OBJECTIVE energy: the rule refuses it
  // before its value is read (an earlier fix checked its value there, as it
  // was read in curve mode only; until the curve knobs were refused under
  // energy this case expected "tadah refit: REFIT_CURVE_RELAX must be a
  // number in (0, 1], got '1.5'")
  {
    Context c = with("REFIT_CURVE_RELAX", {"1.5"});
    c.remove("REFIT_OBJECTIVE");
    c.add("REFIT_OBJECTIVE", std::string("energy"));
    Task task("refit", c);
    CHECK_THROWS_WITH(RefitEngine().validate(task),
                      "tadah refit: REFIT_CURVE_RELAX has no effect under "
                      "REFIT_OBJECTIVE energy: only REFIT_OBJECTIVE curve "
                      "reads it; remove REFIT_CURVE_RELAX, or set "
                      "REFIT_OBJECTIVE curve");
  }
  // The ends of the ranges are accepted.
  for (const auto &[key, tok] :
       std::vector<std::pair<std::string, std::string>>{
           {"REFIT_COMPACT_SEED", "3000000000"},
           {"REFIT_COMPACT_SEED", "4294967295"},
           {"REFIT_COMPACT_SEED", "0"},
           {"REFIT_COMPACT_JITTERS", "0"},
           {"REFIT_COMPACT_JITTERS", "2147483646"},
           {"REFIT_COMPACT_NMAX", "536870911"},
           {"REFIT_MAX_ESCALATIONS", "0"},
           {"REFIT_CURVE_DERIV_WEIGHT", "0"},
           {"REFIT_CURVE_RELAX", "1"},
           {"REFIT_REACH_SMIN", "1"},
           {"REFIT_REACH_SMIN", "0.1"},
           {"N2B", "1024"},
           {"REFIT_TOL_AMBIENT", ".5"},
           {"REFIT_R_LO_FRAC", "0.15"}}) {
    INFO(key << " " << tok);
    Task task("refit", with(key, {tok}));
    // A compact mode key: the numbers pass, and this engine, which has no
    // compact fitter, then refuses REFIT_MODE compact (HPO's engine accepts
    // them: test_compact_refit_stages.cpp).
    if (compact_key(key))
      CHECK_THROWS_WITH(RefitEngine().validate(task), no_compact_fitter());
    else
      CHECK_NOTHROW(RefitEngine().validate(task));
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: an OUTPUT_GRID_SCALE whose refit.eam grid a "
                 "setfl table cannot have is refused before any fitting",
                 "[refit][strict_numbers]") {
  /* Pins: run() builds the refit.eam grid (EAMSetflIO::scaled_grid) right
   * after it reads the EAM file, before any fitting: an OUTPUT_GRID_SCALE
   * whose scaled point counts fall below 3 or above 2147483647 throws
   * naming the key, the token and the counts, and writes nothing (no
   * pot.tadah, refit.eam or refit_opt.log); a scale that gives 3 points
   * (400 x 0.00625 = 2.5) writes a 3-point refit.eam.
   * Before the fix: the grid was built after the fit, when pot.tadah had
   * been written, so the run fails late with scaled_grid's own message
   * and leaves pot.tadah (before scaled_grid refused such a scale it
   * clamped the counts to 2: exit 0 and a 2-point refit.eam). */
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);  // nr = nrho = 400
  const std::string needs =
      " points (the input table has 400 and 400); a setfl table needs 3 to "
      "2147483647 points (the spline that LAMMPS and tadah fit through it "
      "reads its third point)";
  for (const auto &[tok, shown, counts] :
       std::vector<std::tuple<std::string, std::string, std::string>>{
           {"0.0001", "0.0001", "nrho = 0 and nr = 0"},
           {"1e7", "10000000", "nrho = 4000000000 and nr = 4000000000"}}) {
    INFO("OUTPUT_GRID_SCALE " << tok);
    for (const char *f : {"pot.tadah", "refit.eam", "refit_opt.log"})
      fs::remove(f);
    Context c = dense_refit_context("synth_fcc.eam.alloy");
    c.add("OUTPUT_GRID_SCALE", tok);
    Task task("refit", c);
    CHECK_THROWS_WITH(RefitEngine().run(task),
                      "tadah refit: OUTPUT_GRID_SCALE '" + tok +
                          "' for refit.eam: EAMSetflIO: a grid scale of " +
                          shown + " gives " + counts + needs);
    CHECK_FALSE(fs::exists("pot.tadah"));
    CHECK_FALSE(fs::exists("refit.eam"));
    CHECK_FALSE(fs::exists("refit_opt.log"));
  }
  Context c = dense_refit_context("synth_fcc.eam.alloy");
  c.add("OUTPUT_GRID_SCALE", std::string("0.00625"));
  (void)run_refit(c);
  tadah::models::EAMSetflIO setfl;
  setfl.read_setfl("refit.eam");
  CHECK(setfl.nr() == 3);
  CHECK(setfl.nrho() == 3);
}

// ---------------------------------------------------------------------------
//  The mode rule: a key of the other REFIT_MODE is refused
// ---------------------------------------------------------------------------
namespace {
// validate() and run() both throw `message` for `c` (run() before it reads
// the EAM file, which is absent here).
void check_refused(const Context &c, const std::string &message) {
  Task task("refit", c);
  CHECK_THROWS_WITH(RefitEngine().validate(task), message);
  CHECK_THROWS_WITH(RefitEngine().run(task), message);
}

// A stage block as CORE's config parser stores it: one token per value.
void add_block(Context &c, const std::string &key, const std::string &body) {
  std::istringstream iss(body);
  for (std::string t; iss >> t;) c.add(key, t);
}

// The keys only REFIT_MODE compact reads, each with a valid value.
const std::vector<std::pair<std::string, std::string>> kCompactOnlyKeys = {
    {"REFIT_COMPACT_NMAX", "3"},        {"REFIT_COMPACT_SAMPLES", "400"},
    {"REFIT_COMPACT_OPT_EVERY", "2"},   {"REFIT_COMPACT_STALL", "4"},
    {"REFIT_COMPACT_INSERT_TOPK", "2"}, {"REFIT_COMPACT_JITTERS", "1"},
    {"REFIT_COMPACT_SEED", "7"},        {"REFIT_COMPACT_SHARE", "0.4"},
    {"REFIT_COMPACT_MERGE", "true"},    {"REFIT_CURVE_DERIV_WEIGHT", "0.1"},
    {"REFIT_CURVE_WEIGHTING", "false"}, {"REFIT_CURVE_RELAX", "0.1"}};

// The stage blocks only REFIT_MODE compact reads. (A top-level OPTIMIZER
// block, the grammar before them, is read by neither mode: kTopLevelOptimizer.)
const std::vector<std::pair<std::string, std::string>> kCompactBlocks = {
    {"INIT", "PHI XMIN PIN"},
    {"EXPLORE", "MAX_ADD 4"},
    // keys REFIT_MODE compact refuses (THREADS, a negative GTOL): the block
    // is refused as a whole, before its contents are read
    {"REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 5 THREADS 4 GTOL -1 "
               "ENDOPTIMIZER"},
    {"FUSE", "ENABLED false"}};

// The refusal of a top-level OPTIMIZER block in either REFIT_MODE: the
// migration text of the compact fitter's parser (HPO's RefitParser::parse,
// whose own test pins it), to which REFIT_MODE dense adds its last sentence.
const std::string kTopLevelOptimizer =
    "tadah refit: a top-level OPTIMIZER block is no longer supported. The "
    "stage pipeline is now INIT -> EXPLORE -> REFINE -> FUSE, each its own "
    "block, with the OPTIMIZER block NESTED inside EXPLORE/REFINE/FUSE. "
    "Migration: GLOBAL folded into EXPLORE rung 0 (set its global ALGO in "
    "EXPLORE's OPTIMIZER, with an INNER local); POLISH -> REFINE; JOINT -> "
    "FUSE. The Refitting page of the documentation has examples (Stage "
    "blocks).";
const std::string kTopLevelOptimizerDense =
    kTopLevelOptimizer +
    " Under REFIT_MODE dense, which reads no stage block, remove the "
    "OPTIMIZER block.";

// The keys only REFIT_MODE dense reads, each with a valid value.
const std::vector<std::pair<std::string, std::string>> kDenseOnlyKeys = {
    {"REFIT_LEVEL", "accurate"},  {"N2B", "50"},
    {"NRHO_BASIS", "40"},         {"NF_BASIS", "30"},
    {"REFIT_MAX_ESCALATIONS", "9"}, {"REFIT_PLACEMENT", "uniform"},
    {"REFIT_CONSTRAIN_F0", "false"}};
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: a key of the other REFIT_MODE is refused, "
                 "naming the key and the mode that reads it",
                 "[refit][mode_keys]") {
  /* Pins: RefitEngine::validate(), and run() before it reads the EAM file,
   * refuse a key that only the other REFIT_MODE reads, naming it and both
   * modes ("tadah refit: N2B has no effect under REFIT_MODE compact (the
   * default): only REFIT_MODE dense reads it; remove N2B, or set
   * REFIT_MODE dense"; "(the default)" when there is no REFIT_MODE key):
   *  - REFIT_MODE dense: every REFIT_COMPACT_* key, REFIT_CURVE_WEIGHTING,
   *    REFIT_CURVE_RELAX, REFIT_CURVE_DERIV_WEIGHT, the INIT, EXPLORE,
   *    REFINE and FUSE blocks ("the REFINE block has no effect ..."; a
   *    REFINE block with keys compact mode refuses is refused as a whole),
   *    and REFIT_OBJECTIVE curve in any letter case; REFIT_OBJECTIVE energy
   *    is accepted, and another value is refused with the message compact
   *    mode gives it (a top-level OPTIMIZER block, which neither mode
   *    reads: the next test case);
   *  - REFIT_MODE compact, explicit or the default: REFIT_LEVEL, N2B,
   *    NRHO_BASIS, NF_BASIS, REFIT_MAX_ESCALATIONS, REFIT_PLACEMENT and
   *    REFIT_CONSTRAIN_F0 (read by the dense fit only);
   *  - REFIT_MODE compact under REFIT_OBJECTIVE energy (explicit or the
   *    default): REFIT_CURVE_WEIGHTING ("... has no effect under
   *    REFIT_OBJECTIVE energy (the default): only REFIT_OBJECTIVE curve
   *    reads it; ..."); under curve it passes the rule, after which this
   *    engine, which has no compact fitter, refuses REFIT_MODE compact
   *    (REFIT_CURVE_RELAX and REFIT_CURVE_DERIV_WEIGHT: the test case
   *    "every curve knob is refused under REFIT_OBJECTIVE energy");
   *  - a key the task factory stored as its blueprint default
   *    (KeyProvenance: Default; REFIT_COMPACT_MERGE, REFIT_CURVE_*,
   *    REFIT_OBJECTIVE energy under dense, REFIT_CONSTRAIN_F0 under
   *    compact) is not the configuration's and passes; the same key from
   *    the configuration file is refused.
   * Before the fix: validate() checked none of this and did not throw;
   * run() ignored the keys of the other mode (dense never parsed the stage
   * blocks; compact never read the dense keys and, in this engine, fell
   * back to the dense fit) and failed reading the absent EAM file instead
   * ("EAMSetflIO: setfl file does not exist: absent.eam.alloy");
   * REFIT_CURVE_WEIGHTING was ignored under REFIT_OBJECTIVE energy. The
   * blueprint defaults passed (a pin). */
  ScopedTempDir tmp;
  SECTION("REFIT_MODE dense refuses the keys and blocks of compact mode") {
    for (const auto &[key, value] : kCompactOnlyKeys) {
      INFO(key << " " << value);
      Context c = dense_refit_context("absent.eam.alloy");
      c.add(key, value);
      check_refused(c, no_effect_in_mode(key, "dense"));
    }
    for (const auto &[key, body] : kCompactBlocks) {
      INFO(key << " block: " << body);
      Context c = dense_refit_context("absent.eam.alloy");
      add_block(c, key, body);
      check_refused(c, no_effect_in_mode("the " + key + " block", "dense"));
    }
    for (const char *curve : {"curve", "CURVE"}) {
      INFO("REFIT_OBJECTIVE " << curve);
      Context c = dense_refit_context("absent.eam.alloy");
      c.add("REFIT_OBJECTIVE", std::string(curve));
      check_refused(c, no_effect_in_mode("REFIT_OBJECTIVE curve", "dense"));
    }
    Context energy = dense_refit_context("absent.eam.alloy");
    energy.add("REFIT_OBJECTIVE", std::string("energy"));
    Task task("refit", energy);
    CHECK_NOTHROW(RefitEngine().validate(task));
    Context other = dense_refit_context("absent.eam.alloy");
    other.add("REFIT_OBJECTIVE", std::string("Shape"));
    check_refused(other, "tadah refit: REFIT_OBJECTIVE must be 'energy' "
                         "(default) or 'curve'; got 'shape'");
  }
  SECTION("REFIT_MODE compact refuses the keys of dense mode") {
    for (const bool defaulted : {false, true}) {
      for (const auto &[key, value] : kDenseOnlyKeys) {
        INFO(key << " " << value << (defaulted ? " (default mode)" : ""));
        Context c = compact_refit_context("absent.eam.alloy");
        if (defaulted) c.remove("REFIT_MODE");
        c.add(key, value);
        check_refused(c, no_effect_in_mode(key, "compact", defaulted));
      }
    }
  }
  SECTION("REFIT_CURVE_WEIGHTING is refused under REFIT_OBJECTIVE energy") {
    const std::string weighting =
        "tadah refit: REFIT_CURVE_WEIGHTING has no effect under "
        "REFIT_OBJECTIVE energy";
    const std::string rest =
        ": only REFIT_OBJECTIVE curve reads it; remove REFIT_CURVE_WEIGHTING, "
        "or set REFIT_OBJECTIVE curve";
    for (const char *value : {"false", "true"}) {
      INFO("REFIT_CURVE_WEIGHTING " << value);
      Context c = compact_refit_context("absent.eam.alloy");
      c.add("REFIT_CURVE_WEIGHTING", std::string(value));
      check_refused(c, weighting + " (the default)" + rest);
      c.add("REFIT_OBJECTIVE", std::string("energy"));
      check_refused(c, weighting + rest);
      // under curve the rule passes; this engine then refuses compact mode
      c.remove("REFIT_OBJECTIVE");
      c.add("REFIT_OBJECTIVE", std::string("curve"));
      check_refused(c, no_compact_fitter());
    }
  }
  SECTION("a blueprint default the task factory stored is not refused") {
    using tadah::mlip::KeyOrigin;
    using tadah::mlip::KeyProvenance;
    // what TaskFactory stores for every refit key with a blueprint default
    const std::vector<std::pair<std::string, std::string>> defaults = {
        {"REFIT_COMPACT_MERGE", "auto"},     {"REFIT_CURVE_WEIGHTING", "true"},
        {"REFIT_CURVE_RELAX", "0.05"},       {"REFIT_CURVE_DERIV_WEIGHT", "0.0"},
        {"REFIT_OBJECTIVE", "energy"},       {"REFIT_CONSTRAIN_F0", "true"}};
    const auto task_with = [&](const Context &file, bool as_defaults) {
      Context c = file;
      std::vector<std::string> defaulted;
      Context from_file = file;
      for (const auto &[key, value] : defaults) {
        if (c.has_stored(key)) continue;
        c.add(key, value);
        if (as_defaults)
          defaulted.push_back(key);
        else
          from_file.add(key, value);
      }
      Task task("refit", c);
      task.setKeyProvenance(KeyProvenance(Context(), from_file, "config.refit",
                                          KeyOrigin::Kind::ConfigFile,
                                          defaulted));
      return task;
    };
    // dense: the compact defaults pass
    const Task dense = task_with(dense_refit_context("absent.eam.alloy"), true);
    CHECK_NOTHROW(RefitEngine().validate(dense));
    // compact: REFIT_CONSTRAIN_F0's default passes (this engine then refuses
    // compact mode), and so do REFIT_CURVE_WEIGHTING's under energy
    const Task compact =
        task_with(compact_refit_context("absent.eam.alloy"), true);
    CHECK_THROWS_WITH(RefitEngine().validate(compact), no_compact_fitter());
    // written in the configuration file, the same keys are refused
    const Task dense_file =
        task_with(dense_refit_context("absent.eam.alloy"), false);
    CHECK_THROWS_WITH(RefitEngine().validate(dense_file),
                      no_effect_in_mode("REFIT_COMPACT_MERGE", "dense"));
    const Task compact_file =
        task_with(compact_refit_context("absent.eam.alloy"), false);
    CHECK_THROWS_WITH(RefitEngine().validate(compact_file),
                      no_effect_in_mode("REFIT_CONSTRAIN_F0", "compact"));
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: every curve knob is refused under "
                 "REFIT_OBJECTIVE energy, and REFIT_CURVE_RELAX under "
                 "REFIT_CURVE_WEIGHTING false",
                 "[refit][mode_keys]") {
  /* Pins: under REFIT_MODE compact, RefitEngine::validate() and run()
   * (before the EAM file is read) refuse each of REFIT_CURVE_WEIGHTING,
   * REFIT_CURVE_RELAX and REFIT_CURVE_DERIV_WEIGHT under REFIT_OBJECTIVE
   * energy, written or the default ("tadah refit: REFIT_CURVE_RELAX has no
   * effect under REFIT_OBJECTIVE energy (the default): only REFIT_OBJECTIVE
   * curve reads it; remove REFIT_CURVE_RELAX, or set REFIT_OBJECTIVE
   * curve"), before its value is read (REFIT_CURVE_RELAX 1.5, out of its
   * range, gets this refusal too); under curve the three pass the rule
   * (this engine, which has no compact fitter, then refuses REFIT_MODE
   * compact), except REFIT_CURVE_RELAX with REFIT_CURVE_WEIGHTING false,
   * the floor of a weighting that is off: "tadah refit: REFIT_CURVE_RELAX
   * has no effect under REFIT_CURVE_WEIGHTING false: only
   * REFIT_CURVE_WEIGHTING true (the default) reads it; remove
   * REFIT_CURVE_RELAX, or set REFIT_CURVE_WEIGHTING true"
   * (REFIT_CURVE_DERIV_WEIGHT passes there, and REFIT_CURVE_RELAX with
   * REFIT_CURVE_WEIGHTING true). The blueprint defaults of the three
   * (KeyProvenance Default) pass under energy.
   * Before the fix: REFIT_CURVE_RELAX and REFIT_CURVE_DERIV_WEIGHT passed
   * the rule under energy (no effect there, their values checked: the
   * earlier test case pinned this), so their refusal is not thrown and this
   * engine refuses REFIT_MODE compact instead ("tadah refit: REFIT_MODE
   * compact needs the compact fitter, ..."; REFIT_CURVE_RELAX 1.5: "tadah
   * refit: REFIT_CURVE_RELAX must be a number in (0, 1], got '1.5'"), and
   * REFIT_CURVE_RELAX passed under REFIT_CURVE_WEIGHTING false; the
   * REFIT_CURVE_WEIGHTING cases and the curve cases pass as well (a pin). */
  ScopedTempDir tmp;
  const auto refused_under_energy = [](const std::string &key,
                                       bool defaulted) {
    return "tadah refit: " + key + " has no effect under REFIT_OBJECTIVE "
           "energy" + (defaulted ? " (the default)" : "") +
           ": only REFIT_OBJECTIVE curve reads it; remove " + key +
           ", or set REFIT_OBJECTIVE curve";
  };
  for (const auto &[key, value] :
       std::vector<std::pair<std::string, std::string>>{
           {"REFIT_CURVE_WEIGHTING", "false"},
           {"REFIT_CURVE_RELAX", "0.1"},
           {"REFIT_CURVE_RELAX", "1.5"},
           {"REFIT_CURVE_DERIV_WEIGHT", "0.1"}}) {
    for (const bool defaulted : {true, false}) {
      INFO(key << " " << value
               << (defaulted ? "" : " (REFIT_OBJECTIVE energy)"));
      Context c = compact_refit_context("absent.eam.alloy");
      c.add(key, value);
      if (!defaulted) c.add("REFIT_OBJECTIVE", std::string("energy"));
      check_refused(c, refused_under_energy(key, defaulted));
    }
  }
  const std::string relax_unweighted =
      "tadah refit: REFIT_CURVE_RELAX has no effect under "
      "REFIT_CURVE_WEIGHTING false: only REFIT_CURVE_WEIGHTING true (the "
      "default) reads it; remove REFIT_CURVE_RELAX, or set "
      "REFIT_CURVE_WEIGHTING true";
  {
    Context c = compact_refit_context("absent.eam.alloy");
    c.add("REFIT_OBJECTIVE", std::string("curve"));
    c.add("REFIT_CURVE_WEIGHTING", std::string("false"));
    c.add("REFIT_CURVE_RELAX", std::string("0.1"));
    check_refused(c, relax_unweighted);
    c.remove("REFIT_CURVE_RELAX");
    c.add("REFIT_CURVE_DERIV_WEIGHT", std::string("0.1"));
    check_refused(c, no_compact_fitter());
  }
  for (const char *weighting : {static_cast<const char *>(nullptr), "true"}) {
    INFO("REFIT_CURVE_WEIGHTING " << (weighting ? weighting : "(default)"));
    Context c = compact_refit_context("absent.eam.alloy");
    c.add("REFIT_OBJECTIVE", std::string("curve"));
    if (weighting) c.add("REFIT_CURVE_WEIGHTING", std::string(weighting));
    c.add("REFIT_CURVE_RELAX", std::string("0.1"));
    c.add("REFIT_CURVE_DERIV_WEIGHT", std::string("0.1"));
    check_refused(c, no_compact_fitter());
  }
  {
    // the blueprint defaults the task factory stores pass under energy
    using tadah::mlip::KeyOrigin;
    using tadah::mlip::KeyProvenance;
    Context c = compact_refit_context("absent.eam.alloy");
    const Context file = c;
    c.add("REFIT_CURVE_WEIGHTING", std::string("true"));
    c.add("REFIT_CURVE_RELAX", std::string("0.05"));
    c.add("REFIT_CURVE_DERIV_WEIGHT", std::string("0.0"));
    c.add("REFIT_OBJECTIVE", std::string("energy"));
    Task task("refit", c);
    task.setKeyProvenance(KeyProvenance(
        Context(), file, "config.refit", KeyOrigin::Kind::ConfigFile,
        {"REFIT_CURVE_WEIGHTING", "REFIT_CURVE_RELAX",
         "REFIT_CURVE_DERIV_WEIGHT", "REFIT_OBJECTIVE"}));
    CHECK_THROWS_WITH(RefitEngine().validate(task), no_compact_fitter());
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: a key of the knot families is refused under "
                 "the Blip families",
                 "[refit][mode_keys]") {
  /* Pins: under REFIT_MODE compact, RefitEngine::validate() and run()
   * (before the EAM file is read) refuse each of REFIT_COMPACT_MERGE,
   * REFIT_COMPACT_OPT_EVERY and REFIT_COMPACT_INSERT_TOPK, which the
   * compact fitter reads for the one-sided knot families only (the
   * coalescence guard, the insertion-only rungs, the top-k insertion),
   * when REFIT_BASIS is a Blip family, in each spelling tadah refit fits as
   * one (Blip, blip, Blip5, BLIP5, blip5, M6), naming the key and the basis
   * as written: "tadah refit: REFIT_COMPACT_MERGE has no effect under
   * REFIT_BASIS Blip5: only the knot families (REFIT_BASIS Knot5, the
   * compact default, Knot3 or KnotMix) read it; remove REFIT_COMPACT_MERGE,
   * or set REFIT_BASIS Knot5"; REFIT_COMPACT_MERGE auto written in the file
   * too. Under a knot family (Knot5, knot5, Knot3, KnotMix, Mixed) and
   * without REFIT_BASIS (the compact default, Knot5) the three pass the
   * rule, and this engine, which has no compact fitter, then refuses
   * REFIT_MODE compact; so does REFIT_COMPACT_MERGE auto as the blueprint
   * default the task factory stores (KeyProvenance Default) under Blip5.
   * Before the fix: the three passed the rule under the Blip families too
   * (the compact fitter read and ignored them), so their refusal is not
   * thrown and this engine refuses REFIT_MODE compact instead ("tadah
   * refit: REFIT_MODE compact needs the compact fitter, ..."); the knot
   * and default cases pass as well (a pin). */
  ScopedTempDir tmp;
  const auto refused = [](const std::string &key, const std::string &basis) {
    return "tadah refit: " + key + " has no effect under REFIT_BASIS " +
           basis + ": only the knot families (REFIT_BASIS Knot5, the "
           "compact default, Knot3 or KnotMix) read it; remove " + key +
           ", or set REFIT_BASIS Knot5";
  };
  const std::vector<std::pair<std::string, std::string>> knot_keys = {
      {"REFIT_COMPACT_MERGE", "true"},
      {"REFIT_COMPACT_MERGE", "auto"},
      {"REFIT_COMPACT_OPT_EVERY", "2"},
      {"REFIT_COMPACT_INSERT_TOPK", "2"}};
  const auto with_basis = [](const char *basis) {
    Context c = compact_refit_context("absent.eam.alloy");
    c.remove("REFIT_BASIS");
    if (basis) c.add("REFIT_BASIS", std::string(basis));
    return c;
  };
  for (const auto &[key, value] : knot_keys) {
    for (const char *basis : {"Blip", "blip", "Blip5", "BLIP5", "blip5",
                              "M6"}) {
      INFO(key << " " << value << " under REFIT_BASIS " << basis);
      Context c = with_basis(basis);
      c.add(key, value);
      check_refused(c, refused(key, basis));
    }
    for (const char *basis : {static_cast<const char *>(nullptr), "Knot5",
                              "knot5", "Knot3", "KnotMix", "Mixed"}) {
      INFO(key << " " << value << " under REFIT_BASIS "
               << (basis ? basis : "(the default)"));
      Context c = with_basis(basis);
      c.add(key, value);
      check_refused(c, no_compact_fitter());
    }
  }
  // the default the task factory stores passes under a Blip family
  using tadah::mlip::KeyOrigin;
  using tadah::mlip::KeyProvenance;
  Context c = with_basis("Blip5");
  const Context file = c;
  c.add("REFIT_COMPACT_MERGE", std::string("auto"));
  Task task("refit", c);
  task.setKeyProvenance(KeyProvenance(Context(), file, "config.refit",
                                      KeyOrigin::Kind::ConfigFile,
                                      {"REFIT_COMPACT_MERGE"}));
  CHECK_THROWS_WITH(RefitEngine().validate(task), no_compact_fitter());
  CHECK_THROWS_WITH(RefitEngine().run(task), no_compact_fitter());
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: a top-level OPTIMIZER block is refused in "
                 "either REFIT_MODE with the migration text",
                 "[refit][mode_keys]") {
  /* Pins: RefitEngine::validate(), and run() before it reads the EAM file,
   * refuse a top-level OPTIMIZER block (the grammar before the stage
   * blocks), which neither REFIT_MODE reads, with the migration text of the
   * compact fitter's parser: "tadah refit: a top-level OPTIMIZER block is
   * no longer supported. The stage pipeline is now INIT -> EXPLORE ->
   * REFINE -> FUSE, ... The Refitting page of the documentation has
   * examples (Stage blocks)."; under REFIT_MODE dense it ends " Under
   * REFIT_MODE dense, which reads no stage block, remove the OPTIMIZER
   * block.". Under REFIT_MODE compact, written or the default, it comes
   * before this engine's refusal of compact mode (it has no compact
   * fitter); in either mode before the refusal of the other mode's keys
   * (REFIT_COMPACT_NMAX under dense, N2B under compact).
   * Before the fix: the mode rule refused the block under REFIT_MODE dense
   * as a key of compact mode, "tadah refit: the OPTIMIZER block has no
   * effect under REFIT_MODE dense: only REFIT_MODE compact reads it; remove
   * the OPTIMIZER block, or set REFIT_MODE compact", although compact mode
   * refuses it too, and REFIT_COMPACT_NMAX was refused before it; under
   * compact it passed the rule, so this engine refused compact mode
   * ("tadah refit: REFIT_MODE compact needs the compact fitter, ...") and
   * N2B was refused first (HPO's engine refused the block in
   * RefitParser::parse only, after it had read the table). On v1-dev,
   * before the mode rule, validate() accepted the block in either mode and
   * run() failed reading the absent EAM file. */
  ScopedTempDir tmp;
  {
    Context c = dense_refit_context("absent.eam.alloy");
    add_block(c, "OPTIMIZER", "GLOBAL ALGO MLSL ENDGLOBAL");
    check_refused(c, kTopLevelOptimizerDense);
    c.add("REFIT_COMPACT_NMAX", std::string("3"));
    check_refused(c, kTopLevelOptimizerDense);
  }
  for (const bool defaulted : {false, true}) {
    INFO((defaulted ? "REFIT_MODE compact (the default)"
                    : "REFIT_MODE compact"));
    Context c = compact_refit_context("absent.eam.alloy");
    if (defaulted) c.remove("REFIT_MODE");
    add_block(c, "OPTIMIZER", "GLOBAL ALGO MLSL ENDGLOBAL");
    check_refused(c, kTopLevelOptimizer);
    c.add("N2B", std::string("50"));
    check_refused(c, kTopLevelOptimizer);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: REFIT_MODE is compact or dense in any letter "
                 "case, and another value is refused",
                 "[refit][mode_keys]") {
  /* Pins: REFIT_MODE is read in any letter case, as REFIT_OBJECTIVE is:
   * 'Dense' and 'DENSE' run the dense fit (refit_opt.log: mode=dense, no
   * WARNING), 'COMPACT' is compact mode (which this engine, without a
   * compact fitter, refuses); another value is refused in validate() and
   * run(), before the EAM file is read ("tadah refit: REFIT_MODE must be
   * 'compact' (the default) or 'dense', got 'sparse'"), and writes nothing.
   * Before the fix: only 'compact' and 'dense' were known; any other
   * value, 'Dense' and 'COMPACT' included, ran the dense fit after
   * "WARNING: tadah refit: unknown REFIT_MODE '<value>' — using dense."
   * (validate() accepted it and run() wrote pot.tadah). */
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
  for (const char *dense : {"Dense", "DENSE"}) {
    INFO("REFIT_MODE " << dense);
    Context c = dense_refit_context("synth_fcc.eam.alloy");
    c.remove("REFIT_MODE");
    c.add("REFIT_MODE", std::string(dense));
    const RefitOutputs out = run_refit(c);
    INFO(out.log);
    CHECK(out.log.find("unknown REFIT_MODE") == std::string::npos);
    CHECK(slurp("refit_opt.log").find("mode=dense") != std::string::npos);
  }
  {
    Context c = compact_refit_context("synth_fcc.eam.alloy");
    c.remove("REFIT_MODE");
    c.add("REFIT_MODE", std::string("COMPACT"));
    check_refused(c, no_compact_fitter());
  }
  for (const char *other : {"sparse", "compact2"}) {
    INFO("REFIT_MODE " << other);
    for (const char *f : {"pot.tadah", "refit.eam", "refit_opt.log"})
      fs::remove(f);
    Context c = dense_refit_context("synth_fcc.eam.alloy");
    c.remove("REFIT_MODE");
    c.add("REFIT_MODE", std::string(other));
    check_refused(c, std::string("tadah refit: REFIT_MODE must be 'compact' "
                                 "(the default) or 'dense', got '") +
                         other + "'");
    CHECK_FALSE(fs::exists("pot.tadah"));
    CHECK_FALSE(fs::exists("refit_opt.log"));
  }
}

TEST_CASE_METHOD(PTFixture,
                 "tadah refit: REFIT_MODE compact without a compact fitter is "
                 "refused before the table is read",
                 "[refit][mode_keys]") {
  /* Pins: RefitEngine without a compact fitter (the engine of a tadah built
   * without TADAH_ENABLE_NFP or TADAH_ENABLE_CERES; HPO's engine has one)
   * refuses REFIT_MODE compact, written or the default, in validate() and
   * in run() before it reads the EAM file, in every RHOFORM and with the
   * default or a chosen basis: "tadah refit: REFIT_MODE compact (the
   * default) needs the compact fitter, which this tadah was built without
   * (it needs TADAH_ENABLE_NFP and TADAH_ENABLE_CERES); set REFIT_MODE
   * dense"; it writes nothing (no pot.tadah, refit.eam or refit_opt.log).
   * Before the fix: validate() accepted it, and run() fitted the table in
   * dense mode after "WARNING: tadah refit: REFIT_MODE compact requires
   * the HPO build (Ceres); falling back to dense fitting." (a defaulted
   * Knot5 basis became Blip) and wrote pot.tadah. */
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
  for (const char *form : {"SIGNED", "NONNEG", "MONOTONE"}) {
    for (const char *basis : {static_cast<const char *>(nullptr), "Blip5"}) {
      for (const bool defaulted : {false, true}) {
        // MONOTONE is not defined for the defaulted Knot5 (a refusal that
        // needs the table, after this one)
        INFO("RHOFORM " << form << " REFIT_BASIS "
                        << (basis ? basis : "(default)")
                        << (defaulted ? " (default mode)" : ""));
        for (const char *f : {"pot.tadah", "refit.eam", "refit_opt.log"})
          fs::remove(f);
        Context c = compact_refit_context("synth_fcc.eam.alloy", form);
        c.remove("REFIT_BASIS");
        if (basis) c.add("REFIT_BASIS", std::string(basis));
        if (defaulted) c.remove("REFIT_MODE");
        check_refused(c, no_compact_fitter(defaulted));
        CHECK_FALSE(fs::exists("pot.tadah"));
        CHECK_FALSE(fs::exists("refit.eam"));
        CHECK_FALSE(fs::exists("refit_opt.log"));
      }
    }
  }
}

// ---------------------------------------------------------------------------
//  refit_range_report.dat describes the fit that ran
// ---------------------------------------------------------------------------
namespace {
// One function row of refit_range_report.dat ("F   lo hi wlo whi  # note").
struct RangeRow {
  double full_lo = 0, full_hi = 0, win_lo = 0, win_hi = 0;
  std::string note;  ///< the text after "# "
};
RangeRow range_row(const std::string &function) {
  std::istringstream in(slurp("refit_range_report.dat"));
  for (std::string line; std::getline(in, line);) {
    std::istringstream ls(line);
    std::string name;
    if (!(ls >> name) || name != function) continue;
    RangeRow row;
    ls >> row.full_lo >> row.full_hi >> row.win_lo >> row.win_hi;
    const auto hash = line.find("# ");
    if (hash != std::string::npos) row.note = line.substr(hash + 2);
    return row;
  }
  FAIL("refit_range_report.dat has no " << function << " row");
  return {};
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit dense: refit_range_report.dat's F row describes "
                 "the dense fit's basis placement",
                 "[refit][engine][integration]") {
  /* Pins: after a dense fit with REFIT_PLACEMENT window (the default) the F
   * row of refit_range_report.dat has the fit window [0, rho_reach], below
   * the table's rhomax (60), and the note "tail keeps a small uniform
   * basis; control: REFIT_REACH_MARGIN"; with REFIT_PLACEMENT uniform (one
   * uniform basis over the table, no reachable window) its window is the
   * full range [0, 60] and its note "one uniform basis fitted on the full
   * range (REFIT_PLACEMENT uniform); control: REFIT_PLACEMENT", and the
   * report names no tail basis.
   * Before the fix: the window case passes as well (a pin); under uniform
   * the note was the window placement's "tail keeps a small uniform basis;
   * control: REFIT_REACH_MARGIN", a tail the uniform basis does not have. */
  ScopedTempDir tmp;
  write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
  {
    (void)run_refit(dense_refit_context("synth_fcc.eam.alloy"));
    const RangeRow F = range_row("F");
    CHECK(F.full_lo == 0.0);
    CHECK(F.full_hi == Approx(60.0));
    CHECK(F.win_lo == 0.0);
    CHECK(F.win_hi > 0.0);
    CHECK(F.win_hi < 0.999 * F.full_hi);
    CHECK(F.note == "tail keeps a small uniform basis; control: "
                    "REFIT_REACH_MARGIN");
  }
  {
    Context c = dense_refit_context("synth_fcc.eam.alloy");
    c.add("REFIT_PLACEMENT", std::string("uniform"));
    (void)run_refit(c);
    const RangeRow F = range_row("F");
    CHECK(F.win_lo == 0.0);
    CHECK(F.win_hi == F.full_hi);
    CHECK(F.note == "one uniform basis fitted on the full range "
                    "(REFIT_PLACEMENT uniform); control: REFIT_PLACEMENT");
    CHECK(slurp("refit_range_report.dat").find("small uniform basis") ==
          std::string::npos);
  }
}

namespace {
// The rows of a refit_curve_<fn>.dat file: x and its in_window flag.
std::vector<std::pair<double, int>> curve_window_flags(const std::string &f) {
  std::vector<std::pair<double, int>> rows;
  std::istringstream in(slurp(f));
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::vector<double> cols;
    for (double v; ls >> v;) cols.push_back(v);
    REQUIRE(cols.size() == 9);
    rows.emplace_back(cols[0], static_cast<int>(cols[8]));
  }
  return rows;
}

// Every row of refit_curve_<fn>.dat is flagged in_window exactly when its x
// lies in [lo, hi] (a row within 1e-9 of an edge is not judged).
void check_curve_window(const std::string &f, double lo, double hi) {
  const auto rows = curve_window_flags(f);
  REQUIRE(rows.size() > 100);
  int wrong = 0;
  double first_wrong = 0.0;
  for (const auto &[x, flag] : rows) {
    if (std::fabs(x - lo) < 1e-9 || std::fabs(x - hi) < 1e-9) continue;
    const int want = (x >= lo && x <= hi) ? 1 : 0;
    if (flag != want && wrong++ == 0) first_wrong = x;
  }
  INFO(f << ": window [" << lo << ", " << hi << "], first wrong flag at x = "
         << first_wrong);
  CHECK(wrong == 0);
}

// The value after `mark` in refit_range_report.dat ("r_1nn-compressed=").
double report_value(const std::string &mark) {
  const std::string report = slurp("refit_range_report.dat");
  const auto at = report.find(mark);
  REQUIRE(at != std::string::npos);
  return std::stod(report.substr(at + mark.size()));
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "tadah refit dense: refit_range_report.dat and the curve "
                 "files give the window the dense fit used",
                 "[refit][engine][integration]") {
  /* Pins: after a dense fit (REFIT_PLACEMENT window) the phi and rho rows
   * of refit_range_report.dat have fit_window_lo = the lower edge of the
   * radial window EAMRefit::fit fitted on, min(max(r_lo, r_K), 0.8 r_reach)
   * and never below r_lo (where its samples start), and fit_window_hi =
   * rcut; refit_curve_phi.dat and refit_curve_rho.dat flag in_window
   * exactly the x in [fit_window_lo, rcut], refit_curve_F.dat the rho in
   * [0, F's fit_window_hi]. The synthetic fcc table (a0 3.6 A, rcut 5 A,
   * r_reach = 0.75 a0/sqrt 2 = 1.909 A, its reachability line):
   *  - as written (no wall): fit_window_lo = r_lo = 0.2 rcut = 1 A, the
   *    row's full_lo;
   *  - with a steep pair wall, phi + 100 exp(-(r - 1.6)/0.05), above 30
   *    times its reachable scale up to r = 1.67 A > 0.8 r_reach: 0.8 r_reach
   *    (1.527 A);
   *  - with REFIT_R_LO_FRAC 0.35 (r_lo 1.75 A above 0.8 r_reach): r_lo.
   * Before the fix: fit_window_lo was 0.9 r_reach = 1.718 A in all three
   * (below the row's full_lo of 1.75 A with REFIT_R_LO_FRAC 0.35), and the
   * curve files flagged in_window from 0.9 r_reach, so x in [1, 1.718) is
   * flagged 0 in the first case and [1.527, 1.718) in the second. */
  ScopedTempDir tmp;
  const double rcut = 5.0;
  const double r_reach = 0.75 * 3.6 / std::sqrt(2.0);
  struct Case {
    std::string name;
    std::function<double(double)> phi;
    const char *r_lo_frac;
    double window_lo;
  };
  const std::vector<Case> cases = {
      {"no wall", nullptr, nullptr, 1.0},
      {"a steep pair wall",
       [](double r) {
         const double x = 1.0 - r / 5.0;
         return (x > 0 ? 3.0 * x * x * x : 0.0) +
                100.0 * std::exp(-(r - 1.6) / 0.05);
       },
       nullptr, 0.8 * r_reach},
      {"REFIT_R_LO_FRAC 0.35", nullptr, "0.35", 1.75}};
  for (const Case &cs : cases) {
    INFO(cs.name);
    write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6, 400, 400, 0.0, nullptr,
                    cs.phi);
    Context c = dense_refit_context("synth_fcc.eam.alloy");
    if (cs.r_lo_frac) c.add("REFIT_R_LO_FRAC", std::string(cs.r_lo_frac));
    (void)run_refit(c);
    CHECK(report_value("r_1nn-compressed=") == Approx(r_reach));
    for (const char *fn : {"phi", "rho"}) {
      INFO(fn);
      const RangeRow row = range_row(fn);
      CHECK(row.full_hi == Approx(rcut));
      CHECK(row.win_lo == Approx(cs.window_lo));
      CHECK(row.win_lo >= row.full_lo);
      CHECK(row.win_hi == Approx(rcut));
      check_curve_window(std::string("refit_curve_") + fn + ".dat",
                         cs.window_lo, rcut);
    }
    const RangeRow F = range_row("F");
    check_curve_window("refit_curve_F.dat", 0.0, F.win_hi);
  }
}

/* Pins: the dense fit's word keys are read as written or refused, before the
 * EAM file is read (RefitEngine::validate, and run(): the EAM file here does
 * not exist):
 *  - REFIT_BASIS: a spelling of no basis family ('Blipp', 'BLIPP') is
 *    refused in either REFIT_MODE, naming it and the families; 'BLIP' is
 *    the Blip family, as BLIP5 and KNOT5 are theirs: a dense refit with it
 *    writes the TYPE2B of REFIT_BASIS Blip and no EXPERIMENTAL (Gaussian)
 *    WARNING;
 *  - REFIT_LEVEL in any letter case ('COARSE' is coarse: the same WEIGHTS),
 *    another value ('acurate', 'Fine-ish') refused naming the four;
 *  - REFIT_PLACEMENT in any letter case ('Uniform'), another value
 *    ('unifrom') refused naming window and uniform;
 *  - REFIT_CONSTRAIN_F0, which only the F(0) pin of EMBF0 FREE reads, true
 *    or false, is refused under EMBF0 ZERO (the default, and written),
 *    naming EMBF0 FREE; under EMBF0 FREE it passes, and so does the
 *    blueprint default the task factory stores (KeyOrigin::Default).
 * Before the fix: every one passed validate(); a dense refit with 'BLIP' or
 * 'Blipp' fitted the Blip basis (MODELS' lenient basis_kind_from_string)
 * while warning that the basis was the experimental Gaussian one; another
 * REFIT_LEVEL fitted the balanced preset and another REFIT_PLACEMENT the
 * window placement; REFIT_CONSTRAIN_F0 true was accepted without a word
 * under EMBF0 ZERO and false with a WARNING only. */
TEST_CASE_METHOD(PTFixture,
                 "REFIT_CONSTRAIN_F0, REFIT_LEVEL, REFIT_PLACEMENT and "
                 "REFIT_BASIS are read as written or refused",
                 "[refit][engine][strict_keys]") {
  const auto refusal = [](const Context &c) {
    Task task("refit", c);
    std::string what;
    try {
      RefitEngine().validate(task);
    } catch (const std::exception &e) {
      what = e.what();
    }
    return what;
  };
  const auto with = [](Context c, const std::string &key,
                       const std::string &value) {
    c.remove(key);
    c.add(key, value);
    return c;
  };
  const Context dense = dense_refit_context("missing.eam.alloy");
  SECTION("REFIT_BASIS") {
    for (const char *mode : {"dense", "compact"}) {
      for (const char *basis : {"Blipp", "BLIPP"}) {
        INFO(mode << " " << basis);
        Context c = with(dense, "REFIT_BASIS", basis);
        c = with(c, "REFIT_MODE", mode);
        if (std::string(mode) == "compact") {
          c.remove("REFIT_LEVEL");
          c.remove("REFIT_MAX_ESCALATIONS");
        }
        CHECK(refusal(c) ==
              std::string("tadah refit: REFIT_BASIS '") + basis +
                  "' is no basis family of tadah refit: Blip (Blip, BLIP, "
                  "blip), Blip5 (Blip5, BLIP5, blip5, M6), the knot families "
                  "Knot5, Knot3 and KnotMix (Knot5, KNOT5, knot5, Knot3, "
                  "KNOT3, knot3, KnotMix, KNOTMIX, knotmix, Mixed, MIXED), "
                  "or, under REFIT_MODE dense only, the experimental "
                  "Gaussian (Gaussian, gaussian, BP, D2_BP)");
      }
    }
    CHECK(refusal(with(dense, "REFIT_BASIS", "BLIP")).find("REFIT_BASIS") ==
          std::string::npos);
    ScopedTempDir tmp;
    write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
    Context c = dense_refit_context("synth_fcc.eam.alloy");
    c = with(c, "REFIT_BASIS", "BLIP");
    c = with(c, "REFIT_MAX_ESCALATIONS", "0");
    const RefitOutputs upper = run_refit(c);
    INFO(upper.log);
    CHECK(upper.log.find("EXPERIMENTAL") == std::string::npos);
    const std::string pot_upper = slurp("pot.tadah");
    const RefitOutputs blip = run_refit(with(c, "REFIT_BASIS", "Blip"));
    CHECK(upper.weights == blip.weights);
    CHECK(pot_upper == slurp("pot.tadah"));
  }
  SECTION("REFIT_LEVEL") {
    for (const char *level : {"acurate", "Fine-ish"})
      CHECK(refusal(with(dense, "REFIT_LEVEL", level)) ==
            std::string("tadah refit: REFIT_LEVEL must be 'coarse', "
                        "'balanced' (the default), 'fine' or 'accurate' (in "
                        "any letter case), got '") + level + "'");
    CHECK(refusal(with(dense, "REFIT_LEVEL", "Fine")).empty());
    ScopedTempDir tmp;
    write_synth_eam("synth_fcc.eam.alloy", "fcc", 3.6);
    Context c = with(dense_refit_context("synth_fcc.eam.alloy"),
                     "REFIT_MAX_ESCALATIONS", "0");
    const RefitOutputs lower = run_refit(c);
    const RefitOutputs upper = run_refit(with(c, "REFIT_LEVEL", "COARSE"));
    REQUIRE_FALSE(lower.weights.empty());
    CHECK(upper.weights == lower.weights);
  }
  SECTION("REFIT_PLACEMENT") {
    CHECK(refusal(with(dense, "REFIT_PLACEMENT", "unifrom")) ==
          "tadah refit: REFIT_PLACEMENT must be 'window' (the default) or "
          "'uniform' (in any letter case), got 'unifrom'");
    CHECK(refusal(with(dense, "REFIT_PLACEMENT", "Uniform")).empty());
    CHECK(refusal(with(dense, "REFIT_PLACEMENT", "WINDOW")).empty());
  }
  SECTION("REFIT_CONSTRAIN_F0") {
    for (const char *v : {"true", "false"}) {
      INFO("REFIT_CONSTRAIN_F0 " << v);
      const Context c = with(dense, "REFIT_CONSTRAIN_F0", v);
      CHECK(refusal(c) ==
            "tadah refit: REFIT_CONSTRAIN_F0 has no effect under EMBF0 ZERO "
            "(the default): only EMBF0 FREE reads it; remove "
            "REFIT_CONSTRAIN_F0, or set EMBF0 FREE");
      CHECK(refusal(with(c, "EMBF0", "ZERO")) ==
            "tadah refit: REFIT_CONSTRAIN_F0 has no effect under EMBF0 ZERO: "
            "only EMBF0 FREE reads it; remove REFIT_CONSTRAIN_F0, or set "
            "EMBF0 FREE");
      CHECK(refusal(with(c, "EMBF0", "FREE")).empty());
    }
    using tadah::mlip::KeyOrigin;
    using tadah::mlip::KeyProvenance;
    Context c = with(dense, "REFIT_CONSTRAIN_F0", "true");
    Task task("refit", c);
    task.setKeyProvenance(KeyProvenance(Context(), dense, "dense.refit",
                                        KeyOrigin::Kind::ConfigFile,
                                        {"REFIT_CONSTRAIN_F0"}));
    CHECK_NOTHROW(RefitEngine().validate(task));
  }
}
