// Full-engine integration tests for `tadah refit` (RefitEngine::run): a real
// end-to-end run on a synthesised single-element setfl, and the crystal gate
// that hard-errors on an unspecified / unsupported lattice. The EAM is written
// from smooth analytic phi/rho/F so the test needs no checked-in binary fixture.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/engines/refit_engine.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>

using tadah::core::Context;
using tadah::mlip::Task;
using tadah::mlip::engines::RefitEngine;
namespace fs = std::filesystem;

namespace {
struct PTFixture { PTFixture() { tadah::core::PeriodicTable::initialize(); } };

// A smooth, physically-sane single-element setfl: rho(r)=(1-r/rcut)^2,
// phi(r)=3(1-r/rcut)^3 (stored as z2r=phi*r), F(rho)=-2*sqrt(rho)+0.05*rho.
// rhomax is generous so the crystal stays in range even under compression.
void write_synth_eam(const std::string &path, const std::string &lat,
                     double a0, int nr = 400, int nrho = 400) {
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
    os << (-2.0 * std::sqrt(rho) + 0.05 * rho) << ((i % 5 == 4) ? "\n" : " ");
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
