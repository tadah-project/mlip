// End-to-end tests for `tadah refit --edit` through RefitEngine::run: file
// round-trip (parse -> edit -> dump), default output naming (never clobbers
// the input), curve bit-identity for --add, and the mode-key validation the
// CLI cannot enforce on the config-file path.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/mlip/engines/refit_modes.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/models/eam/pot_curve_edit.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

using tadah::core::Context;
using tadah::mlip::Task;
using tadah::mlip::engines::RefitEngine;
using tadah::mlip::engines::TrainEngine;
using tadah::mlip::engines::validate_refit_mode_keys;
namespace potedit = tadah::models::potedit;
namespace fs = std::filesystem;

namespace {

struct PTFixture { PTFixture() { tadah::core::PeriodicTable::initialize(); } };

Context make_knot_pot() {
  Context c;
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("ATOM", "Ta");
  c.add("WATOM", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("LAMBDA", 0);
  c.add("OALGO", 2);
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Knot5");
  c.add("TYPE2B", 3); c.add("TYPE2B", 0);
  c.add("TYPE2B", "Ta"); c.add("TYPE2B", "Ta");
  for (double v : {5.0, 3.0, 4.2}) c.add("CGRID2B", v);
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_REAM");
  c.add("TYPEMB", 2); c.add("TYPEMB", 2); c.add("TYPEMB", 0);
  c.add("TYPEMB", 2); c.add("TYPEMB", 3); c.add("TYPEMB", 0);
  c.add("TYPEMB", 0); c.add("TYPEMB", 0);
  c.add("TYPEMB", "Ta"); c.add("TYPEMB", "Ta");
  for (double v : {4.8, 2.5}) c.add("CGRIDMB", v);
  for (double v : {0.01, -1.2}) c.add("AMPGRIDMB", v);
  for (double v : {0.9, -2.9, 0.1}) c.add("CEMBFUNC", v);
  c.add("RCUTMB", 4.8);
  c.add("RCTYPEMB", "Cut_Dummy");
  for (double v : {1.5, -900.0, 12.0, 220.0, -160.0, 5300.0})
    c.add("WEIGHTS", v);
  c.add("FIXINDEX", "1-6");
  for (double v : {1.5, -900.0, 12.0, 220.0, -160.0, 5300.0})
    c.add("FIXWEIGHT", v);
  return c;
}

struct ScopedTempDir {
  fs::path old_cwd, dir;
  ScopedTempDir() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_editmode_" + std::to_string(::getpid()) + "_" +
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

// A potential file as tadah <= 1.5.0 wrote it: dumpPotFile now declares
// EMBF0 for every DM_REAM potential, so drop that line again.
void strip_embf0_line(const std::string &file) {
  std::ifstream in(file);
  std::string line, body;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key != "EMBF0") body += line + "\n";
  }
  in.close();
  std::ofstream(file) << body;
}

std::string stored_embf0_text(const Context &c) {
  return c.stored_value("EMBF0").value_or("");
}

// Runs one refit --edit and returns what it logged at INFO and above.
std::string run_edit_capturing(Context c) {
  Task task("refit", c);
  auto &log = tadah::core::Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(tadah::core::LogLevel::INFO);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  int rc = -1;
  try {
    rc = RefitEngine().run(task);
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

double curve_maxdiff(const potedit::CurveView &a, const potedit::CurveView &b,
                     double lo, double hi, int N = 1500) {
  double mx = 0.0;
  for (int i = 0; i < N; ++i) {
    const double x = lo + (hi - lo) * i / (N - 1);
    mx = std::max(mx, std::fabs(a.value(x) - b.value(x)));
    mx = std::max(mx, std::fabs(a.deriv(x) - b.deriv(x)));
    mx = std::max(mx, std::fabs(a.deriv2(x) - b.deriv2(x)));
  }
  return mx;
}

}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "refit --edit --add: file round-trip with identical curves",
                 "[refit][edit][integration]") {
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("in.tadah", make_knot_pot());

  Context c;
  c.add("EDIT", std::string("in.tadah"));
  c.add("FUNC", std::string("phi"));
  c.add("ADD", 3.6);
  Task task("refit", c);

  RefitEngine engine;
  REQUIRE_NOTHROW(engine.validate(task));
  REQUIRE(engine.run(task) == EXIT_SUCCESS);

  // default names: the input is never clobbered
  REQUIRE(fs::exists("pot_edited.tadah"));
  REQUIRE(fs::exists("refit_edited.hpotarget"));

  const Context in = tadah::core::CoreUserConfigParser().parse("in.tadah");
  const Context out =
      tadah::core::CoreUserConfigParser().parse("pot_edited.tadah");
  potedit::PotEAMView vi = potedit::parse_pot_view(in);
  potedit::PotEAMView vo = potedit::parse_pot_view(out);
  REQUIRE(vo.phi.n() == vi.phi.n() + 1);
  REQUIRE(vo.phi.coeff.back() == 0.0);
  // through the 18-decimal text round-trip the curves stay bit-identical
  REQUIRE(curve_maxdiff(vi.phi, vo.phi, 0.2, 8.0) == 0.0);
  REQUIRE(curve_maxdiff(vi.rho, vo.rho, 0.2, 8.0) == 0.0);
  REQUIRE(curve_maxdiff(vi.F, vo.F, -4.0, 4.0) == 0.0);
  // bookkeeping followed
  REQUIRE(out.get<std::string>("FIXINDEX") == "1-7");
  REQUIRE(out.size("FIXWEIGHT") == 7);
  REQUIRE(out.get<int>("TYPE2B", 1) == 4);
  // dumpPotFile declared the default on the input; the edit keeps it
  REQUIRE(stored_embf0_text(in) == "ZERO");
  REQUIRE(stored_embf0_text(out) == "ZERO");
  REQUIRE(vo.F.zero_ref);
}

TEST_CASE_METHOD(PTFixture,
                 "refit --edit: NUMERIC is honoured and the messages say when "
                 "the edit is not exact", "[refit][edit][numeric][lossless]") {
  ScopedTempDir tmp;
  Context pot = make_knot_pot();
  pot.remove("WEIGHTS");
  pot.remove("FIXWEIGHT");
  const std::vector<double> w = {1.0 / 3.0, -900.0, 12.0, 220.0, -160.0,
                                 5300.0};
  for (double v : w) pot.add("WEIGHTS", v);
  for (double v : w) pot.add("FIXWEIGHT", v);
  TrainEngine::dumpPotFile("in.tadah", pot);
  {  // a 1.5.0 hpo best_pot carries an auto-inserted NUMERIC 12
    std::ofstream app("in12.tadah");
    std::ifstream src("in.tadah");
    app << src.rdbuf() << "NUMERIC 12\n";
  }

  auto run_edit = [](const std::string &in, const std::string &out,
                     const char *numeric) {
    Context c;
    c.add("EDIT", in);
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 3.6);
    c.add("OUTFILE", out);
    c.add("HPOFILEOUT", out + ".hpotarget");
    if (numeric) c.add("NUMERIC", std::string(numeric));
    Task task("refit", c);
    auto &log = tadah::core::Logger::instance();
    const auto old_level = log.getLogLevel();
    log.setLogLevel(tadah::core::LogLevel::WARNING);
    std::ostringstream captured;
    std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
    int rc = -1;
    try {
      rc = RefitEngine().run(task);
    } catch (...) {
      std::cout.rdbuf(old_buf);
      log.setLogLevel(old_level);
      throw;
    }
    std::cout.rdbuf(old_buf);
    log.setLogLevel(old_level);
    REQUIRE(rc == EXIT_SUCCESS);
    return captured.str();
  };
  auto first_weight = [](const std::string &file) {
    const Context c = tadah::core::CoreUserConfigParser().parse(file);
    return c.get<std::string>("WEIGHTS", 0);
  };

  SECTION("no NUMERIC: exact edit, no warning") {
    const std::string log = run_edit("in.tadah", "out.tadah", nullptr);
    CHECK(log.find("NUMERIC") == std::string::npos);
    const Context out = tadah::core::CoreUserConfigParser().parse("out.tadah");
    CHECK(out.get<double>("WEIGHTS", 0) == 1.0 / 3.0);
  }
  SECTION("NUMERIC 12 carried by the input: honoured, warning names it") {
    const std::string log = run_edit("in12.tadah", "out12.tadah", nullptr);
    CHECK(log.find("NUMERIC 12 (carried by in12.tadah)") != std::string::npos);
    CHECK(log.find("Delete the NUMERIC line from in12.tadah") !=
          std::string::npos);
    CHECK(log.find("only up to that rounding") != std::string::npos);
    CHECK(first_weight("out12.tadah") == "0.333333333333");
    std::ifstream f("out12.tadah");
    const std::string body((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    CHECK(body.find("NUMERIC") == std::string::npos);
  }
  SECTION("a NUMERIC set for the run wins over the input's") {
    const std::string log = run_edit("in12.tadah", "out18.tadah", "18");
    CHECK(log.find("rounding") == std::string::npos);
    const Context out =
        tadah::core::CoreUserConfigParser().parse("out18.tadah");
    CHECK(out.get<double>("WEIGHTS", 0) == 1.0 / 3.0);
  }
}

TEST_CASE_METHOD(PTFixture, "refit --edit --remove: end-to-end",
                 "[refit][edit][integration][embf0]") {
  ScopedTempDir tmp;
  Context c;
  c.add("EDIT", std::string("in.tadah"));
  c.add("FUNC", std::string("F"));
  c.add("REMOVE", -2.9);  // knot with g(0) = 2.9^5 != 0

  // The layout expectations hold in both embedding references; the
  // reference itself decides the compensation region (ZERO widens it).
  auto check_layout = [](const Context &out) {
    REQUIRE(out.size("CEMBFUNC") == 2);
    REQUIRE(out.get<int>("TYPEMB", 5) == 2);  // nEc
    REQUIRE(out.size("WEIGHTS") == 5);
    REQUIRE(out.get<std::string>("FIXINDEX") == "1-5");
  };

  SECTION("legacy input (no EMBF0 line): edited and written as ZERO") {
    TrainEngine::dumpPotFile("in.tadah", make_knot_pot());
    strip_embf0_line("in.tadah");
    REQUIRE_FALSE(tadah::core::CoreUserConfigParser()
                      .parse("in.tadah")
                      .has_stored("EMBF0"));
    const std::string log = run_edit_capturing(c);
    CHECK(log.find("EMBF0 ZERO (key absent -> default ZERO") !=
          std::string::npos);
    CHECK(log.find("widened to the F domain") != std::string::npos);
    const Context out =
        tadah::core::CoreUserConfigParser().parse("pot_edited.tadah");
    check_layout(out);
    REQUIRE(stored_embf0_text(out) == "ZERO");
    const potedit::PotEAMView vo = potedit::parse_pot_view(out);
    REQUIRE(vo.F.zero_ref);
    REQUIRE(vo.F.value(0.0) == 0.0);
  }
  SECTION("EMBF0 FREE input: edited with the raw basis, written FREE") {
    Context pot = make_knot_pot();
    pot.add("EMBF0", std::string("FREE"));
    TrainEngine::dumpPotFile("in.tadah", pot);
    const std::string log = run_edit_capturing(c);
    CHECK(log.find("EMBF0 FREE: F is the raw embedding") != std::string::npos);
    CHECK(log.find("key absent") == std::string::npos);
    CHECK(log.find("widened") == std::string::npos);
    const Context out =
        tadah::core::CoreUserConfigParser().parse("pot_edited.tadah");
    check_layout(out);
    REQUIRE(stored_embf0_text(out) == "FREE");
    REQUIRE_FALSE(potedit::parse_pot_view(out).F.zero_ref);
  }
}

TEST_CASE_METHOD(PTFixture,
                 "refit --edit: pure two-body potentials (no many-body term)",
                 "[refit][edit][integration]") {
  ScopedTempDir tmp;
  // the HPO_two_body_fit seed shape: TYPE2B only, WEIGHTS = pair block,
  // RCUT2B deliberately stale (headroom-style, beyond the largest knot)
  Context pot;
  pot.add("MODEL", "M_BLR");
  pot.add("MODEL", "BF_Linear");
  pot.add("ATOM", "Kr");
  pot.add("WATOM", 1.0);
  pot.add("BIAS", "false");
  pot.add("NORM", "false");
  pot.add("FORCE", "false");
  pot.add("STRESS", "false");
  pot.add("LAMBDA", 0);
  pot.add("OALGO", 2);
  pot.add("INIT2B", "true");
  pot.add("TYPE2B", "D2_Knot5");
  pot.add("TYPE2B", 3); pot.add("TYPE2B", 0);
  pot.add("TYPE2B", "Kr"); pot.add("TYPE2B", "Kr");
  for (double v : {5.0, 3.0, 4.2}) pot.add("CGRID2B", v);
  pot.add("RCUT2B", 6.3);
  pot.add("RCTYPE2B", "Cut_Dummy");
  for (double v : {1.5, -900.0, 12.0}) pot.add("WEIGHTS", v);
  pot.add("FIXINDEX", "1-3");
  for (double v : {1.5, -900.0, 12.0}) pot.add("FIXWEIGHT", v);
  TrainEngine::dumpPotFile("in.tadah", pot);

  SECTION("--add beyond the largest knot grows the derived cutoff") {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 6.8);
    Task task("refit", c);
    RefitEngine engine;
    REQUIRE(engine.run(task) == EXIT_SUCCESS);

    const Context in = tadah::core::CoreUserConfigParser().parse("in.tadah");
    const Context out =
        tadah::core::CoreUserConfigParser().parse("pot_edited.tadah");
    potedit::PotEAMView vi = potedit::parse_pot_view(in);
    potedit::PotEAMView vo = potedit::parse_pot_view(out);
    REQUIRE(vi.mb == potedit::MBKind::None);
    REQUIRE(vo.phi.n() == 4);
    REQUIRE(vo.phi.coeff.back() == 0.0);
    REQUIRE(curve_maxdiff(vi.phi, vo.phi, 0.2, 9.0) == 0.0);
    REQUIRE(out.get<double>("RCUT2B") == Approx(6.8));
    REQUIRE(out.get<std::string>("FIXINDEX") == "1-4");
    REQUIRE(!out.contains("TYPEMB"));
    REQUIRE_FALSE(out.has_stored("EMBF0"));  // no DM_REAM block, no key
    // the regenerated hpotarget has no many-body sections
    std::ifstream hs("refit_edited.hpotarget");
    const std::string body((std::istreambuf_iterator<char>(hs)),
                           std::istreambuf_iterator<char>());
    REQUIRE(body.find("OPTIM FIXWEIGHT (4)") != std::string::npos);
    REQUIRE(body.find("density function") == std::string::npos);
    REQUIRE(body.find("embedding function") == std::string::npos);
  }
  SECTION("--func rho/F is rejected with a pure-2B explanation") {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("F"));
    c.add("REMOVE", 0.5);
    Task task("refit", c);
    RefitEngine engine;
    REQUIRE_THROWS_WITH(engine.run(task),
                        Catch::Contains("pure two-body"));
  }
}

TEST_CASE_METHOD(PTFixture, "refit --edit: output naming and clobber guard",
                 "[refit][edit][integration]") {
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("in.tadah", make_knot_pot());

  SECTION("-o is honoured") {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 3.6);
    c.add("OUTFILE", std::string("custom.tadah"));
    c.add("HPOFILEOUT", std::string("custom.hpotarget"));
    Task task("refit", c);
    RefitEngine engine;
    REQUIRE(engine.run(task) == EXIT_SUCCESS);
    REQUIRE(fs::exists("custom.tadah"));
    REQUIRE(fs::exists("custom.hpotarget"));
    REQUIRE_FALSE(fs::exists("pot_edited.tadah"));
  }
  SECTION("writing onto the input is refused") {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 3.6);
    c.add("OUTFILE", std::string("in.tadah"));
    Task task("refit", c);
    RefitEngine engine;
    REQUIRE_THROWS_WITH(engine.run(task),
                        Catch::Contains("refusing to overwrite"));
  }
}

TEST_CASE_METHOD(PTFixture, "refit mode-key validation (config-file path)",
                 "[refit][edit]") {
  auto ctx_with = [](std::initializer_list<std::pair<const char *, double>>
                         dkeys,
                     std::initializer_list<std::pair<const char *, const char *>>
                         skeys) {
    Context c;
    for (const auto &[k, v] : dkeys) c.add(k, v);
    for (const auto &[k, v] : skeys) c.add(k, std::string(v));
    return c;
  };

  // edit + retarget
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx_with(
          {}, {{"EDIT", "a"}, {"RETARGET", "b"}})),
      Catch::Contains("mutually exclusive"));
  // EAM alongside edit
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx_with({}, {{"EDIT", "a"}, {"EAM", "x"}})),
      Catch::Contains("cannot be combined"));
  // edit without FUNC
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx_with({{"ADD", 1.0}}, {{"EDIT", "a"}})),
      Catch::Contains("--func"));
  // edit with both ADD and REMOVE
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx_with({{"ADD", 1.0}, {"REMOVE", 2.0}},
                                        {{"EDIT", "a"}, {"FUNC", "phi"}})),
      Catch::Contains("exactly one"));
  // edit with neither
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(
          ctx_with({}, {{"EDIT", "a"}, {"FUNC", "phi"}})),
      Catch::Contains("exactly one"));
  // ETA without ADD
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx_with({{"REMOVE", 2.0}, {"ETA", 1.0}},
                                        {{"EDIT", "a"}, {"FUNC", "phi"}})),
      Catch::Contains("--eta requires --add"));
  // stray edit keys without --edit
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx_with({{"ADD", 1.0}}, {{"EAM", "x"}})),
      Catch::Contains("requires --edit"));
  // PCT without retarget
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx_with({}, {{"EAM", "x"}, {"PCT", "10"}})),
      Catch::Contains("requires --retarget"));
  // valid combinations pass
  REQUIRE_NOTHROW(validate_refit_mode_keys(
      ctx_with({{"ADD", 1.0}}, {{"EDIT", "a"}, {"FUNC", "phi"}})));
  REQUIRE_NOTHROW(validate_refit_mode_keys(
      ctx_with({{"REMOVE", 1.0}}, {{"EDIT", "a"}, {"FUNC", "rho"}})));
  REQUIRE_NOTHROW(validate_refit_mode_keys(
      ctx_with({}, {{"RETARGET", "b"}, {"PCT", "10"}})));
  REQUIRE_NOTHROW(validate_refit_mode_keys(ctx_with({}, {{"EAM", "x"}})));
}
