// End-to-end tests for `tadah refit --edit` through RefitEngine::run: file
// round-trip (parse -> edit -> dump), default output naming (never clobbers
// the input), curve bit-identity for --add, and the mode-key validation the
// CLI cannot enforce on the config-file path.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/key_provenance.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/mlip/engines/refit_modes.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/models/eam/pot_curve_edit.h>

#include <algorithm>
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
  c.add("RHOFORM", "SIGNED");  // a negative amplitude: SIGNED only
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

/* The opening and end lines of the training record of a potential file.
 * The opening line said "(not read by tadah)" before MK's decision of
 * 2026-10-14 (option 2): refit --edit reads the block, to copy it. */
const std::string kRecordOpen =
    "# --- training record (no key is read from it) ---";
const std::string kRecordEnd = "# --- end of training record ---";

std::string slurp_file(const std::string &f) {
  std::ifstream in(f);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/* The lines of the training record of the potential file @p f, without
 * their leading "# ": empty when it has none. */
std::vector<std::string> record_lines(const std::string &f) {
  std::ifstream in(f);
  std::vector<std::string> out;
  bool in_block = false;
  for (std::string l; std::getline(in, l);) {
    if (!in_block) {
      in_block = l == kRecordOpen;
      continue;
    }
    if (l == kRecordEnd) break;
    out.push_back(l.rfind("# ", 0) == 0 ? l.substr(2) : l);
  }
  return out;
}

/* The tokens after @p key on its key line of the file @p f (a comment
 * line is not one); empty when there is none. */
std::vector<std::string> key_line(const std::string &f, const std::string &key) {
  std::ifstream in(f);
  for (std::string l; std::getline(in, l);) {
    std::istringstream ls(l);
    std::string k;
    if (!(ls >> k) || k != key) continue;
    std::vector<std::string> v;
    for (std::string t; ls >> t;) v.push_back(t);
    return v;
  }
  return {};
}

Context edit_of(const std::string &in, const std::string &out) {
  Context c;
  c.add("EDIT", in);
  c.add("FUNC", std::string("phi"));
  c.add("ADD", 3.6);
  c.add("OUTFILE", out);
  c.add("HPOFILEOUT", out + ".hpotarget");
  return c;
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

/* Pins: refit --edit takes NUMERIC from its own run only: an input with no
 * NUMERIC is edited exactly, with no NUMERIC message; an input potential with
 * the key line "NUMERIC 12" (a 1.5.0 tadah hpo best_pot.tadah has one) is
 * edited exactly too -- its NUMERIC, a setting of the run that wrote it, is
 * not read: no NUMERIC message, the first weight of out12.tadah is 1/3
 * read back exactly -- and the line moves into the edit's training record
 * ("edited by tadah refit --edit from in12.tadah", "NUMERIC 12"), no key
 * line of it; an input with "VERBOSE 2" is edited at the run's log level
 * (WARNING: no INFO line "added a basis function" is printed) and its
 * record lists "VERBOSE 2"; a NUMERIC set for the run applies.
 * Before the fix: the input's NUMERIC 12 was honoured ("NUMERIC 12 (carried
 * by in12.tadah)", the edit written with 12 significant digits, the first
 * weight 0.333333333333; this case pinned that), and the input's VERBOSE 2
 * set the log level to INFO for the edit (its INFO lines printed): the
 * NUMERIC and INFO checks fail. */
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
  SECTION("NUMERIC 12 of the input: not read, the edit is exact") {
    const std::string log = run_edit("in12.tadah", "out12.tadah", nullptr);
    CHECK(log.find("NUMERIC") == std::string::npos);
    CHECK(std::stod(first_weight("out12.tadah")) == 1.0 / 3.0);
    // NUMERIC is no key line of the edit; the input's NUMERIC 12 key line
    // (as a 1.5.0 tadah hpo potential has it) is listed in the edit's
    // training record, after the line naming the input (owner decision,
    // MK, 2026-10-14). Before the training record the check was that
    // NUMERIC appears nowhere in out12.tadah; before the fix this fails:
    // out12.tadah has no record.
    CHECK(key_line("out12.tadah", "NUMERIC").empty());
    CHECK(record_lines("out12.tadah") ==
          std::vector<std::string>{
              "edited by tadah refit --edit from in12.tadah", "NUMERIC 12"});
  }
  SECTION("VERBOSE 2 of the input: the edit runs at its own level") {
    {
      std::ofstream app("in_v2.tadah");
      std::ifstream src("in.tadah");
      app << src.rdbuf() << "VERBOSE 2\n";
    }
    const std::string log = run_edit("in_v2.tadah", "out_v2.tadah", nullptr);
    CHECK(log.find("added a basis function") == std::string::npos);
    CHECK(log.find("INFO:") == std::string::npos);
    CHECK(key_line("out_v2.tadah", "VERBOSE").empty());
    CHECK(record_lines("out_v2.tadah") ==
          std::vector<std::string>{
              "edited by tadah refit --edit from in_v2.tadah", "VERBOSE 2"});
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

TEST_CASE_METHOD(PTFixture,
                 "refit --edit: OPTIM_PCT is read strictly before anything is "
                 "written",
                 "[refit][edit][strict_numbers]") {
  /* Pins: refit --edit reads OPTIM_PCT as tadah refit and --retarget read
   * it (a positive number in the hpotarget grammar) in validate() and,
   * in run(), before it writes a file: OPTIM_PCT 0 or -5 throws naming the
   * key and the token, and neither the edited potential nor its hpotarget
   * is written; OPTIM_PCT 7.5 writes both.
   * Before the fix: validate() did not read it, and run() wrote the edited
   * potential before it read OPTIM_PCT, so after the error
   * pot_edited.tadah exists (a partial result without its hpotarget). */
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("in.tadah", make_knot_pot());
  const auto edit = [](const std::string &pct) {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 3.6);
    c.add("OPTIM_PCT", pct);
    return c;
  };
  for (const std::string pct : {"0", "-5"}) {
    INFO("OPTIM_PCT " << pct);
    const std::string msg =
        "tadah refit: OPTIM_PCT must be a positive number, got '" + pct + "'";
    Task task("refit", edit(pct));
    CHECK_THROWS_WITH(RefitEngine().validate(task), msg);
    CHECK_THROWS_WITH(RefitEngine().run(task), msg);
    CHECK_FALSE(fs::exists("pot_edited.tadah"));
    CHECK_FALSE(fs::exists("refit_edited.hpotarget"));
  }
  Task task("refit", edit("7.5"));
  REQUIRE_NOTHROW(RefitEngine().validate(task));
  REQUIRE(RefitEngine().run(task) == EXIT_SUCCESS);
  CHECK(fs::exists("pot_edited.tadah"));
  CHECK(fs::exists("refit_edited.hpotarget"));
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

// ---------------------------------------------------------------------------
//  RHOFORM (FEAT-02): --edit keeps and logs the potential's density form
// ---------------------------------------------------------------------------
namespace {
// Blip pair channel + DM_REAM Blip density (non-negative amplitudes, every
// support inside RCUTMB 4.8) declared in `form`.
Context make_blip_ream_pot(const std::string &form) {
  Context c;
  c.add("MODEL", "M_BLR");
  c.add("MODEL", "BF_Linear");
  c.add("ATOM", "Cu");
  c.add("WATOM", 1.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("FORCE", "false");
  c.add("STRESS", "false");
  c.add("LAMBDA", 0);
  c.add("OALGO", 2);
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Blip");
  c.add("TYPE2B", 3); c.add("TYPE2B", 3);
  c.add("TYPE2B", "Cu"); c.add("TYPE2B", "Cu");
  for (double v : {2.4, 3.2, 4.0}) c.add("CGRID2B", v);
  for (double v : {2.0, 2.0, 2.0}) c.add("SGRID2B", v);
  c.add("RCUT2B", 5.0);
  c.add("RCTYPE2B", "Cut_Dummy");
  c.add("INITMB", "true");
  c.add("TYPEMB", "DM_REAM");
  for (int v : {0, 5, 5, 5, 3, 3, 0, 0}) c.add("TYPEMB", v);
  c.add("TYPEMB", "Cu"); c.add("TYPEMB", "Cu");
  for (double v : {1.2, 3.0, 3.4, 3.8, 4.3}) c.add("CGRIDMB", v);
  for (double v : {2.0, 2.0, 2.0, 2.0, 4.0}) c.add("SGRIDMB", v);
  for (double v : {0.5, 0.2, 0.05, 0.01, 0.02}) c.add("AMPGRIDMB", v);
  c.add("RHOFORM", form);
  for (double v : {0.9, 2.9, 0.1}) c.add("CEMBFUNC", v);
  for (double v : {1.5, 2.0, 1.8}) c.add("SEMBFUNC", v);
  c.add("RCUTMB", 4.8);
  c.add("RCTYPEMB", "Cut_Dummy");
  for (double v : {1.5, -9.0, 1.2, 2.2, -1.6, 5.3}) c.add("WEIGHTS", v);
  c.add("FIXINDEX", "1-6");
  for (double v : {1.5, -9.0, 1.2, 2.2, -1.6, 5.3}) c.add("FIXWEIGHT", v);
  return c;
}

void strip_key_line(const std::string &file, const std::string &drop) {
  std::ifstream in(file);
  std::string line, body;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key != drop) body += line + "\n";
  }
  in.close();
  std::ofstream(file) << body;
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "refit --edit: a DM_REAM input without RHOFORM gets the "
                 "migration text",
                 "[refit][edit][rhoform]") {
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("in.tadah", make_knot_pot());
  strip_key_line("in.tadah", "RHOFORM");  // as tadah wrote it before 1.6.0
  Context c;
  c.add("EDIT", std::string("in.tadah"));
  c.add("FUNC", std::string("phi"));
  c.add("ADD", 3.6);
  Task task("refit", c);
  REQUIRE_THROWS_AS(RefitEngine().run(task), tadah::models::DensityFormError);
  REQUIRE_THROWS_WITH(RefitEngine().run(task),
                      Catch::Contains("pot edit: DM_REAM: RHOFORM is required: "
                                      "SIGNED") &&
                          !Catch::Contains("internal error"));
  REQUIRE_FALSE(fs::exists("pot_edited.tadah"));
}

TEST_CASE_METHOD(PTFixture,
                 "refit --edit: the density form is kept, logged and "
                 "honoured by --remove",
                 "[refit][edit][integration][rhoform]") {
  for (const char *form : {"NONNEG", "monotone"}) {
    INFO(form);
    ScopedTempDir tmp;
    TrainEngine::dumpPotFile("in.tadah", make_blip_ream_pot(form));
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("rho"));
    c.add("REMOVE", 3.0);
    const std::string log = run_edit_capturing(c);
    const std::string name = tadah::models::rhoform_name(
        tadah::models::parse_rhoform(form));
    CHECK(log.find("tadah refit --edit: RHOFORM " + name + ": rho is sum_k") !=
          std::string::npos);
    CHECK(log.find("the edited potential declares RHOFORM " + name) !=
          std::string::npos);
    CHECK(log.find("held at its bound 0") != std::string::npos);
    const Context out =
        tadah::core::CoreUserConfigParser().parse("pot_edited.tadah");
    REQUIRE(out.stored_value("RHOFORM") == std::optional<std::string>(name));
    REQUIRE(out.size("AMPGRIDMB") == 4);
    for (std::size_t n = 0; n < out.size("AMPGRIDMB"); ++n)
      REQUIRE(out.get<double>("AMPGRIDMB", n) >= 0.0);
    // the regenerated hpotarget keeps the AMPGRIDMB boxes non-negative
    std::ifstream hs("refit_edited.hpotarget");
    const std::string body((std::istreambuf_iterator<char>(hs)),
                           std::istreambuf_iterator<char>());
    CHECK(body.find("# RHOFORM " + name +
                    ": AMPGRIDMB bounds are non-negative") !=
          std::string::npos);
  }
}

// ---------------------------------------------------------------------------
//  RHOFORM / EMBF0 in the configuration of --edit and --retarget (C7)
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(PTFixture,
                 "refit --edit: a RHOFORM or EMBF0 in the configuration that "
                 "differs from the potential's is a conflict",
                 "[refit][edit][rhoform][embf0]") {
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("in.tadah", make_blip_ream_pot("NONNEG"));
  auto edit = [](const char *key, const char *value) {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 3.6);
    if (key) c.add(key, std::string(value));
    return Task("refit", c);
  };
  SECTION("RHOFORM") {
    Task t = edit("RHOFORM", "MONOTONE");
    REQUIRE_THROWS_AS(RefitEngine().run(t), tadah::models::DensityFormError);
    REQUIRE_THROWS_WITH(RefitEngine().run(t),
                        Catch::Contains("RHOFORM conflict: in.tadah declares "
                                        "RHOFORM NONNEG but the task "
                                        "configuration sets RHOFORM "
                                        "MONOTONE"));
    REQUIRE_FALSE(fs::exists("pot_edited.tadah"));
  }
  SECTION("EMBF0") {
    Task t = edit("EMBF0", "FREE");
    REQUIRE_THROWS_WITH(RefitEngine().run(t),
                        Catch::Contains("EMBF0 conflict: in.tadah declares "
                                        "EMBF0 ZERO but the task "
                                        "configuration sets EMBF0 FREE"));
    REQUIRE_FALSE(fs::exists("pot_edited.tadah"));
  }
  SECTION("the potential's own values are accepted") {
    for (const auto &kv : std::vector<std::pair<const char *, const char *>>{
             {"RHOFORM", "nonneg"}, {"EMBF0", "ZERO"}}) {
      INFO(kv.first);
      Task t = edit(kv.first, kv.second);
      REQUIRE(RefitEngine().run(t) == EXIT_SUCCESS);
      REQUIRE(fs::exists("pot_edited.tadah"));
      fs::remove("pot_edited.tadah");
    }
  }
}

TEST_CASE_METHOD(PTFixture,
                 "refit --retarget: RHOFORM and EMBF0 in the configuration are "
                 "refused; --rhoform needs --retarget",
                 "[refit][retarget][rhoform][embf0]") {
  auto ctx = [](std::initializer_list<std::pair<const char *, const char *>>
                    keys) {
    Context c;
    for (const auto &[k, v] : keys) c.add(k, std::string(v));
    return c;
  };
  // an hpotarget carries no potential: the keys would have no effect
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(
          ctx({{"RETARGET", "b"}, {"RHOFORM", "NONNEG"}})),
      Catch::Contains("RHOFORM has no effect on --retarget") &&
          Catch::Contains("--rhoform"));
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx({{"RETARGET", "b"}, {"EMBF0", "FREE"}})),
      Catch::Contains("EMBF0 has no effect on --retarget"));
  // through the engine's validate (the config-file path) as well
  Task t("refit", ctx({{"RETARGET", "b"}, {"RHOFORM", "SIGNED"}}));
  REQUIRE_THROWS_WITH(RefitEngine().validate(t),
                      Catch::Contains("RHOFORM has no effect on --retarget"));
  // --rhoform belongs to --retarget
  REQUIRE_THROWS_WITH(
      validate_refit_mode_keys(ctx({{"EAM", "x"},
                                    {"RETARGET_RHOFORM", "NONNEG"}})),
      Catch::Contains("requires --retarget"));
  REQUIRE_NOTHROW(validate_refit_mode_keys(
      ctx({{"RETARGET", "b"}, {"RETARGET_RHOFORM", "NONNEG"}})));
}

// ---------------------------------------------------------------------------
//  --edit and --retarget refuse the keys of a refit of an EAM file
// ---------------------------------------------------------------------------
namespace {
// What reads a key that --edit or --retarget does not (the refusal's middle
// part), and the keys of each, each with a valid value.
const std::string kFitReads = "only a refit of an EAM file reads it";
const std::string kDenseReads =
    "only a refit of an EAM file in REFIT_MODE dense reads it";
const std::string kCompactReads =
    "only a refit of an EAM file in REFIT_MODE compact reads it";
const std::string kFitEditReads =
    "only a refit of an EAM file and --edit read it";

struct FitKey {
  std::string key, value;
  const std::string *reader;
  bool block;  ///< a stage block (one token per value, "the <KEY> block")
};
const std::vector<FitKey> kFitOnlyKeys = {
    // read in both REFIT_MODEs
    {"REFIT_MODE", "dense", &kFitReads, false},
    {"REFIT_BASIS", "Blip", &kFitReads, false},
    {"REFIT_TOL_AMBIENT", "0.5", &kFitReads, false},
    {"REFIT_TOL_COMPRESSED", "2", &kFitReads, false},
    {"REFIT_REACH_SMIN", "0.8", &kFitReads, false},
    {"REFIT_REACH_MARGIN", "1.5", &kFitReads, false},
    {"REFIT_R_LO_FRAC", "0.15", &kFitReads, false},
    {"REFIT_OBJECTIVE", "energy", &kFitReads, false},
    {"OUTPUT_GRID_SCALE", "2", &kFitReads, false},
    {"SETFL_OUT", "x.eam", &kFitReads, false},
    {"REFIT_LOG", "x.log", &kFitReads, false},
    // REFIT_MODE dense
    {"REFIT_LEVEL", "accurate", &kDenseReads, false},
    {"N2B", "50", &kDenseReads, false},
    {"NRHO_BASIS", "40", &kDenseReads, false},
    {"NF_BASIS", "30", &kDenseReads, false},
    {"REFIT_MAX_ESCALATIONS", "9", &kDenseReads, false},
    {"REFIT_PLACEMENT", "uniform", &kDenseReads, false},
    {"REFIT_CONSTRAIN_F0", "false", &kDenseReads, false},
    // REFIT_MODE compact
    {"REFIT_COMPACT_NMAX", "3", &kCompactReads, false},
    {"REFIT_COMPACT_SAMPLES", "400", &kCompactReads, false},
    {"REFIT_COMPACT_OPT_EVERY", "2", &kCompactReads, false},
    {"REFIT_COMPACT_STALL", "4", &kCompactReads, false},
    {"REFIT_COMPACT_INSERT_TOPK", "2", &kCompactReads, false},
    {"REFIT_COMPACT_JITTERS", "1", &kCompactReads, false},
    {"REFIT_COMPACT_SEED", "7", &kCompactReads, false},
    {"REFIT_COMPACT_SHARE", "0.4", &kCompactReads, false},
    {"REFIT_COMPACT_MERGE", "true", &kCompactReads, false},
    {"REFIT_CURVE_DERIV_WEIGHT", "0.1", &kCompactReads, false},
    {"REFIT_CURVE_WEIGHTING", "false", &kCompactReads, false},
    {"REFIT_CURVE_RELAX", "0.1", &kCompactReads, false},
    {"INIT", "PHI XMIN PIN", &kCompactReads, true},
    {"EXPLORE", "MAX_ADD 4", &kCompactReads, true},
    {"REFINE", "OPTIMIZER LIB CERES ALGO LM MAXEVAL 5 ENDOPTIMIZER",
     &kCompactReads, true},
    {"FUSE", "ENABLED false", &kCompactReads, true}};

// The refusal of `what` under `mode` ("--edit" or "--retarget").
std::string output_mode_refusal(const std::string &mode,
                                const std::string &what,
                                const std::string &reader,
                                const std::string &tail = "") {
  return "tadah refit " + mode + ": " + what + " has no effect under " +
         mode + ": " + reader + "; remove " + what + tail;
}

const std::string kOptimizerUnread =
    "no mode of tadah refit reads a top-level OPTIMIZER block";

void add_tokens(Context &c, const std::string &key, const std::string &body) {
  std::istringstream iss(body);
  for (std::string t; iss >> t;) c.add(key, t);
}
}  // namespace

TEST_CASE_METHOD(PTFixture,
                 "refit --edit and --retarget: a key of a refit of an EAM "
                 "file is refused, naming the key and what reads it",
                 "[refit][edit][retarget][mode_keys]") {
  /* Pins: RefitEngine::validate(), and run() before it reads or writes a
   * file, refuse under --edit and under --retarget every key that only a
   * refit of an EAM file reads, naming the key and what reads it:
   * "tadah refit --edit: N2B has no effect under --edit: only a refit of an
   * EAM file in REFIT_MODE dense reads it; remove N2B" (the dense keys),
   * "... in REFIT_MODE compact reads it; ..." (REFIT_COMPACT_*,
   * REFIT_CURVE_*; the INIT, EXPLORE, REFINE and FUSE blocks as "the
   * REFINE block"), "... only a refit of an EAM file reads it; ..." (the
   * keys both REFIT_MODEs read: REFIT_MODE, REFIT_BASIS, REFIT_TOL_*,
   * REFIT_REACH_*, REFIT_R_LO_FRAC, REFIT_OBJECTIVE, OUTPUT_GRID_SCALE,
   * SETFL_OUT, REFIT_LOG); a top-level OPTIMIZER block: "the OPTIMIZER
   * block has no effect under --edit: no mode of tadah refit reads a
   * top-level OPTIMIZER block; remove the OPTIMIZER block"; under
   * --retarget also HPOFILEOUT, which --edit reads ("... only a refit of an
   * EAM file and --edit read it; remove HPOFILEOUT (the file --retarget
   * writes is -o, OUTFILE)"). Nothing is written (no pot_edited.tadah,
   * refit_edited.hpotarget or a_retargeted.hpotarget, the --retarget
   * default; it was a_edited.hpotarget before --retarget's default name
   * became <input>_retargeted). NUMERIC, the precision
   * of the potential and dataset files a command writes and no key of the
   * refit, passes under --retarget, which writes neither: validate()
   * accepts NUMERIC 12 and run() writes a_retargeted.hpotarget byte-identical
   * to the run without it (its bounds are printed %.8e whatever NUMERIC
   * is); --edit reads HPOFILEOUT and NUMERIC (new.hpotarget written). The
   * blueprint defaults the task factory stores for every refit
   * (KeyProvenance Default: HPOFILEOUT refit.hpotarget, OUTFILE pot.tadah,
   * REFIT_OBJECTIVE energy, REFIT_CURVE_*, REFIT_COMPACT_MERGE auto,
   * REFIT_CONSTRAIN_F0 true, IN_PLACE false, VERBOSE 1) pass and both
   * modes run; the same keys written in the configuration file are refused
   * (REFIT_OBJECTIVE energy, the first of them in the table, under --edit
   * and under --retarget).
   * Before the fix: validate() accepted every key and run() ignored it:
   * --edit wrote pot_edited.tadah and its hpotarget, --retarget wrote
   * a_edited.hpotarget, its default then (with HPOFILEOUT set as well,
   * which it never read);
   * in the defaults section the defaults passed as well (a pin) and the
   * keys written in the file were accepted (no exception, the outputs
   * written). NUMERIC under --retarget passed as well (a pin); the first
   * version of the rule refused it ("tadah refit --retarget: NUMERIC has
   * no effect under --retarget: only a refit of an EAM file and --edit
   * read it; remove NUMERIC"), which stopped every --retarget TASK of a
   * task file that sets NUMERIC above its first TASK line, for every
   * task. */
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("in.tadah", make_knot_pot());
  tadah::mlip::engines::write_hpotarget("a.hpotarget", make_knot_pot(), 5.0);
  const auto edit = [] {
    Context c;
    c.add("EDIT", std::string("in.tadah"));
    c.add("FUNC", std::string("phi"));
    c.add("ADD", 3.6);
    return c;
  };
  const auto retarget = [] {
    Context c;
    c.add("RETARGET", std::string("a.hpotarget"));
    return c;
  };
  const std::vector<std::string> outputs = {
      "pot_edited.tadah", "refit_edited.hpotarget", "a_retargeted.hpotarget"};
  const auto check_refused = [&](const Task &task, const std::string &msg) {
    for (const auto &f : outputs) fs::remove(f);
    CHECK_THROWS_WITH(RefitEngine().validate(task), msg);
    Task copy = task;
    CHECK_THROWS_WITH(RefitEngine().run(copy), msg);
    for (const auto &f : outputs) {
      INFO("written: " << f);
      CHECK_FALSE(fs::exists(f));
    }
  };
  SECTION("the keys of a refit of an EAM file") {
    for (const char *mode : {"--edit", "--retarget"}) {
      for (const FitKey &k : kFitOnlyKeys) {
        INFO(mode << " " << k.key << " " << k.value);
        Context c = std::string(mode) == "--edit" ? edit() : retarget();
        if (k.block)
          add_tokens(c, k.key, k.value);
        else
          c.add(k.key, k.value);
        const std::string what =
            k.block ? "the " + k.key + " block" : k.key;
        check_refused(Task("refit", c),
                      output_mode_refusal(mode, what, *k.reader));
      }
      INFO(mode << " a top-level OPTIMIZER block");
      Context c = std::string(mode) == "--edit" ? edit() : retarget();
      add_tokens(c, "OPTIMIZER", "GLOBAL ALGO MLSL ENDGLOBAL");
      check_refused(Task("refit", c),
                    output_mode_refusal(mode, "the OPTIMIZER block",
                                        kOptimizerUnread));
    }
  }
  SECTION("--retarget refuses HPOFILEOUT, which --edit reads, and passes "
          "NUMERIC") {
    Context c = retarget();
    c.add("HPOFILEOUT", std::string("new.hpotarget"));
    check_refused(Task("refit", c),
                  output_mode_refusal("--retarget", "HPOFILEOUT",
                                      kFitEditReads,
                                      " (the file --retarget writes is -o, "
                                      "OUTFILE)"));
    CHECK_FALSE(fs::exists("new.hpotarget"));
    // NUMERIC is the precision of a written potential or dataset, which
    // --retarget does not write: it passes, and the hpotarget is the one
    // written without it
    const auto slurp = [](const std::string &f) {
      std::ifstream is(f);
      return std::string((std::istreambuf_iterator<char>(is)),
                         std::istreambuf_iterator<char>());
    };
    for (const auto &f : outputs) fs::remove(f);
    Task plain("refit", retarget());
    REQUIRE(RefitEngine().run(plain) == EXIT_SUCCESS);
    const std::string without = slurp("a_retargeted.hpotarget");
    REQUIRE_FALSE(without.empty());
    fs::remove("a_retargeted.hpotarget");
    Context n = retarget();
    n.add("NUMERIC", std::string("12"));
    Task numeric("refit", n);
    CHECK_NOTHROW(RefitEngine().validate(numeric));
    CHECK(RefitEngine().run(numeric) == EXIT_SUCCESS);
    CHECK(slurp("a_retargeted.hpotarget") == without);
    // --edit reads both
    Context e = edit();
    e.add("HPOFILEOUT", std::string("new.hpotarget"));
    e.add("NUMERIC", std::string("12"));
    Task task("refit", e);
    CHECK_NOTHROW(RefitEngine().validate(task));
    CHECK(RefitEngine().run(task) == EXIT_SUCCESS);
    CHECK(fs::exists("new.hpotarget"));
  }
  SECTION("the blueprint defaults the task factory stores pass") {
    using tadah::mlip::KeyOrigin;
    using tadah::mlip::KeyProvenance;
    const std::vector<std::pair<std::string, std::string>> defaults = {
        {"HPOFILEOUT", "refit.hpotarget"}, {"OUTFILE", "pot.tadah"},
        {"REFIT_OBJECTIVE", "energy"},     {"REFIT_CURVE_WEIGHTING", "true"},
        {"REFIT_CURVE_RELAX", "0.05"},     {"REFIT_CURVE_DERIV_WEIGHT", "0.0"},
        {"REFIT_COMPACT_MERGE", "auto"},   {"REFIT_CONSTRAIN_F0", "true"},
        {"IN_PLACE", "false"},             {"VERBOSE", "1"}};
    // `file`: the configuration file's keys; the defaults are added as the
    // blueprint's (as_defaults) or as written in the file
    const auto task_with = [&](const Context &file, bool as_defaults) {
      Context c = file, from_file = file;
      std::vector<std::string> defaulted;
      for (const auto &[key, value] : defaults) {
        c.add(key, value);
        if (as_defaults)
          defaulted.push_back(key);
        else
          from_file.add(key, value);
      }
      Task task("refit", c);
      task.setKeyProvenance(KeyProvenance(Context(), from_file, "edit.refit",
                                          KeyOrigin::Kind::ConfigFile,
                                          defaulted));
      return task;
    };
    for (const auto &f : outputs) fs::remove(f);
    Task e = task_with(edit(), true);
    CHECK_NOTHROW(RefitEngine().validate(e));
    CHECK(RefitEngine().run(e) == EXIT_SUCCESS);
    CHECK(fs::exists("pot_edited.tadah"));
    Task r = task_with(retarget(), true);
    CHECK_NOTHROW(RefitEngine().validate(r));
    CHECK(RefitEngine().run(r) == EXIT_SUCCESS);
    CHECK(fs::exists("a_retargeted.hpotarget"));
    // written in the configuration file, they are refused (the first of
    // the table each mode does not read)
    check_refused(task_with(edit(), false),
                  output_mode_refusal("--edit", "REFIT_OBJECTIVE", kFitReads));
    check_refused(task_with(retarget(), false),
                  output_mode_refusal("--retarget", "REFIT_OBJECTIVE",
                                      kFitReads));
  }
}

/* Pins: `tadah refit --edit` writes no weight covariance of its input: the
 * edited weights are not those any fit computed a covariance for, so the
 * edited potential (an --add of a phi basis function) has none of SIGMA,
 * WCOV and WNOISE, whichever the input carries (a SIGMA line, as Tadah!
 * 1.0.0 to 1.5.0 and the 1.6.0 development builds wrote it; WCOV and
 * WNOISE, as an OALGO 3 fit writes them, for its 6 weights).
 * Before the fix: the input's SIGMA was copied into the edited potential
 * beside the edited weights; WCOV and WNOISE were not keys (Context::add
 * threw "Key not allowed: WCOV"). */
TEST_CASE_METHOD(PTFixture,
                 "refit --edit writes no weight covariance of its input",
                 "[refit][edit][weight_covariance]") {
  for (const char *which : {"SIGMA", "WCOV and WNOISE"}) {
    DYNAMIC_SECTION(which) {
      ScopedTempDir tmp;
      Context in = make_knot_pot();
      if (std::string(which) == "SIGMA") {
        in.add("SIGMA", 6);
        for (int k = 0; k < 36; ++k) in.add("SIGMA", k % 7 == 0 ? 1.0 : 0.0);
      } else {
        in.add("WCOV", 6);
        for (int k = 0; k < 36; ++k) in.add("WCOV", k % 7 == 0 ? 1.0 : 0.0);
        in.add("WNOISE", "E 0.01");
      }
      TrainEngine::dumpPotFile("in.tadah", in);
      Context c;
      c.add("EDIT", std::string("in.tadah"));
      c.add("FUNC", std::string("phi"));
      c.add("ADD", 3.6);
      Task task("refit", c);
      REQUIRE(RefitEngine().run(task) == EXIT_SUCCESS);
      std::ifstream f("pot_edited.tadah");
      const std::string text((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
      INFO(text);
      for (const char *key : {"SIGMA", "WCOV", "WNOISE"}) {
        INFO(key);
        std::istringstream lines(text);
        bool found = false;
        for (std::string line; std::getline(lines, line);) {
          std::istringstream ls(line);
          std::string k;
          if ((ls >> k) && k == key) found = true;
        }
        CHECK_FALSE(found);
      }
    }
  }
}

TEST_CASE_METHOD(PTFixture,
                 "refit --edit and --retarget write different default "
                 "hpotargets",
                 "[refit][edit][retarget][integration]") {
  /* Pins: the default outputs of --edit and --retarget differ, so neither
   * overwrites the other's. On a potential pot.tadah and its hpotarget
   * refit.hpotarget (the names a refit writes), --edit (EDIT pot.tadah,
   * FUNC phi, ADD 3.6) writes pot_edited.tadah and refit_edited.hpotarget,
   * and --retarget refit.hpotarget (PCT 10) then writes
   * refit_retargeted.hpotarget, byte-identical to the file it writes with
   * -o, and leaves refit_edited.hpotarget byte-identical to what --edit
   * wrote. The derived name keeps the input's directory and extension
   * (sub/b.hpo -> sub/b_retargeted.hpo), and --retarget of the --edit
   * hpotarget writes refit_edited_retargeted.hpotarget.
   * Before the fix: --retarget's default was <input>_edited<ext>, so
   * --retarget refit.hpotarget wrote refit_edited.hpotarget over the
   * --edit file (the bounds around the unedited potential, which has one
   * pair function fewer) and refit_retargeted.hpotarget does not exist;
   * sub/b.hpo gave
   * sub/b_edited.hpo, refit_edited.hpotarget refit_edited_edited.hpotarget. */
  ScopedTempDir tmp;
  const auto slurp = [](const std::string &f) {
    std::ifstream is(f);
    return std::string((std::istreambuf_iterator<char>(is)),
                       std::istreambuf_iterator<char>());
  };
  TrainEngine::dumpPotFile("pot.tadah", make_knot_pot());
  tadah::mlip::engines::write_hpotarget("refit.hpotarget", make_knot_pot(),
                                        5.0);
  Context e;
  e.add("EDIT", std::string("pot.tadah"));
  e.add("FUNC", std::string("phi"));
  e.add("ADD", 3.6);
  Task edit("refit", e);
  REQUIRE(RefitEngine().run(edit) == EXIT_SUCCESS);
  REQUIRE(fs::exists("pot_edited.tadah"));
  const std::string edited = slurp("refit_edited.hpotarget");
  REQUIRE_FALSE(edited.empty());

  const auto retarget = [](const std::string &in, const std::string &out) {
    Context c;
    c.add("RETARGET", in);
    c.add("PCT", std::string("10"));
    if (!out.empty()) c.add("OUTFILE", out);
    Task task("refit", c);
    return RefitEngine().run(task);
  };
  REQUIRE(retarget("refit.hpotarget", "") == EXIT_SUCCESS);
  CHECK(slurp("refit_edited.hpotarget") == edited);
  CHECK(fs::exists("refit_retargeted.hpotarget"));
  REQUIRE(retarget("refit.hpotarget", "explicit.hpotarget") == EXIT_SUCCESS);
  CHECK(slurp("refit_retargeted.hpotarget") == slurp("explicit.hpotarget"));
  CHECK_FALSE(slurp("explicit.hpotarget").empty());

  fs::create_directories("sub");
  tadah::mlip::engines::write_hpotarget("sub/b.hpo", make_knot_pot(), 5.0);
  REQUIRE(retarget("sub/b.hpo", "") == EXIT_SUCCESS);
  CHECK(fs::exists("sub/b_retargeted.hpo"));
  CHECK_FALSE(fs::exists("sub/b_edited.hpo"));
  REQUIRE(retarget("refit_edited.hpotarget", "") == EXIT_SUCCESS);
  CHECK(fs::exists("refit_edited_retargeted.hpotarget"));
  CHECK(slurp("refit_edited.hpotarget") == edited);
}

/* Pins: --edit and --retarget write the files an explicit -o / --hpofileout
 * names, also when the name is the default one of a refit (pot.tadah,
 * refit.hpotarget): set on the command line or in the configuration file
 * (KeyProvenance CommandLine, ConfigFile), --edit writes pot.tadah and
 * refit.hpotarget and no pot_edited.tadah or refit_edited.hpotarget, and
 * --retarget writes pot.tadah (the hpotarget, under the name asked for) and
 * no a_retargeted.hpotarget; the blueprint defaults the task factory stores
 * (KeyOrigin::Default) still give the edited names. --retarget --in-place
 * with an explicit OUTFILE pot.tadah (a configuration or task file, which
 * the command line's excludes cannot see) is refused as -o is.
 * Before the fix: a value equal to the default string counted as unset, so
 * both modes silently wrote pot_edited.tadah / refit_edited.hpotarget /
 * a_retargeted.hpotarget instead, and --in-place ran with OUTFILE
 * pot.tadah. */
TEST_CASE_METHOD(PTFixture,
                 "refit --edit and --retarget honour an explicit output name "
                 "that is the default one",
                 "[refit][edit][retarget][output_names]") {
  using tadah::mlip::KeyOrigin;
  using tadah::mlip::KeyProvenance;
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("in.tadah", make_knot_pot());
  tadah::mlip::engines::write_hpotarget("a.hpotarget", make_knot_pot(), 5.0);
  const auto slurp = [](const std::string &f) {
    std::ifstream is(f);
    return std::string((std::istreambuf_iterator<char>(is)),
                       std::istreambuf_iterator<char>());
  };
  const std::vector<std::string> all = {
      "pot.tadah", "refit.hpotarget", "pot_edited.tadah",
      "refit_edited.hpotarget", "a_retargeted.hpotarget"};
  const auto clean = [&] {
    for (const auto &f : all) fs::remove(f);
  };
  using Keys = std::vector<std::pair<std::string, std::string>>;
  // `keys` set as `kind` (or added as the blueprint defaults `defaulted`
  // names); the other keys are the configuration file's
  const auto task_of = [](const Context &base, const Keys &keys,
                          KeyOrigin::Kind kind,
                          const std::vector<std::string> &defaulted = {}) {
    Context c = base, set;
    for (const auto &[k, v] : keys) {
      c.add(k, v);
      set.add(k, v);
    }
    Task task("refit", c);
    task.setKeyProvenance(
        kind == KeyOrigin::Kind::CommandLine
            ? KeyProvenance(set, base, "edit.refit",
                            KeyOrigin::Kind::ConfigFile, defaulted)
            : KeyProvenance(Context(), defaulted.empty() ? c : base,
                            "edit.refit", kind, defaulted));
    return task;
  };
  Context edit;
  edit.add("EDIT", std::string("in.tadah"));
  edit.add("FUNC", std::string("phi"));
  edit.add("ADD", 3.6);
  Context retarget;
  retarget.add("RETARGET", std::string("a.hpotarget"));
  const Keys names = {{"OUTFILE", "pot.tadah"},
                      {"HPOFILEOUT", "refit.hpotarget"}};
  const Keys out_only = {{"OUTFILE", "pot.tadah"}};
  for (const auto kind :
       {KeyOrigin::Kind::CommandLine, KeyOrigin::Kind::ConfigFile}) {
    INFO((kind == KeyOrigin::Kind::CommandLine ? "command line"
                                               : "configuration file"));
    clean();
    Task e = task_of(edit, names, kind);
    REQUIRE(RefitEngine().run(e) == EXIT_SUCCESS);
    CHECK(fs::exists("pot.tadah"));
    CHECK(fs::exists("refit.hpotarget"));
    CHECK_FALSE(fs::exists("pot_edited.tadah"));
    CHECK_FALSE(fs::exists("refit_edited.hpotarget"));
    clean();
    Task r = task_of(retarget, out_only, kind);
    REQUIRE(RefitEngine().run(r) == EXIT_SUCCESS);
    CHECK(fs::exists("pot.tadah"));
    CHECK(slurp("pot.tadah").find("OPTIM") != std::string::npos);
    CHECK_FALSE(fs::exists("a_retargeted.hpotarget"));
  }
  clean();
  Task e = task_of(edit, names, KeyOrigin::Kind::ConfigFile,
                   {"OUTFILE", "HPOFILEOUT"});
  REQUIRE(RefitEngine().run(e) == EXIT_SUCCESS);
  CHECK(fs::exists("pot_edited.tadah"));
  CHECK(fs::exists("refit_edited.hpotarget"));
  CHECK_FALSE(fs::exists("pot.tadah"));
  Keys in_place = out_only;
  in_place.push_back({"IN_PLACE", "true"});
  Task ip = task_of(retarget, in_place, KeyOrigin::Kind::TaskFile);
  CHECK_THROWS_WITH(RefitEngine().run(ip),
                    "tadah refit --retarget: --in-place and -o are mutually "
                    "exclusive.");
}


/* Pins: refit --edit of in.tadah -- the knot potential with the training
 * record "DBFILE train.db other.db", "VALIDATION config.val", "VERBOSE 0"
 * above its key lines -- writes out.tadah whose training record is "edited
 * by tadah refit --edit from in.tadah" followed by the input's three lines
 * (the edit fits nothing: its weights are the input's, edited, so the
 * input's record still describes them); an edit of out.tadah (twice.tadah)
 * lists both "edited by" lines, newest first, then the three lines; and
 * out.tadah reads back (CoreUserConfigParser) to the context, key by key,
 * of the edit of the same potential without the record (plain_out.tadah):
 * the record is no key of the edit.
 * Before the fix: neither edit had a training record (no potential had
 * one; the decision taken in MK's absence on 2026-10-12, replaced by MK's
 * of 2026-10-14).
 * Before the opening line was corrected (MK, 2026-10-14, option 2): the
 * edit's opening line was "# --- training record (not read by tadah) ---"
 * and in.tadah's (this one) was not recognised: refused at its end line. */
TEST_CASE_METHOD(PTFixture,
                 "refit --edit copies the training record of its input and "
                 "names the input",
                 "[refit][edit][potential_content][training_record]") {
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("plain.tadah", make_knot_pot());
  std::ofstream("in.tadah") << kRecordOpen << "\n"
                            << "# DBFILE train.db other.db\n"
                            << "# VALIDATION config.val\n"
                            << "# VERBOSE 0\n"
                            << kRecordEnd << "\n"
                            << slurp_file("plain.tadah");
  run_edit_capturing(edit_of("in.tadah", "out.tadah"));
  INFO("out.tadah:\n" << slurp_file("out.tadah"));
  CHECK(record_lines("out.tadah") ==
        std::vector<std::string>{"edited by tadah refit --edit from in.tadah",
                                 "DBFILE train.db other.db",
                                 "VALIDATION config.val", "VERBOSE 0"});
  run_edit_capturing(edit_of("out.tadah", "twice.tadah"));
  CHECK(record_lines("twice.tadah") ==
        std::vector<std::string>{
            "edited by tadah refit --edit from out.tadah",
            "edited by tadah refit --edit from in.tadah",
            "DBFILE train.db other.db", "VALIDATION config.val",
            "VERBOSE 0"});
  for (const char *key : {"DBFILE", "VALIDATION", "VERBOSE"}) {
    INFO("key " << key);
    CHECK(key_line("out.tadah", key).empty());
    CHECK(key_line("twice.tadah", key).empty());
  }
  run_edit_capturing(edit_of("plain.tadah", "plain_out.tadah"));
  const Context out = tadah::core::CoreUserConfigParser().parse("out.tadah");
  const Context plain =
      tadah::core::CoreUserConfigParser().parse("plain_out.tadah");
  CHECK(out.storage_snapshot() == plain.storage_snapshot());
}

/* Pins: refit --edit of a potential that has DBFILE, UNCERTAINTY and
 * VERBOSE key lines and no training record (as tadah 1.5.x and 1.6.0
 * development builds wrote them: the knot potential with "DBFILE old.db",
 * "UNCERTAINTY true" and "VERBOSE 0" appended, old.tadah) writes the edit
 * without those key lines and with the training record "edited by tadah
 * refit --edit from old.tadah", "DBFILE old.db", "UNCERTAINTY true",
 * "VERBOSE 0": the input's keys are kept in the record, not dropped. The
 * same for the keys that prepared the training data, which such a
 * potential has as key lines (old_data.tadah: "ESHIFT_ATOM true", "EFILTER
 * -9 -1", "WDBFILE 1 0.5" and "ZERO_COM_FORCE false" appended, as tadah
 * 1.5.x wrote every key of its training configuration): the edit's record
 * is "edited by tadah refit --edit from old_data.tadah", "EFILTER -9 -1",
 * "ESHIFT_ATOM true", "ZERO_COM_FORCE false", "WDBFILE 1 0.5" (the order of
 * the record), with no key line of them.
 * Before the fix: the key lines were dropped (a pin of that part) and the
 * edit had no record.
 * Before the dataset keys moved into the record (MK, 2026-10-14, option 2):
 * old_data.tadah's edit kept the four as key lines and its record had the
 * "edited by" line alone; the opening line of both records read
 * "(not read by tadah)", so record_lines found none (the record checks fail
 * for old_out.tadah too). */
TEST_CASE_METHOD(PTFixture,
                 "refit --edit of an older potential keeps its DBFILE, "
                 "UNCERTAINTY and VERBOSE in the edit's training record",
                 "[refit][edit][potential_content][training_record]") {
  ScopedTempDir tmp;
  const auto level = tadah::core::Logger::instance().getLogLevel();
  TrainEngine::dumpPotFile("plain.tadah", make_knot_pot());
  std::ofstream("old.tadah") << slurp_file("plain.tadah")
                             << "DBFILE old.db\nUNCERTAINTY true\n"
                                "VERBOSE 0\n";
  run_edit_capturing(edit_of("old.tadah", "old_out.tadah"));
  tadah::core::Logger::instance().setLogLevel(level);  // VERBOSE 0 was read
  INFO("old_out.tadah:\n" << slurp_file("old_out.tadah"));
  CHECK(record_lines("old_out.tadah") ==
        std::vector<std::string>{"edited by tadah refit --edit from old.tadah",
                                 "DBFILE old.db", "UNCERTAINTY true",
                                 "VERBOSE 0"});
  for (const char *key : {"DBFILE", "UNCERTAINTY", "VERBOSE"}) {
    INFO("key " << key);
    CHECK(key_line("old_out.tadah", key).empty());
  }
  std::ofstream("old_data.tadah") << slurp_file("plain.tadah")
                                  << "ESHIFT_ATOM true\nEFILTER -9 -1\n"
                                     "WDBFILE 1 0.5\nZERO_COM_FORCE false\n";
  run_edit_capturing(edit_of("old_data.tadah", "old_data_out.tadah"));
  INFO("old_data_out.tadah:\n" << slurp_file("old_data_out.tadah"));
  CHECK(record_lines("old_data_out.tadah") ==
        std::vector<std::string>{
            "edited by tadah refit --edit from old_data.tadah",
            "EFILTER -9 -1", "ESHIFT_ATOM true", "ZERO_COM_FORCE false",
            "WDBFILE 1 0.5"});
  for (const char *key :
       {"ESHIFT_ATOM", "EFILTER", "WDBFILE", "ZERO_COM_FORCE"}) {
    INFO("key " << key);
    CHECK(key_line("old_data_out.tadah", key).empty());
  }
}

/* Pins: refit --edit refuses an input whose training record cannot be
 * copied, naming the file and the cause, and writes no edit:
 *  - no end line, key lines after the record (open.tadah: the opening line,
 *    "# DBFILE x.db", then the key lines from line 3: every writer puts the
 *    key lines right after the record, so a potential whose end line was
 *    deleted has this shape): "open.tadah: line 1 opens a training record
 *    ('<opening line>') that no end line ('<end line>') closes before line
 *    3, the first line after it that is not a comment: restore the end line
 *    after the record, before line 3, or delete the record.", and not the
 *    key-line refusal ("is not a comment: every reader ...", whose advice
 *    would comment out line 3, a key line);
 *  - no end line before the end of the file (eof.tadah: the key lines, then
 *    the opening line and "# DBFILE x.db"): "eof.tadah: line <n> opens a
 *    training record ('<opening line>') that no end line ('<end line>')
 *    closes: restore the end line after the record, or delete the record.";
 *  - a second opening line before the end line (twice_open.tadah: lines 1
 *    and 3 open, line 5 ends): "twice_open.tadah: line 1 opens a training
 *    record (...) that no end line (...) closes before line 3, which opens
 *    another: restore the end line after the record, before line 3, ...";
 *  - an end line without an opening line (no_open.tadah: "# DBFILE x.db",
 *    the end line on line 2): "no_open.tadah: line 2 ends a training record
 *    ('<end line>') that no opening line ('<opening line>') opens: restore
 *    the opening line before the record, or delete the end line.";
 *  - the opening line as it read before its correction (old_open.tadah:
 *    "# --- training record (not read by tadah) ---", "# DBFILE x.db", the
 *    end line on line 3): "old_open.tadah: line 3 ends a training record
 *    ('<end line>') that no opening line ('<opening line>') opens: restore
 *    the opening line before the record, or delete the end line." (no
 *    shipped potential has that line);
 *  - a key line inside the block (keyline.tadah: line 2, "DBFILE x.db",
 *    between the opening line 1 and the end line 3): "keyline.tadah: line
 *    2, inside the training record that line 1 opens and line 3 ends, is
 *    not a comment: every reader of the potential reads it as a key. Begin
 *    it with '#' if it belongs to the record, or move it after the end
 *    line.", and not the missing-end-line refusal.
 * Before the fix: open.tadah was refused as a key line inside the record
 * ("open.tadah: line 3, inside the training record that line 1 opens, is
 * not a comment: ... Begin each line of the record with '#', ...": line 3
 * is its first key line), and this case passed on that message, which
 * holds "line 1", "training record" and the end line as well; eof.tadah
 * was refused as now (a pin of that part); twice_open.tadah and
 * no_open.tadah were edited (exit 0: the second opening line was copied as
 * a record line, and the record without its opening line was not copied);
 * keyline.tadah's refusal named no end line ("that line 1 opens, is not a
 * comment").
 * Before the training record (MK's decision of 2026-10-14): every edit was
 * written (exit 0).
 * Before the opening line was corrected (MK, 2026-10-14, option 2): every
 * section fails. The opening line of the files of the case was no opening
 * line: open.tadah and eof.tadah were edited (exit 0); twice_open.tadah,
 * no_open.tadah and keyline.tadah were refused at their end line, quoting
 * the opening line "(not read by tadah)", whose quote the checks do not
 * hold; old_open.tadah, whose opening line it was, was edited. */
TEST_CASE_METHOD(PTFixture,
                 "refit --edit refuses a training record without its end "
                 "line or with a key line inside",
                 "[refit][edit][potential_content][training_record]") {
  ScopedTempDir tmp;
  TrainEngine::dumpPotFile("plain.tadah", make_knot_pot());
  const std::string plain = slurp_file("plain.tadah");
  const std::string open_quoted = "('" + kRecordOpen + "')";
  const std::string end_quoted = "('" + kRecordEnd + "')";
  SECTION("no end line, key lines after the record") {
    std::ofstream("open.tadah") << kRecordOpen << "\n# DBFILE x.db\n"
                                << plain;
    CHECK_THROWS_WITH(
        run_edit_capturing(edit_of("open.tadah", "o.tadah")),
        Catch::Contains("open.tadah: line 1 opens a training record " +
                        open_quoted + " that no end line " + end_quoted +
                        " closes before line 3, the first line after it "
                        "that is not a comment: restore the end line after "
                        "the record, before line 3, or delete the record.") &&
            !Catch::Contains("is not a comment: every reader"));
    CHECK_FALSE(fs::exists("o.tadah"));
  }
  SECTION("no end line before the end of the file") {
    std::ofstream("eof.tadah") << plain << kRecordOpen << "\n# DBFILE x.db\n";
    const std::size_t opening =
        std::count(plain.begin(), plain.end(), '\n') + 1;
    CHECK_THROWS_WITH(
        run_edit_capturing(edit_of("eof.tadah", "e.tadah")),
        Catch::Contains("eof.tadah: line " + std::to_string(opening) +
                        " opens a training record " + open_quoted +
                        " that no end line " + end_quoted +
                        " closes: restore the end line after the record, or "
                        "delete the record."));
    CHECK_FALSE(fs::exists("e.tadah"));
  }
  SECTION("a second opening line before the end line") {
    std::ofstream("twice_open.tadah")
        << kRecordOpen << "\n# DBFILE x.db\n"
        << kRecordOpen << "\n# VERBOSE 0\n"
        << kRecordEnd << "\n"
        << plain;
    CHECK_THROWS_WITH(
        run_edit_capturing(edit_of("twice_open.tadah", "t.tadah")),
        Catch::Contains("twice_open.tadah: line 1 opens a training record " +
                        open_quoted + " that no end line " + end_quoted +
                        " closes before line 3, which opens another: "
                        "restore the end line after the record, before line "
                        "3, or delete the record."));
    CHECK_FALSE(fs::exists("t.tadah"));
  }
  SECTION("an end line without an opening line") {
    std::ofstream("no_open.tadah") << "# DBFILE x.db\n"
                                   << kRecordEnd << "\n"
                                   << plain;
    CHECK_THROWS_WITH(
        run_edit_capturing(edit_of("no_open.tadah", "n.tadah")),
        Catch::Contains("no_open.tadah: line 2 ends a training record " +
                        end_quoted + " that no opening line " + open_quoted +
                        " opens: restore the opening line before the record, "
                        "or delete the end line."));
    CHECK_FALSE(fs::exists("n.tadah"));
  }
  SECTION("the opening line before its correction") {
    // "(not read by tadah)": no writer of this version writes it, and no
    // shipped potential has it.
    std::ofstream("old_open.tadah")
        << "# --- training record (not read by tadah) ---\n"
        << "# DBFILE x.db\n"
        << kRecordEnd << "\n"
        << plain;
    CHECK_THROWS_WITH(
        run_edit_capturing(edit_of("old_open.tadah", "oo.tadah")),
        Catch::Contains("old_open.tadah: line 3 ends a training record " +
                        end_quoted + " that no opening line " + open_quoted +
                        " opens: restore the opening line before the record, "
                        "or delete the end line."));
    CHECK_FALSE(fs::exists("oo.tadah"));
  }
  SECTION("a key line inside the block") {
    std::ofstream("keyline.tadah") << kRecordOpen << "\nDBFILE x.db\n"
                                   << kRecordEnd << "\n"
                                   << plain;
    CHECK_THROWS_WITH(
        run_edit_capturing(edit_of("keyline.tadah", "k.tadah")),
        Catch::Contains("keyline.tadah: line 2, inside the training record "
                        "that line 1 opens and line 3 ends, is not a "
                        "comment: every reader of the potential reads it as "
                        "a key. Begin it with '#' if it belongs to the "
                        "record, or move it after the end line.") &&
            !Catch::Contains("no end line"));
    CHECK_FALSE(fs::exists("k.tadah"));
  }
}
