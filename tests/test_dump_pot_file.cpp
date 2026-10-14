// TrainEngine::dumpPotFile (train, refit, refit --edit): numbers follow
// NUMERIC as SIGNIFICANT digits, default 18 = bit-exact round trip; NUMERIC
// itself is never written; an explicit NUMERIC < 17 is honoured with a
// warning.
#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/number_format.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/models/eam/pot_curve_edit.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using tadah::core::Context;
using tadah::core::CoreUserConfigParser;
using tadah::mlip::engines::TrainEngine;
namespace fs = std::filesystem;

namespace {

std::uint64_t dp_bits(double v) {
  std::uint64_t u;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

const std::vector<double> kWeights = {0.1, 1e-20, -2.0 / 3,
                                      0.123456789012345678, 3.0, -0.0};

Context make_pot() {
  Context c;
  c.add("MODEL", "M_BLR BF_Linear");
  c.add("ATOM", "Ta");
  c.add("INIT2B", "true");
  c.add("TYPE2B", "D2_Blip 3 3 Ta Ta");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", 5.3);
  c.add("CGRID2B", "LIN 3 0.1 3.0");
  c.add("SGRID2B", std::vector<double>{0.7, 1.1, 1.0 / 3});
  c.add("LSCALE", 1.0 / 0.997);
  c.add("ESHIFT", -3.1234567890123456);
  c.add("WEIGHTS", kWeights);
  return c;
}

struct DumpTempDir {
  fs::path old_cwd, dir;
  DumpTempDir() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_dumppot_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~DumpTempDir() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

// Runs dumpPotFile with WARNING logging on and std::cout captured.
std::string dump_capturing(const std::string &file, const Context &c) {
  auto &log = tadah::core::Logger::instance();
  const auto old_level = log.getLogLevel();
  log.setLogLevel(tadah::core::LogLevel::WARNING);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  try {
    TrainEngine::dumpPotFile(file, c);
  } catch (...) {
    std::cout.rdbuf(old_buf);
    log.setLogLevel(old_level);
    throw;
  }
  std::cout.rdbuf(old_buf);
  log.setLogLevel(old_level);
  return captured.str();
}

std::string slurp(const std::string &file) {
  std::ifstream in(file);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

std::vector<std::string> line_tokens(const std::string &text,
                                     const std::string &key) {
  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    std::istringstream ls(line);
    std::string k;
    if (!(ls >> k) || k != key) continue;
    std::vector<std::string> t;
    std::string s;
    while (ls >> s) t.push_back(s);
    return t;
  }
  return {};
}

}  // namespace

TEST_CASE("dumpPotFile: default writes 18 significant digits, no NUMERIC, "
          "exact re-parse",
          "[train][dumpPotFile][lossless]") {
  DumpTempDir tmp;
  const Context pot = make_pot();
  REQUIRE_FALSE(pot.has_stored("NUMERIC"));
  const std::string log = dump_capturing("pot.tadah", pot);
  CHECK(log.find("NUMERIC") == std::string::npos);  // no warning

  const std::string text = slurp("pot.tadah");
  INFO(text);
  CHECK(text.find("NUMERIC") == std::string::npos);
  CHECK(line_tokens(text, "WEIGHTS") ==
        std::vector<std::string>{"0.100000000000000006",
                                 "9.99999999999999945e-21",
                                 "-0.66666666666666663",
                                 "0.123456789012345677", "3", "-0"});
  CHECK(line_tokens(text, "CGRID2B") ==
        std::vector<std::string>{"LIN", "3", "0.100000000000000006", "3"});
  CHECK(line_tokens(text, "TYPE2B") ==
        std::vector<std::string>{"D2_Blip", "3", "3", "Ta", "Ta"});
  CHECK(line_tokens(text, "RCUT2B") ==
        std::vector<std::string>{"5.29999999999999982"});

  const Context back = CoreUserConfigParser().parse("pot.tadah");
  CHECK_FALSE(back.has_stored("NUMERIC"));
  REQUIRE(back.size("WEIGHTS") == kWeights.size());
  for (std::size_t i = 0; i < kWeights.size(); ++i)
    CHECK(dp_bits(back.get<double>("WEIGHTS", i)) == dp_bits(kWeights[i]));
  CHECK(dp_bits(back.get<double>("LSCALE")) == dp_bits(1.0 / 0.997));
  CHECK(dp_bits(back.get<double>("ESHIFT")) == dp_bits(-3.1234567890123456));
  CHECK(dp_bits(back.get<double>("SGRID2B", 2)) == dp_bits(1.0 / 3));
  CHECK(dp_bits(back.get<double>("RCUT2B")) == dp_bits(5.3));

  // A second dump of the re-parsed potential is byte-identical (fixed point).
  const std::string log2 = dump_capturing("pot2.tadah", back);
  (void)log2;
  CHECK(slurp("pot2.tadah") == text);
}

TEST_CASE("dumpPotFile: explicit NUMERIC 12 -> 12 digits and a warning",
          "[train][dumpPotFile][lossless]") {
  DumpTempDir tmp;
  Context pot = make_pot();
  pot.add("NUMERIC", 12);
  const std::string log = dump_capturing("pot.tadah", pot);
  CHECK(log.find("WARNING") != std::string::npos);
  CHECK(log.find("NUMERIC 12") != std::string::npos);

  const std::string text = slurp("pot.tadah");
  INFO(text);
  CHECK(text.find("NUMERIC") == std::string::npos);
  CHECK(line_tokens(text, "WEIGHTS") ==
        std::vector<std::string>{"0.1", "1e-20", "-0.666666666667",
                                 "0.123456789012", "3", "-0"});
  CHECK(line_tokens(text, "LSCALE") ==
        std::vector<std::string>{tadah::core::format_sig(1.0 / 0.997, 12)});

  // NUMERIC 17 and above: exact, no warning.
  Context pot17 = make_pot();
  pot17.add("NUMERIC", 17);
  const std::string log17 = dump_capturing("pot17.tadah", pot17);
  CHECK(log17.find("NUMERIC") == std::string::npos);
  const Context back = CoreUserConfigParser().parse("pot17.tadah");
  for (std::size_t i = 0; i < kWeights.size(); ++i)
    CHECK(dp_bits(back.get<double>("WEIGHTS", i)) == dp_bits(kWeights[i]));
}

/* Pins: dumpPotFile (tadah train, refit, refit --edit) with NUMERIC 12, 6 and
 * 3 writes the numbers of the cutoff and grid keys exactly, as the shortest
 * text that reads back as the same double: RCUT2B '5.3', CGRID2B 'LIN 3 0.1
 * 3', SGRID2B '0.7 1.1 0.3333333333333333' (they read back bit-exact), and
 * every other number with NUMERIC significant digits (WEIGHTS, LSCALE,
 * ESHIFT); its WARNING says which keys are written exactly. A D2_Knot5
 * potential whose RCUT2B is its largest knot (5.58190742024) reloads with
 * that RCUT2B bit-exact and normalize_knot_rcut (train, predict, refit --edit
 * and hpo startup) leaves it unchanged.
 * Before the fix: the cutoff and grid keys were rounded with the rest
 * (SGRID2B '0.7 1.1 0.333333333333' at NUMERIC 12, RCUT2B '5.58' for the
 * knot potential at 3; the knot relation held only because both were
 * rounded alike), and the WARNING said NUMERIC limits every non-integer
 * number. */
TEST_CASE("dumpPotFile: NUMERIC below 17 writes the cutoff and grid keys "
          "exactly and rounds the other numbers",
          "[train][dumpPotFile][lossless][exact_in_potential]") {
  DumpTempDir tmp;
  for (const int n : {12, 6, 3}) {
    INFO("NUMERIC " << n);
    Context pot = make_pot();
    pot.add("NUMERIC", n);
    const std::string log = dump_capturing("pot.tadah", pot);
    CHECK(log.find("and so are the numbers of CGRID2B, CGRIDMB, RCUT2B, "
                   "RCUTMB, SGRID2B and SGRIDMB") != std::string::npos);
    const std::string text = slurp("pot.tadah");
    INFO(text);
    CHECK(line_tokens(text, "RCUT2B") == std::vector<std::string>{"5.3"});
    CHECK(line_tokens(text, "CGRID2B") ==
          std::vector<std::string>{"LIN", "3", "0.1", "3"});
    CHECK(line_tokens(text, "SGRID2B") ==
          std::vector<std::string>{"0.7", "1.1", "0.3333333333333333"});
    std::vector<std::string> w;
    for (double v : kWeights) {
      const std::string t = tadah::core::format_roundtrip(v);
      w.push_back(tadah::core::is_integer_literal(t)
                      ? t
                      : tadah::core::format_sig(v, n));
    }
    CHECK(line_tokens(text, "WEIGHTS") == w);
    CHECK(line_tokens(text, "LSCALE") ==
          std::vector<std::string>{tadah::core::format_sig(1.0 / 0.997, n)});
    CHECK(line_tokens(text, "ESHIFT") ==
          std::vector<std::string>{
              tadah::core::format_sig(-3.1234567890123456, n)});
    const Context back = CoreUserConfigParser().parse("pot.tadah");
    CHECK(dp_bits(back.get<double>("SGRID2B", 2)) == dp_bits(1.0 / 3));
    CHECK(dp_bits(back.get<double>("RCUT2B")) == dp_bits(5.3));
  }

  for (const int n : {6, 3}) {
    INFO("D2_Knot5, NUMERIC " << n);
    Context knot;
    knot.add("MODEL", "M_BLR BF_Linear");
    knot.add("ATOM", "Ta");
    knot.add("INIT2B", "true");
    knot.add("TYPE2B", "D2_Knot5 3 Ta Ta");
    knot.add("RCTYPE2B", "Cut_Dummy");
    knot.add("RCUT2B", "5.58190742024");
    knot.add("CGRID2B", "2.123456789 3.987654321 5.58190742024");
    knot.add("WEIGHTS", std::vector<double>{1.0 / 3, -2.0 / 3, 0.25});
    knot.add("NUMERIC", n);
    (void)dump_capturing("knot.tadah", knot);
    const std::string text = slurp("knot.tadah");
    INFO(text);
    CHECK(line_tokens(text, "RCUT2B") ==
          std::vector<std::string>{"5.58190742024"});
    CHECK(line_tokens(text, "CGRID2B") ==
          std::vector<std::string>{"2.123456789", "3.987654321",
                                   "5.58190742024"});
    Context back = CoreUserConfigParser().parse("knot.tadah");
    CHECK_FALSE(tadah::models::potedit::normalize_knot_rcut(back, "test"));
    CHECK(dp_bits(back.get<double>("RCUT2B")) == dp_bits(5.58190742024));
  }
}

TEST_CASE("dumpPotFile and write_hpotarget: an output that cannot be written "
          "is an error naming the path",
          "[dump_pot][writer]") {
  // dumpPotFile logged an ERROR and returned, write_hpotarget a WARNING, so
  // `tadah train -o no_such_dir/pot.tadah` exited 0 without a potential.
  DumpTempDir tmp;
  const Context pot = make_pot();
  REQUIRE_THROWS_WITH(TrainEngine::dumpPotFile("no_such_dir/pot.tadah", pot),
                      Catch::Contains("no_such_dir/pot.tadah"));
  REQUIRE_THROWS_WITH(tadah::mlip::engines::write_hpotarget(
                          "no_such_dir/refit.hpotarget", pot, 5.0),
                      Catch::Contains("no_such_dir/refit.hpotarget"));
  REQUIRE_FALSE(fs::exists("no_such_dir"));
  // a directory in place of the file
  fs::create_directories("a_dir");
  REQUIRE_THROWS_WITH(TrainEngine::dumpPotFile("a_dir", pot),
                      Catch::Contains("a_dir"));
  REQUIRE_THROWS_WITH(
      tadah::mlip::engines::write_hpotarget("a_dir", pot, 5.0),
      Catch::Contains("a_dir"));
}

/* Pins: dumpPotFile, the writer of tadah train, tadah refit and refit --edit
 * (whose potential is a copy of its input's context), writes no DBFILE line
 * for a context with DBFILE db1.tadah db2.tadah (an M_BLR potential written
 * before the potential-content fix records the DBFILE it was trained on),
 * and keeps the settings of the fit beside it: ALPHA 0.25 and BETA 4; no
 * WDBFILE line either for its WDBFILE 1 0.5, the per-dataset weights of the
 * training data, which a potential keeps in its training record only (MK's
 * decision of 2026-10-14, option 2). Given no training record (the record
 * is its caller's, from the run's configuration: TrainEngine::run,
 * RefitEngine::run, run_refit_edit), it writes none, so db1.tadah appears
 * nowhere in the file.
 * Before the fix: the DBFILE line was written (`DBFILE db1.tadah
 * db2.tadah`), so the potential, used as a training configuration with the
 * DBFILE lines of the retrain added, listed the datasets twice.
 * Before the training record (MK's decision of 2026-10-14): passes as well
 * (a pin); the case was named "dumpPotFile writes no DBFILE: a potential
 * records no training data".
 * Before the dataset keys moved into the record (MK, 2026-10-14, option 2):
 * the case pinned the key line "WDBFILE 1 0.5" (a retrain listing the same
 * datasets in the same order read it as the first fit did): its WDBFILE
 * check fails. */
TEST_CASE("dumpPotFile writes no DBFILE line: the DBFILE of its context is "
          "no key of the potential",
          "[train][dumpPotFile][potential_content]") {
  DumpTempDir tmp;
  Context pot = make_pot();
  pot.add("DBFILE", "db1.tadah");
  pot.add("DBFILE", "db2.tadah");
  pot.add("ALPHA", "0.25");
  pot.add("BETA", "4");
  pot.add("WDBFILE", "1");
  pot.add("WDBFILE", "0.5");
  dump_capturing("pot.tadah", pot);

  const std::string text = slurp("pot.tadah");
  INFO(text);
  CHECK(line_tokens(text, "DBFILE").empty());
  CHECK(text.find("db1.tadah") == std::string::npos);
  CHECK(line_tokens(text, "ALPHA") == std::vector<std::string>{"0.25"});
  CHECK(line_tokens(text, "BETA") == std::vector<std::string>{"4"});
  CHECK(line_tokens(text, "WDBFILE").empty());
}

/* Pins: dumpPotFile (tadah train, tadah refit and refit --edit) writes no
 * key line of an input of the run that made its context, whatever the
 * context holds: no HPOTARGET and no VALIDATION line (the target and
 * validation files of a tadah hpo run, which a best_pot.tadah used as a
 * training configuration carried: HPOTARGET config.hpo, VALIDATION
 * config.val), and no VERBOSE (1), NUMERIC (18), OUTFILE (out.tadah),
 * AUDIT (warn), AUDIT_SAMPLE (64) or HEALTH_LOG (full) line; the model's
 * settings and fit beside them stay (MODEL M_BLR BF_Linear, RCUT2B, the
 * six WEIGHTS). Given no training record (its caller's, from the run's
 * configuration), it writes none: config.hpo and config.val appear
 * nowhere in the file.
 * Before the fix: the HPOTARGET, VALIDATION and AUDIT_SAMPLE lines were
 * written ("HPOTARGET config.hpo", "VALIDATION config.val", "AUDIT_SAMPLE
 * 64"); the other five were not (a pin of those).
 * Before the training record (MK's decision of 2026-10-14): passes as well
 * (a pin); the case was named "dumpPotFile writes no input of the run:
 * ...". */
TEST_CASE("dumpPotFile writes no key line of the run's inputs: HPOTARGET, "
          "VALIDATION, VERBOSE, NUMERIC, OUTFILE, AUDIT, AUDIT_SAMPLE, "
          "HEALTH_LOG",
          "[train][dumpPotFile][potential_content]") {
  DumpTempDir tmp;
  auto &log = tadah::core::Logger::instance();
  const auto old_level = log.getLogLevel();
  Context pot = make_pot();
  pot.add("HPOTARGET", "config.hpo");
  pot.add("VALIDATION", "config.val");
  pot.add("VERBOSE", 1);  // Context::add sets the Logger level: restored
  pot.add("NUMERIC", 18);
  pot.add("OUTFILE", "out.tadah");
  pot.add("AUDIT", "warn");
  pot.add("AUDIT_SAMPLE", 64);
  pot.add("HEALTH_LOG", "full");
  log.setLogLevel(old_level);
  dump_capturing("pot.tadah", pot);

  const std::string text = slurp("pot.tadah");
  INFO(text);
  for (const char *key : {"HPOTARGET", "VALIDATION", "VERBOSE", "NUMERIC",
                          "OUTFILE", "AUDIT", "AUDIT_SAMPLE", "HEALTH_LOG"}) {
    INFO("key " << key);
    CHECK(line_tokens(text, key).empty());
  }
  CHECK(text.find("config.hpo") == std::string::npos);
  CHECK(text.find("config.val") == std::string::npos);
  CHECK(line_tokens(text, "MODEL") ==
        std::vector<std::string>{"M_BLR", "BF_Linear"});
  CHECK(line_tokens(text, "RCUT2B") ==
        std::vector<std::string>{"5.29999999999999982"});
  CHECK(line_tokens(text, "WEIGHTS").size() == kWeights.size());
  log.setLogLevel(old_level);
}

/* Pins: dumpPotFile writes no UNCERTAINTY line, whatever its context holds:
 * UNCERTAINTY is a switch of the train task (weights.tadah), not part of
 * the model, and no key of a potential (a potential's training record, a
 * comment block, lists the value of the run that wrote it). A context with
 * UNCERTAINTY true -- refit --edit copies its input's context, and a
 * potential of tadah 1.5.x or a 1.6.0 development build has the line --
 * is written without it; the model's settings and fit stay (MODEL, the six
 * WEIGHTS).
 * Before the fix: remove_run_inputs left UNCERTAINTY, which only M_BLR's
 * and M_KRR's get_param_file removed, so refit --edit of such a potential
 * wrote "UNCERTAINTY true" (lanes/D/tmp/review/training-verify/unc: edited
 * twice, the line kept; the retrain then warned that 1.5.x and development
 * builds wrote it).
 * Before the training record (MK's decision of 2026-10-14): passes as well
 * (a pin); the case was named "dumpPotFile writes no UNCERTAINTY: a
 * potential does not record the switch of the run". */
TEST_CASE("dumpPotFile writes no UNCERTAINTY line: the switch of the run "
          "is no key of the potential",
          "[train][dumpPotFile][potential_content][uncertainty]") {
  DumpTempDir tmp;
  Context pot = make_pot();
  pot.add("UNCERTAINTY", "true");
  dump_capturing("pot.tadah", pot);

  const std::string text = slurp("pot.tadah");
  INFO(text);
  CHECK(line_tokens(text, "UNCERTAINTY").empty());
  CHECK(line_tokens(text, "MODEL") ==
        std::vector<std::string>{"M_BLR", "BF_Linear"});
  CHECK(line_tokens(text, "WEIGHTS").size() == kWeights.size());
}
