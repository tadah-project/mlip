// test_train_engine_datasets.cpp
//
// TrainEngine::run (tadah train, a TASK train of a task file) and its
// training datasets: a potential has no DBFILE line (its training record, a
// comment block no loader of a potential reads, lists the datasets of its
// fit when there were any), so `tadah train -c pot.tadah` without a DBFILE
// line is the likely mistake on the documented retrain route; the
// configuration is refused before any file is read, naming the key, saying
// that a potential has no DBFILE line and how to retrain from one. The
// training record a potential of tadah train carries: the run's own keys, a
// retrain's too.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/task.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::mlip::Task;
using tadah::mlip::engines::TrainEngine;
namespace fs = std::filesystem;

/* A Ta pair potential as tadah train writes it (D2_BP, two functions, its
 * WEIGHTS), no DBFILE line, with the model @p model. */
Context potential_context(const std::string &model) {
  Context c;
  c.add("ATOM", "Ta");
  c.add("WATOM", 73);
  c.add("INIT2B", "true");
  c.add("INITMB", "false");
  c.add("TYPE2B", "D2_BP 2 2 Ta Ta");
  c.add("CGRID2B", "2.6 3.3");
  c.add("SGRID2B", "1.0 1.0");
  c.add("RCTYPE2B", "Cut_Cos");
  c.add("RCUT2B", 5.0);
  c.add("BIAS", "false");
  c.add("NORM", "false");
  c.add("LAMBDA", 0);
  c.add("OALGO", 1);
  c.add("MODEL", model);
  if (model.rfind("M_KRR Kern_RBF", 0) == 0) {
    c.add("MPARAMS", 1.0);
    c.add("SBASIS", 4);
  }
  c.add("WEIGHTS", "0.25 -0.5");
  return c;
}

/* The working directory of a case: tadah train writes its OUTFILE there. */
struct ScratchCwd {
  fs::path old_cwd, dir;
  ScratchCwd() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_train_datasets_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++));
    fs::create_directories(dir);
    fs::current_path(dir);
  }
  ~ScratchCwd() {
    fs::current_path(old_cwd);
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

/* The advice every refusal of a configuration without a training DBFILE
 * ends with (potential_has_no_datasets_advice). */
const std::string kNoDatasetsAdvice =
    "A potential has no DBFILE line (no potential tadah writes one; its "
    "training record, the comment block above its key lines, lists the "
    "datasets its weights were fitted to in a '# DBFILE' line when there "
    "were any: a refit has none, and a potential of an earlier version has "
    "no record): to train from one, add a DBFILE line to a copy of it; a "
    "potential of tadah refit or refit --edit also pins every weight with "
    "its FIXINDEX and FIXWEIGHT lines, the seed of a tadah hpo coefficient "
    "search, so delete them from the copy for the fit to change the weights "
    "(the Training page of the documentation, Retraining from a potential).";

/* What TrainEngine::run throws for @p c, "" when it does not throw; what
 * it prints on std::cout (the Logger's lines) goes to @p printed. */
std::string train_error(const Context &c, std::string &printed) {
  Task task("train", c);
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  std::string what;
  try {
    TrainEngine().run(task);
  } catch (const std::exception &e) {
    what = e.what();
  }
  std::cout.rdbuf(old_buf);
  printed = captured.str();
  return what;
}

}  // namespace

/* Pins: TrainEngine::run of a potential with no DBFILE line (an M_BLR
 * BF_Linear and an M_KRR Kern_RBF 1 (SBASIS 4) Ta pair potential, OUTFILE
 * pot2.tadah) throws "The training configuration has no DBFILE: tadah train
 * fits the datasets it lists." followed by the advice every refusal of a
 * configuration without a training DBFILE gives (kNoDatasetsAdvice: a
 * potential has no DBFILE line, its training record lists the datasets
 * its weights were fitted to when there were any, which a refit and a
 * potential of an earlier version have not; add a DBFILE line to a copy of
 * it, and delete the FIXINDEX and FIXWEIGHT lines with which a refit
 * potential pins every weight), prints no "Key not found" line, and writes
 * no pot2.tadah.
 * Before the fix: the DBFILE read of find_unique_elements threw "Key not
 * found in the Context object: DBFILE", after the Logger's ERROR line
 * "Context::operator() -> Key not found: DBFILE".
 * Before the advice named the pins (cross-lane review 2): it ended "add a
 * DBFILE line to a copy of it (the Training page of the documentation,
 * Retraining from a potential).", and a refit potential so retrained
 * fitted nothing (the next test case).
 * Before the training record (MK's decision of 2026-10-14): the advice
 * began "A potential records no DBFILE (no potential tadah writes has
 * one): to train from one, ...", and the case was named "..., saying that
 * a potential records none".
 * Before the advice said when the record lists a DBFILE (review of the
 * training record): it said "the DBFILE of the run that wrote it is in its
 * training record, a comment block tadah does not read", which a refit
 * potential, an hpo run's without training datasets and one of an earlier
 * version contradict: this check fails. */
TEST_CASE("TrainEngine::run refuses a configuration without DBFILE, saying "
          "that a potential has no DBFILE line",
          "[train][potential_content]") {
  tadah::core::PeriodicTable::initialize();
  const std::string advice = kNoDatasetsAdvice;
  for (const char *model : {"M_BLR BF_Linear", "M_KRR Kern_RBF 1"})
    DYNAMIC_SECTION(model) {
      ScratchCwd cwd;
      Context c = potential_context(model);
      c.add("OUTFILE", "pot2.tadah");
      std::string printed;
      const std::string what = train_error(c, printed);
      INFO("thrown: " << what << "\nprinted:\n" << printed);
      CHECK(what == "The training configuration has no DBFILE: tadah train "
                    "fits the datasets it lists. " + advice);
      CHECK(printed.find("Key not found") == std::string::npos);
      CHECK_FALSE(fs::exists("pot2.tadah"));
    }
}

namespace {

/* Three four-atom bcc Ta structures (energies -0.001, -0.04, 0.02, each
 * plus @p de), no forces, written to @p path. */
void write_ta_dataset(const fs::path &path, double de = 0.0) {
  std::ofstream os(path);
  const double energies[] = {-0.001 + de, -0.04 + de, 0.02 + de};
  const double offsets[] = {0.0, 0.07, -0.05};
  for (int k = 0; k < 3; ++k) {
    const double x = offsets[k];
    os << "ta_" << k << "\n1.0 1.0 1.0\n" << energies[k] << " 0.0\n"
       << "3.301 0.0 0.0\n0.0 3.301 0.0\n0.0 0.0 6.602\n"
       << "0.0 0.0 0.0\n0.0 0.0 0.0\n0.0 0.0 0.0\n"
       << "Ta " << x << " 0.0 0.0  0.0 0.0 0.0\n"
       << "Ta " << 1.6505 + x << " 1.6505 1.6505  0.0 0.0 0.0\n"
       << "Ta " << x << " 0.0 3.301  0.0 0.0 0.0\n"
       << "Ta " << 1.6505 + x << " 1.6505 4.9515  0.0 0.0 0.0\n\n";
  }
}

/* The Logger at LogLevel::WARNING for a scope (its WARNING lines are
 * printed), its level restored after. */
struct LoggerAtWarning {
  tadah::core::LogLevel old = tadah::core::Logger::instance().getLogLevel();
  LoggerAtWarning() {
    tadah::core::Logger::instance().setLogLevel(
        tadah::core::LogLevel::WARNING);
  }
  ~LoggerAtWarning() { tadah::core::Logger::instance().setLogLevel(old); }
};

/* The tokens of the line of @p key in the potential file @p p. */
std::vector<std::string> pot_line(const fs::path &p, const std::string &key) {
  std::ifstream in(p);
  for (std::string line; std::getline(in, line);) {
    std::istringstream ls(line);
    std::string k;
    if (!(ls >> k) || k != key) continue;
    std::vector<std::string> tok;
    for (std::string t; ls >> t;) tok.push_back(t);
    return tok;
  }
  return {};
}

}  // namespace

/* Pins: a training whose FIXINDEX and FIXWEIGHT pin every weight fits none
 * (the regressor is skipped: MODELS M_BLR_Train, every column fixed), and
 * TrainEngine::run says so in a WARNING, naming the potentials that pin
 * every weight and what to delete: "tadah train: FIXINDEX and FIXWEIGHT pin
 * every weight (all 2 of them), so the fit changes none: the WEIGHTS of the
 * potential written are the FIXWEIGHT values. tadah refit and refit --edit
 * write FIXINDEX and FIXWEIGHT for every weight of their potentials, the
 * seed of a tadah hpo coefficient search, and so does a tadah hpo run that
 * pins every weight in its best_pot.tadah: to fit the weights to the
 * DBFILE data, delete both lines from the training configuration (the
 * Training page of the documentation, Retraining from a potential)." -- the
 * Ta pair potential of the case above with a DBFILE of three bcc Ta
 * structures:
 *   - M_BLR BF_Linear, FIXINDEX 1-2, FIXWEIGHT 0.25 -0.5: the WARNING, once,
 *     and pot2.tadah has WEIGHTS 0.25 -0.5;
 *   - M_KRR Kern_RBF 1 (SBASIS 4), FIXINDEX 1-4: the WARNING with "all 4 of
 *     them";
 *   - M_BLR with FIXINDEX 1 alone (FIXWEIGHT 0.25), which fits weight 2: no
 *     such WARNING.
 * Not a refusal: such a fit can be deliberate (under OALGO 3 it gives the
 * noise of the data about the pinned weights).
 * Before the fix: no WARNING: tadah train exited 0 with the input weights
 * and said nothing (lanes/D/tmp/review/training2/refit: a dense refit
 * potential, FIXINDEX 1-171, retrained on eos_bcc.tadahdb, WEIGHTS
 * byte-identical). */
TEST_CASE("TrainEngine::run warns when FIXINDEX pins every weight, so the "
          "fit changes none",
          "[train][potential_content][fixindex]") {
  tadah::core::PeriodicTable::initialize();
  const auto warning = [](std::size_t k) {
    return "tadah train: FIXINDEX and FIXWEIGHT pin every weight (all " +
           std::to_string(k) +
           " of them), so the fit changes none: the WEIGHTS of the potential "
           "written are the FIXWEIGHT values. tadah refit and refit --edit "
           "write FIXINDEX and FIXWEIGHT for every weight of their "
           "potentials, the seed of a tadah hpo coefficient search, and so "
           "does a tadah hpo run that pins every weight in its "
           "best_pot.tadah: to fit the weights to the DBFILE data, delete "
           "both lines from the training configuration (the Training page of "
           "the documentation, Retraining from a potential).";
  };
  const std::string head = "FIXINDEX and FIXWEIGHT pin every weight";
  struct Case {
    const char *model;
    const char *fixindex;
    const char *fixweight;
    std::size_t pinned_all;  // 0: not every weight
  };
  for (const Case &cs : {Case{"M_BLR BF_Linear", "1-2", "0.25 -0.5", 2},
                         Case{"M_KRR Kern_RBF 1", "1-4",
                              "0.25 -0.5 0.125 0.0625", 4},
                         Case{"M_BLR BF_Linear", "1", "0.25", 0}})
    DYNAMIC_SECTION(cs.model << ", FIXINDEX " << cs.fixindex) {
      ScratchCwd cwd;
      LoggerAtWarning at_warning;
      write_ta_dataset(cwd.dir / "db.tadah");
      Context c = potential_context(cs.model);
      c.add("DBFILE", (cwd.dir / "db.tadah").string());
      c.add("FIXINDEX", cs.fixindex);
      c.add("FIXWEIGHT", cs.fixweight);
      c.add("OUTFILE", "pot2.tadah");
      std::string printed;
      const std::string what = train_error(c, printed);
      INFO("thrown: " << what << "\nprinted:\n" << printed);
      REQUIRE(what.empty());
      REQUIRE(fs::exists("pot2.tadah"));
      if (cs.pinned_all) {
        CHECK(printed.find("WARNING: " + warning(cs.pinned_all)) !=
              std::string::npos);
        CHECK(printed.find(head) == printed.rfind(head));
        std::istringstream fw(cs.fixweight);
        std::vector<std::string> pins;
        for (std::string t; fw >> t;) pins.push_back(t);
        const std::vector<std::string> w = pot_line("pot2.tadah", "WEIGHTS");
        REQUIRE(w.size() == pins.size());
        for (std::size_t i = 0; i < w.size(); ++i)
          CHECK(std::stod(w[i]) == std::stod(pins[i]));
      } else {
        CHECK(printed.find(head) == std::string::npos);
      }
    }
}

namespace {

/* The opening and end lines of the training record of a potential file. */
const std::string kRecordOpen = "# --- training record (not read by tadah) ---";
const std::string kRecordEnd = "# --- end of training record ---";

std::string slurp(const fs::path &p) {
  std::ifstream in(p);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

std::vector<std::string> file_lines(const fs::path &p) {
  std::ifstream in(p);
  std::vector<std::string> lines;
  for (std::string l; std::getline(in, l);) lines.push_back(l);
  return lines;
}

/* The training record of the potential file @p p as it is written: the
 * index of its opening line (found), of its end line (closed), and the
 * lines between them without their leading "# ". */
struct Record {
  bool found = false, closed = false;
  std::size_t open = 0, end = 0;
  std::vector<std::string> lines;
};
Record read_record(const fs::path &p) {
  Record r;
  const std::vector<std::string> lines = file_lines(p);
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (!r.found) {
      if (lines[i] == kRecordOpen) {
        r.found = true;
        r.open = i;
      }
      continue;
    }
    if (lines[i] == kRecordEnd) {
      r.closed = true;
      r.end = i;
      break;
    }
    r.lines.push_back(lines[i].rfind("# ", 0) == 0 ? lines[i].substr(2)
                                                    : lines[i]);
  }
  return r;
}

/* The index of the first line of @p p that is neither blank nor a
 * comment: the first key line. */
std::size_t first_key_line(const fs::path &p) {
  const std::vector<std::string> lines = file_lines(p);
  for (std::size_t i = 0; i < lines.size(); ++i) {
    std::istringstream ls(lines[i]);
    std::string k;
    if ((ls >> k) && k[0] != '#') return i;
  }
  return lines.size();
}

/* @p p without the lines of its training record (opening and end lines
 * included). */
std::string without_record(const fs::path &p) {
  std::string out;
  bool in_block = false;
  for (const std::string &l : file_lines(p)) {
    if (l == kRecordOpen) in_block = true;
    if (!in_block) out += l + "\n";
    if (l == kRecordEnd) in_block = false;
  }
  return out;
}

/* The keys a potential keeps only in its training record. */
const char *const kRecordKeys[] = {"DBFILE",  "VALIDATION", "HPOTARGET",
                                   "UNCERTAINTY", "VERBOSE", "NUMERIC",
                                   "OUTFILE", "AUDIT",      "AUDIT_SAMPLE",
                                   "HEALTH_LOG"};

}  // namespace

/* Pins: TrainEngine::run of an M_BLR BF_Linear Ta pair potential whose
 * configuration sets every key a potential leaves out of its key lines --
 * DBFILE ./db1.tadah ./db2.tadah (two datasets), VALIDATION config.val,
 * HPOTARGET config.hpo (a best_pot.tadah of an earlier version used as the
 * configuration had them), UNCERTAINTY false, VERBOSE 0, NUMERIC 9,
 * OUTFILE pot.tadah, AUDIT off, AUDIT_SAMPLE 64, HEALTH_LOG full -- writes
 * pot.tadah whose first line opens its training record ("# --- training
 * record (not read by tadah) ---"), then one "# KEY values" line per key
 * in that order with the values as the configuration has them (./db1.tadah
 * not normalised; under NUMERIC 9, which governs the key lines only), then
 * "# --- end of training record ---", before the first key line; no key
 * line of any of the ten; and the file parses (CoreUserConfigParser, the
 * reader of tadah predict, tadah train -c, tadah hpo -c and pair_style
 * tadah) to the same context, key by key, as the file without the block,
 * with none of the ten keys.
 * Before the fix: no training record: the ten keys were dropped from the
 * potential (the decision taken in MK's absence on 2026-10-12, replaced by
 * MK's of 2026-10-14). */
TEST_CASE("TrainEngine::run records the run's DBFILE and run inputs in the "
          "potential's training record, as written",
          "[train][potential_content][training_record]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  LoggerAtWarning at_warning;  // VERBOSE 0 below sets the level: restored
  write_ta_dataset(cwd.dir / "db1.tadah");
  write_ta_dataset(cwd.dir / "db2.tadah", 0.01);
  Context c = potential_context("M_BLR BF_Linear");
  c.add("DBFILE", "./db1.tadah");
  c.add("DBFILE", "./db2.tadah");
  c.add("VALIDATION", "config.val");
  c.add("HPOTARGET", "config.hpo");
  c.add("UNCERTAINTY", "false");
  c.add("VERBOSE", 0);
  c.add("NUMERIC", 9);
  c.add("OUTFILE", "pot.tadah");
  c.add("AUDIT", "off");
  c.add("AUDIT_SAMPLE", 64);
  c.add("HEALTH_LOG", "full");
  std::string printed;
  const std::string what = train_error(c, printed);
  INFO("thrown: " << what << "\nprinted:\n" << printed);
  REQUIRE(what.empty());
  INFO("pot.tadah:\n" << slurp("pot.tadah"));
  const Record r = read_record("pot.tadah");
  REQUIRE(r.found);
  REQUIRE(r.closed);
  CHECK(r.open == 0u);
  CHECK(r.end < first_key_line("pot.tadah"));
  CHECK(r.lines == std::vector<std::string>{
                       "DBFILE ./db1.tadah ./db2.tadah",
                       "VALIDATION config.val", "HPOTARGET config.hpo",
                       "UNCERTAINTY false", "VERBOSE 0", "NUMERIC 9",
                       "OUTFILE pot.tadah", "AUDIT off", "AUDIT_SAMPLE 64",
                       "HEALTH_LOG full"});
  for (const char *key : kRecordKeys) {
    INFO("key " << key);
    CHECK(pot_line("pot.tadah", key).empty());
  }
  CHECK(pot_line("pot.tadah", "WEIGHTS").size() == 2u);
  const Context back = tadah::core::CoreUserConfigParser().parse("pot.tadah");
  for (const char *key : kRecordKeys) {
    INFO("read back: key " << key);
    CHECK_FALSE(back.has_stored(key));
  }
  std::ofstream("bare.tadah") << without_record("pot.tadah");
  const Context bare = tadah::core::CoreUserConfigParser().parse("bare.tadah");
  CHECK(back.storage_snapshot() == bare.storage_snapshot());
}

/* Pins: a retrain from a potential -- pot1.tadah, trained on db1.tadah
 * (its record: DBFILE db1.tadah, OUTFILE pot1.tadah), parsed as the
 * configuration (no DBFILE: its record is a comment) with DBFILE db2.tadah
 * and OUTFILE pot2.tadah added -- writes pot2.tadah whose record is that
 * of the retrain: "DBFILE db2.tadah" and "OUTFILE pot2.tadah", and no
 * db1.tadah anywhere in the file (the old record is not kept); its WEIGHTS
 * are those of a fit of db2.tadah from the configuration pot1.tadah was
 * trained with (pot3.tadah), token for token: the retrain fits its own
 * DBFILE only.
 * Before the fix: neither potential has a training record: the check of
 * pot1.tadah's record fails, and the REQUIRE of pot2.tadah's stops the case
 * before the WEIGHTS comparison. */
TEST_CASE("A retrain from a potential records the retrain's DBFILE, not "
          "the one of the potential",
          "[train][potential_content][training_record]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  LoggerAtWarning at_warning;
  write_ta_dataset(cwd.dir / "db1.tadah");
  write_ta_dataset(cwd.dir / "db2.tadah", 0.01);
  std::string printed;
  {
    Context c1 = potential_context("M_BLR BF_Linear");
    c1.add("DBFILE", "db1.tadah");
    c1.add("OUTFILE", "pot1.tadah");
    const std::string what = train_error(c1, printed);
    INFO("pot1: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
  }
  INFO("pot1.tadah:\n" << slurp("pot1.tadah"));
  CHECK(read_record("pot1.tadah").lines ==
        std::vector<std::string>{"DBFILE db1.tadah", "OUTFILE pot1.tadah"});
  {
    Context c2 = tadah::core::CoreUserConfigParser().parse("pot1.tadah");
    REQUIRE_FALSE(c2.has_stored("DBFILE"));
    c2.add("DBFILE", "db2.tadah");
    c2.add("OUTFILE", "pot2.tadah");
    const std::string what = train_error(c2, printed);
    INFO("pot2: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
  }
  {
    Context c3 = potential_context("M_BLR BF_Linear");
    c3.add("DBFILE", "db2.tadah");
    c3.add("OUTFILE", "pot3.tadah");
    const std::string what = train_error(c3, printed);
    INFO("pot3: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
  }
  INFO("pot2.tadah:\n" << slurp("pot2.tadah"));
  const Record r = read_record("pot2.tadah");
  REQUIRE(r.found);
  CHECK(r.closed);
  CHECK(r.lines ==
        std::vector<std::string>{"DBFILE db2.tadah", "OUTFILE pot2.tadah"});
  CHECK(slurp("pot2.tadah").find("db1.tadah") == std::string::npos);
  REQUIRE(pot_line("pot2.tadah", "WEIGHTS").size() == 2u);
  CHECK(pot_line("pot2.tadah", "WEIGHTS") == pot_line("pot3.tadah", "WEIGHTS"));
}
