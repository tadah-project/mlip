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
// retrain's too, and the keys that prepared the training data (MK's
// decision of 2026-10-14, option 2), which a retrain from a potential does
// not take from it.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/key_provenance.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
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

/* The opening and end lines of the training record of a potential file.
 * The opening line said "(not read by tadah)" before MK's decision of
 * 2026-10-14 (option 2): refit --edit reads the block, to copy it. */
const std::string kRecordOpen =
    "# --- training record (no key is read from it) ---";
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
 * record (no key is read from it) ---"), then one "# KEY values" line per key
 * in that order with the values as the configuration has them (./db1.tadah
 * not normalised; under NUMERIC 9, which governs the key lines only), then
 * "# --- end of training record ---", before the first key line -- every
 * key but UNCERTAINTY: the configuration has WEIGHTS, so its UNCERTAINTY
 * false is the potential's line, which a run does not read (the rule of
 * TaskFactory's parse; a context without a KeyProvenance is one file), and
 * the record lists the nine others; no key line of any of the ten; and the
 * file parses (CoreUserConfigParser, the reader of tadah predict, tadah
 * train -c, tadah hpo -c and pair_style tadah) to the same context, key by
 * key, as the file without the block, with none of the ten keys.
 * Before the fix: no training record: the ten keys were dropped from the
 * potential (the decision taken in MK's absence on 2026-10-12, replaced by
 * MK's of 2026-10-14).
 * Before the opening line was corrected (MK, 2026-10-14, option 2): it read
 * "# --- training record (not read by tadah) ---", so no opening line is
 * found and the REQUIRE stops the case.
 * Before the fix that reads no ESHIFT_ATOM, ESHIFT_DBATOM or UNCERTAINTY
 * line of a potential: the potential's UNCERTAINTY false was read as the
 * run's and the record listed "UNCERTAINTY false" after "HPOTARGET
 * config.hpo" (this case pinned it): the record check fails. */
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
                       "VERBOSE 0", "NUMERIC 9", "OUTFILE pot.tadah",
                       "AUDIT off", "AUDIT_SAMPLE 64", "HEALTH_LOG full"});
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
 * before the WEIGHTS comparison.
 * Before the opening line was corrected (MK, 2026-10-14, option 2): the
 * same, the opening line being "# --- training record (not read by tadah)
 * ---". */
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

/* ---- the keys that prepared the training data (MK's decision of
 * 2026-10-14, option 2): a potential's key lines are what is needed to use
 * the model (the model keys, the resolved ESHIFT and LSCALE); EFILTER,
 * FFILTER, ESHIFT_ATOM, ESHIFT_DBATOM, EWEIGHT_TEMP, ZERO_COM_FORCE,
 * WDBFILE and WDBFILE_AUTO are written into its training record. */

namespace {

/* The keys that prepared the training data, in the order their transforms
 * apply (apply_load_transforms) and the training record lists them. */
const char *const kDatasetKeys[] = {"EFILTER",       "FFILTER",
                                    "ESHIFT_ATOM",   "ESHIFT_DBATOM",
                                    "EWEIGHT_TEMP",  "ZERO_COM_FORCE",
                                    "WDBFILE",       "WDBFILE_AUTO"};

/* potential_context(@p model) without its WEIGHTS: a training
 * configuration, not a potential. */
Context training_context_of(const std::string &model) {
  Context c = potential_context(model);
  c.remove("WEIGHTS");
  return c;
}

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* What TaskFactory gets from the command line: the task @p task (the
 * blueprint name, "TRAIN") and its options keyed by the blueprint's
 * lower-case key names ({"config", {"retrain.train"}}); with no task name
 * and {"task", {file}}, `tadah --task file`. */
struct CommandInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  CommandInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* What TaskFactory, and TrainEngine::run of each task it makes, throw for
 * @p input, "" when they do not throw; std::cout goes to @p printed. */
std::string train_input_error(const tadah::mlip::ITaskInput &input,
                              std::string &printed) {
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  std::string what;
  try {
    std::vector<Task> tasks =
        tadah::mlip::TaskFactory().createTasksFromInput(input);
    for (Task &t : tasks) TrainEngine().run(t);
  } catch (const std::exception &e) {
    what = e.what();
  }
  std::cout.rdbuf(old_buf);
  printed = captured.str();
  return what;
}

/* Writes @p c to @p path as the key lines of a potential file. */
void write_pot_file(const fs::path &path, const Context &c) {
  std::ofstream os(path);
  c.write_potential(os, 18);
}

}  // namespace

/* Pins: TrainEngine::run of a training configuration (the Ta pair keys,
 * M_BLR BF_Linear, no WEIGHTS) with DBFILE db1.tadah db2.tadah and every
 * key that prepares the training data -- EFILTER -1 1 and FFILTER 100
 * (which keep every structure), ESHIFT_DBATOM true, EWEIGHT_TEMP 300,
 * ZERO_COM_FORCE true, WDBFILE 1 0.5, WDBFILE_AUTO 0.5 -- writes pot.tadah
 * whose training record is "DBFILE db1.tadah db2.tadah", then those keys
 * as the configuration has them, in that order, then "OUTFILE pot.tadah";
 * no key line of any of the eight; and the ESHIFT line of the shift
 * ESHIFT_DBATOM resolved (one value, not 0: the energies are not 0), the
 * model's energy reference, a key line.
 * Before the fix: the eight were key lines of the potential and the record
 * listed DBFILE and OUTFILE alone, after the opening line "# --- training
 * record (not read by tadah) ---": REQUIRE(r.found) stops the case. */
TEST_CASE("TrainEngine::run writes the keys that prepared the training data "
          "into the training record, and the ESHIFT they resolved as a key "
          "line",
          "[train][potential_content][training_record][dataset_keys]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  LoggerAtWarning at_warning;
  write_ta_dataset(cwd.dir / "db1.tadah", -1.0);
  write_ta_dataset(cwd.dir / "db2.tadah", -0.99);
  Context c = training_context_of("M_BLR BF_Linear");
  c.add("DBFILE", "db1.tadah");
  c.add("DBFILE", "db2.tadah");
  c.add("EFILTER", "-1");
  c.add("EFILTER", "1");
  c.add("FFILTER", "100");
  c.add("ESHIFT_DBATOM", "true");
  c.add("EWEIGHT_TEMP", "300");
  c.add("ZERO_COM_FORCE", "true");
  c.add("WDBFILE", "1");
  c.add("WDBFILE", "0.5");
  c.add("WDBFILE_AUTO", "0.5");
  c.add("OUTFILE", "pot.tadah");
  std::string printed;
  const std::string what = train_error(c, printed);
  INFO("thrown: " << what << "\nprinted:\n" << printed);
  REQUIRE(what.empty());
  INFO("pot.tadah:\n" << slurp("pot.tadah"));
  const Record r = read_record("pot.tadah");
  REQUIRE(r.found);
  CHECK(r.closed);
  CHECK(r.lines == std::vector<std::string>{
                       "DBFILE db1.tadah db2.tadah", "EFILTER -1 1",
                       "FFILTER 100", "ESHIFT_DBATOM true",
                       "EWEIGHT_TEMP 300", "ZERO_COM_FORCE true",
                       "WDBFILE 1 0.5", "WDBFILE_AUTO 0.5",
                       "OUTFILE pot.tadah"});
  for (const char *key : kDatasetKeys) {
    INFO("key " << key);
    CHECK(pot_line("pot.tadah", key).empty());
  }
  const std::vector<std::string> eshift = pot_line("pot.tadah", "ESHIFT");
  REQUIRE(eshift.size() == 1u);
  CHECK(std::stod(eshift[0]) != 0.0);
  CHECK(pot_line("pot.tadah", "WEIGHTS").size() == 2u);
}

/* Pins: TrainEngine::run of a potential used as the configuration whose
 * key lines include the keys that prepared the data of its own fit, as a
 * potential of an earlier version has them -- the Ta pair potential
 * (WEIGHTS 0.25 -0.5) with ESHIFT 0.001, ESHIFT_DBATOM true, EFILTER 5 6
 * (it would drop every structure of the retrain: E/N is about -0.25 eV),
 * EWEIGHT_TEMP 300, WDBFILE 1 1 (two values: the retrain has one dataset),
 * WDBFILE_AUTO 0 and ZERO_COM_FORCE false -- and the DBFILE db.tadah of a
 * retrain: none of the eight applies to the retrain:
 *   - without a KeyProvenance (every key the configuration's): the retrain
 *     trains (no "WDBFILE size" refusal, no structure dropped), prints the
 *     WARNING "tadah train: the configuration is a potential (it has
 *     WEIGHTS): its EFILTER 5 6, ESHIFT_DBATOM true, EWEIGHT_TEMP 300 and
 *     WDBFILE 1 1 prepared the training data of the fit that wrote it, not
 *     of this run, and are ignored (tadah keeps such keys in a potential's
 *     training record, not as key lines). Give --efilter, --eshift-dbatom,
 *     --eweight-temp or --wdbfile on the command line to apply one to this
 *     run. The ESHIFT of the configuration, the energy reference of the
 *     model, overrules ESHIFT_ATOM and ESHIFT_DBATOM: to derive ESHIFT from
 *     the data of this run, delete the ESHIFT line and give the run one of
 *     them (the Training page of the documentation, Retraining from a
 *     potential)." (WDBFILE_AUTO 0 and
 *     ZERO_COM_FORCE false, their defaults, change nothing and are not
 *     named), and no "Multiple ESHIFT sources" WARNING; pot2.tadah has the
 *     potential's ESHIFT 0.001, a training record of DBFILE and OUTFILE
 *     alone, and the WEIGHTS of a training configuration of the same keys
 *     without the eight (pot3.tadah), token for token;
 *   - `tadah train -c retrain.train --wdbfile 2` (TaskFactory and
 *     TrainEngine::run), retrain.train being that potential as a file: the
 *     command line's WDBFILE 2 applies to the retrain and is recorded
 *     ("WDBFILE 2" after "DBFILE db.tadah" in pot2.tadah's record), the
 *     file's other dataset keys are not read, and the WARNING is "tadah
 *     train: the configuration file retrain.train is a potential (it has
 *     WEIGHTS): its EFILTER 5 6, ESHIFT_DBATOM true and EWEIGHT_TEMP 300
 *     prepared the training data of the fit that wrote it, not of this
 *     run, and are ignored (tadah keeps such keys in a potential's
 *     training record, not as key lines). Give --efilter, --eshift-dbatom
 *     or --eweight-temp on the command line, or an EFILTER, ESHIFT_DBATOM
 *     or EWEIGHT_TEMP line in a configuration file that includes the
 *     potential with CONFIG, to apply one to this run. The ESHIFT of the
 *     configuration, ..." (no WDBFILE: the run gives its own).
 * Before the fix: the potential's keys applied to the retrain: EFILTER 5 6
 * dropped every structure, the WARNING "Multiple ESHIFT sources set;
 * precedence: explicit > ATOM > DBATOM." was printed, and TrainEngine::run
 * threw "ESHIFT size (1) does not match number of unique elements (0)."
 * (no structure left, so no element for the ESHIFT value; the WDBFILE 1 1
 * of two values for one dataset, applied later, was not reached): the
 * REQUIRE stops each section.
 * Before the run's own dataset keys of a configuration that includes a
 * potential applied (review of the dataset keys): the second section built
 * the KeyProvenance by hand, and the WARNING's advice was "Give --efilter
 * or --eweight-temp on the command line to apply one to this run.", which
 * a run of a task file cannot follow: the check of the WARNING fails.
 * Before the fix that reads no ESHIFT_ATOM, ESHIFT_DBATOM or UNCERTAINTY
 * line of a potential (the toggles were skipped only beside the potential's
 * ESHIFT): the advice named neither toggle among the
 * keys the run can give ("Give --efilter, --eweight-temp or --wdbfile on
 * the command line ...") and ended "The potential's ESHIFT, the energy
 * reference of the model, is used: to derive ESHIFT from the data of this
 * run, delete the ESHIFT line and set ESHIFT_ATOM or ESHIFT_DBATOM", which
 * a toggle set in the copy of the potential no longer follows: both
 * WARNING checks fail. */
TEST_CASE("TrainEngine::run of a potential used as the configuration "
          "ignores the keys that prepared its own training data",
          "[train][potential_content][retrain][dataset_keys]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  LoggerAtWarning at_warning;
  write_ta_dataset(cwd.dir / "db.tadah", -1.0);
  Context older = potential_context("M_BLR BF_Linear");
  older.add("ESHIFT", "0.001");
  older.add("ESHIFT_DBATOM", "true");
  older.add("EFILTER", "5");
  older.add("EFILTER", "6");
  older.add("EWEIGHT_TEMP", "300");
  older.add("WDBFILE", "1");
  older.add("WDBFILE", "1");
  older.add("WDBFILE_AUTO", "0");
  older.add("ZERO_COM_FORCE", "false");
  older.add("DBFILE", "db.tadah");
  older.add("OUTFILE", "pot2.tadah");
  std::string printed;
  SECTION("no KeyProvenance") {
    {
      Context c3 = training_context_of("M_BLR BF_Linear");
      c3.add("ESHIFT", "0.001");
      c3.add("DBFILE", "db.tadah");
      c3.add("OUTFILE", "pot3.tadah");
      const std::string what = train_error(c3, printed);
      INFO("pot3: thrown: " << what << "\nprinted:\n" << printed);
      REQUIRE(what.empty());
    }
    const std::string what = train_error(older, printed);
    INFO("pot2: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK(printed.find(
              "WARNING: tadah train: the configuration is a potential (it "
              "has WEIGHTS): its EFILTER 5 6, ESHIFT_DBATOM true, "
              "EWEIGHT_TEMP 300 and WDBFILE 1 1 prepared the training data "
              "of the fit that wrote it, not of this run, and are ignored "
              "(tadah keeps such keys in a potential's training record, not "
              "as key lines). Give --efilter, --eshift-dbatom, "
              "--eweight-temp or --wdbfile on the command line to apply one "
              "to this run. The ESHIFT of the configuration, the energy "
              "reference of the model, overrules ESHIFT_ATOM and "
              "ESHIFT_DBATOM: to derive ESHIFT from the data of this run, "
              "delete the ESHIFT line and give the run one of them (the "
              "Training page of the documentation, Retraining from a "
              "potential).") != std::string::npos);
    CHECK(printed.find("Multiple ESHIFT sources") == std::string::npos);
    INFO("pot2.tadah:\n" << slurp("pot2.tadah"));
    const std::vector<std::string> eshift = pot_line("pot2.tadah", "ESHIFT");
    REQUIRE(eshift.size() == 1u);
    CHECK(std::stod(eshift[0]) == 0.001);
    CHECK(read_record("pot2.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "OUTFILE pot2.tadah"});
    REQUIRE(pot_line("pot2.tadah", "WEIGHTS").size() == 2u);
    CHECK(pot_line("pot2.tadah", "WEIGHTS") ==
          pot_line("pot3.tadah", "WEIGHTS"));
  }
  SECTION("WDBFILE on the command line") {
    write_pot_file("retrain.train", older);
    const CommandInput input("TRAIN", {{"config", {"retrain.train"}},
                                       {"wdbfile", {"2"}}});
    const std::string what = train_input_error(input, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK(printed.find(
              "WARNING: tadah train: the configuration file retrain.train is "
              "a potential (it has WEIGHTS): its EFILTER 5 6, ESHIFT_DBATOM "
              "true and EWEIGHT_TEMP 300 prepared the training data of the "
              "fit that wrote it, not of this run, and are ignored (tadah "
              "keeps such keys in a potential's training record, not as key "
              "lines). Give --efilter, --eshift-dbatom or --eweight-temp on "
              "the command line, or an EFILTER, ESHIFT_DBATOM or "
              "EWEIGHT_TEMP line in a configuration file that includes the "
              "potential with CONFIG, to apply one to this run. The ESHIFT "
              "of the configuration, the energy reference of the model, "
              "overrules ESHIFT_ATOM and ESHIFT_DBATOM: to derive ESHIFT from "
              "the data of this run, delete the ESHIFT line and give the run "
              "one of them (the Training page of the documentation, "
              "Retraining from a potential).") != std::string::npos);
    CHECK(printed.find("WDBFILE 1 1") == std::string::npos);
    INFO("pot2.tadah:\n" << slurp("pot2.tadah"));
    CHECK(read_record("pot2.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "WDBFILE 2",
                                   "OUTFILE pot2.tadah"});
    CHECK(pot_line("pot2.tadah", "WDBFILE").empty());
  }
}

/* Pins: the keys that prepared the training data that a run's own
 * configuration gives apply to a retrain from a potential it includes with
 * CONFIG, and are recorded; only the potential's lines of them are not
 * read. old.tadah is the Ta pair potential (WEIGHTS 0.25 -0.5) with the key
 * lines of an earlier version: ESHIFT 0.001, ESHIFT_DBATOM true, EFILTER 5
 * 6 (it would drop every structure: E/N is about -0.25 eV), EWEIGHT_TEMP
 * 300, WDBFILE 1 1, WDBFILE_AUTO 0 and ZERO_COM_FORCE false; db1.tadah,
 * db2.tadah and db3.tadah are three datasets (TaskFactory and
 * TrainEngine::run, as `tadah --task` and `tadah train -c` run them):
 *   - `tadah --task run.task`, its TASK train section "ZERO_COM_FORCE
 *     true", "CONFIG old.tadah", "DBFILE db1.tadah db2.tadah db3.tadah",
 *     "WDBFILE 1 0.001 1", "EFILTER -1 1", "OUTFILE pot2.tadah": trains;
 *     pot2.tadah has the training record "DBFILE db1.tadah db2.tadah
 *     db3.tadah", "EFILTER -1 1", "ZERO_COM_FORCE true", "WDBFILE 1 0.001
 *     1", "OUTFILE pot2.tadah" (the line before CONFIG too: the potential's
 *     ZERO_COM_FORCE false does not replace it), the potential's ESHIFT
 *     0.001, and the WEIGHTS of a training configuration of the same keys
 *     (pot3.tadah), token for token; the WARNING names the potential's
 *     lines the run does not give itself: "tadah train: old.tadah, which
 *     the task file run.task includes with CONFIG, is a potential (it has
 *     WEIGHTS): its ESHIFT_DBATOM true and EWEIGHT_TEMP 300 prepared the
 *     training data of the fit that wrote it, not of this run, and are
 *     ignored (tadah keeps such keys in a potential's training record, not
 *     as key lines). Write ESHIFT_DBATOM or EWEIGHT_TEMP in the task file
 *     run.task to apply one to this run. The ESHIFT of the configuration,
 *     the energy reference of the model, overrules ESHIFT_ATOM and
 *     ESHIFT_DBATOM: ... (the Training page of the documentation,
 *     Retraining from a potential)."; no "Multiple ESHIFT sources";
 *   - a TASK train section with none of them ("CONFIG old.tadah", "DBFILE
 *     db1.tadah", "OUTFILE pot4.tadah": WDBFILE 1 1 has two values for its
 *     one dataset): trains, and the WARNING names EFILTER 5 6,
 *     ESHIFT_DBATOM true, EWEIGHT_TEMP 300 and WDBFILE 1 1, with "Write
 *     EFILTER, ESHIFT_DBATOM, EWEIGHT_TEMP or WDBFILE in the task file
 *     run.task to apply one to this run.";
 *   - `tadah train -c outer.train`, outer.train being "CONFIG old.tadah",
 *     "DBFILE db1.tadah db2.tadah db3.tadah", "WDBFILE 1 0.001 1", "OUTFILE
 *     pot5.tadah": pot5.tadah's record is "DBFILE db1.tadah db2.tadah
 *     db3.tadah", "WDBFILE 1 0.001 1", "OUTFILE pot5.tadah", and the WARNING
 *     "tadah train: old.tadah, which the configuration file outer.train
 *     includes with CONFIG, is a potential (it has WEIGHTS): its EFILTER 5
 *     6, ESHIFT_DBATOM true and EWEIGHT_TEMP 300 prepared ... Give
 *     --efilter, --eshift-dbatom or --eweight-temp on the command line, or
 *     an EFILTER, ESHIFT_DBATOM or EWEIGHT_TEMP line in outer.train, to
 *     apply one to this run. ...".
 * Before the fix: the CONFIG include mixed the potential's lines with the
 * run's, and the run's keys of a configuration with WEIGHTS that were not
 * command-line options were removed with the potential's: the first
 * section's TaskFactory threw "Exceeded capacity for key: EFILTER" (the
 * potential's two values were there), so the REQUIRE stops it; the second
 * warned "tadah train: the task file run.task is a potential (it has
 * WEIGHTS): ... Give --efilter, --eweight-temp or --wdbfile on the command
 * line to apply one to this run. ..." (the task file called the potential,
 * advice --task refuses): the WARNING check fails; the third's WDBFILE was
 * 1 1 1 0.001 1 (the run's values after the potential's) and was removed
 * with the WARNING "the configuration file outer.train is a potential ...":
 * the record has no WDBFILE, and the checks fail.
 * Before the fix that reads no ESHIFT_ATOM, ESHIFT_DBATOM or UNCERTAINTY
 * line of a potential: the advice did not name ESHIFT_DBATOM among the
 * keys the run can give ("Write EWEIGHT_TEMP in the task file run.task to
 * apply it to this run.", "Write EFILTER, EWEIGHT_TEMP or WDBFILE ...",
 * "Give --efilter or --eweight-temp ...") and ended "The potential's
 * ESHIFT, the energy reference of the model, is used: to derive ESHIFT
 * from the data of this run, delete the ESHIFT line and set ESHIFT_ATOM
 * or ESHIFT_DBATOM": the three WARNING checks fail. */
TEST_CASE("A retrain from a potential included with CONFIG applies the "
          "run's own keys that prepare the training data",
          "[train][potential_content][retrain][dataset_keys][task_file]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  LoggerAtWarning at_warning;
  write_ta_dataset(cwd.dir / "db1.tadah", -1.0);
  write_ta_dataset(cwd.dir / "db2.tadah", -0.99);
  write_ta_dataset(cwd.dir / "db3.tadah", -0.98);
  Context older = potential_context("M_BLR BF_Linear");
  older.add("ESHIFT", "0.001");
  older.add("ESHIFT_DBATOM", "true");
  older.add("EFILTER", "5");
  older.add("EFILTER", "6");
  older.add("EWEIGHT_TEMP", "300");
  older.add("WDBFILE", "1");
  older.add("WDBFILE", "1");
  older.add("WDBFILE_AUTO", "0");
  older.add("ZERO_COM_FORCE", "false");
  write_pot_file("old.tadah", older);
  const std::string datasets = "db1.tadah db2.tadah db3.tadah";
  const std::string eshift_advice =
      " The ESHIFT of the configuration, the energy reference of the model, "
      "overrules ESHIFT_ATOM and ESHIFT_DBATOM: to derive ESHIFT from the "
      "data of this run, delete the ESHIFT line and give the run one of them "
      "(the Training page of the documentation, Retraining from a "
      "potential).";
  const CommandInput task_file("", {{"task", {"run.task"}}});
  std::string printed;
  SECTION("the task file's keys, after and before the CONFIG line") {
    std::ofstream("run.task") << "TASK train\n"
                                 "  ZERO_COM_FORCE true\n"
                                 "  CONFIG old.tadah\n"
                                 "  DBFILE " << datasets << "\n"
                                 "  WDBFILE 1 0.001 1\n"
                                 "  EFILTER -1 1\n"
                                 "  OUTFILE pot2.tadah\n";
    const std::string what = train_input_error(task_file, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    INFO("pot2.tadah:\n" << slurp("pot2.tadah"));
    CHECK(read_record("pot2.tadah").lines ==
          std::vector<std::string>{"DBFILE " + datasets, "EFILTER -1 1",
                                   "ZERO_COM_FORCE true",
                                   "WDBFILE 1 0.001 1",
                                   "OUTFILE pot2.tadah"});
    const std::vector<std::string> eshift = pot_line("pot2.tadah", "ESHIFT");
    REQUIRE(eshift.size() == 1u);
    CHECK(std::stod(eshift[0]) == 0.001);
    CHECK(printed.find(
              "WARNING: tadah train: old.tadah, which the task file run.task "
              "includes with CONFIG, is a potential (it has WEIGHTS): its "
              "ESHIFT_DBATOM true and EWEIGHT_TEMP 300 prepared the training "
              "data of the fit that wrote it, not of this run, and are "
              "ignored (tadah keeps such keys in a potential's training "
              "record, not as key lines). Write ESHIFT_DBATOM or "
              "EWEIGHT_TEMP in the task file run.task to apply one to this "
              "run." + eshift_advice) !=
          std::string::npos);
    CHECK(printed.find("Multiple ESHIFT sources") == std::string::npos);
    Context c3 = training_context_of("M_BLR BF_Linear");
    c3.add("ESHIFT", "0.001");
    for (const char *db : {"db1.tadah", "db2.tadah", "db3.tadah"})
      c3.add("DBFILE", db);
    c3.add("WDBFILE", "1 0.001 1");
    c3.add("EFILTER", "-1 1");
    c3.add("ZERO_COM_FORCE", "true");
    c3.add("OUTFILE", "pot3.tadah");
    const std::string what3 = train_error(c3, printed);
    INFO("pot3: thrown: " << what3 << "\nprinted:\n" << printed);
    REQUIRE(what3.empty());
    REQUIRE(pot_line("pot2.tadah", "WEIGHTS").size() == 2u);
    CHECK(pot_line("pot2.tadah", "WEIGHTS") ==
          pot_line("pot3.tadah", "WEIGHTS"));
  }
  SECTION("a task file without such keys of its own") {
    std::ofstream("run.task") << "TASK train\n"
                                 "  CONFIG old.tadah\n"
                                 "  DBFILE db1.tadah\n"
                                 "  OUTFILE pot4.tadah\n";
    const std::string what = train_input_error(task_file, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK(printed.find(
              "WARNING: tadah train: old.tadah, which the task file run.task "
              "includes with CONFIG, is a potential (it has WEIGHTS): its "
              "EFILTER 5 6, ESHIFT_DBATOM true, EWEIGHT_TEMP 300 and WDBFILE "
              "1 1 prepared the training data of the fit that wrote it, not "
              "of this run, and are ignored (tadah keeps such keys in a "
              "potential's training record, not as key lines). Write "
              "EFILTER, ESHIFT_DBATOM, EWEIGHT_TEMP or WDBFILE in the task "
              "file run.task to apply one to this run." + eshift_advice) !=
          std::string::npos);
    CHECK(read_record("pot4.tadah").lines ==
          std::vector<std::string>{"DBFILE db1.tadah", "OUTFILE pot4.tadah"});
  }
  SECTION("a configuration file that includes the potential") {
    std::ofstream("outer.train") << "CONFIG old.tadah\n"
                                    "DBFILE " << datasets << "\n"
                                    "WDBFILE 1 0.001 1\n"
                                    "OUTFILE pot5.tadah\n";
    const CommandInput input("TRAIN", {{"config", {"outer.train"}}});
    const std::string what = train_input_error(input, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    INFO("pot5.tadah:\n" << slurp("pot5.tadah"));
    CHECK(read_record("pot5.tadah").lines ==
          std::vector<std::string>{"DBFILE " + datasets, "WDBFILE 1 0.001 1",
                                   "OUTFILE pot5.tadah"});
    CHECK(printed.find(
              "WARNING: tadah train: old.tadah, which the configuration file "
              "outer.train includes with CONFIG, is a potential (it has "
              "WEIGHTS): its EFILTER 5 6, ESHIFT_DBATOM true and EWEIGHT_TEMP "
              "300 prepared the training data of the fit that wrote it, not "
              "of this run, and are ignored (tadah keeps such keys in a "
              "potential's training record, not as key lines). Give "
              "--efilter, --eshift-dbatom or --eweight-temp on the command "
              "line, or an EFILTER, ESHIFT_DBATOM or EWEIGHT_TEMP line in "
              "outer.train, to apply one to this run." + eshift_advice) !=
          std::string::npos);
  }
}

/* Pins: a retrain reads no ESHIFT_ATOM or ESHIFT_DBATOM line of a
 * potential, with or without its ESHIFT line, and the run's own toggle
 * derives ESHIFT from the retrain's data (the documented way: delete the
 * ESHIFT line from the copy of the potential and give the run the toggle).
 * The Ta pair potential (WEIGHTS 0.25 -0.5) has no ESHIFT line; db.tadah
 * is one dataset (E/N about -0.25 eV); pot3.tadah is a training
 * configuration of the same keys with ESHIFT_DBATOM true (one ESHIFT
 * value):
 *   - without a KeyProvenance, the potential with "ESHIFT_DBATOM true":
 *     pot2.tadah has no ESHIFT line and the WEIGHTS of the training
 *     configuration without the toggle (pot5.tadah), token for token, and
 *     the WARNING is "tadah train: the configuration is a potential (it has
 *     WEIGHTS): its ESHIFT_DBATOM true prepared the training data of the
 *     fit that wrote it, not of this run, and is ignored (tadah keeps such
 *     keys in a potential's training record, not as key lines). Give
 *     --eshift-dbatom on the command line to apply it to this run (the
 *     Training page of the documentation, Retraining from a potential).";
 *   - `tadah train -c copy.train` (TaskFactory and TrainEngine::run),
 *     copy.train being that potential with "ESHIFT_DBATOM true": no ESHIFT
 *     line, the record "DBFILE db.tadah", "OUTFILE pot7.tadah", and the
 *     WARNING "tadah train: the configuration file copy.train is a
 *     potential (it has WEIGHTS): its ESHIFT_DBATOM true prepared ... Give
 *     --eshift-dbatom on the command line, or an ESHIFT_DBATOM line in a
 *     configuration file that includes the potential with CONFIG, to apply
 *     it to this run (...).";
 *   - `tadah train -c rederive.train --eshift-dbatom`, rederive.train being
 *     the potential without a toggle, and `tadah train -c outer.train`,
 *     outer.train being "CONFIG rederive.tadah" and "ESHIFT_DBATOM true":
 *     both write the ESHIFT of pot3.tadah, token for token, outer.train's
 *     record lists "ESHIFT_DBATOM true", and no WARNING calls a file a
 *     potential (a pin: the run's own toggle was read before the fix too);
 *   - the run's own toggle beside the potential's other one (the two choose
 *     one thing, the source of the shift, and exclude each other):
 *     `tadah train -c atom.train --eshift-dbatom`, atom.train being the
 *     potential with "ESHIFT_ATOM true", writes the ESHIFT of pot3.tadah,
 *     token for token, pot9.tadah's record is "DBFILE db.tadah",
 *     "ESHIFT_DBATOM true", "OUTFILE pot9.tadah", and nothing printed calls
 *     a file a potential or says "Multiple ESHIFT sources";
 *     `tadah train -c dbatom.train --eshift-atom`, dbatom.train being the
 *     potential with "ESHIFT_DBATOM true", writes ESHIFT 0 (db.tadah has
 *     no isolated atom), the record "DBFILE db.tadah", "ESHIFT_ATOM true",
 *     "OUTFILE pot10.tadah", and the same nothing; a run's toggle given
 *     false chooses no source: `tadah train -c outer.train`, outer.train
 *     being "CONFIG dbatom.tadah" (the potential with "ESHIFT_DBATOM
 *     true"), "ESHIFT_ATOM false", "DBFILE db.tadah", "OUTFILE
 *     pot11.tadah", writes no ESHIFT line and the WARNING "tadah train:
 *     dbatom.tadah, which the configuration file outer.train includes with
 *     CONFIG, is a potential (it has WEIGHTS): its ESHIFT_DBATOM true
 *     prepared ... Give --eshift-dbatom on the command line, or an
 *     ESHIFT_DBATOM line in outer.train, to apply it to this run (...)."
 *     (that WARNING a pin: it was printed before this rule too);
 *   - `tadah train -c old.train`, old.train being the potential with the
 *     key lines "ESHIFT_ATOM false" and "ESHIFT_DBATOM false" that every
 *     potential of an earlier tadah train has when its run set no shift:
 *     pot6.tadah's training record is "DBFILE db.tadah", "OUTFILE
 *     pot6.tadah" (the potential's lines are not the run's), and no WARNING
 *     (the two are at their defaults).
 * Before the fix: a toggle of a potential without an ESHIFT line was read:
 * the first two sections derived ESHIFT from the retrain's data (pot2.tadah
 * and pot7.tadah have an ESHIFT line) and printed no WARNING about it; the
 * last recorded "ESHIFT_ATOM false" and "ESHIFT_DBATOM false" as the run's.
 * Before that fix this case pinned the derivation from the potential's own
 * toggle (pot2.tadah's ESHIFT that of pot3.tadah).
 * Before the toggles were one setting of the notice (review of the
 * dev-history messages): a potential's toggle was left out only when the
 * run gave that same toggle, so atom.train with --eshift-dbatom printed
 * "WARNING: tadah train: the configuration file atom.train is a potential
 * (it has WEIGHTS): its ESHIFT_ATOM true ... Give --eshift-atom on the
 * command line, ... to apply it to this run ...", advice that sets both
 * toggles (ATOM then overrules the run's DBATOM, "Multiple ESHIFT sources
 * set"), and dbatom.train with --eshift-atom "... its ESHIFT_DBATOM true
 * ... Give --eshift-dbatom ...", advice that does not apply it (ATOM
 * wins): the two "is a potential" CHECKs fail. */
TEST_CASE("A retrain reads no ESHIFT_ATOM or ESHIFT_DBATOM line of a "
          "potential; the run's own derives ESHIFT",
          "[train][potential_content][retrain][dataset_keys]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  LoggerAtWarning at_warning;
  write_ta_dataset(cwd.dir / "db.tadah", -1.0);
  std::string printed;
  {
    Context c3 = training_context_of("M_BLR BF_Linear");
    c3.add("ESHIFT_DBATOM", "true");
    c3.add("DBFILE", "db.tadah");
    c3.add("OUTFILE", "pot3.tadah");
    const std::string what = train_error(c3, printed);
    INFO("pot3: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
  }
  const std::vector<std::string> derived = pot_line("pot3.tadah", "ESHIFT");
  REQUIRE(derived.size() == 1u);
  SECTION("no KeyProvenance") {
    {
      Context c5 = training_context_of("M_BLR BF_Linear");
      c5.add("DBFILE", "db.tadah");
      c5.add("OUTFILE", "pot5.tadah");
      const std::string what = train_error(c5, printed);
      INFO("pot5: thrown: " << what << "\nprinted:\n" << printed);
      REQUIRE(what.empty());
    }
    Context c = potential_context("M_BLR BF_Linear");
    c.add("ESHIFT_DBATOM", "true");
    c.add("DBFILE", "db.tadah");
    c.add("OUTFILE", "pot2.tadah");
    const std::string what = train_error(c, printed);
    INFO("pot2: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    INFO("pot2.tadah:\n" << slurp("pot2.tadah"));
    CHECK(pot_line("pot2.tadah", "ESHIFT").empty());
    REQUIRE(pot_line("pot2.tadah", "WEIGHTS").size() == 2u);
    CHECK(pot_line("pot2.tadah", "WEIGHTS") ==
          pot_line("pot5.tadah", "WEIGHTS"));
    CHECK(printed.find(
              "WARNING: tadah train: the configuration is a potential (it "
              "has WEIGHTS): its ESHIFT_DBATOM true prepared the training "
              "data of the fit that wrote it, not of this run, and is "
              "ignored (tadah keeps such keys in a potential's training "
              "record, not as key lines). Give --eshift-dbatom on the "
              "command line to apply it to this run (the Training page of "
              "the documentation, Retraining from a potential).") !=
          std::string::npos);
  }
  SECTION("a copy of the potential with the toggle") {
    Context copy = potential_context("M_BLR BF_Linear");
    copy.add("ESHIFT_DBATOM", "true");
    copy.add("DBFILE", "db.tadah");
    copy.add("OUTFILE", "pot7.tadah");
    write_pot_file("copy.train", copy);
    const CommandInput input("TRAIN", {{"config", {"copy.train"}}});
    const std::string what = train_input_error(input, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    INFO("pot7.tadah:\n" << slurp("pot7.tadah"));
    CHECK(pot_line("pot7.tadah", "ESHIFT").empty());
    CHECK(read_record("pot7.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "OUTFILE pot7.tadah"});
    CHECK(printed.find(
              "WARNING: tadah train: the configuration file copy.train is a "
              "potential (it has WEIGHTS): its ESHIFT_DBATOM true prepared "
              "the training data of the fit that wrote it, not of this run, "
              "and is ignored (tadah keeps such keys in a potential's "
              "training record, not as key lines). Give --eshift-dbatom on "
              "the command line, or an ESHIFT_DBATOM line in a configuration "
              "file that includes the potential with CONFIG, to apply it to "
              "this run (the Training page of the documentation, Retraining "
              "from a potential).") != std::string::npos);
  }
  SECTION("the run's own toggle") {
    const Context base = potential_context("M_BLR BF_Linear");
    write_pot_file("rederive.tadah", base);
    Context re = base;
    re.add("DBFILE", "db.tadah");
    re.add("OUTFILE", "pot4.tadah");
    write_pot_file("rederive.train", re);
    {
      const CommandInput input("TRAIN", {{"config", {"rederive.train"}},
                                         {"eshift_dbatom", {"true"}}});
      const std::string what = train_input_error(input, printed);
      INFO("--eshift-dbatom: thrown: " << what << "\nprinted:\n" << printed);
      REQUIRE(what.empty());
      CHECK(printed.find("is a potential") == std::string::npos);
      CHECK(pot_line("pot4.tadah", "ESHIFT") == derived);
    }
    std::ofstream("outer.train") << "CONFIG rederive.tadah\n"
                                    "ESHIFT_DBATOM true\n"
                                    "DBFILE db.tadah\n"
                                    "OUTFILE pot8.tadah\n";
    const CommandInput input("TRAIN", {{"config", {"outer.train"}}});
    const std::string what = train_input_error(input, printed);
    INFO("outer.train: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK(printed.find("is a potential") == std::string::npos);
    CHECK(pot_line("pot8.tadah", "ESHIFT") == derived);
    CHECK(read_record("pot8.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "ESHIFT_DBATOM true",
                                   "OUTFILE pot8.tadah"});
  }
  SECTION("the run's own toggle beside the other toggle of the potential") {
    {
      Context copy = potential_context("M_BLR BF_Linear");
      copy.add("ESHIFT_ATOM", "true");
      copy.add("DBFILE", "db.tadah");
      copy.add("OUTFILE", "pot9.tadah");
      write_pot_file("atom.train", copy);
      const CommandInput input("TRAIN", {{"config", {"atom.train"}},
                                         {"eshift_dbatom", {"true"}}});
      const std::string what = train_input_error(input, printed);
      INFO("atom.train --eshift-dbatom: thrown: " << what << "\nprinted:\n"
                                                  << printed);
      REQUIRE(what.empty());
      INFO("pot9.tadah:\n" << slurp("pot9.tadah"));
      CHECK(printed.find("is a potential") == std::string::npos);
      CHECK(printed.find("Multiple ESHIFT sources") == std::string::npos);
      CHECK(pot_line("pot9.tadah", "ESHIFT") == derived);
      CHECK(read_record("pot9.tadah").lines ==
            std::vector<std::string>{"DBFILE db.tadah", "ESHIFT_DBATOM true",
                                     "OUTFILE pot9.tadah"});
    }
    {
      Context copy = potential_context("M_BLR BF_Linear");
      copy.add("ESHIFT_DBATOM", "true");
      copy.add("DBFILE", "db.tadah");
      copy.add("OUTFILE", "pot10.tadah");
      write_pot_file("dbatom.train", copy);
      const CommandInput input("TRAIN", {{"config", {"dbatom.train"}},
                                         {"eshift_atom", {"true"}}});
      const std::string what = train_input_error(input, printed);
      INFO("dbatom.train --eshift-atom: thrown: " << what << "\nprinted:\n"
                                                  << printed);
      REQUIRE(what.empty());
      INFO("pot10.tadah:\n" << slurp("pot10.tadah"));
      CHECK(printed.find("is a potential") == std::string::npos);
      CHECK(printed.find("Multiple ESHIFT sources") == std::string::npos);
      const std::vector<std::string> atom = pot_line("pot10.tadah", "ESHIFT");
      REQUIRE(atom.size() == 1u);
      CHECK(std::stod(atom[0]) == 0.0);  // db.tadah has no isolated atom
      CHECK(read_record("pot10.tadah").lines ==
            std::vector<std::string>{"DBFILE db.tadah", "ESHIFT_ATOM true",
                                     "OUTFILE pot10.tadah"});
    }
    Context pot = potential_context("M_BLR BF_Linear");
    pot.add("ESHIFT_DBATOM", "true");
    write_pot_file("dbatom.tadah", pot);
    std::ofstream("outer.train") << "CONFIG dbatom.tadah\n"
                                    "ESHIFT_ATOM false\n"
                                    "DBFILE db.tadah\n"
                                    "OUTFILE pot11.tadah\n";
    const CommandInput input("TRAIN", {{"config", {"outer.train"}}});
    const std::string what = train_input_error(input, printed);
    INFO("outer.train: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    INFO("pot11.tadah:\n" << slurp("pot11.tadah"));
    CHECK(pot_line("pot11.tadah", "ESHIFT").empty());
    CHECK(printed.find(
              "WARNING: tadah train: dbatom.tadah, which the configuration "
              "file outer.train includes with CONFIG, is a potential (it has "
              "WEIGHTS): its ESHIFT_DBATOM true prepared the training data of "
              "the fit that wrote it, not of this run, and is ignored (tadah "
              "keeps such keys in a potential's training record, not as key "
              "lines). Give --eshift-dbatom on the command line, or an "
              "ESHIFT_DBATOM line in outer.train, to apply it to this run "
              "(the Training page of the documentation, Retraining from a "
              "potential).") != std::string::npos);
  }
  SECTION("an earlier version's toggles at their defaults") {
    Context old = potential_context("M_BLR BF_Linear");
    old.add("ESHIFT_ATOM", "false");
    old.add("ESHIFT_DBATOM", "false");
    old.add("DBFILE", "db.tadah");
    old.add("OUTFILE", "pot6.tadah");
    write_pot_file("old.train", old);
    const CommandInput input("TRAIN", {{"config", {"old.train"}}});
    const std::string what = train_input_error(input, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK(printed.find("is a potential") == std::string::npos);
    INFO("pot6.tadah:\n" << slurp("pot6.tadah"));
    CHECK(read_record("pot6.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "OUTFILE pot6.tadah"});
  }
}

namespace {

/* What TaskFactory, and TrainEngine::validate and run of each task it
 * makes (as tadah runs a task), throw for @p input, "" when they do not
 * throw; std::cout goes to @p printed. */
std::string validated_train_input_error(const tadah::mlip::ITaskInput &input,
                                        std::string &printed) {
  std::ostringstream captured;
  std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
  std::string what;
  try {
    std::vector<Task> tasks =
        tadah::mlip::TaskFactory().createTasksFromInput(input);
    for (Task &t : tasks) {
      TrainEngine().validate(t);
      TrainEngine().run(t);
    }
  } catch (const std::exception &e) {
    what = e.what();
  }
  std::cout.rdbuf(old_buf);
  printed = captured.str();
  return what;
}

/* potential_context(@p model) with OALGO @p oalgo. */
Context potential_with_oalgo(const std::string &model, int oalgo) {
  Context c = potential_context(model);
  c.remove("OALGO");
  c.add("OALGO", oalgo);
  return c;
}

}  // namespace

/* Pins: a retrain reads no UNCERTAINTY line of a potential, and the run's
 * own UNCERTAINTY (--uncertainty, a line of the task file or of a
 * configuration file that includes the potential) asks for weights.tadah
 * as in any run. db.tadah is one dataset of three structures; old.tadah is
 * the Ta pair potential (2 WEIGHTS) with OALGO 1 and "UNCERTAINTY true",
 * old3.tadah the same with OALGO 3 and no UNCERTAINTY line (TaskFactory,
 * then TrainEngine::validate and run, as tadah runs a task):
 *   - `tadah --task run.task`, its TASK train section "CONFIG old.tadah",
 *     "DBFILE db.tadah", "OUTFILE pot2.tadah": trains (no OALGO 3
 *     refusal), writes no weights.tadah, pot2.tadah's record is "DBFILE
 *     db.tadah", "OUTFILE pot2.tadah", and the WARNING is "tadah train:
 *     old.tadah, which the task file run.task includes with CONFIG, is a
 *     potential (it has WEIGHTS): its UNCERTAINTY true asked the run that
 *     wrote it for weights.tadah, not this run, and is ignored (tadah keeps
 *     such keys in a potential's training record, not as key lines). Write
 *     UNCERTAINTY in the task file run.task to apply it to this run (the
 *     Training page of the documentation, Retraining from a potential).";
 *   - `tadah train -c outer.train`, outer.train being "CONFIG old3.tadah",
 *     "UNCERTAINTY true", "DBFILE db.tadah", "OUTFILE pot3.tadah": writes
 *     weights.tadah, pot3.tadah's record is "DBFILE db.tadah",
 *     "UNCERTAINTY true", "OUTFILE pot3.tadah", and no WARNING calls a file
 *     a potential;
 *   - `tadah train -c retrain.train`, retrain.train being old3.tadah with
 *     "UNCERTAINTY true", "DBFILE db.tadah" and "OUTFILE pot4.tadah": no
 *     weights.tadah, the record "DBFILE db.tadah", "OUTFILE pot4.tadah",
 *     the WARNING "tadah train: the configuration file retrain.train is a
 *     potential (it has WEIGHTS): its UNCERTAINTY true asked the run that
 *     wrote it for weights.tadah, not this run, and is ignored (...). Give
 *     --uncertainty on the command line, or an UNCERTAINTY line in a
 *     configuration file that includes the potential with CONFIG, to apply
 *     it to this run (...).", and nothing printed names a development
 *     stage ("development"); with --uncertainty (pot5.tadah): weights.tadah,
 *     the record "DBFILE db.tadah", "UNCERTAINTY true", "OUTFILE
 *     pot5.tadah", no WARNING (a pin: --uncertainty was a request before
 *     the fix too);
 *   - `tadah train -c both.train`, both.train being the potential (OALGO
 *     1) with "EFILTER -9 9", "UNCERTAINTY true", "DBFILE db.tadah" and
 *     "OUTFILE pot7.tadah": no weights.tadah, the record "DBFILE db.tadah",
 *     "OUTFILE pot7.tadah", and the one WARNING "tadah train: the
 *     configuration file both.train is a potential (it has WEIGHTS): its
 *     EFILTER -9 9 prepared the training data of the fit that wrote it, not
 *     of this run, and is ignored (...). Its UNCERTAINTY true asked the run
 *     that wrote it for weights.tadah, not this run, and is ignored too.
 *     Give --efilter or --uncertainty on the command line, or an EFILTER or
 *     UNCERTAINTY line in a configuration file that includes the potential
 *     with CONFIG, to apply one to this run (...).";
 *   - a Task("train") a program built (no KeyProvenance) from the
 *     potential (OALGO 1) with "UNCERTAINTY true", "DBFILE db.tadah" and
 *     "OUTFILE pot6.tadah": TrainEngine::validate and run do not throw, no
 *     weights.tadah, the record "DBFILE db.tadah", "OUTFILE pot6.tadah",
 *     and the WARNING "tadah train: the configuration is a potential (it
 *     has WEIGHTS): its UNCERTAINTY true asked the run that wrote it for
 *     weights.tadah, not this run, and is ignored (...). Give --uncertainty
 *     on the command line to apply it to this run (...)." printed once
 *     (validate applies the rule to a copy of the configuration and prints
 *     nothing).
 * Before the fix: UNCERTAINTY was a request unless it came from the
 * configuration file of a context with WEIGHTS: the task file's section was
 * refused by validate ("tadah train --uncertainty (UNCERTAINTY true: task
 * file run.task) needs OALGO 3 ..."), so the REQUIRE stops it; outer.train's
 * own line was ignored with the WARNING "tadah train: the configuration
 * file outer.train is a potential (it has WEIGHTS), and its UNCERTAINTY
 * true is the record of the run that trained it ...", and no weights.tadah
 * was written; retrain.train's WARNING said "(tadah 1.5.x and 1.6.0
 * development builds wrote UNCERTAINTY into every potential)", and its
 * record listed "UNCERTAINTY true"; both.train printed that WARNING apart
 * from the notice of its EFILTER line ("... and is ignored (tadah keeps
 * such keys in a potential's training record, not as key lines). Give
 * --efilter on the command line, or an EFILTER line ..."), so the exact
 * WARNING is not found, and its record listed "UNCERTAINTY true" too; and
 * the Task without a KeyProvenance was refused by
 * validate ("tadah train --uncertainty needs OALGO 3 ..."), since without
 * a provenance UNCERTAINTY true was a request: the REQUIRE stops it.
 * (These two sections were added in the review of the fix, and were run on
 * the code before it.) */
TEST_CASE("A retrain reads no UNCERTAINTY line of a potential; the run's own "
          "UNCERTAINTY applies",
          "[train][potential_content][retrain][uncertainty]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  LoggerAtWarning at_warning;
  write_ta_dataset(cwd.dir / "db.tadah", -1.0);
  {
    Context old = potential_context("M_BLR BF_Linear");
    old.add("UNCERTAINTY", "true");
    write_pot_file("old.tadah", old);
  }
  write_pot_file("old3.tadah", potential_with_oalgo("M_BLR BF_Linear", 3));
  std::string printed;
  SECTION("a task file that includes the potential") {
    std::ofstream("run.task") << "TASK train\n"
                                 "  CONFIG old.tadah\n"
                                 "  DBFILE db.tadah\n"
                                 "  OUTFILE pot2.tadah\n";
    const CommandInput input("", {{"task", {"run.task"}}});
    const std::string what = validated_train_input_error(input, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK_FALSE(fs::exists("weights.tadah"));
    INFO("pot2.tadah:\n" << slurp("pot2.tadah"));
    CHECK(read_record("pot2.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "OUTFILE pot2.tadah"});
    CHECK(printed.find(
              "WARNING: tadah train: old.tadah, which the task file run.task "
              "includes with CONFIG, is a potential (it has WEIGHTS): its "
              "UNCERTAINTY true asked the run that wrote it for "
              "weights.tadah, not this run, and is ignored (tadah keeps such "
              "keys in a potential's training record, not as key lines). "
              "Write UNCERTAINTY in the task file run.task to apply it to "
              "this run (the Training page of the documentation, Retraining "
              "from a potential).") != std::string::npos);
  }
  SECTION("the run's own line in a file that includes the potential") {
    std::ofstream("outer.train") << "CONFIG old3.tadah\n"
                                    "UNCERTAINTY true\n"
                                    "DBFILE db.tadah\n"
                                    "OUTFILE pot3.tadah\n";
    const CommandInput input("TRAIN", {{"config", {"outer.train"}}});
    const std::string what = validated_train_input_error(input, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK(fs::exists("weights.tadah"));
    CHECK(printed.find("is a potential") == std::string::npos);
    INFO("pot3.tadah:\n" << slurp("pot3.tadah"));
    CHECK(read_record("pot3.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "UNCERTAINTY true",
                                   "OUTFILE pot3.tadah"});
  }
  SECTION("a copy of the potential with the line") {
    Context copy = potential_with_oalgo("M_BLR BF_Linear", 3);
    copy.add("UNCERTAINTY", "true");
    copy.add("DBFILE", "db.tadah");
    copy.add("OUTFILE", "pot4.tadah");
    write_pot_file("retrain.train", copy);
    {
      const CommandInput input("TRAIN", {{"config", {"retrain.train"}}});
      const std::string what = validated_train_input_error(input, printed);
      INFO("thrown: " << what << "\nprinted:\n" << printed);
      REQUIRE(what.empty());
      CHECK_FALSE(fs::exists("weights.tadah"));
      INFO("pot4.tadah:\n" << slurp("pot4.tadah"));
      CHECK(read_record("pot4.tadah").lines ==
            std::vector<std::string>{"DBFILE db.tadah", "OUTFILE pot4.tadah"});
      CHECK(printed.find(
                "WARNING: tadah train: the configuration file retrain.train "
                "is a potential (it has WEIGHTS): its UNCERTAINTY true asked "
                "the run that wrote it for weights.tadah, not this run, and "
                "is ignored (tadah keeps such keys in a potential's training "
                "record, not as key lines). Give --uncertainty on the command "
                "line, or an UNCERTAINTY line in a configuration file that "
                "includes the potential with CONFIG, to apply it to this run "
                "(the Training page of the documentation, Retraining from a "
                "potential).") != std::string::npos);
      CHECK(printed.find("development") == std::string::npos);
    }
    const CommandInput input("TRAIN", {{"config", {"retrain.train"}},
                                       {"uncertainty", {"true"}},
                                       {"outfile", {"pot5.tadah"}}});
    const std::string what = validated_train_input_error(input, printed);
    INFO("--uncertainty: thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK(fs::exists("weights.tadah"));
    CHECK(printed.find("is a potential") == std::string::npos);
    INFO("pot5.tadah:\n" << slurp("pot5.tadah"));
    CHECK(read_record("pot5.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "UNCERTAINTY true",
                                   "OUTFILE pot5.tadah"});
  }
  SECTION("a copy of the potential with a dataset key and the line") {
    Context copy = potential_context("M_BLR BF_Linear");
    copy.add("EFILTER", "-9 9");
    copy.add("UNCERTAINTY", "true");
    copy.add("DBFILE", "db.tadah");
    copy.add("OUTFILE", "pot7.tadah");
    write_pot_file("both.train", copy);
    const CommandInput input("TRAIN", {{"config", {"both.train"}}});
    const std::string what = validated_train_input_error(input, printed);
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK_FALSE(fs::exists("weights.tadah"));
    INFO("pot7.tadah:\n" << slurp("pot7.tadah"));
    CHECK(read_record("pot7.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "OUTFILE pot7.tadah"});
    CHECK(printed.find(
              "WARNING: tadah train: the configuration file both.train is a "
              "potential (it has WEIGHTS): its EFILTER -9 9 prepared the "
              "training data of the fit that wrote it, not of this run, and "
              "is ignored (tadah keeps such keys in a potential's training "
              "record, not as key lines). Its UNCERTAINTY true asked the run "
              "that wrote it for weights.tadah, not this run, and is ignored "
              "too. Give --efilter or --uncertainty on the command line, or "
              "an EFILTER or UNCERTAINTY line in a configuration file that "
              "includes the potential with CONFIG, to apply one to this run "
              "(the Training page of the documentation, Retraining from a "
              "potential).") != std::string::npos);
  }
  SECTION("no KeyProvenance") {
    Context c = potential_context("M_BLR BF_Linear");
    c.add("UNCERTAINTY", "true");
    c.add("DBFILE", "db.tadah");
    c.add("OUTFILE", "pot6.tadah");
    Task task("train", c);
    std::ostringstream captured;
    std::streambuf *old_buf = std::cout.rdbuf(captured.rdbuf());
    std::string what;
    try {
      TrainEngine().validate(task);
      TrainEngine().run(task);
    } catch (const std::exception &e) {
      what = e.what();
    }
    std::cout.rdbuf(old_buf);
    printed = captured.str();
    INFO("thrown: " << what << "\nprinted:\n" << printed);
    REQUIRE(what.empty());
    CHECK_FALSE(fs::exists("weights.tadah"));
    INFO("pot6.tadah:\n" << slurp("pot6.tadah"));
    CHECK(read_record("pot6.tadah").lines ==
          std::vector<std::string>{"DBFILE db.tadah", "OUTFILE pot6.tadah"});
    const std::string warning =
        "WARNING: tadah train: the configuration is a potential (it has "
        "WEIGHTS): its UNCERTAINTY true asked the run that wrote it for "
        "weights.tadah, not this run, and is ignored (tadah keeps such keys "
        "in a potential's training record, not as key lines). Give "
        "--uncertainty on the command line to apply it to this run (the "
        "Training page of the documentation, Retraining from a potential).";
    const std::size_t at = printed.find(warning);
    CHECK(at != std::string::npos);
    // Once: validate applies the rule to a copy and prints nothing.
    if (at != std::string::npos)
      CHECK(printed.find(warning, at + 1) == std::string::npos);
  }
}

/* Pins: TrainEngine::run refuses a DBFILE that lists one dataset twice
 * (the Ta pair training configuration with "DBFILE db.tadah db.tadah")
 * with exactly "DBFILE lists one dataset twice: entry 1 (db.tadah) and
 * entry 2 (db.tadah) are one file, so each of its structures would be
 * fitted twice. List each dataset once; to give a dataset more weight, use
 * WDBFILE. An M_BLR potential written by an earlier version of Tadah!
 * records the DBFILE it was trained on: remove that line from a potential
 * used as the training configuration.", and writes no pot.tadah.
 * Before the fix: the advice named a development stage, "An M_BLR
 * potential written by Tadah! 1.5.x or earlier, or by a 1.6.0 development
 * build, records the DBFILE it was trained on: ...": the CHECK fails. */
TEST_CASE("TrainEngine::run refuses a DBFILE that lists one dataset twice, "
          "naming an earlier version's DBFILE line",
          "[train][potential_content][dbfile]") {
  tadah::core::PeriodicTable::initialize();
  ScratchCwd cwd;
  write_ta_dataset(cwd.dir / "db.tadah");
  Context c = training_context_of("M_BLR BF_Linear");
  c.add("DBFILE", "db.tadah");
  c.add("DBFILE", "db.tadah");
  c.add("OUTFILE", "pot.tadah");
  std::string printed;
  const std::string what = train_error(c, printed);
  INFO("printed:\n" << printed);
  CHECK(what ==
        "DBFILE lists one dataset twice: entry 1 (db.tadah) and entry 2 "
        "(db.tadah) are one file, so each of its structures would be fitted "
        "twice. List each dataset once; to give a dataset more weight, use "
        "WDBFILE. An M_BLR potential written by an earlier version of Tadah! "
        "records the DBFILE it was trained on: remove that line from a "
        "potential used as the training configuration.");
  CHECK_FALSE(fs::exists("pot.tadah"));
}
