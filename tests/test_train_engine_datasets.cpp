// test_train_engine_datasets.cpp
//
// TrainEngine::run (tadah train, a TASK train of a task file) and its
// training datasets: a potential records no DBFILE (training_datasets.h),
// so `tadah train -c pot.tadah` without a DBFILE line is the likely mistake
// on the documented retrain route; the configuration is refused before any
// file is read, naming the key, saying that a potential records none and
// how to retrain from one.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/task.h>

#include <filesystem>
#include <fstream>
#include <iostream>
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
    "A potential records no DBFILE (no potential tadah writes has one): to "
    "train from one, add a DBFILE line to a copy of it; a potential of tadah "
    "refit or refit --edit also pins every weight with its FIXINDEX and "
    "FIXWEIGHT lines, the seed of a tadah hpo coefficient search, so delete "
    "them from the copy for the fit to change the weights (the Training page "
    "of the documentation, Retraining from a potential).";

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
 * configuration without a training DBFILE gives (kNoDatasetsAdvice: add a
 * DBFILE line to a copy of it, and delete the FIXINDEX and FIXWEIGHT lines
 * with which a refit potential pins every weight), prints no "Key not
 * found" line, and writes no pot2.tadah.
 * Before the fix: the DBFILE read of find_unique_elements threw "Key not
 * found in the Context object: DBFILE", after the Logger's ERROR line
 * "Context::operator() -> Key not found: DBFILE".
 * Before the advice named the pins (cross-lane review 2): it ended "add a
 * DBFILE line to a copy of it (the Training page of the documentation,
 * Retraining from a potential).", and a refit potential so retrained
 * fitted nothing (the next test case). */
TEST_CASE("TrainEngine::run refuses a configuration without DBFILE, saying "
          "that a potential records none",
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

/* Three four-atom bcc Ta structures (energies -0.001, -0.04, 0.02), no
 * forces, written to @p path. */
void write_ta_dataset(const fs::path &path) {
  std::ofstream os(path);
  const double energies[] = {-0.001, -0.04, 0.02};
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
