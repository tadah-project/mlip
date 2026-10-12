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
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>

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
 * configuration without a training DBFILE gives ("A potential records no
 * DBFILE (no potential tadah writes has one): to train from one, add a
 * DBFILE line to a copy of it (the Training page of the documentation,
 * Retraining from a potential)."), prints no "Key not found" line, and
 * writes no pot2.tadah.
 * Before the fix: the DBFILE read of find_unique_elements threw "Key not
 * found in the Context object: DBFILE", after the Logger's ERROR line
 * "Context::operator() -> Key not found: DBFILE". */
TEST_CASE("TrainEngine::run refuses a configuration without DBFILE, saying "
          "that a potential records none",
          "[train][potential_content]") {
  tadah::core::PeriodicTable::initialize();
  const std::string advice =
      "A potential records no DBFILE (no potential tadah writes has one): "
      "to train from one, add a DBFILE line to a copy of it (the Training "
      "page of the documentation, Retraining from a potential).";
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
