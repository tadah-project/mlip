// test_uncertainty_output_retired.cpp
//
// Uncertainty output is not part of Tadah! 1.6.0: the errors of Tadah!
// 1.0.0 to 1.5.0 were not standard deviations, and no calibrated estimate
// replaces them. The predictions and the fit are unaffected.
//
//  - tadah predict --error and tadah properties pairwise --error are
//    retired options of those tasks, refused on the command line and in a
//    task file (a TASK section, or the GLOBAL section, which reaches every
//    task of the file), with the reason;
//  - tadah train --uncertainty is the option of the retired key
//    UNCERTAINTY, refused with the reason;
//  - a fit writes no WCOV or WNOISE, and refuses a configuration with
//    WCOV, WNOISE, UNCERTAINTY or ERROR lines (a potential of a development
//    build of Tadah! 1.6.0, which wrote WCOV and WNOISE for an OALGO 3 fit,
//    used as the configuration of a retrain), with the reason and the
//    conversion: deleting the lines.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/key_notice.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task_factory.h>
#include <tadah/mlip/trainer.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Catch::Matchers::Contains;
using Catch::Matchers::StartsWith;
using tadah::core::Context;
using tadah::core::Element;
using tadah::core::IgnoredKeyNoticeCapture;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
namespace fs = std::filesystem;

const std::string kNotPart =
    "Uncertainty output is not part of Tadah! 1.6.0: the errors of Tadah! "
    "1.0.0 to 1.5.0 were not standard deviations, and no calibrated "
    "estimate replaces them.";

/* The working directory of a case. */
struct ScratchCwd {
  fs::path old_cwd, dir;
  ScratchCwd() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_uncertainty_retired_" + std::to_string(::getpid()) + "_" +
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

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* A task input: a task name and its options, as the command line gives
 * them. */
struct Input : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  Input(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* What TaskFactory throws for @p input, "" when it makes the tasks. */
std::string factory_error(const Input &input) {
  try {
    (void)tadah::mlip::TaskFactory().createTasksFromInput(input);
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

/* A 2-atom Ta cell with smooth labels. */
Structure cell(double a, double shift, std::size_t s) {
  Structure st;
  st.cell.load(a, 0.0, 0.0, 0.03, a, 0.0, -0.02, 0.04, a);
  st.add_atom(Atom(Element("Ta"), 0.01, 0.02, -0.01, 0.1, -0.2, 0.05));
  st.add_atom(Atom(Element("Ta"), 0.5 * a + shift, 0.5 * a - shift,
                   0.5 * a, -0.1, 0.2, -0.05));
  st.energy = -23.4 + 0.1 * std::sin(1.3 * s);
  return st;
}

StructureDB training_db() {
  StructureDB db;
  for (std::size_t s = 0; s < 6; ++s)
    db.add(cell(3.25 + 0.03 * s, 0.02 * s, s));
  return db;
}

/* The lines of an OALGO 3 configuration of @p model, and @p extra. */
std::vector<std::string> model_lines(const std::string &model,
                                     const std::vector<std::string> &extra) {
  std::vector<std::string> l = {
      "ATOM Ta",          "WATOM 73",           "INIT2B true",
      "INITMB false",     "TYPE2B D2_Blip 3 3 Ta Ta",
      "CGRID2B 2.8 3.5 4.2", "SGRID2B 2 2 2",   "RCUT2B 5",
      "RCTYPE2B Cut_Cos", "FORCE true",         "LAMBDA 1e-3",
      "OALGO 3",          "BIAS false",         "NORM false",
      "MODEL " + model};
  l.insert(l.end(), extra.begin(), extra.end());
  return l;
}

/* What Trainer::train throws for the configuration of @p lines, "" when
 * it fits; the potential it writes goes to @p pot. */
std::string fit(const std::vector<std::string> &lines, Context &pot) {
  IgnoredKeyNoticeCapture quiet;
  Context c = tadah::core::CoreUserConfigParser().parse(lines);
  MLIPWorkspaceManager ws;
  try {
    Trainer tr(c, ws);
    StructureDB db = training_db();
    tr.train(db);
    pot = tr.get_param_file();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

}  // namespace

/* Pins: the command line of tadah predict --error, tadah properties
 * pairwise --error and tadah train --uncertainty (TaskFactory of their
 * key-value data) is refused before any file is read, with the reason, the
 * option's own (the task's, although its key ERROR is retired too):
 * "--error of tadah predict is retired: tadah predict --error wrote an
 * error computed from SIGMA beside every predicted value (some of the
 * errors negative). Uncertainty output is not part of Tadah! 1.6.0: the
 * errors of Tadah! 1.0.0 to 1.5.0 were not standard deviations, and no
 * calibrated estimate replaces them. Replaced by: nothing in Tadah! 1.6.0.
 * Deleting the option converts it: ...", likewise "--error of tadah
 * properties pairwise is retired: ...", and "--uncertainty: UNCERTAINTY is
 * a retired key: tadah train --uncertainty wrote weights.tadah, ...". The
 * same commands without the option make their task.
 * Before the fix: each option was taken; then the reasons said that the
 * options gave standard deviations, too narrow for many structures (what
 * a development build gave, not a release). */
TEST_CASE("tadah predict, properties pairwise and train refuse --error and "
          "--uncertainty",
          "[uncertainty][retired][cli]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  std::ofstream("pot.tadah") << "ATOM Ta\n";
  std::ofstream("db.tadah") << "\n";
  std::ofstream("config.train") << "ATOM Ta\n";
  const KV predict = {{"potential", {"pot.tadah"}}, {"dbfile", {"db.tadah"}}};
  KV with = predict;
  with["error"] = {"true"};
  CHECK_THAT(factory_error(Input("PREDICT", with)),
             StartsWith("--error of tadah predict is retired: tadah predict "
                        "--error wrote an error computed from SIGMA beside "
                        "every predicted value") &&
                 Contains(kNotPart) &&
                 Contains("Replaced by: nothing in Tadah! 1.6.0. Deleting "
                          "the option converts it"));
  CHECK(factory_error(Input("PREDICT", predict)).empty());

  const KV pairwise = {{"potential", {"pot.tadah"}},
                       {"outfile", {"pair.txt"}},
                       {"range", {"2", "5", "4"}},
                       {"atompair", {"Ta", "Ta"}}};
  with = pairwise;
  with["error"] = {"true"};
  CHECK_THAT(factory_error(Input("PROPERTIES.pairwise", with)),
             StartsWith("--error of tadah properties pairwise is retired: "
                        "tadah properties pairwise --error wrote an error "
                        "computed from SIGMA beside every value of the "
                        "energy curve") &&
                 Contains(kNotPart) &&
                 Contains("Replaced by: nothing in Tadah! 1.6.0."));
  CHECK(factory_error(Input("PROPERTIES.pairwise", pairwise)).empty());

  CHECK_THAT(factory_error(Input("TRAIN", {{"config", {"config.train"}},
                                           {"uncertainty", {"true"}}})),
             StartsWith("--uncertainty: UNCERTAINTY is a retired key: tadah "
                        "train --uncertainty wrote weights.tadah, each "
                        "coefficient of the fit beside a diagonal entry of "
                        "SIGMA.") &&
                 Contains(kNotPart) &&
                 Contains("Replaced by: nothing in Tadah! 1.6.0."));
}

/* Pins: a task file that gives ERROR in a TASK predict or TASK properties
 * pairwise section, or in its GLOBAL section with such a TASK section
 * (the GLOBAL section sets the keys of every task of the file), is refused
 * before any task runs, naming the line and the reason of the task's
 * option ("ERROR (line 3 of p.task) in TASK predict: --error of tadah
 * predict is retired: ..."), the key ERROR being retired too.
 * Before the fix: the tasks were made, and predict wrote errors. */
TEST_CASE("A task file is refused for ERROR in a predict or properties "
          "pairwise task",
          "[uncertainty][retired][cli]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  std::ofstream("pot.tadah") << "ATOM Ta\n";
  std::ofstream("db.tadah") << "\n";
  auto run = [](const std::string &file, const std::string &text) {
    std::ofstream(file) << text;
    return factory_error(Input("", {{"task", {file}}}));
  };
  CHECK_THAT(run("p.task", "TASK predict\n  POTENTIAL pot.tadah\n"
                           "  ERROR true\n  DBFILE db.tadah\n"),
             StartsWith("ERROR (line 3 of p.task) in TASK predict: --error of "
                        "tadah predict is retired: ") &&
                 Contains(kNotPart));
  CHECK_THAT(run("q.task", "TASK properties pairwise\n  POTENTIAL pot.tadah\n"
                           "  OUTFILE pair.txt\n  RANGE 2 5 4\n"
                           "  ATOMPAIR Ta Ta\n  ERROR false\n"),
             StartsWith("ERROR (line 6 of q.task) in TASK properties "
                        "pairwise: --error of tadah properties pairwise is "
                        "retired: ") &&
                 Contains(kNotPart));
  CHECK_THAT(run("g.task", "ERROR true\nTASK predict\n  POTENTIAL pot.tadah\n"
                           "  DBFILE db.tadah\n"),
             StartsWith("ERROR (line 1 of g.task) in the GLOBAL section "
                        "reaches TASK predict") &&
                 Contains(kNotPart));
  CHECK(run("ok.task", "TASK predict\n  POTENTIAL pot.tadah\n"
                       "  DBFILE db.tadah\n")
            .empty());
}

/* Pins: the potential of an OALGO 3 fit (Trainer, M_BLR BF_Linear and
 * M_KRR Kern_Linear, LAMBDA 1e-3 and LAMBDA -1, FORCE true) has no WCOV
 * and no WNOISE.
 * Before the fix: it had both (the covariance of the WEIGHTS and the noise
 * variance of each quantity). */
TEST_CASE("An OALGO 3 fit writes no WCOV or WNOISE",
          "[uncertainty][fit]") {
  PeriodicTable::initialize();
  for (const std::string model : {"M_BLR BF_Linear", "M_KRR Kern_Linear"})
    for (const std::string lambda : {"1e-3", "-1"}) {
      DYNAMIC_SECTION(model << ", LAMBDA " << lambda) {
        std::vector<std::string> lines = model_lines(model, {});
        lines[10] = "LAMBDA " + lambda;
        Context pot;
        REQUIRE(fit(lines, pot).empty());
        CHECK(pot.has_stored("WEIGHTS"));
        CHECK_FALSE(pot.has_stored("WCOV"));
        CHECK_FALSE(pot.has_stored("WNOISE"));
      }
    }
}

/* Pins: a fit (Trainer, as the C++ API trains) refuses a configuration
 * with the WCOV and WNOISE lines of a potential of a development build of
 * Tadah! 1.6.0, naming each line (WCOV by its first values), marked as
 * converted by deleting it, with the reason and the conversion of each;
 * and one with an UNCERTAINTY line (as tadah train 1.0.0 to 1.5.0 wrote
 * into every potential) or an ERROR line (the key of --error of tadah
 * predict), likewise. Without the lines the configuration fits.
 * Before the fix: both fitted, WCOV and WNOISE replaced by those of the
 * fit and UNCERTAINTY read as the switch of weights.tadah; then the lines
 * were not marked and no reason given, and an ERROR line was taken and
 * written into the potential, where nothing read it. */
TEST_CASE("A fit refuses a configuration with WCOV, WNOISE or UNCERTAINTY",
          "[uncertainty][retired][fit]") {
  PeriodicTable::initialize();
  const auto &keys = tadah::core::BlueprintParser::getKeysMapRef();
  Context pot;
  const std::string covariance = fit(
      model_lines("M_BLR BF_Linear", {"WCOV 3 1 0 0 0 1 0 0 0 1",
                                      "WNOISE E 1e-6 F 1e-3"}),
      pot);
  const std::string mark = "; converted by deleting the line)";
  CHECK_THAT(covariance,
             Contains("WCOV 3 1 0 and 7 more values (line 16" + mark) &&
                 Contains("WNOISE E 1e-6 F 1e-3 (line 17" + mark) &&
                 Contains("retired") &&
                 Contains(keys.at("WCOV").retired.reason) &&
                 Contains(keys.at("WCOV").retired.conversion) &&
                 Contains(keys.at("WNOISE").retired.reason) &&
                 Contains(keys.at("WNOISE").retired.conversion) &&
                 Contains("With these lines deleted"));
  const std::string uncertainty =
      fit(model_lines("M_KRR Kern_Linear", {"UNCERTAINTY true"}), pot);
  CHECK_THAT(uncertainty,
             Contains("UNCERTAINTY true (line 16" + mark) &&
                 Contains(keys.at("UNCERTAINTY").retired.reason) &&
                 Contains(keys.at("UNCERTAINTY").retired.conversion) &&
                 Contains("With this line deleted"));
  pot = Context();
  const std::string error =
      fit(model_lines("M_BLR BF_Linear", {"ERROR true"}), pot);
  CHECK_THAT(error, Contains("ERROR true (line 16" + mark) &&
                        Contains(kNotPart) &&
                        Contains(keys.at("ERROR").retired.conversion));
  CHECK_FALSE(pot.has_stored("ERROR"));
  CHECK(fit(model_lines("M_BLR BF_Linear", {}), pot).empty());
}
