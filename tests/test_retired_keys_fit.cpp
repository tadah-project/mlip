// test_retired_keys_fit.cpp
//
// The keys retired in 1.6.0 (EWEIGHT, FWEIGHT, SWEIGHT, WDBFILE_AUTO,
// EWEIGHT_TEMP, and UNCERTAINTY, which tadah train 1.5.0 wrote into every
// potential) in the programs that load and fit potentials.
//
//  - every load of a potential that has them works, with one notice per
//    load listing them: tadah predict, tadah properties pairwise, tadah
//    analysis descriptor and tadah refit --edit;
//  - every fit refuses a configuration that records them, with the
//    conversion: the C++ API (Trainer, M_BLR and M_KRR train), tadah train
//    -c with a potential, and tadah train -c of the potential refit --edit
//    wrote from one (the edit keeps the retired lines as they are and adds
//    no error scale); a potential converted as the refusal says, its
//    WDBFILE values outside it, reproduces the fit of Tadah! 1.5.0, and
//    under M_KRR implementation 2 the refusal gives that implementation's
//    conversion;
//  - prediction neither validates nor reads the error scales: a potential
//    trained with FORCE true and FSIGMA 0.2 predicts without -F;
//  - a potential a fit writes has no retired line and retrains.
//
// The potential of Tadah! 1.5.0 is tests_data/potentials_1_5/
// ta_blip_1.5.0.tadah, written by `tadah train -c config.train` of tadah
// 1.5.0 (D2_Blip 3 3 Ta Ta, CGRID2B 2.8 3.5 4.2, SGRID2B 2 2 2, RCUT2B 5,
// Cut_Cos, M_BLR BF_Linear, FORCE true, LAMBDA 0, OALGO 3, EWEIGHT 1.0,
// FWEIGHT 100, three 54-atom bcc Ta cells), as it wrote it: its EWEIGHT,
// FWEIGHT, SWEIGHT, UNCERTAINTY and WDBFILE_AUTO lines are lines 8, 10,
// 23, 25 and 27, and it has SIGMA, DBFILE and the ESHIFT toggles.
//
// tests_data/potentials_1_5/wdbfile_auto/ holds a fit of tadah 1.5.0 with
// WDBFILE and WDBFILE_AUTO both set: config.train (DBFILE a.dat b.dat, the
// first 4 and the next 2 structures of the Quick Start's db.train, 54 atoms
// each; D2_Blip 4 4 Ta Ta, M_BLR BF_Linear, BIAS true, OALGO 1, LAMBDA
// 1e-3, FORCE true, EWEIGHT 2.5, FWEIGHT 50, WDBFILE 1 3, WDBFILE_AUTO
// 0.7) and pot_1.5.0.tadah, the potential `tadah train -c config.train` of
// tadah 1.5.0 wrote there.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/key_notice.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/analysis_descriptor_engine.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/properties_pairwise_engine.h>
#include <tadah/mlip/engines/refit_engine.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>
#include <tadah/mlip/trainer.h>

#include <algorithm>
#include <cmath>
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

using Catch::Matchers::Contains;
using tadah::core::Context;
using tadah::core::Element;
using tadah::core::IgnoredKeyNoticeCapture;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Task;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
namespace fs = std::filesystem;

const fs::path kOldPot =
    fs::absolute("tests_data/potentials_1_5/ta_blip_1.5.0.tadah");

/* The working directory of a case. */
struct ScratchCwd {
  fs::path old_cwd, dir;
  ScratchCwd() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_retired_fit_" + std::to_string(::getpid()) + "_" +
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

std::string slurp(const fs::path &p) {
  std::ifstream in(p);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

/* The lines of @p text whose first word is @p key. */
std::vector<std::string> key_lines(const std::string &text,
                                   const std::string &key) {
  std::vector<std::string> out;
  std::istringstream in(text);
  for (std::string l; std::getline(in, l);) {
    std::istringstream ls(l);
    std::string k;
    if ((ls >> k) && k == key) out.push_back(l);
  }
  return out;
}

/* A 2-atom Ta cell with smooth labels. */
Structure cell(double a, double shift, std::size_t s) {
  Structure st;
  st.label = "cell " + std::to_string(s);
  st.cell.load(a, 0.0, 0.0, 0.03, a, 0.0, -0.02, 0.04, a);
  st.add_atom(Atom(Element("Ta"), 0.01, 0.02, -0.01, 0.1, -0.2, 0.05));
  st.add_atom(Atom(Element("Ta"), 0.5 * a + shift, 0.5 * a - shift,
                   0.5 * a, -0.1, 0.2, -0.05));
  st.energy = -23.4 + 0.1 * std::sin(1.3 * s);
  return st;
}

/* Writes a dataset of four cells to @p path. */
void write_db(const std::string &path) {
  StructureDB db;
  for (std::size_t s = 0; s < 4; ++s)
    db.add(cell(3.25 + 0.03 * s, 0.02 * s, s));
  db.dump_to_file(path, false);
}

/* The configuration keys of the 1.5.0 potential's model, and @p extra. */
std::vector<std::string> model_lines(const std::vector<std::string> &extra) {
  std::vector<std::string> l = {
      "ATOM Ta",          "WATOM 73",           "INIT2B true",
      "INITMB false",     "TYPE2B D2_Blip 3 3 Ta Ta",
      "CGRID2B 2.8 3.5 4.2", "SGRID2B 2 2 2",   "RCUT2B 5",
      "RCTYPE2B Cut_Cos", "FORCE true",         "LAMBDA 0",
      "OALGO 3",          "BIAS false",         "NORM false",
      "MODEL M_BLR BF_Linear"};
  l.insert(l.end(), extra.begin(), extra.end());
  return l;
}

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* What TaskFactory gets from the command line. */
struct CommandInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  CommandInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* Runs @p run on each task TaskFactory makes of @p input; what it throws,
 * "" when nothing; the notices of the run go to @p notices. */
template <typename Run>
std::string run_input(const CommandInput &input, Run run,
                      std::vector<std::string> &notices) {
  IgnoredKeyNoticeCapture cap;
  std::string what;
  try {
    for (Task &t : tadah::mlip::TaskFactory().createTasksFromInput(input))
      run(t);
  } catch (const std::exception &e) {
    what = e.what();
  }
  notices = cap.notices();
  return what;
}

std::size_t retired_notices(const std::vector<std::string> &notices) {
  std::size_t n = 0;
  for (const std::string &s : notices)
    if (s.find("Retired keys") != std::string::npos) ++n;
  return n;
}

/* The WEIGHTS of the potential @p p. */
std::vector<double> weights_of(const fs::path &p) {
  IgnoredKeyNoticeCapture quiet;
  const Context c = tadah::core::CoreUserConfigParser().parse(p.string());
  std::vector<double> w(c.size("WEIGHTS"));
  c.get("WEIGHTS", w);
  return w;
}

/* The largest difference of @p got from @p ref over the largest magnitude
 * of @p ref (both of one size, not empty). */
double relative_difference(const std::vector<double> &got,
                           const std::vector<double> &ref) {
  REQUIRE(got.size() == ref.size());
  REQUIRE_FALSE(ref.empty());
  double scale = 0.0, diff = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    scale = std::max(scale, std::fabs(ref[i]));
    diff = std::max(diff, std::fabs(got[i] - ref[i]));
  }
  return diff / scale;
}

}  // namespace

/* Pins: a fit refuses a configuration that records a retired key, through
 * the C++ API (a configuration parsed with EWEIGHT 1.0 and FWEIGHT 100,
 * trained by Trainer, as the API examples do), with the lines and the
 * conversion of each; the same configuration without them trains.
 * Before the fix: the fit ran, EWEIGHT and FWEIGHT weighting the rows. */
TEST_CASE("The C++ API refuses a fit of a configuration with retired keys",
          "[retired][fit]") {
  PeriodicTable::initialize();
  IgnoredKeyNoticeCapture quiet;
  StructureDB db;
  for (std::size_t s = 0; s < 4; ++s) db.add(cell(3.25 + 0.03 * s, 0.02 * s, s));
  for (const std::string model : {"M_BLR BF_Linear", "M_KRR Kern_Linear"}) {
    DYNAMIC_SECTION(model) {
      std::vector<std::string> lines =
          model_lines({"EWEIGHT 1.0", "FWEIGHT 100"});
      lines[14] = "MODEL " + model;
      Context c = tadah::core::CoreUserConfigParser().parse(lines);
      MLIPWorkspaceManager ws;
      Trainer tr(c, ws);
      StructureDB d = db;
      std::string what;
      try {
        tr.train(d);
      } catch (const std::exception &e) {
        what = e.what();
      }
      CHECK_THAT(what, Contains("EWEIGHT 1.0 (line 16)") &&
                           Contains("FWEIGHT 100 (line 17)") &&
                           Contains("retired") &&
                           Contains("FSIGMA = 1/sqrt(FWEIGHT)") &&
                           Contains("ESIGMA = 1000/(N sqrt(EWEIGHT))"));
      lines.resize(15);
      Context ok = tadah::core::CoreUserConfigParser().parse(lines);
      MLIPWorkspaceManager ws2;
      Trainer tr2(ok, ws2);
      StructureDB d2 = db;
      CHECK_NOTHROW(tr2.train(d2));
    }
  }
}

/* Pins: tadah train -c of a configuration that includes the 1.5.0
 * potential (CONFIG old.tadah and a DBFILE line) is refused before any
 * structure is fitted, naming the five retired lines of the potential file
 * (its UNCERTAINTY false among them) and the conversion of each, and
 * writes no potential; the parse gives one notice listing them.
 * Before the fix: the retrain ran with EWEIGHT 1.0, FWEIGHT 100 and
 * SWEIGHT 1.0 and wrote pot.tadah. Before UNCERTAINTY was retired: the
 * refusal named four lines, the UNCERTAINTY line not read. */
TEST_CASE("tadah train -c refuses a potential of Tadah! 1.5.0 with the "
          "conversion",
          "[retired][fit]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  write_db("db.tadah");
  {
    std::ofstream cfg("retrain.train");
    cfg << "CONFIG " << kOldPot.string() << "\nDBFILE db.tadah\n";
  }
  std::vector<std::string> notices;
  const std::string what = run_input(
      CommandInput("TRAIN", {{"config", {"retrain.train"}}}),
      [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); }, notices);
  const std::string f = kOldPot.string();
  CHECK_THAT(what, Contains("tadah train") &&
                       Contains("EWEIGHT 1.0 (line 8 of " + f + ")") &&
                       Contains("FWEIGHT 100 (line 10 of " + f + ")") &&
                       Contains("SWEIGHT 1.0 (line 23 of " + f) &&
                       Contains("WDBFILE_AUTO 0.0 (line 27 of " + f) &&
                       Contains("UNCERTAINTY false (line 25 of " + f + ")") &&
                       Contains("SSIGMA = 160.2/sqrt(SWEIGHT)"));
  CHECK_FALSE(fs::exists("pot.tadah"));
  CHECK(retired_notices(notices) == 1);
}

/* Pins: the 1.5.0 potential loads, with one notice per load that lists its
 * five retired keys (UNCERTAINTY among them) and their lines, in tadah
 * predict (energies and forces of a dataset), tadah properties pairwise
 * and tadah analysis descriptor.
 * Before the fix: the keys were stored (no notice), as keys of a model
 * nothing read. Before UNCERTAINTY was retired: the notice listed four. */
TEST_CASE("A potential of Tadah! 1.5.0 loads with one notice per load",
          "[retired][load]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  write_db("db.tadah");
  const std::string pot = kOldPot.string();
  std::vector<std::string> notices;
  SECTION("tadah predict") {
    const std::string what = run_input(
        CommandInput("PREDICT", {{"potential", {pot}},
                                 {"dbfile", {"db.tadah"}},
                                 {"force", {"true"}}}),
        [](Task &t) { tadah::mlip::engines::PredictEngine().run(t); },
        notices);
    CHECK(what.empty());
    CHECK(fs::exists("energy.pred"));
  }
  SECTION("tadah properties pairwise") {
    const std::string what = run_input(
        CommandInput("PROPERTIES.pairwise", {{"potential", {pot}},
                                             {"range", {"2.5", "4.5", "5"}},
                                             {"atompair", {"Ta", "Ta"}},
                                             {"outfile", {"pair.txt"}}}),
        [](Task &t) {
          tadah::mlip::engines::PropertiesPairwiseEngine().run(t);
        },
        notices);
    CHECK(what.empty());
  }
  SECTION("tadah analysis descriptor") {
    const std::string what = run_input(
        CommandInput("ANALYSIS.descriptor", {{"potential", {pot}},
                                             {"dbfile", {"db.tadah"}},
                                             {"outfile", {"desc.txt"}}}),
        [](Task &t) {
          tadah::mlip::engines::AnalysisDescriptorEngine().run(t);
        },
        notices);
    CHECK(what.empty());
  }
  REQUIRE(retired_notices(notices) == 1);
  for (const std::string &n : notices)
    if (n.find("Retired keys") != std::string::npos)
      CHECK_THAT(n, Contains("EWEIGHT (line 8 of " + pot + ")") &&
                        Contains("FWEIGHT (line 10 of " + pot + ")") &&
                        Contains("SWEIGHT (line 23 of " + pot + ")") &&
                        Contains("UNCERTAINTY (line 25 of " + pot + ")") &&
                        Contains("WDBFILE_AUTO (line 27 of " + pot + ")"));
}

/* Pins: tadah predict neither validates nor reads the error scales of a
 * potential: one trained with FORCE true and FSIGMA 0.2 predicts energies
 * without -F (the task's FORCE false over the potential's), where a fit
 * would refuse FSIGMA 0.2 with FORCE false.
 * Before the fix: FSIGMA was not a key. */
TEST_CASE("A potential trained with forces predicts without -F",
          "[retired][load]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  write_db("db.tadah");
  Context c = tadah::core::CoreUserConfigParser().parse(
      model_lines({"FSIGMA 0.2", "DBFILE db.tadah"}));
  {
    Task t("train", c);
    t.getContext().add("OUTFILE", std::string("pot.tadah"));
    REQUIRE_NOTHROW(tadah::mlip::engines::TrainEngine().run(t));
  }
  REQUIRE(key_lines(slurp("pot.tadah"), "FSIGMA").size() == 1);
  std::vector<std::string> notices;
  const std::string what = run_input(
      CommandInput("PREDICT", {{"potential", {"pot.tadah"}},
                               {"dbfile", {"db.tadah"}}}),
      [](Task &t) { tadah::mlip::engines::PredictEngine().run(t); }, notices);
  CHECK(what.empty());
  CHECK(retired_notices(notices) == 0);
}

/* Pins: a potential a fit writes has no line of a retired key, and a
 * retrain from it (tadah train -c with a DBFILE line added) runs.
 * Before the fix: the potential had EWEIGHT, FWEIGHT and SWEIGHT lines
 * (their defaults). */
TEST_CASE("A potential written by a fit has no retired line and retrains",
          "[retired][fit]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  write_db("db.tadah");
  {
    std::ofstream cfg("config.train");
    for (const std::string &l : model_lines({"DBFILE db.tadah"}))
      cfg << l << "\n";
  }
  std::vector<std::string> notices;
  CHECK(run_input(CommandInput("TRAIN", {{"config", {"config.train"}}}),
                  [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); },
                  notices)
            .empty());
  const std::string text = slurp("pot.tadah");
  for (const char *key : {"EWEIGHT", "FWEIGHT", "SWEIGHT", "WDBFILE_AUTO",
                          "EWEIGHT_TEMP"}) {
    INFO(key);
    CHECK(key_lines(text, key).empty());
  }
  CHECK(key_lines(text, "ESIGMA").size() == 1);
  {
    std::ofstream cfg("retrain.train");
    cfg << "CONFIG pot.tadah\nDBFILE db.tadah\n";
  }
  CHECK(run_input(CommandInput("TRAIN", {{"config", {"retrain.train"}},
                                         {"outfile", {"pot2.tadah"}}}),
                  [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); },
                  notices)
            .empty());
  CHECK(fs::exists("pot2.tadah"));
  CHECK(retired_notices(notices) == 0);
}

/* Pins: tadah refit --edit of the potential of Tadah! 1.5.0 (FUNC phi,
 * ADD 3.0) loads it with one notice, writes the edited potential with its
 * EWEIGHT, FWEIGHT, SWEIGHT, UNCERTAINTY and WDBFILE_AUTO lines as they
 * are and with no ESIGMA, FSIGMA, SSIGMA, ENORM or ENERGY line, and tadah
 * train -c of the edited potential (with a DBFILE line) is refused naming
 * the retired lines of the edited file: the edit cannot turn a potential
 * of retired keys into one that retrains with the default error scales.
 * Before the fix: the parse stored the four weight keys and the edit wrote
 * them back as key lines, and a retrain read them as weights. Before
 * UNCERTAINTY was retired: the edit moved its UNCERTAINTY line into the
 * training record. */
TEST_CASE("refit --edit keeps the retired lines, and a fit refuses the "
          "edited potential",
          "[retired][refit]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  const std::string old_text = slurp(kOldPot);
  std::vector<std::string> retired;
  for (const char *key :
       {"EWEIGHT", "FWEIGHT", "SWEIGHT", "UNCERTAINTY", "WDBFILE_AUTO"}) {
    const std::vector<std::string> l = key_lines(old_text, key);
    REQUIRE(l.size() == 1);
    retired.push_back(l.front());
  }
  Context c;
  c.add("EDIT", kOldPot.string());
  c.add("FUNC", std::string("phi"));
  c.add("ADD", 3.0);
  c.add("OUTFILE", std::string("edited.tadah"));
  c.add("HPOFILEOUT", std::string("edited.hpotarget"));
  Task task("refit", c);
  std::vector<std::string> notices;
  {
    IgnoredKeyNoticeCapture cap;
    REQUIRE(tadah::mlip::engines::RefitEngine().run(task) == EXIT_SUCCESS);
    notices = cap.notices();
  }
  CHECK(retired_notices(notices) == 1);
  const std::string text = slurp("edited.tadah");
  for (const std::string &l : retired) {
    INFO(l << " in:\n" << text);
    std::istringstream ls(l);
    std::string key;
    ls >> key;
    CHECK(key_lines(text, key) == std::vector<std::string>{l});
  }
  for (const char *key : {"ESIGMA", "FSIGMA", "SSIGMA", "ENORM", "ENERGY"}) {
    INFO(key);
    CHECK(key_lines(text, key).empty());
  }
  write_db("db.tadah");
  {
    std::ofstream cfg("retrain.train");
    cfg << "CONFIG edited.tadah\nDBFILE db.tadah\n";
  }
  const std::string what = run_input(
      CommandInput("TRAIN", {{"config", {"retrain.train"}}}),
      [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); }, notices);
  CHECK_THAT(what, Contains("EWEIGHT 1.0 (line") &&
                       Contains("of edited.tadah") &&
                       Contains("WDBFILE_AUTO 0.0") && Contains("retired"));
}

/* Pins: the refusal of a retrain of the potential of Tadah! 1.5.0 (FORCE
 * true, STRESS false; CONFIG of it and a DBFILE of four 2-atom cells)
 * marks the lines a deletion converts, SWEIGHT 1.0 (STRESS false here)
 * and WDBFILE_AUTO 0.0, and not EWEIGHT and FWEIGHT, and names the
 * UNCERTAINTY false line, whose conversion is its deletion; the
 * configuration converted as the refusal says -- those three lines
 * deleted, EWEIGHT 1.0 as ESIGMA = 1000/(N sqrt(EWEIGHT)) = 500 for the
 * 2-atom cells, FWEIGHT 100 as FSIGMA = 1/sqrt(FWEIGHT) = 0.1, its DBFILE
 * line (the data of the 1.5.0 fit) for db.tadah -- trains and writes a
 * potential.
 * Before the fix: the refusal gave SSIGMA = 160.2/sqrt(SWEIGHT) for the
 * SWEIGHT line too, and the configuration converted so (SSIGMA 160.2) was
 * refused again, as SSIGMA other than its default with STRESS false. */
TEST_CASE("A potential of Tadah! 1.5.0 converted as its refusal says "
          "trains",
          "[retired][fit]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  write_db("db.tadah");
  {
    std::ofstream cfg("retrain.train");
    cfg << "CONFIG " << kOldPot.string() << "\nDBFILE db.tadah\n";
  }
  std::vector<std::string> notices;
  const std::string what = run_input(
      CommandInput("TRAIN", {{"config", {"retrain.train"}}}),
      [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); }, notices);
  const std::string f = kOldPot.string();
  CHECK_THAT(what,
             Contains("EWEIGHT 1.0 (line 8 of " + f + ")") &&
                 Contains("FWEIGHT 100 (line 10 of " + f + ")") &&
                 Contains("SWEIGHT 1.0 (line 23 of " + f +
                          "; STRESS false here: converted by deleting the "
                          "line)") &&
                 Contains("WDBFILE_AUTO 0.0 (line 27 of " + f +
                          "; 0: converted by deleting the line)") &&
                 Contains("UNCERTAINTY false (line 25 of " + f + ")"));
  {
    std::istringstream in(slurp(kOldPot));
    std::ofstream out("converted.tadah");
    for (std::string l; std::getline(in, l);) {
      std::istringstream ls(l);
      std::string key;
      ls >> key;
      // DBFILE: the data of the 1.5.0 fit, which the test does not have
      // (the retrain reads db.tadah).
      if (key == "SWEIGHT" || key == "WDBFILE_AUTO" || key == "DBFILE" ||
          key == "UNCERTAINTY")
        continue;
      if (key == "EWEIGHT") l = "ESIGMA 500";
      if (key == "FWEIGHT") l = "FSIGMA 0.1";
      out << l << "\n";
    }
  }
  {
    std::ofstream cfg("converted.train");
    cfg << "CONFIG converted.tadah\nDBFILE db.tadah\n";
  }
  const std::string converted = run_input(
      CommandInput("TRAIN", {{"config", {"converted.train"}},
                             {"outfile", {"new.tadah"}}}),
      [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); }, notices);
  INFO(converted);
  CHECK(converted.empty());
  CHECK(fs::exists("new.tadah"));
}

/* Pins (an oracle of Tadah! 1.5.0): WDBFILE_AUTO a multiplied the WDBFILE
 * value of each dataset by n_d^-a. The refusal of the 1.5.0 configuration
 * with WDBFILE 1 3 and WDBFILE_AUTO 0.7 gives that conversion, and the
 * converted configuration -- WDBFILE 1 x 4^-0.7 3 x 2^-0.7 (4 and 2
 * structures), ESIGMA = 1000/(54 sqrt(2.5)) for EWEIGHT 2.5 and the
 * 54-atom cells, FSIGMA = 1/sqrt(50) for FWEIGHT 50 -- fits the WEIGHTS of
 * the potential tadah 1.5.0 wrote, to 1e-10 relative.
 * Before the fix: the conversion said WDBFILE with the values n_d^-a
 * (WDBFILE_AUTO 0 no WDBFILE), which leaves out WDBFILE 1 3: WEIGHTS
 * -23.17 and 1.656 for 1.5.0's -23.84 and 1.588 in such a fit. */
TEST_CASE("WDBFILE_AUTO converts as each WDBFILE value times n_d^-a: a fit "
          "of Tadah! 1.5.0 is reproduced",
          "[retired][fit][wdbfile_auto]") {
  PeriodicTable::initialize();
  const fs::path dir =
      fs::absolute("tests_data/potentials_1_5/wdbfile_auto");
  ScratchCwd cwd;
  for (const char *f : {"a.dat", "b.dat", "config.train"})
    fs::copy_file(dir / f, f);
  std::vector<std::string> notices;
  const std::string what = run_input(
      CommandInput("TRAIN", {{"config", {"config.train"}}}),
      [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); }, notices);
  CHECK_THAT(what, Contains("WDBFILE_AUTO 0.7 (line 21 of config.train)") &&
                       Contains("Each WDBFILE value (1 for every dataset "
                                "without WDBFILE) multiplied by n_d^-a"));
  {
    std::istringstream in(slurp("config.train"));
    std::ofstream out("converted.train");
    for (std::string l; std::getline(in, l);) {
      std::istringstream ls(l);
      std::string key;
      ls >> key;
      if (key == "EWEIGHT" || key == "FWEIGHT" || key == "WDBFILE" ||
          key == "WDBFILE_AUTO")
        continue;
      out << l << "\n";
    }
    out.precision(17);
    out << "ESIGMA " << 1000.0 / (54.0 * std::sqrt(2.5)) << "\n"
        << "FSIGMA " << 1.0 / std::sqrt(50.0) << "\n"
        << "WDBFILE " << 1.0 * std::pow(4.0, -0.7) << " "
        << 3.0 * std::pow(2.0, -0.7) << "\n";
  }
  REQUIRE(run_input(CommandInput("TRAIN", {{"config", {"converted.train"}},
                                           {"outfile", {"new.tadah"}}}),
                    [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); },
                    notices)
              .empty());
  const std::vector<double> ref = weights_of(dir / "pot_1.5.0.tadah");
  const std::vector<double> got = weights_of("new.tadah");
  REQUIRE(got.size() == ref.size());
  REQUIRE_FALSE(ref.empty());
  double scale = 0.0;
  for (double v : ref) scale = std::max(scale, std::fabs(v));
  for (std::size_t i = 0; i < ref.size(); ++i) {
    INFO("weight " << i << ": " << got[i] << " against " << ref[i]);
    CHECK(std::fabs(got[i] - ref[i]) <= 1e-10 * scale);
  }
}

/* Pins (an oracle of Tadah! 1.5.0): the potential tadah 1.5.0 wrote with
 * WDBFILE 1 3 and WDBFILE_AUTO 0.7 (tests_data/potentials_1_5/
 * wdbfile_auto/pot_1.5.0.tadah, its key lines and DBFILE a.dat b.dat),
 * retrained from itself, is refused with the conversion, which says that a
 * retrain does not read a potential's WDBFILE line and where the WDBFILE
 * values go; the potential converted as the refusal says -- EWEIGHT 2.5 as
 * ESIGMA = 1000/(54 sqrt(2.5)), FWEIGHT 50 as FSIGMA = 1/sqrt(50), the
 * SWEIGHT line (STRESS false), the WDBFILE_AUTO line and the UNCERTAINTY
 * false line deleted, its
 * WDBFILE line left as it is -- with the WDBFILE values 1 x 4^-0.7 and 3 x
 * 2^-0.7 (4 and 2 structures) given in a file that includes it with
 * CONFIG, or with --wdbfile, fits the WEIGHTS of the potential, to 1e-10
 * relative.
 * Before the fix: the refusal said to put the converted keys in place of
 * the lines, a potential's included; a WDBFILE line of the converted
 * values in the potential was ignored (a WARNING), and the retrain fitted
 * WEIGHTS -5.362 and 0.5288 for 1.5.0's -5.334 and 0.5203. */
TEST_CASE("A potential of Tadah! 1.5.0 with WDBFILE and WDBFILE_AUTO, "
          "converted as its refusal says, reproduces its fit",
          "[retired][fit][wdbfile_auto]") {
  PeriodicTable::initialize();
  const fs::path dir =
      fs::absolute("tests_data/potentials_1_5/wdbfile_auto");
  ScratchCwd cwd;
  for (const char *f : {"a.dat", "b.dat"}) fs::copy_file(dir / f, f);
  fs::copy_file(dir / "pot_1.5.0.tadah", "pot.tadah");
  std::vector<std::string> notices;
  const std::string what = run_input(
      CommandInput("TRAIN", {{"config", {"pot.tadah"}},
                             {"outfile", {"refused.tadah"}}}),
      [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); }, notices);
  CHECK_THAT(what,
             Contains("SWEIGHT 1.0 (line 23 of pot.tadah; STRESS false "
                      "here: converted by deleting the line)") &&
                 Contains("WDBFILE_AUTO 0.7 (line 28 of pot.tadah)") &&
                 Contains("UNCERTAINTY false (line 25 of pot.tadah)") &&
                 Contains("A retrain does not read a potential's WDBFILE "
                          "line") &&
                 Contains("the WDBFILE values of a potential on the command "
                          "line or in a file that includes it"));
  CHECK_FALSE(fs::exists("refused.tadah"));
  {
    std::istringstream in(slurp("pot.tadah"));
    std::ofstream out("converted.tadah");
    out.precision(17);
    for (std::string l; std::getline(in, l);) {
      std::istringstream ls(l);
      std::string key;
      ls >> key;
      if (key == "SWEIGHT" || key == "WDBFILE_AUTO" || key == "UNCERTAINTY")
        continue;
      if (key == "EWEIGHT")
        out << "ESIGMA " << 1000.0 / (54.0 * std::sqrt(2.5)) << "\n";
      else if (key == "FWEIGHT")
        out << "FSIGMA " << 1.0 / std::sqrt(50.0) << "\n";
      else
        out << l << "\n";
    }
  }
  std::ostringstream w1, w2;
  w1.precision(17);
  w2.precision(17);
  w1 << 1.0 * std::pow(4.0, -0.7);
  w2 << 3.0 * std::pow(2.0, -0.7);
  {
    std::ofstream inc("retrain.train");
    inc << "CONFIG converted.tadah\nWDBFILE " << w1.str() << " " << w2.str()
        << "\n";
  }
  const std::vector<double> ref = weights_of(dir / "pot_1.5.0.tadah");
  SECTION("WDBFILE in a file that includes the potential") {
    REQUIRE(run_input(CommandInput("TRAIN", {{"config", {"retrain.train"}},
                                             {"outfile", {"new.tadah"}}}),
                      [](Task &t) {
                        tadah::mlip::engines::TrainEngine().run(t);
                      },
                      notices)
                .empty());
    CHECK(relative_difference(weights_of("new.tadah"), ref) <= 1e-10);
  }
  SECTION("WDBFILE on the command line") {
    REQUIRE(run_input(CommandInput("TRAIN", {{"config", {"converted.tadah"}},
                                             {"wdbfile", {w1.str(), w2.str()}},
                                             {"outfile", {"new.tadah"}}}),
                      [](Task &t) {
                        tadah::mlip::engines::TrainEngine().run(t);
                      },
                      notices)
                .empty());
    CHECK(relative_difference(weights_of("new.tadah"), ref) <= 1e-10);
  }
}

/* Pins: M_KRR implementation 2 of Tadah! 1.0.0 to 1.5.0 solved (K +
 * LAMBDA I) w = t on the energies per atom in eV and read none of the
 * retired keys. A configuration of it with EWEIGHT 1.0 and WDBFILE_AUTO
 * 0.5 is refused with that conversion: each line marked as converted by
 * deleting it, and ESIGMA 1000, or LAMBDA times 10^6 with ESIGMA 1, for
 * that solve (not ESIGMA = 1000/(N sqrt(EWEIGHT)), which weighs the
 * energies otherwise); converted so (MODEL M_KRR Kern_Linear 2, SBASIS 3,
 * LAMBDA 1e-3, four 2-atom cells), the two forms train to the same
 * WEIGHTS, to 1e-10 relative, and the lines deleted alone (ESIGMA 1) give
 * other WEIGHTS. That ESIGMA 1000 is the solve (K + LAMBDA I) w = t is
 * pinned against a ridge oracle in test_krr_implementation2_weights.cpp.
 * Before the fix: the refusal gave the conversions of implementation 1,
 * ESIGMA = 1000/(N sqrt(EWEIGHT)) and WDBFILE values n_d^-a, which change
 * the fit of implementation 2 (it refuses WDBFILE values other than 1). */
TEST_CASE("A configuration of M_KRR implementation 2 with retired keys is "
          "refused with the conversion of implementation 2",
          "[retired][fit][krr]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  write_db("db.tadah");
  const std::vector<std::string> model = {
      "ATOM Ta",          "WATOM 73",        "INIT2B true",
      "INITMB false",     "TYPE2B D2_Blip 3 3 Ta Ta",
      "CGRID2B 2.8 3.5 4.2", "SGRID2B 2 2 2", "RCUT2B 5",
      "RCTYPE2B Cut_Cos", "BIAS false",      "NORM false",
      "MODEL M_KRR Kern_Linear 2",           "SBASIS 3",
      "DBFILE db.tadah"};
  auto write_config = [&](const std::string &name,
                          const std::vector<std::string> &extra) {
    std::ofstream cfg(name);
    for (const std::string &l : model) cfg << l << "\n";
    for (const std::string &l : extra) cfg << l << "\n";
  };
  auto train = [](const std::string &config, const std::string &outfile,
                  std::vector<std::string> &notices) {
    return run_input(
        CommandInput("TRAIN",
                     {{"config", {config}}, {"outfile", {outfile}}}),
        [](Task &t) { tadah::mlip::engines::TrainEngine().run(t); }, notices);
  };
  std::vector<std::string> notices;
  write_config("old.train",
               {"LAMBDA 1e-3", "EWEIGHT 1.0", "WDBFILE_AUTO 0.5"});
  const std::string what = train("old.train", "old.tadah", notices);
  const std::string mark =
      "; M_KRR implementation 2 here: converted by deleting the line)";
  CHECK_THAT(what, Contains("EWEIGHT 1.0 (line 16 of old.train" + mark) &&
                       Contains("WDBFILE_AUTO 0.5 (line 17 of old.train" +
                                mark) &&
                       Contains("(K + LAMBDA I) w = t") &&
                       Contains("ESIGMA 1000, or LAMBDA times 10^6 with "
                                "ESIGMA 1,"));
  CHECK(what.find("ESIGMA = 1000/(N sqrt(EWEIGHT))") == std::string::npos);
  CHECK_FALSE(fs::exists("old.tadah"));

  write_config("esigma.train", {"LAMBDA 1e-3", "ESIGMA 1000"});
  write_config("lambda.train", {"LAMBDA 1e3"});
  write_config("deleted.train", {"LAMBDA 1e-3"});
  REQUIRE(train("esigma.train", "esigma.tadah", notices).empty());
  REQUIRE(train("lambda.train", "lambda.tadah", notices).empty());
  REQUIRE(train("deleted.train", "deleted.tadah", notices).empty());
  const std::vector<double> w_esigma = weights_of("esigma.tadah");
  CHECK(relative_difference(weights_of("lambda.tadah"), w_esigma) <= 1e-10);
  CHECK(relative_difference(weights_of("deleted.tadah"), w_esigma) > 1e-6);
}

/* Pins: the command-line options of the retired keys, --wdbfile-auto and
 * --eweight-temp of tadah train (TaskFactory of their key-value data, as
 * the command line gives it), are refused before any file is read, naming
 * the option, the key and its conversion.
 * Before the fix: the options were taken (WDBFILE_AUTO and EWEIGHT_TEMP
 * reweighted the training data). */
TEST_CASE("The options of the retired keys give their conversion",
          "[retired][cli]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  std::ofstream("config.train") << "ATOM Ta\n";
  const std::vector<std::pair<std::string, std::string>> options = {
      {"wdbfile_auto", "--wdbfile-auto"}, {"eweight_temp", "--eweight-temp"}};
  for (const auto &o : options) {
    INFO(o.second);
    std::string what;
    try {
      (void)tadah::mlip::TaskFactory().createTasksFromInput(CommandInput(
          "TRAIN", {{"config", {"config.train"}}, {o.first, {"0.5"}}}));
    } catch (const std::exception &e) {
      what = e.what();
    }
    CHECK_THAT(what, Contains(o.second + ": ") && Contains("retired") &&
                         Contains("WDBFILE"));
  }
}
