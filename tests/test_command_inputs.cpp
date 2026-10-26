// test_command_inputs.cpp
//
// What a run of one command takes as input (TaskFactory and the engines,
// through the C++ API): a configuration file given with -c (and the files
// it includes with CONFIG) holds keys only -- a TASK line is an unknown key
// at its line --, a potential holds no TASK line either, an input has no
// key "task", and tadah predict, tadah properties pairwise and tadah
// analysis descriptor take no key that is not one of their options: their
// model is the potential's, as it was fitted.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/key_notice.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/engines/analysis_descriptor_engine.h>
#include <tadah/mlip/engines/data_sample_engine.h>
#include <tadah/mlip/engines/predict_engine.h>
#include <tadah/mlip/engines/properties_pairwise_engine.h>
#include <tadah/mlip/engines/train_engine.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
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
using tadah::mlip::Task;
using tadah::mlip::TaskFactory;
namespace fs = std::filesystem;

/* The working directory of a case, removed on exit. */
struct ScratchCwd {
  fs::path old_cwd, dir;
  ScratchCwd() {
    static int counter = 0;
    old_cwd = fs::current_path();
    dir = fs::temp_directory_path() /
          ("tadah_command_inputs_" + std::to_string(::getpid()) + "_" +
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

/* A 2-atom Ta cell. */
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

/* The lines of a Ta pair training configuration on db.tadah. */
const char *const kTrainLines =
    "ATOM Ta\nWATOM 73\nINIT2B true\nINITMB false\n"
    "TYPE2B D2_Blip 3 3 Ta Ta\nCGRID2B 2.8 3.5 4.2\nSGRID2B 2 2 2\n"
    "RCUT2B 5\nRCTYPE2B Cut_Cos\nLAMBDA 0\nOALGO 3\nBIAS false\n"
    "NORM false\nMODEL M_BLR BF_Linear\nDBFILE db.tadah\n";

/* Trains pot.tadah on db.tadah (both in the working directory). */
void train_potential() {
  write_db("db.tadah");
  std::ofstream("pair.train") << kTrainLines;
  Context c = tadah::core::CoreUserConfigParser().parse("pair.train");
  Task t("train", c);
  t.getContext().add("OUTFILE", std::string("pot.tadah"));
  REQUIRE(tadah::mlip::engines::TrainEngine().run(t) == EXIT_SUCCESS);
  REQUIRE(fs::exists("pot.tadah"));
}

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* What TaskFactory gets from the command line, or from a program. */
struct CommandInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  CommandInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* The message of what @p f throws, "" when it throws nothing. */
std::string error_of(const std::function<void()> &f) {
  try {
    f();
  } catch (const std::exception &e) {
    return e.what();
  }
  return "";
}

/* The tasks TaskFactory makes of @p input. */
std::vector<Task> tasks_of(const CommandInput &input) {
  return TaskFactory().createTasksFromInput(input);
}

}  // namespace

/* Pins: TaskFactory reads a configuration file given with -c (CONFIG) with
 * the configuration parser of CORE: a TASK line, in the file or in a file
 * it includes with CONFIG, is an unknown key, refused naming its line and
 * its file before any task is made ("Invalid key TASK on line 3 of
 * run.train (closest key: ...)"), and so is any line whose key begins with
 * TASK ("Invalid key TASKS on line 2 of tasks.train"); the file of the
 * other keys alone makes the one task.
 * Before the fix: TaskFactory read the file with MLIPUserConfigParser,
 * which took a line beginning with the four letters TASK for the start of a
 * TASK section: run.train made one TRAIN task of the lines above it
 * (LAMBDA 5 after the TASK line dropped, no exception), a TASK line of an
 * included file was stored as the key TASK (no exception), and "TASKS
 * predict" opened a TASK predict section, dropped the same way. */
TEST_CASE("TaskFactory: a TASK line in a configuration file, or in a file it "
          "includes, is refused naming its line and its file",
          "[task_factory][config][task_line]") {
  ScratchCwd cwd;
  std::ofstream("run.train") << "ATOM Ta\nWATOM 73\nTASK predict\n"
                                "LAMBDA 5\n";
  CHECK_THAT(error_of([] {
               (void)tasks_of(CommandInput("TRAIN",
                                           {{"config", {"run.train"}}}));
             }),
             StartsWith("Invalid key TASK on line 3 of run.train (closest "
                        "key: "));
  std::ofstream("inc.cfg") << "# included\nTASK train\n";
  std::ofstream("outer.train") << "ATOM Ta\nCONFIG inc.cfg\nLAMBDA 5\n";
  CHECK_THAT(error_of([] {
               (void)tasks_of(CommandInput("TRAIN",
                                           {{"config", {"outer.train"}}}));
             }),
             StartsWith("Invalid key TASK on line 2 of inc.cfg (closest "
                        "key: "));
  std::ofstream("tasks.train") << "ATOM Ta\nTASKS predict\n";
  CHECK_THAT(error_of([] {
               (void)tasks_of(CommandInput("TRAIN",
                                           {{"config", {"tasks.train"}}}));
             }),
             StartsWith("Invalid key TASKS on line 2 of tasks.train"));
  std::ofstream("keys.train") << "ATOM Ta\nWATOM 73\nLAMBDA 5\n";
  std::vector<Task> tasks;
  REQUIRE_NOTHROW(tasks = tasks_of(CommandInput(
                      "TRAIN", {{"config", {"keys.train"}}})));
  REQUIRE(tasks.size() == 1u);
  CHECK(tasks[0].getContext().get<double>("LAMBDA") == 5.0);
}

/* Pins: a potential with a TASK line is refused when tadah predict loads
 * it, naming the line and the file ("Invalid key TASK on line N of
 * old.tadah"), and nothing is predicted (no energy.pred); the potential
 * without the line predicts.
 * Before the fix: TASK was a configuration key (the file of a task file),
 * so the line loaded as a key of the potential and the prediction ran
 * (energy.pred written, no exception). Only a training configuration that
 * included a file with TASK lines wrote such a line into a potential. */
TEST_CASE("A potential with a TASK line is refused when it is loaded",
          "[predict][task_line]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  IgnoredKeyNoticeCapture quiet;
  train_potential();
  std::string pot;
  {
    std::ifstream in("pot.tadah");
    pot.assign(std::istreambuf_iterator<char>(in), {});
  }
  std::ofstream("old.tadah") << "TASK predict\n" << pot;
  const auto predict = [](const std::string &file) {
    for (Task &t : tasks_of(CommandInput("PREDICT",
                                         {{"potential", {file}},
                                          {"dbfile", {"db.tadah"}}}))) {
      tadah::mlip::engines::PredictEngine e;
      e.validate(t);
      e.run(t);
    }
  };
  CHECK_THAT(error_of([&] { predict("old.tadah"); }),
             StartsWith("Invalid key TASK on line 1 of old.tadah"));
  CHECK_FALSE(fs::exists("energy.pred"));
  CHECK(error_of([&] { predict("pot.tadah"); }).empty());
  CHECK(fs::exists("energy.pred"));
}

/* Pins: TaskFactory refuses an input with the key "task" (a program's
 * ITaskInput: the command line has no --task) as an unknown key,
 * "TaskFactory::createTasksFromInput - Invalid blueprint key: task", before
 * it reads the file it names; and a configuration file it cannot open is
 * named in the message of CORE's parser, "Failed to open config file:
 * missing.train".
 * Before the fix: the key named a task file, whose TASK train section
 * TaskFactory made into a TRAIN task (no exception), and a missing -c file
 * gave "MLIPUserConfigParser::parse(file): Cannot open file:
 * missing.train". */
TEST_CASE("TaskFactory: an input with the key task is refused as an unknown "
          "key, and a missing configuration file is named",
          "[task_factory][config][task_line]") {
  ScratchCwd cwd;
  std::ofstream("run.task") << "TASK train\nATOM Ta\n";
  CHECK(error_of([] {
          (void)tasks_of(CommandInput("", {{"task", {"run.task"}}}));
        }) == "TaskFactory::createTasksFromInput - Invalid blueprint key: "
              "task");
  CHECK(error_of([] {
          (void)tasks_of(
              CommandInput("TRAIN", {{"task", {"run.task"}}}));
        }) == "TaskFactory::createTasksFromInput - Invalid blueprint key: "
              "task");
  CHECK(error_of([] {
          (void)tasks_of(
              CommandInput("TRAIN", {{"config", {"missing.train"}}}));
        }) == "Failed to open config file: missing.train");
}

/* Pins: tadah predict, tadah properties pairwise and tadah analysis
 * descriptor run the potential's model as it was fitted: validate() and
 * run(), before run() reads or writes a file, refuse a run whose context
 * has a key that is not an option of the command (its blueprint keys and
 * those of its groups), naming each in upper case, sorted ("tadah predict:
 * RCUT2B, RHOFORM and WDBFILE are not options of tadah predict: the model
 * is the potential's, as it was fitted (its keys are those of the potential
 * file)"; "tadah properties pairwise: LAMBDA is not an option of ...";
 * "tadah analysis descriptor: EMBF0 is not ..."), and nothing is written;
 * the options alone run (energy.pred, pair.txt, desc.txt written). Only a
 * program reaches this: the command line gives the options alone.
 * Before the fix: the run merged its keys over the potential's (a TASK
 * predict section of a task file gave them: RCUT2B 4.0 changed the
 * prediction), validate() accepted them and run() wrote energy.pred,
 * pair.txt and desc.txt: every refusal check fails. */
TEST_CASE("tadah predict, properties pairwise and analysis descriptor "
          "refuse a key that is not one of their options",
          "[predict][pairwise][descriptor][command_options]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  IgnoredKeyNoticeCapture quiet;
  train_potential();
  const std::string tail =
      ": the model is the potential's, as it was fitted (its keys are those "
      "of the potential file)";
  struct Case {
    const char *name;
    std::string blueprint;
    KV options;
    KV extra;
    std::string message;
    const char *output;
    std::function<void(Task &, bool)> engine;
  };
  const std::vector<Case> cases = {
      {"predict", "PREDICT",
       {{"potential", {"pot.tadah"}}, {"dbfile", {"db.tadah"}}},
       {{"rcut2b", {"4.0"}}, {"wdbfile", {"2"}}, {"rhoform", {"SIGNED"}}},
       "tadah predict: RCUT2B, RHOFORM and WDBFILE are not options of tadah "
       "predict" + tail,
       "energy.pred",
       [](Task &t, bool run) {
         tadah::mlip::engines::PredictEngine e;
         if (run) (void)e.run(t); else e.validate(t);
       }},
      {"properties pairwise", "PROPERTIES.pairwise",
       {{"potential", {"pot.tadah"}},
        {"range", {"2.5", "4.5", "5"}},
        {"atompair", {"Ta", "Ta"}},
        {"outfile", {"pair.txt"}}},
       {{"lambda", {"5"}}},
       "tadah properties pairwise: LAMBDA is not an option of tadah "
       "properties pairwise" + tail,
       "pair.txt",
       [](Task &t, bool run) {
         tadah::mlip::engines::PropertiesPairwiseEngine e;
         if (run) (void)e.run(t); else e.validate(t);
       }},
      {"analysis descriptor", "ANALYSIS.descriptor",
       {{"potential", {"pot.tadah"}},
        {"dbfile", {"db.tadah"}},
        {"outfile", {"desc.txt"}}},
       {{"embf0", {"FREE"}}},
       "tadah analysis descriptor: EMBF0 is not an option of tadah analysis "
       "descriptor" + tail,
       "desc.txt",
       [](Task &t, bool run) {
         tadah::mlip::engines::AnalysisDescriptorEngine e;
         if (run) (void)e.run(t); else e.validate(t);
       }},
  };
  for (const Case &c : cases) {
    INFO(c.name);
    KV with = c.options;
    with.insert(c.extra.begin(), c.extra.end());
    std::vector<Task> tasks = tasks_of(CommandInput(c.blueprint, with));
    REQUIRE(tasks.size() == 1u);
    CHECK(error_of([&] { c.engine(tasks[0], false); }) == c.message);
    CHECK(error_of([&] { c.engine(tasks[0], true); }) == c.message);
    CHECK_FALSE(fs::exists(c.output));
    std::vector<Task> plain = tasks_of(CommandInput(c.blueprint, c.options));
    REQUIRE(plain.size() == 1u);
    CHECK(error_of([&] { c.engine(plain[0], false); }).empty());
    CHECK(error_of([&] { c.engine(plain[0], true); }).empty());
    CHECK(fs::exists(c.output));
  }
}

/* Pins: tadah data sample takes one selection: a program that gives two
 * (UNIFORM 2 and RANDOM 2, which the command line refuses as excluding each
 * other) is refused by validate(), naming both: "tadah data sample: give
 * one of UNIFORM, RANDOM and INDEX, not UNIFORM and RANDOM".
 * Before the fix: passes as well (a pin; a task file reached the check from
 * the command line). */
TEST_CASE("tadah data sample refuses two selections given by a program",
          "[data][sample][command_options]") {
  PeriodicTable::initialize();
  ScratchCwd cwd;
  write_db("db.tadah");
  std::vector<Task> tasks = tasks_of(CommandInput(
      "DATA.sample", {{"dbfile", {"db.tadah"}},
                      {"outfile", {"out.tadah"}},
                      {"uniform", {"2"}},
                      {"random", {"2"}}}));
  REQUIRE(tasks.size() == 1u);
  tadah::mlip::engines::DataSampleEngine e;
  CHECK(error_of([&] { e.validate(tasks[0]); }) ==
        "tadah data sample: give one of UNIFORM, RANDOM and INDEX, not "
        "UNIFORM and RANDOM");
  CHECK_FALSE(fs::exists("out.tadah"));
}
