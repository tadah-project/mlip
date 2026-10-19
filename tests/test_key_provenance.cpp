// test_key_provenance.cpp
//
// KeyProvenance / KeyOrigin (key_provenance.h) and what TaskFactory records
// on the Task (Task::getKeyProvenance()): where each key of a task's context
// came from -- a command-line option, the -c/--config file (or a file it
// includes with CONFIG), the --task file, or the blueprint default -- so that
// a run journal can say, e.g., "FORCE true (command line -F/--force;
// overrides FORCE false in c.tadah)".
//
// These pin behaviour that already holds. The trap they guard against is
// Context::contains(): it INSERTS the blueprint default of a missing key and
// then answers true (FORCE and STRESS both default to false). A provenance
// or a TaskFactory that asked contains() instead of has_stored() would call a
// key nobody set "command line" or "configuration file", and a TaskFactory
// that did so would never see a key as defaulted. Each TEST_CASE names the
// mutation that makes it fail.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/key_provenance.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>
#include <tadah/mlip/training_record.h>

#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::mlip::KeyOrigin;
using tadah::mlip::KeyProvenance;
using tadah::mlip::Task;
using tadah::mlip::TaskFactory;
using Kind = tadah::mlip::KeyOrigin::Kind;
using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* A file in the working directory, removed on exit. */
struct ProvFile {
  std::string path;
  ProvFile(const std::string &stem, const std::string &text)
      : path("kprov_" + stem + "_" + std::to_string(::getpid()) + ".tadah") {
    std::ofstream os(path);
    os << text;
  }
  ~ProvFile() { std::remove(path.c_str()); }
};

/* What TaskFactory gets from the CLI: the subcommand's blueprint task name
 * and its options, keyed by the blueprint's lower-case key names (as
 * TadahCLI::addCmdOptions stores them), e.g. {"force", {"true"}}. */
struct OptionsInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  OptionsInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* TaskFactory stores the blueprint default VERBOSE 1 into each task's
 * context, and Context::add("VERBOSE") sets the global log level: put the
 * level back for the tests that follow. */
struct LogLevelGuard {
  tadah::core::LogLevel saved = tadah::core::Logger::instance().getLogLevel();
  ~LogLevelGuard() { tadah::core::Logger::instance().setLogLevel(saved); }
};

std::vector<Task> make_tasks(const std::string &task, KV kv) {
  return TaskFactory().createTasksFromInput(
      OptionsInput(task, std::move(kv)));
}

const KeyProvenance &provenance_of(const Task &t) {
  const KeyProvenance *p = t.getKeyProvenance();
  REQUIRE(p != nullptr);
  return *p;
}

} // namespace

/* KeyOrigin::describe(): the words a run journal prints for each kind.
 * Pins existing behaviour; fails if the format of any kind changes, e.g. the
 * "; overrides KEY values in file" clause is dropped, the key is not upper
 * cased, several overridden values are not joined by single spaces, or
 * NotRecorded/Default stop saying "source not recorded"/"default". */
TEST_CASE("KeyProvenance: KeyOrigin::describe says where a key came from",
          "[key_provenance]") {
  KeyOrigin o;
  CHECK(o.kind == Kind::NotRecorded);
  CHECK(o.describe("FORCE") == "source not recorded");

  o.kind = Kind::Default;
  CHECK(o.describe("FORCE") == "default");

  o = KeyOrigin();
  o.kind = Kind::ConfigFile;
  o.file = "c.tadah";
  CHECK(o.describe("FORCE") == "configuration file c.tadah");

  o.kind = Kind::TaskFile;
  o.file = "run.task";
  CHECK(o.describe("FORCE") == "task file run.task");

  o = KeyOrigin();
  o.kind = Kind::CommandLine;
  CHECK(o.describe("FORCE") == "command line");
  o.flag = "-F/--force";
  CHECK(o.describe("FORCE") == "command line -F/--force");
  // The file alone (no overridden values) adds nothing.
  o.file = "c.tadah";
  CHECK(o.describe("FORCE") == "command line -F/--force");
  o.overridden = {"false"};
  CHECK(o.describe("force") ==
        "command line -F/--force; overrides FORCE false in c.tadah");

  KeyOrigin db;
  db.kind = Kind::CommandLine;
  db.flag = "-d/--dbfile";
  db.file = "c.tadah";
  db.overridden = {"a.db", "b.db"};
  CHECK(db.describe("DBFILE") ==
        "command line -d/--dbfile; overrides DBFILE a.db b.db in c.tadah");
}

/* KeyProvenance::origin() on contexts built by hand. Pins existing
 * behaviour. Fails under each of these mutations of key_provenance.cpp:
 *  - `command_line_.has_stored(uKey)` -> `command_line_.contains(uKey)`:
 *    STRESS (set nowhere, in the defaults) comes back CommandLine;
 *  - `file_.has_stored(uKey)` -> `file_.contains(uKey)` in the file branch:
 *    STRESS comes back ConfigFile;
 *  - the same in the command-line branch: -S/--stress, which the file does
 *    not set, comes back "overrides STRESS false in c.tadah";
 *  - the defaults not upper cased/sorted in the constructor: "stress"
 *    (given in lower case) comes back NotRecorded;
 *  - option_names() returning only one of the names: "-F/--force" or
 *    "--hpotarget" (a key without a short option) differs. */
TEST_CASE("KeyProvenance: origin() tells the command line, the file, the "
          "defaults and the unrecorded apart",
          "[key_provenance]") {
  // The trap is real for this blueprint: contains() answers true for a key
  // nobody stored, because it inserts the FORCE/STRESS default (false).
  {
    const Context probe;
    REQUIRE_FALSE(probe.has_stored("STRESS"));
    REQUIRE(probe.contains("STRESS"));
    REQUIRE(probe.has_stored("STRESS"));
  }

  Context command_line;
  command_line.add("FORCE", "true");
  command_line.add("DBFILE", "c1.db");
  command_line.add("HPOTARGET", "t.hpotarget");

  Context file;
  file.add("FORCE", "false");
  file.add("DBFILE", "a.db");
  file.add("DBFILE", "b.db");
  file.add("ALPHA", "0.25");

  const KeyProvenance p(command_line, file, "c.tadah", Kind::ConfigFile,
                        {"verbose", "stress", "VERBOSE"});

  SECTION("a command-line option that replaced the file's value") {
    const KeyOrigin o = p.origin("FORCE");
    CHECK(o.kind == Kind::CommandLine);
    CHECK(o.flag == "-F/--force");
    CHECK(o.file == "c.tadah");
    CHECK(o.overridden == std::vector<std::string>{"false"});
    CHECK(o.describe("FORCE") ==
          "command line -F/--force; overrides FORCE false in c.tadah");
    // Any letter case.
    CHECK(p.origin("force").describe("force") == o.describe("FORCE"));

    const KeyOrigin d = p.origin("dbfile");
    CHECK(d.kind == Kind::CommandLine);
    CHECK(d.flag == "-d/--dbfile");
    CHECK(d.overridden == std::vector<std::string>{"a.db", "b.db"});
    CHECK(d.describe("dbfile") ==
          "command line -d/--dbfile; overrides DBFILE a.db b.db in c.tadah");
  }
  SECTION("a command-line option the file does not set overrides nothing") {
    const KeyOrigin o = p.origin("HPOTARGET");
    CHECK(o.kind == Kind::CommandLine);
    CHECK(o.flag == "--hpotarget");
    CHECK(o.file.empty());
    CHECK(o.overridden.empty());
    CHECK(o.describe("HPOTARGET") == "command line --hpotarget");

    // -S/--stress given, file silent about STRESS (it would answer
    // contains("STRESS") with its default false).
    Context cl_stress;
    cl_stress.add("STRESS", "true");
    const KeyProvenance ps(cl_stress, file, "c.tadah", Kind::ConfigFile, {});
    const KeyOrigin s = ps.origin("STRESS");
    CHECK(s.kind == Kind::CommandLine);
    CHECK(s.flag == "-S/--stress");
    CHECK(s.file.empty());
    CHECK(s.overridden.empty());
    CHECK(s.describe("STRESS") == "command line -S/--stress");
  }
  SECTION("a key only the file sets") {
    const KeyOrigin o = p.origin("alpha");
    CHECK(o.kind == Kind::ConfigFile);
    CHECK(o.file == "c.tadah");
    CHECK(o.flag.empty());
    CHECK(o.overridden.empty());
    CHECK(o.describe("ALPHA") == "configuration file c.tadah");
  }
  SECTION("a key nobody set that the defaults list is Default, not the "
          "command line or the file") {
    for (const char *k : {"STRESS", "stress", "VERBOSE"}) {
      INFO("key " << k);
      const KeyOrigin o = p.origin(k);
      CHECK(o.kind == Kind::Default);
      CHECK(o.describe(k) == "default");
    }
    // Asking twice gives the same answer (no default was stored on the way).
    CHECK(p.origin("STRESS").kind == Kind::Default);
  }
  SECTION("a key nobody set and nobody defaulted is not recorded") {
    // SETFL has no blueprint default; BETA has one (1.0) that the provenance
    // was not told about: neither the command line nor the file stored it.
    for (const char *k : {"SETFL", "BETA"}) {
      INFO("key " << k);
      const KeyOrigin o = p.origin(k);
      CHECK(o.kind == Kind::NotRecorded);
      CHECK(o.flag.empty());
      CHECK(o.file.empty());
      CHECK(o.describe(k) == "source not recorded");
    }
  }
  SECTION("a task file is named as one") {
    const KeyProvenance t(Context(), file, "run.task", Kind::TaskFile,
                          {"STRESS"});
    CHECK(t.origin("FORCE").kind == Kind::TaskFile);
    CHECK(t.origin("FORCE").file == "run.task");
    CHECK(t.origin("FORCE").describe("FORCE") == "task file run.task");
    CHECK(t.origin("STRESS").kind == Kind::Default);
  }
  SECTION("copies answer the same; an empty provenance records nothing") {
    const KeyProvenance copy(p);
    CHECK(copy.origin("FORCE").describe("FORCE") ==
          p.origin("FORCE").describe("FORCE"));
    KeyProvenance tmp(copy);
    const KeyProvenance moved(std::move(tmp));
    CHECK(moved.origin("STRESS").kind == Kind::Default);
    CHECK(moved.origin("ALPHA").kind == Kind::ConfigFile);

    const KeyProvenance none;
    CHECK(none.origin("FORCE").kind == Kind::NotRecorded);
    CHECK(none.origin("FORCE").describe("FORCE") == "source not recorded");
  }
}

/* A Task built directly carries no provenance (Task::getKeyProvenance()
 * returns nullptr); TaskFactory sets one on every task it makes. Pins
 * existing behaviour; fails if Task default-constructs a provenance (a
 * journal would then claim "source not recorded" for keys TaskFactory never
 * saw instead of omitting the source), or if TaskFactory stops calling
 * setKeyProvenance. */
TEST_CASE("KeyProvenance: a Task built directly has no provenance",
          "[key_provenance][task_factory]") {
  Context c;
  c.add("FORCE", "true");
  const Task t("TRAIN", c);
  CHECK(t.getKeyProvenance() == nullptr);
  const Task copy = t;
  CHECK(copy.getKeyProvenance() == nullptr);
}

/* tadah predict -p pot.tadah -d a.db b.db -F: every key given is a
 * command-line key named by its options; keys the task defaults are Default;
 * keys set nowhere without a default are NotRecorded. Pins existing
 * behaviour. Fails under:
 *  - KeyProvenance::origin `command_line_.has_stored` -> `contains`: STRESS
 *    and VERBOSE come back "command line -S/--stress" / "command line
 *    -v/--verbose";
 *  - TaskFactory `!taskContext.has_stored(key)` -> `!taskContext.contains`
 *    when collecting `defaulted`: STRESS, VERBOSE and OUTFILE come back
 *    NotRecorded ("source not recorded");
 *  - TaskFactory handing KeyProvenance the task's context after the defaults
 *    were added instead of the options alone: STRESS comes back CommandLine. */
TEST_CASE("KeyProvenance: TaskFactory attributes command-line options to "
          "their option names",
          "[key_provenance][task_factory]") {
  const LogLevelGuard level;
  const auto tasks =
      make_tasks("PREDICT", {{"potential", {"kprov_pot.tadah"}},
                             {"dbfile", {"kprov_a.db", "kprov_b.db"}},
                             {"force", {"true"}}});
  REQUIRE(tasks.size() == 1);
  const Task &t = tasks[0];
  const KeyProvenance &p = provenance_of(t);
  CHECK(t.getContext().get<bool>("FORCE"));

  struct Expect { const char *key, *flag; };
  for (const Expect &e : {Expect{"FORCE", "-F/--force"},
                          Expect{"POTENTIAL", "-p/--potential"},
                          Expect{"DBFILE", "-d/--dbfile"}}) {
    INFO("key " << e.key);
    const KeyOrigin o = p.origin(e.key);
    CHECK(o.kind == Kind::CommandLine);
    CHECK(o.flag == e.flag);
    CHECK(o.file.empty());
    CHECK(o.overridden.empty());
    CHECK(o.describe(e.key) == std::string("command line ") + e.flag);
  }

  // Set by nobody; the predict blueprint defaults them (STRESS false,
  // VERBOSE 1, OUTFILE energy.pred forces.pred stress.pred).
  for (const char *k : {"STRESS", "VERBOSE", "OUTFILE"}) {
    INFO("key " << k);
    REQUIRE(t.getContext().has_stored(k));
    CHECK(p.origin(k).kind == Kind::Default);
    CHECK(p.origin(k).describe(k) == "default");
  }
  CHECK_FALSE(t.getContext().get<bool>("STRESS"));

  // Set by nobody, no default: SETFL.
  CHECK_FALSE(t.getContext().has_stored("SETFL"));
  CHECK(p.origin("SETFL").kind == Kind::NotRecorded);
  CHECK(p.origin("SETFL").describe("SETFL") == "source not recorded");
}

/* tadah train -c c.tadah, where c.tadah sets FORCE false and DBFILE and
 * includes (CONFIG) a file that sets ALPHA: the file's keys, the included
 * file's too, are the -c file's; keys nobody set are Default. Pins existing
 * behaviour. Fails under:
 *  - KeyProvenance::origin `file_.has_stored` -> `file_.contains` (file
 *    branch): STRESS and VERBOSE come back "configuration file <c>" (their
 *    ALLOWED_KEYS default is inserted);
 *  - `command_line_.has_stored` -> `contains`: STRESS comes back CommandLine;
 *  - TaskFactory recording the included file's path, or ConfigFile replaced
 *    by TaskFile, or the merged context given as the command line (then
 *    FORCE comes back CommandLine);
 *  - TaskFactory `has_stored` -> `contains` when collecting `defaulted`:
 *    STRESS/VERBOSE/OUTFILE come back NotRecorded. */
TEST_CASE("KeyProvenance: TaskFactory attributes a -c file's keys, and the "
          "keys of a file it includes, to the -c file",
          "[key_provenance][task_factory][config]") {
  const LogLevelGuard level;
  const ProvFile inc("inc", "ALPHA 0.25\n");
  const ProvFile cfg("cfg", "# provenance test\n"
                            "DBFILE kprov_train.db\n"
                            "FORCE false\n"
                            "CONFIG " + inc.path + "\n");

  const auto tasks = make_tasks("TRAIN", {{"config", {cfg.path}}});
  REQUIRE(tasks.size() == 1);
  const Task &t = tasks[0];
  const KeyProvenance &p = provenance_of(t);
  CHECK_FALSE(t.getContext().get<bool>("FORCE"));
  CHECK(t.getContext().get<double>("ALPHA") == 0.25);

  for (const char *k : {"DBFILE", "FORCE", "ALPHA"}) {
    INFO("key " << k);
    const KeyOrigin o = p.origin(k);
    CHECK(o.kind == Kind::ConfigFile);
    CHECK(o.file == cfg.path);
    CHECK(o.flag.empty());
    CHECK(o.overridden.empty());
    CHECK(o.describe(k) == "configuration file " + cfg.path);
  }

  for (const char *k : {"STRESS", "VERBOSE", "OUTFILE"}) {
    INFO("key " << k);
    REQUIRE(t.getContext().has_stored(k));
    CHECK(p.origin(k).kind == Kind::Default);
    CHECK(p.origin(k).describe(k) == "default");
  }
  CHECK(t.getContext().get<std::string>("OUTFILE") == "pot.tadah");

  // Set nowhere, no default.
  CHECK(p.origin("POTENTIAL").kind == Kind::NotRecorded);
  CHECK(p.origin("POTENTIAL").describe("POTENTIAL") == "source not recorded");
}

/* tadah hpo -c c.tadah --hpotarget t -F -S, where c.tadah sets FORCE false
 * and not STRESS: -F replaced the file's FORCE false (and says so), -S
 * replaced nothing, --hpotarget is named by its long option. Pins existing
 * behaviour. Fails under:
 *  - KeyProvenance::origin `file_.has_stored` -> `contains` in the
 *    command-line branch: STRESS comes back "command line -S/--stress;
 *    overrides STRESS false in <c>";
 *  - TaskFactory passing the merged context as the file (then the file
 *    "sets" FORCE true and STRESS true) or no file path (the FORCE line loses
 *    " in <c>");
 *  - the task's context not taking the command line's value: FORCE false. */
TEST_CASE("KeyProvenance: TaskFactory records a command-line option that "
          "overrides the -c file, and the value it replaced",
          "[key_provenance][task_factory][config]") {
  const LogLevelGuard level;
  const ProvFile cfg("override", "DBFILE kprov_train.db\n"
                                 "FORCE false\n");

  const auto tasks = make_tasks("HPO", {{"config", {cfg.path}},
                                        {"hpotarget", {"kprov.hpotarget"}},
                                        {"force", {"true"}},
                                        {"stress", {"true"}}});
  REQUIRE(tasks.size() == 1);
  const Task &t = tasks[0];
  const KeyProvenance &p = provenance_of(t);
  CHECK(t.getContext().get<bool>("FORCE"));
  CHECK(t.getContext().get<bool>("STRESS"));

  const KeyOrigin f = p.origin("FORCE");
  CHECK(f.kind == Kind::CommandLine);
  CHECK(f.flag == "-F/--force");
  CHECK(f.file == cfg.path);
  CHECK(f.overridden == std::vector<std::string>{"false"});
  CHECK(f.describe("FORCE") ==
        "command line -F/--force; overrides FORCE false in " + cfg.path);

  const KeyOrigin s = p.origin("STRESS");
  CHECK(s.kind == Kind::CommandLine);
  CHECK(s.flag == "-S/--stress");
  CHECK(s.file.empty());
  CHECK(s.overridden.empty());
  CHECK(s.describe("STRESS") == "command line -S/--stress");

  const KeyOrigin h = p.origin("HPOTARGET");
  CHECK(h.kind == Kind::CommandLine);
  CHECK(h.describe("HPOTARGET") == "command line --hpotarget");

  CHECK(p.origin("DBFILE").kind == Kind::ConfigFile);
  CHECK(p.origin("DBFILE").describe("DBFILE") ==
        "configuration file " + cfg.path);
  CHECK(p.origin("VERBOSE").kind == Kind::Default);
}

/* tadah --task run.task with a GLOBAL section and two TASK sections: every
 * key of each task, GLOBAL ones and included ones too, is the task file's;
 * each task has its own provenance. Pins existing behaviour. Fails under:
 *  - TaskFactory `KeyOrigin::Kind::TaskFile` -> `ConfigFile` in the --task
 *    branch ("configuration file run.task"), or the wrong path recorded;
 *  - one provenance shared by both tasks (POTENTIAL of the train task comes
 *    back TaskFile);
 *  - KeyProvenance::origin `file_.has_stored` -> `contains`: STRESS of the
 *    train task comes back "task file run.task";
 *  - TaskFactory `has_stored` -> `contains` when collecting `defaulted`:
 *    STRESS and VERBOSE come back NotRecorded. */
TEST_CASE("KeyProvenance: TaskFactory attributes a --task file's keys to the "
          "task file, per task",
          "[key_provenance][task_factory][task_file]") {
  const LogLevelGuard level;
  const ProvFile inc("task_inc", "ALPHA 0.5\n");
  const ProvFile run("run", "# GLOBAL section\n"
                            "FORCE true\n"
                            "\n"
                            "TASK train\n"
                            "DBFILE kprov_train.db\n"
                            "CONFIG " + inc.path + "\n"
                            "\n"
                            "TASK predict\n"
                            "POTENTIAL kprov_pot.tadah\n"
                            "DBFILE kprov_test.db\n"
                            "STRESS true\n");

  struct TaskFileInput : tadah::mlip::ITaskInput {
    std::string file;
    explicit TaskFileInput(std::string f) : file(std::move(f)) {}
    std::string getTaskName() const override { return ""; }
    KV getKeyValueData() const override { return {{"task", {file}}}; }
  };
  const auto tasks =
      TaskFactory().createTasksFromInput(TaskFileInput(run.path));
  REQUIRE(tasks.size() == 2);
  REQUIRE(tasks[0].getTaskName() == "TRAIN");
  REQUIRE(tasks[1].getTaskName() == "PREDICT");

  SECTION("the train task") {
    const KeyProvenance &p = provenance_of(tasks[0]);
    for (const char *k : {"FORCE", "DBFILE", "ALPHA"}) {
      INFO("key " << k);
      const KeyOrigin o = p.origin(k);
      CHECK(o.kind == Kind::TaskFile);
      CHECK(o.file == run.path);
      CHECK(o.describe(k) == "task file " + run.path);
    }
    for (const char *k : {"STRESS", "VERBOSE", "OUTFILE"}) {
      INFO("key " << k);
      CHECK(p.origin(k).kind == Kind::Default);
    }
    CHECK(p.origin("POTENTIAL").kind == Kind::NotRecorded);
  }
  SECTION("the predict task") {
    const KeyProvenance &p = provenance_of(tasks[1]);
    for (const char *k : {"FORCE", "POTENTIAL", "DBFILE", "STRESS"}) {
      INFO("key " << k);
      const KeyOrigin o = p.origin(k);
      CHECK(o.kind == Kind::TaskFile);
      CHECK(o.file == run.path);
      CHECK(o.describe(k) == "task file " + run.path);
    }
    CHECK(p.origin("VERBOSE").kind == Kind::Default);
    CHECK(p.origin("ALPHA").kind == Kind::NotRecorded);
    CHECK(tasks[1].getContext().get<std::string>("DBFILE") ==
          "kprov_test.db");
  }
}

/* "source:line KEY v1 v2" for each line TaskFactory did not read. */
std::vector<std::string> skipped_of(const KeyProvenance &p) {
  std::vector<std::string> out;
  for (const auto &l : p.skipped_key_lines()) {
    std::string s = l.source + ":" + std::to_string(l.line) + " " + l.key;
    for (const std::string &v : l.values) s += " " + v;
    out.push_back(s);
  }
  return out;
}

/* Pins: the rule of the parse of a run's configuration,
 * potential_line_not_read(file_keys, key): a key that prepared the
 * training data, and UNCERTAINTY, in a file whose key lines give WEIGHTS
 * (a potential), is not read -- ESHIFT_ATOM and ESHIFT_DBATOM whether or
 * not the file gives ESHIFT; no other key (ESHIFT, LSCALE, DBFILE,
 * WEIGHTS), and no key of a file without WEIGHTS. TaskFactory parses every
 * -c and --task file under it,
 * and the task's KeyProvenance lists the lines it did not read:
 *   - `tadah --task run.task`, its lines above the first TASK line
 *     "WDBFILE 2", its TASK train section "CONFIG pot", "EFILTER -1 1",
 *     "DBFILE kprov_a.db", pot being "ATOM Ta", "WEIGHTS 0.25 -0.5",
 *     "ESHIFT 0.001", "EFILTER 5 6", "ESHIFT_DBATOM true", "WDBFILE 1 1",
 *     "ZERO_COM_FORCE false": the train task's EFILTER is -1 1 and its
 *     WDBFILE 2 (task file), ESHIFT 0.001 and WEIGHTS are the potential's,
 *     ESHIFT_DBATOM and ZERO_COM_FORCE false the blueprint defaults
 *     (Default); skipped_key_lines() is pot:4 EFILTER 5 6, pot:5
 *     ESHIFT_DBATOM true, pot:6 WDBFILE 1 1, pot:7 ZERO_COM_FORCE false;
 *     file_path() run.task, file_kind() TaskFile;
 *   - `tadah train -c pot`: no EFILTER and no WDBFILE in the context, the
 *     same four lines skipped, file_path() pot, file_kind() ConfigFile.
 * Before the fix: does not compile (no potential_line_not_read, no
 * KeyProvenance::skipped_key_lines, file_path or file_kind).
 * Before the fix that reads no ESHIFT_ATOM, ESHIFT_DBATOM or UNCERTAINTY
 * line of a potential: a toggle was read from a potential without ESHIFT
 * (CHECK_FALSE pinned it) and UNCERTAINTY from any file: the toggle checks
 * on pot_keys and the UNCERTAINTY check fail. */
TEST_CASE("KeyProvenance: TaskFactory does not read a potential's lines of "
          "the keys that prepared the training data, and lists them",
          "[key_provenance][task_factory][dataset_keys]") {
  using tadah::mlip::potential_line_not_read;
  const std::set<std::string> pot_keys = {"ATOM", "WEIGHTS"};
  const std::set<std::string> pot_eshift = {"WEIGHTS", "ESHIFT"};
  for (const char *k : {"EFILTER", "FFILTER", "EWEIGHT_TEMP", "ZERO_COM_FORCE",
                        "WDBFILE", "WDBFILE_AUTO"}) {
    INFO("key " << k);
    CHECK(potential_line_not_read(pot_keys, k));
    CHECK_FALSE(potential_line_not_read({"ATOM", "DBFILE"}, k));
  }
  for (const char *k : {"ESHIFT_ATOM", "ESHIFT_DBATOM", "UNCERTAINTY"}) {
    INFO("key " << k);
    CHECK(potential_line_not_read(pot_keys, k));
    CHECK(potential_line_not_read(pot_eshift, k));
    CHECK_FALSE(potential_line_not_read({"ATOM", "DBFILE"}, k));
  }
  for (const char *k : {"ESHIFT", "LSCALE", "DBFILE", "WEIGHTS"}) {
    INFO("key " << k);
    CHECK_FALSE(potential_line_not_read(pot_eshift, k));
  }

  const LogLevelGuard level;
  const ProvFile pot("dk_pot", "ATOM Ta\n"               // 1
                               "WEIGHTS 0.25 -0.5\n"     // 2
                               "ESHIFT 0.001\n"          // 3
                               "EFILTER 5 6\n"           // 4
                               "ESHIFT_DBATOM true\n"    // 5
                               "WDBFILE 1 1\n"           // 6
                               "ZERO_COM_FORCE false\n");  // 7
  const std::vector<std::string> pot_lines = {
      pot.path + ":4 EFILTER 5 6", pot.path + ":5 ESHIFT_DBATOM true",
      pot.path + ":6 WDBFILE 1 1", pot.path + ":7 ZERO_COM_FORCE false"};
  SECTION("a task file that includes the potential") {
    const ProvFile run("dk_run", "WDBFILE 2\n"
                                 "TASK train\n"
                                 "CONFIG " + pot.path + "\n"
                                 "EFILTER -1 1\n"
                                 "DBFILE kprov_a.db\n");
    const auto tasks = make_tasks("", {{"task", {run.path}}});
    REQUIRE(tasks.size() == 1);
    const Context &c = tasks[0].getContext();
    CHECK(c("EFILTER") == std::vector<std::string>{"-1", "1"});
    CHECK(c("WDBFILE") == std::vector<std::string>{"2"});
    CHECK(c("ESHIFT") == std::vector<std::string>{"0.001"});
    CHECK(c("WEIGHTS") == std::vector<std::string>{"0.25", "-0.5"});
    const KeyProvenance &p = provenance_of(tasks[0]);
    CHECK(p.origin("EFILTER").kind == Kind::TaskFile);
    CHECK(p.origin("WDBFILE").kind == Kind::TaskFile);
    CHECK(p.origin("ESHIFT_DBATOM").kind == Kind::Default);
    CHECK(p.origin("ZERO_COM_FORCE").kind == Kind::Default);
    CHECK_FALSE(c.get<bool>("ESHIFT_DBATOM"));
    CHECK(skipped_of(p) == pot_lines);
    CHECK(p.file_path() == run.path);
    CHECK(p.file_kind() == Kind::TaskFile);
  }
  SECTION("the potential as the configuration file") {
    const auto tasks = make_tasks("TRAIN", {{"config", {pot.path}}});
    REQUIRE(tasks.size() == 1);
    const Context &c = tasks[0].getContext();
    CHECK_FALSE(c.has_stored("EFILTER"));
    CHECK_FALSE(c.has_stored("WDBFILE"));
    const KeyProvenance &p = provenance_of(tasks[0]);
    CHECK(skipped_of(p) == pot_lines);
    CHECK(p.file_path() == pot.path);
    CHECK(p.file_kind() == Kind::ConfigFile);
  }
}

/* Pins: the one list of the keys whose lines a run does not read from a
 * potential, keys_not_read_from_a_potential(), is the training record's,
 * training_record_keys(), without DBFILE (which a retrain adds to a copy of
 * the potential), in its order: the keys that prepared the training data,
 * VALIDATION, HPOTARGET, CHECK, UNCERTAINTY, VERBOSE, NUMERIC, OUTFILE,
 * AUDIT, AUDIT_SAMPLE and HEALTH_LOG; potential_line_not_read selects each
 * of the nine run settings in a file whose key lines give WEIGHTS and none
 * in a file without WEIGHTS, DBFILE in neither; and TaskFactory's parse of
 * `tadah train -c pot` (pot: ATOM, WEIGHTS and the eight lines of the run
 * that wrote it, VERBOSE 2 among them) stores none of the eight but the
 * blueprint defaults it adds itself (VERBOSE 1 and OUTFILE pot.tadah,
 * origin Default), lists the eight lines in skipped_key_lines() (lines 3 to
 * 10) and leaves the log level at WARNING, where tadah starts a run (the
 * default it adds sets none); a configuration file that includes pot with
 * CONFIG keeps its own VERBOSE 0 and OUTFILE mine.tadah (origin
 * ConfigFile).
 * Before the fix: the list had the dataset keys and UNCERTAINTY only, the
 * rule selected none of the eight, and the parse read them as the run's
 * (VERBOSE 2 set the level to INFO, OUTFILE old.tadah was the task's, no
 * line skipped; the included potential's VERBOSE 2 replaced the outer
 * file's VERBOSE 0 before it): the list, rule, has_stored, skipped and
 * level checks fail.
 * Before CHECK was a run setting: the lists had no CHECK and the rule did
 * not select it: the list and rule checks fail. */
TEST_CASE("KeyProvenance: TaskFactory does not read a potential's run "
          "settings, and lists them",
          "[key_provenance][task_factory][training_record][run_settings]") {
  using tadah::mlip::keys_not_read_from_a_potential;
  using tadah::mlip::potential_line_not_read;
  using tadah::mlip::training_record_keys;
  const std::vector<std::string> expected = {
      "EFILTER",     "FFILTER",      "ESHIFT_ATOM", "ESHIFT_DBATOM",
      "EWEIGHT_TEMP", "ZERO_COM_FORCE", "WDBFILE",   "WDBFILE_AUTO",
      "VALIDATION",  "HPOTARGET",    "CHECK",       "UNCERTAINTY",
      "VERBOSE",     "NUMERIC",      "OUTFILE",     "AUDIT",
      "AUDIT_SAMPLE", "HEALTH_LOG"};
  CHECK(keys_not_read_from_a_potential() == expected);
  std::vector<std::string> record = training_record_keys();
  REQUIRE_FALSE(record.empty());
  CHECK(record.front() == "DBFILE");
  record.erase(record.begin());
  CHECK(keys_not_read_from_a_potential() == record);

  const std::set<std::string> pot_keys = {"ATOM", "WEIGHTS"};
  const char *const settings[] = {"VALIDATION", "HPOTARGET",    "CHECK",
                                  "VERBOSE",    "NUMERIC",      "OUTFILE",
                                  "AUDIT",      "AUDIT_SAMPLE", "HEALTH_LOG"};
  for (const char *k : settings) {
    INFO("key " << k);
    CHECK(potential_line_not_read(pot_keys, k));
    CHECK_FALSE(potential_line_not_read({"ATOM", "DBFILE"}, k));
  }
  CHECK_FALSE(potential_line_not_read(pot_keys, "DBFILE"));

  const LogLevelGuard level;
  const ProvFile pot("rs_pot", "ATOM Ta\n"                // 1
                               "WEIGHTS 0.25 -0.5\n"      // 2
                               "VALIDATION config.val\n"  // 3
                               "HPOTARGET config.hpo\n"   // 4
                               "VERBOSE 2\n"              // 5
                               "NUMERIC 12\n"             // 6
                               "OUTFILE old.tadah\n"      // 7
                               "AUDIT warn\n"             // 8
                               "AUDIT_SAMPLE 2\n"         // 9
                               "HEALTH_LOG full\n");      // 10
  SECTION("the potential as the configuration file") {
    tadah::core::Logger::instance().setLogLevel(
        tadah::core::LogLevel::WARNING);  // where tadah starts a run
    const auto tasks = make_tasks("TRAIN", {{"config", {pot.path}}});
    REQUIRE(tasks.size() == 1);
    CHECK(tadah::core::Logger::instance().getLogLevel() ==
          tadah::core::LogLevel::WARNING);
    const Context &c = tasks[0].getContext();
    for (const char *k : {"VALIDATION", "HPOTARGET", "NUMERIC", "AUDIT",
                          "AUDIT_SAMPLE", "HEALTH_LOG"}) {
      INFO("key " << k);
      CHECK_FALSE(c.has_stored(k));
    }
    CHECK(c("VERBOSE") == std::vector<std::string>{"1"});
    CHECK(c("OUTFILE") == std::vector<std::string>{"pot.tadah"});
    const KeyProvenance &p = provenance_of(tasks[0]);
    CHECK(p.origin("VERBOSE").kind == Kind::Default);
    CHECK(p.origin("OUTFILE").kind == Kind::Default);
    CHECK(skipped_of(p) ==
          std::vector<std::string>{
              pot.path + ":3 VALIDATION config.val",
              pot.path + ":4 HPOTARGET config.hpo", pot.path + ":5 VERBOSE 2",
              pot.path + ":6 NUMERIC 12", pot.path + ":7 OUTFILE old.tadah",
              pot.path + ":8 AUDIT warn", pot.path + ":9 AUDIT_SAMPLE 2",
              pot.path + ":10 HEALTH_LOG full"});
  }
  SECTION("a configuration file that includes the potential") {
    const ProvFile outer("rs_outer", "VERBOSE 0\n"
                                     "CONFIG " + pot.path + "\n"
                                     "OUTFILE mine.tadah\n");
    const auto tasks = make_tasks("TRAIN", {{"config", {outer.path}}});
    REQUIRE(tasks.size() == 1);
    const Context &c = tasks[0].getContext();
    CHECK(c("VERBOSE") == std::vector<std::string>{"0"});
    CHECK(c("OUTFILE") == std::vector<std::string>{"mine.tadah"});
    CHECK_FALSE(c.has_stored("NUMERIC"));
    const KeyProvenance &p = provenance_of(tasks[0]);
    CHECK(p.origin("VERBOSE").kind == Kind::ConfigFile);
    CHECK(p.origin("OUTFILE").kind == Kind::ConfigFile);
    CHECK(skipped_of(p).size() == 8u);
  }
}
