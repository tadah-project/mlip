// test_user_config_sections.cpp
//
// MLIPUserConfigParser: a configuration file split into a GLOBAL section
// and TASK sections. An error that names a line names the line of the
// file, and the file, whichever section the line is in: the sections were
// handed to the key parser without their blank and comment lines and
// without their offset, so an invalid EMBF0 on line 6 of a task file was
// reported "on line 3" (its index among the TASK section's key lines).
// A TASK line followed by no key line is a section too: its name is
// checked and it takes the GLOBAL keys. A TASK line naming a group of tasks
// (TASK data) is refused.
// TaskFactory: a task file without a TASK section is an error.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/mlip_user_config_parser.h>
#include <tadah/mlip/task.h>
#include <tadah/mlip/task_factory.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Catch::Matchers::Contains;
using tadah::mlip::MLIPUserConfigParser;

/* A file in the working directory, removed on exit. */
struct TempFile {
  std::string path;
  TempFile(const std::string &stem, const std::string &text)
      : path(stem + "_" + std::to_string(::getpid()) + ".task") {
    std::ofstream os(path);
    os << text;
  }
  ~TempFile() { std::remove(path.c_str()); }
};

/* `tadah --task <file>`: the input TaskFactory gets from the CLI. */
struct TaskFileInput : tadah::mlip::ITaskInput {
  std::string file;
  explicit TaskFileInput(std::string f) : file(std::move(f)) {}
  std::string getTaskName() const override { return ""; }
  std::unordered_map<std::string, std::vector<std::string>>
  getKeyValueData() const override {
    return {{"TASK", {file}}};
  }
};

/* Silences the Logger's ERROR lines of a rejected value (std::cout). */
struct Quiet {
  std::ostringstream out;
  std::streambuf *old = std::cout.rdbuf(out.rdbuf());
  ~Quiet() { std::cout.rdbuf(old); }
};

} // namespace

TEST_CASE("Task file: an invalid value names the file's line in every section",
          "[config][task_file][allowed_values]") {
  Quiet q;
  SECTION("in a TASK section after blank and comment lines") {
    const TempFile f("sections_task",
                     "VERBOSE 1\n"               // 1
                     "\n"                        // 2
                     "TASK predict\n"            // 3
                     "POTENTIAL pot.tadah\n"     // 4
                     "# the reference\n"         // 5
                     "EMBF0 frei\n"              // 6
                     "OUTFILE energies.pred\n"); // 7
    REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(f.path),
                        Contains("Invalid value for key EMBF0 on line 6 of " +
                                 f.path + ": 'frei'"));
  }
  SECTION("in a TASK section that starts the file") {
    const TempFile f("sections_first",
                     "TASK predict\n"            // 1
                     "POTENTIAL pot.tadah\n"     // 2
                     "DBFILE train.db\n"         // 3
                     "EMBF0 frei\n");            // 4
    REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(f.path),
                        Contains("on line 4 of " + f.path + ": 'frei'"));
  }
  SECTION("in the GLOBAL section") {
    const TempFile f("sections_global",
                     "# global keys\n"           // 1
                     "\n"                        // 2
                     "EMBF0 zeroo\n"             // 3
                     "TASK predict\n"            // 4
                     "POTENTIAL pot.tadah\n");   // 5
    REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(f.path),
                        Contains("on line 3 of " + f.path + ": 'zeroo'"));
  }
  SECTION("in a second TASK section") {
    const TempFile f("sections_second",
                     "TASK predict\n"            // 1
                     "POTENTIAL a.tadah\n"       // 2
                     "\n"                        // 3
                     "TASK predict\n"            // 4
                     "# again\n"                 // 5
                     "POTENTIAL b.tadah\n"       // 6
                     "EMBF0 frei\n");            // 7
    REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(f.path),
                        Contains("on line 7 of " + f.path + ": 'frei'"));
  }
  SECTION("lines that come from no file name their line only") {
    const std::vector<std::string> lines = {"", "TASK predict",
                                            "POTENTIAL pot.tadah",
                                            "EMBF0 frei"};
    REQUIRE_THROWS_WITH(
        MLIPUserConfigParser().parse(lines),
        Contains("Invalid value for key EMBF0 on line 4: 'frei'"));
  }
}

TEST_CASE("Task file: sections keep their keys; blank and comment lines "
          "alone make no section", "[config][task_file]") {
  const std::vector<std::string> lines = {
      "# only comments before the first task", "", "TASK predict",
      "POTENTIAL a.tadah", "", "EMBF0 FREE", "# trailing comment"};
  const auto sections = MLIPUserConfigParser().parse(lines);
  REQUIRE(sections.size() == 1);
  CHECK(sections[0].first == "PREDICT");
  const tadah::core::Context &ctx = sections[0].second;
  CHECK(ctx.get<std::string>("POTENTIAL") == "a.tadah");
  REQUIRE(ctx.stored_value("EMBF0").has_value());
  CHECK(*ctx.stored_value("EMBF0") == "FREE");

  SECTION("GLOBAL keys are merged into each task") {
    const auto merged = MLIPUserConfigParser().parse(
        std::vector<std::string>{"VERBOSE 2", "# c", "TASK predict",
                                 "POTENTIAL a.tadah"});
    REQUIRE(merged.size() == 2);
    CHECK(merged[0].first == "GLOBAL");
    CHECK(merged[1].first == "PREDICT");
    CHECK(merged[1].second.get<int>("VERBOSE") == 2);
  }
}

/* Pins: a comment on a TASK line is not part of the task name, as on every
 * key line (CoreUserConfigParser drops the comment and the blanks before
 * it): "TASK       train          # First task: training" and
 * "TASK       predict\t# Second task: prediction" are the sections TRAIN
 * and PREDICT, each with its own keys and the GLOBAL keys merged in; a task
 * name of two words keeps both ("TASK data convert   # c" is DATA.convert);
 * "TASK predict# c", without a blank before the '#', is PREDICT; a TASK
 * line with only a comment after TASK names no task, an error that quotes
 * the line; and an invalid value after a commented TASK line names its line
 * of the input.
 * Before the fix: the words of the comment were joined to the task name, so
 * every parse stopped with an unknown task section and every assertion
 * failed: 'Unknown task section "TRAIN.#.first.task:.training"' in the first
 * section, "DATA.convert.#.c", "PREDICT#.c", "#.no.task.here" (instead of
 * the 'not followed by any name' error) and "TRAIN.#.first.task" (instead
 * of the EMBF0 error) in the others. */
TEST_CASE("Task file: a comment on a TASK line is not part of the task name",
          "[config][task_file]") {
  SECTION("after blanks and after a tab") {
    const std::vector<std::string> lines = {
        "VERBOSE    2",
        "TASK       train          # First task: training",
        "DBFILE     db.train",
        "TASK       predict\t# Second task: prediction",
        "POTENTIAL  pot.tadah      # Same as CLI -p/--potential"};
    std::vector<std::pair<std::string, tadah::core::Context>> sections;
    REQUIRE_NOTHROW(sections = MLIPUserConfigParser().parse(lines, true));
    REQUIRE(sections.size() == 3);
    CHECK(sections[0].first == "GLOBAL");
    CHECK(sections[1].first == "TRAIN");
    CHECK(sections[2].first == "PREDICT");
    CHECK(sections[1].second.get<std::string>("DBFILE") == "db.train");
    CHECK(sections[2].second.get<std::string>("POTENTIAL") == "pot.tadah");
    CHECK(sections[1].second.get<int>("VERBOSE") == 2);
    CHECK(sections[2].second.get<int>("VERBOSE") == 2);
  }
  SECTION("after a task name of two words") {
    const std::vector<std::string> lines = {"TASK data convert   # c",
                                            "VERBOSE 0"};
    std::vector<std::pair<std::string, tadah::core::Context>> sections;
    REQUIRE_NOTHROW(sections = MLIPUserConfigParser().parse(lines));
    REQUIRE(sections.size() == 1);
    CHECK(sections[0].first == "DATA.convert");
  }
  SECTION("without a blank before the '#'") {
    const std::vector<std::string> lines = {"TASK predict# c",
                                            "POTENTIAL pot.tadah"};
    std::vector<std::pair<std::string, tadah::core::Context>> sections;
    REQUIRE_NOTHROW(sections = MLIPUserConfigParser().parse(lines));
    REQUIRE(sections.size() == 1);
    CHECK(sections[0].first == "PREDICT");
  }
  SECTION("a comment alone after TASK names no task") {
    const std::vector<std::string> lines = {"TASK   # no task here",
                                            "POTENTIAL pot.tadah"};
    REQUIRE_THROWS_WITH(
        MLIPUserConfigParser().parse(lines),
        Contains("'TASK' keyword not followed by any name") &&
            Contains("TASK   # no task here"));
  }
  SECTION("an invalid value after a commented TASK line names its line") {
    Quiet q;
    const TempFile f("sections_task_comment",
                     "VERBOSE 1\n"                                   // 1
                     "TASK       train          # First task\n"      // 2
                     "DBFILE     db.train\n"                         // 3
                     "TASK       predict\t# Second task\n"           // 4
                     "POTENTIAL  pot.tadah\n"                        // 5
                     "EMBF0      frei           # the reference\n"); // 6
    REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(f.path),
                        Contains("Invalid value for key EMBF0 on line 6 of " +
                                 f.path + ": 'frei'"));
  }
}

/* Pins: a TASK line followed by no key line (the next line is another TASK
 * line, the end of the input, or only blank and comment lines) is a section
 * of its own, so its name is checked like any other and it takes the GLOBAL
 * keys:
 *  - "TASK bogus" followed by "TASK predict" is refused with 'Unknown task
 *    section "BOGUS"', and so is "TASK bogus" on the last line;
 *  - "TASK global" is refused as 'Unknown task section "GLOBAL"', at the
 *    start of the input with keys and on the last line without;
 *  - GLOBAL keys and a keyless "TASK predict" (at the end of the input, or
 *    followed by blank and comment lines) give the sections GLOBAL and
 *    PREDICT, the PREDICT context holding the GLOBAL POTENTIAL and DBFILE;
 *  - two keyless TASK lines in a row are two sections;
 *  - TaskFactory makes a keyless "TASK predict" of a task file a PREDICT
 *    task with the GLOBAL keys.
 * Before the fix: splitSections dropped a TASK section without a key line,
 * so the unknown names were never checked (no throw: the first REQUIRE_
 * THROWS_WITH failed, the second as well); a section named GLOBAL was not
 * checked either, so "TASK global" with keys at the start was taken for the
 * GLOBAL section (no throw); the keyless PREDICT sections
 * were missing (sizes 1 instead of 2, and 1 instead of 3), and TaskFactory
 * found no TASK section ("the task file '...' has no TASK section"). */
TEST_CASE("Task file: a TASK line without key lines is a section of its own",
          "[config][task_file]") {
  SECTION("an unknown task name before another task is refused") {
    const std::vector<std::string> lines = {"TASK bogus", "TASK predict",
                                            "POTENTIAL pot.tadah"};
    REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(lines),
                        Contains("Unknown task section \"BOGUS\""));
  }
  SECTION("an unknown task name on the last line is refused") {
    const std::vector<std::string> lines = {"TASK predict",
                                            "POTENTIAL pot.tadah",
                                            "TASK bogus"};
    REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(lines),
                        Contains("Unknown task section \"BOGUS\""));
  }
  SECTION("'TASK global' is an unknown task name, not the GLOBAL section") {
    for (const auto &lines : std::vector<std::vector<std::string>>{
             {"TASK global", "VERBOSE 0", "TASK predict",
              "POTENTIAL pot.tadah"},
             {"TASK predict", "POTENTIAL pot.tadah", "TASK global"}}) {
      INFO("first line: " << lines[0]);
      REQUIRE_THROWS_WITH(MLIPUserConfigParser().parse(lines),
                          Contains("Unknown task section \"GLOBAL\""));
    }
  }
  SECTION("a keyless task takes the GLOBAL keys") {
    for (const bool trailing_comment : {false, true}) {
      INFO("blank and comment lines after the TASK line: "
           << trailing_comment);
      std::vector<std::string> lines = {"POTENTIAL pot.tadah",
                                        "DBFILE db.tadah", "TASK predict"};
      if (trailing_comment) {
        lines.push_back("");
        lines.push_back("# nothing else");
      }
      std::vector<std::pair<std::string, tadah::core::Context>> sections;
      REQUIRE_NOTHROW(sections = MLIPUserConfigParser().parse(lines, true));
      REQUIRE(sections.size() == 2);
      CHECK(sections[0].first == "GLOBAL");
      CHECK(sections[1].first == "PREDICT");
      CHECK(sections[1].second.get<std::string>("POTENTIAL") == "pot.tadah");
      CHECK(sections[1].second.get<std::string>("DBFILE") == "db.tadah");
    }
  }
  SECTION("two keyless tasks in a row are two sections") {
    const std::vector<std::string> lines = {"POTENTIAL pot.tadah",
                                            "TASK predict", "TASK predict"};
    std::vector<std::pair<std::string, tadah::core::Context>> sections;
    REQUIRE_NOTHROW(sections = MLIPUserConfigParser().parse(lines, true));
    REQUIRE(sections.size() == 3);
    CHECK(sections[1].first == "PREDICT");
    CHECK(sections[2].first == "PREDICT");
    CHECK(sections[2].second.get<std::string>("POTENTIAL") == "pot.tadah");
  }
  SECTION("TaskFactory runs a keyless task on the GLOBAL keys") {
    const TempFile f("sections_keyless",
                     "POTENTIAL pot.tadah\nDBFILE db.tadah\nTASK predict\n");
    std::vector<tadah::mlip::Task> tasks;
    REQUIRE_NOTHROW(tasks = tadah::mlip::TaskFactory().createTasksFromInput(
                        TaskFileInput(f.path)));
    REQUIRE(tasks.size() == 1);
    CHECK(tasks[0].getTaskName() == "PREDICT");
    CHECK(tasks[0].getContext().get<std::string>("POTENTIAL") == "pot.tadah");
    CHECK(tasks[0].getContext().get<std::string>("DBFILE") == "db.tadah");
  }
}

/* Pins: a TASK section that names a group of tasks (a blueprint task with
 * tasks of its own: data, analysis, properties) is refused by the parser,
 * with or without key lines, naming the section and its tasks:
 *  - "TASK data" after "TASK train" (no key line), "TASK analysis" followed
 *    by a key line, and "TASK properties" alone each throw 'TASK section
 *    "<NAME>" is a group of tasks, not a task' and list the group's tasks
 *    ("TASK data <task>" with "convert" and "print" among them; "TASK
 *    analysis <task>" with "descriptor"; "TASK properties <task>" with
 *    "pairwise");
 *  - TaskFactory refuses a task file with "TASK train" (LAMBDA 0) followed
 *    by "TASK data" before it makes any task;
 *  - a task of the group ("TASK data print") is not refused.
 * Before the fix: the parser checked only that the name is a blueprint task,
 * which the group names are, so it returned the sections (no throw), and
 * TaskFactory made the tasks (no throw): `tadah --task` ran the training and
 * then stopped with 'EngineFactory::create: No engine found for DATA'. */
TEST_CASE("Task file: a TASK line naming a group of tasks is refused",
          "[config][task_file]") {
  const std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> cases = {
      {{"TASK train", "VERBOSE 0", "TASK data"},
       {"TASK section \"DATA\" is a group of tasks, not a task", "TASK data <task>",
        "convert", "print"}},
      {{"TASK analysis", "VERBOSE 0"},
       {"TASK section \"ANALYSIS\" is a group of tasks, not a task",
        "TASK analysis <task>", "descriptor"}},
      {{"TASK properties"},
       {"TASK section \"PROPERTIES\" is a group of tasks, not a task",
        "TASK properties <task>", "pairwise"}}};
  for (const auto &c : cases) {
    INFO("lines: " << c.first.front() << " ... " << c.first.back());
    try {
      MLIPUserConfigParser().parse(c.first);
      FAIL_CHECK("no exception");
    } catch (const std::runtime_error &e) {
      for (const auto &part : c.second) CHECK_THAT(e.what(), Contains(part));
    }
  }
  {
    const TempFile f("sections_group",
                     "VERBOSE 0\nTASK train\nLAMBDA 0\nTASK data\n");
    CHECK_THROWS_WITH(
        tadah::mlip::TaskFactory().createTasksFromInput(TaskFileInput(f.path)),
        Contains("TASK section \"DATA\" is a group of tasks"));
  }
  {
    std::vector<std::pair<std::string, tadah::core::Context>> sections;
    REQUIRE_NOTHROW(sections = MLIPUserConfigParser().parse(
                        std::vector<std::string>{"TASK data print", "DBFILE db.tadah"}));
    REQUIRE(sections.size() == 1);
    CHECK(sections[0].first == "DATA.print");
  }
}

/* An input of blank and comment lines has no section; merging them read
 * the first element of an empty list. */
TEST_CASE("Task file: an input of comments only has no sections",
          "[config][task_file]") {
  const std::vector<std::string> lines = {"# nothing", "", "  # else"};
  std::vector<std::pair<std::string, tadah::core::Context>> sections;
  REQUIRE_NOTHROW(sections = MLIPUserConfigParser().parse(lines, true));
  CHECK(sections.empty());
}

/* A task file runs its TASK sections. One without any -- only blank and
 * comment lines, or only GLOBAL keys -- read the first element of an empty
 * list (a segmentation fault, exit status 139, for the comment-only file)
 * or ran nothing and succeeded (the GLOBAL-only file).
 * Pins: the message says what a TASK section is: "the task file '<path>' has
 * no TASK section (a line 'TASK <name>', with or without key lines after
 * it)", for both files.
 * Before the fix (of the message): it said "(a line 'TASK <name>' followed
 * by the task's keys)", which a TASK line has not needed since a keyless
 * TASK line is a task. */
TEST_CASE("Task file: a task file without a TASK section is an error",
          "[config][task_file]") {
  const std::string what = "' has no TASK section (a line 'TASK <name>', with or "
                           "without key lines after it)";
  SECTION("blank and comment lines only") {
    const TempFile f("sections_empty", "# nothing to run\n\n");
    REQUIRE_THROWS_WITH(
        tadah::mlip::TaskFactory().createTasksFromInput(TaskFileInput(f.path)),
        Contains("the task file '" + f.path + what));
  }
  SECTION("GLOBAL keys only") {
    const TempFile f("sections_global_only", "VERBOSE 1\n");
    REQUIRE_THROWS_WITH(
        tadah::mlip::TaskFactory().createTasksFromInput(TaskFileInput(f.path)),
        Contains("the task file '" + f.path + what));
  }
  SECTION("a TASK section is a task") {
    const TempFile f("sections_one_task",
                     "VERBOSE 1\nTASK predict\nPOTENTIAL pot.tadah\n");
    std::vector<tadah::mlip::Task> tasks;
    REQUIRE_NOTHROW(tasks = tadah::mlip::TaskFactory().createTasksFromInput(
                        TaskFileInput(f.path)));
    REQUIRE(tasks.size() == 1);
    CHECK(tasks[0].getTaskName() == "PREDICT");
    CHECK(tasks[0].getContext().get<std::string>("POTENTIAL") == "pot.tadah");
  }
}
