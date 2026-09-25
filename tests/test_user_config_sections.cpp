// test_user_config_sections.cpp
//
// MLIPUserConfigParser: a configuration file split into a GLOBAL section
// and TASK sections. An error that names a line names the line of the
// file, and the file, whichever section the line is in: the sections were
// handed to the key parser without their blank and comment lines and
// without their offset, so an invalid EMBF0 on line 6 of a task file was
// reported "on line 3" (its index among the TASK section's key lines).
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
 * or ran nothing and succeeded (the GLOBAL-only file). */
TEST_CASE("Task file: a task file without a TASK section is an error",
          "[config][task_file]") {
  SECTION("blank and comment lines only") {
    const TempFile f("sections_empty", "# nothing to run\n\n");
    REQUIRE_THROWS_WITH(
        tadah::mlip::TaskFactory().createTasksFromInput(TaskFileInput(f.path)),
        Contains("the task file '" + f.path + "' has no TASK section"));
  }
  SECTION("GLOBAL keys only") {
    const TempFile f("sections_global_only", "VERBOSE 1\n");
    REQUIRE_THROWS_WITH(
        tadah::mlip::TaskFactory().createTasksFromInput(TaskFileInput(f.path)),
        Contains("the task file '" + f.path + "' has no TASK section"));
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
