// TaskFactory and the task name of an ITaskInput without a task file: the
// name of a blueprint task (`DATA.split`, `TRAIN`), as the command line
// passes it. A program that builds its own ITaskInput can pass any string.

#include "catch2/catch.hpp"

#include <tadah/core/logger.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/task_factory.h>

#include <exception>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* A programmatic task input: a task name and its options. */
struct NamedInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  NamedInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* TaskFactory stores the blueprint default VERBOSE 1 into each task's
 * context, which sets the global log level: put it back afterwards. */
struct LogLevelGuard {
  tadah::core::LogLevel saved = tadah::core::Logger::instance().getLogLevel();
  ~LogLevelGuard() { tadah::core::Logger::instance().setLogLevel(saved); }
};

/* "invalid_argument: <message>" for a std::invalid_argument, "<other>:
 * <message>" for another exception, "tasks: <n>" for none. */
std::string outcome(const std::string &task, KV kv = {}) {
  LogLevelGuard guard;
  try {
    const auto tasks = tadah::mlip::TaskFactory().createTasksFromInput(
        NamedInput(task, std::move(kv)));
    return "tasks: " + std::to_string(tasks.size());
  } catch (const std::invalid_argument &e) {
    return std::string("invalid_argument: ") + e.what();
  } catch (const std::exception &e) {
    return std::string("other: ") + e.what();
  }
}

/* The task names every refusal of an unknown name lists: the blueprint
 * tasks without tasks of their own, sorted. */
const char *kTasks =
    "The tasks: ANALYSIS.bfunc, ANALYSIS.check_bounds, ANALYSIS.cutoff, "
    "ANALYSIS.dataset_stats, ANALYSIS.descriptor, ANALYSIS.suggest_bounds, "
    "DATA.balance, DATA.convert, DATA.dedup, DATA.merge, DATA.print, "
    "DATA.sample, DATA.split, DATA.write, EXPLAIN, HPO, PREDICT, "
    "PROPERTIES.pairwise, REFIT, TRAIN.";

}  // namespace

/* Pins: a task name that is not a blueprint task is refused with a
 * std::invalid_argument that names it and lists the task names (those
 * without tasks of their own), and, when the name matches one but for the
 * case of its letters, names that one:
 *   "TaskFactory::createTasksFromInput - unknown task 'DATA.SPLIT': task
 *    names are case-sensitive, did you mean 'DATA.split'? The tasks: ..."
 *   "TaskFactory::createTasksFromInput - unknown task 'bogus'. The tasks:
 *    ..."
 * and the empty name likewise; before any configuration file is read (a
 * CONFIG naming a missing file is not reached).
 * Before the fix: 'DATA.SPLIT', 'train', 'bogus' and '' each died with
 * SIGSEGV (the defaults loop read the blueprint entry of the name without
 * checking that there was one; the test case stops at the first), and
 * 'bogus' with that CONFIG was refused for the file instead
 * ("MLIPUserConfigParser::parse(file): Cannot open file: ..."). */
TEST_CASE("TaskFactory refuses an unknown task name, naming it and the tasks",
          "[task_factory][task_names]") {
  const std::string head =
      "invalid_argument: TaskFactory::createTasksFromInput - unknown task ";
  CHECK(outcome("DATA.SPLIT", {{"percent", {"80", "20"}}}) ==
        head + "'DATA.SPLIT': task names are case-sensitive, did you mean "
               "'DATA.split'? " + kTasks);
  CHECK(outcome("train") == head +
        "'train': task names are case-sensitive, did you mean 'TRAIN'? " +
        kTasks);
  CHECK(outcome("bogus") == head + "'bogus'. " + kTasks);
  CHECK(outcome("") == head + "''. " + kTasks);
  CHECK(outcome("bogus", {{"config", {"/nonexistent_dir/config.train"}}}) ==
        head + "'bogus'. " + kTasks);
}

/* Pins: a blueprint task name is accepted (DATA.split with its options: one
 * task) and the name of a group of tasks is refused as before, with the
 * runtime_error that names it ("... taskInput.getTaskName() is not a valid
 * task: DATA"). Before the fix: passes as well (a pin of the names the
 * check leaves alone). */
TEST_CASE("TaskFactory accepts a task name and refuses a group of tasks as "
          "before",
          "[task_factory][task_names]") {
  CHECK(outcome("DATA.split", {{"percent", {"80", "20"}}}) == "tasks: 1");
  CHECK(outcome("DATA") ==
        "other: TaskFactory::createTasksFromInput - taskInput.getTaskName() "
        "is not a valid task: DATA");
}
