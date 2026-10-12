// TaskFactory's check of the values it is given (a command line's options,
// the keys of a task file): a comma or a '+' between two values, which
// neither CLI11 nor the configuration parser splits at, is refused naming
// the separator and the values as separate words.

#include "catch2/catch.hpp"

#include <tadah/core/logger.h>
#include <tadah/mlip/itask_input.h>
#include <tadah/mlip/task_factory.h>

#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using KV = std::unordered_map<std::string, std::vector<std::string>>;

/* What TaskFactory gets from the CLI: the subcommand's blueprint task name
 * and its options, keyed by the blueprint's lower-case key names (as
 * TadahCLI::addCmdOptions stores them). */
struct OptionsInput : tadah::mlip::ITaskInput {
  std::string task;
  KV kv;
  OptionsInput(std::string t, KV k) : task(std::move(t)), kv(std::move(k)) {}
  std::string getTaskName() const override { return task; }
  KV getKeyValueData() const override { return kv; }
};

/* TaskFactory stores the blueprint default VERBOSE 1 into each task's
 * context, which sets the global log level: put it back afterwards. */
struct LogLevelGuard {
  tadah::core::LogLevel saved = tadah::core::Logger::instance().getLogLevel();
  ~LogLevelGuard() { tadah::core::Logger::instance().setLogLevel(saved); }
};

/* The message createTasksFromInput throws, or "no exception". */
std::string refusal(const std::string &task, KV kv) {
  LogLevelGuard guard;
  try {
    (void)tadah::mlip::TaskFactory().createTasksFromInput(
        OptionsInput(task, std::move(kv)));
  } catch (const std::exception &e) {
    return e.what();
  }
  return "no exception";
}

}  // namespace

/* Pins: an option whose list values are one word joined by a comma or a
 * '+' (CLI11 passes '--percent 80,20' as the one word '80,20') is refused
 * by TaskFactory, naming the key, the value, the separator and the values
 * as separate words:
 *   "TaskFactory::createTasksFromInput - Invalid key-value pair: percent ->
 *    80,20 ('80,20' separates values with ',', which is not a value
 *    separator: give them as separate words, 80 20)"
 * likewise '--chunk 10+20' ('+'); an index list keeps its commas
 * ('--index 1,3,5' passes the check). A value that separate words would not
 * make valid either keeps the bare refusal ('--chunk 1.5').
 * Before the fix: '80,20' and '10+20' passed the blueprint check (its list
 * patterns took a comma or a '+' between two values) and the split was
 * refused later, when PERCENT was read as integers ('Conversion failed for
 * key=PERCENT, value="80,20" ...'): the check threw nothing ("no exception"
 * here, with no dataset to split). */
TEST_CASE("TaskFactory: a comma or '+' between option values is refused, "
          "naming the separator",
          "[task_factory][separators]") {
  const std::string head =
      "TaskFactory::createTasksFromInput - Invalid key-value pair: ";
  CHECK(refusal("DATA.split", {{"percent", {"80,20"}}}) ==
        head + "percent -> 80,20 ('80,20' separates values with ',', which "
               "is not a value separator: give them as separate words, 80 "
               "20)");
  CHECK(refusal("DATA.split", {{"chunk", {"10+20"}}}) ==
        head + "chunk -> 10+20 ('10+20' separates values with '+', which is "
               "not a value separator: give them as separate words, 10 20)");
  CHECK(refusal("DATA.split", {{"chunk", {"1.5"}}}) ==
        head + "chunk -> 1.5");
  const std::string index = refusal("DATA.print", {{"index", {"1,3,5"}}});
  INFO(index);
  CHECK(index.find("Invalid key-value pair") == std::string::npos);
}
