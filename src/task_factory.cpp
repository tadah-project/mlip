#include <tadah/mlip/task_factory.h>
#include <tadah/mlip/retired_options.h>
#include <tadah/mlip/training_record.h>  // potential_line_not_read
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/blueprint_validator.h>
#include <tadah/core/retired_keys.h>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>

namespace tadah {
namespace mlip {

namespace {

/* The blueprint entry of the task @p name. A name that is not a blueprint
 * task (an ITaskInput a program builds can pass any string; the command
 * line passes only blueprint names) is a
 * std::invalid_argument that names it, the task it matches but for the
 * case of its letters, if one, and the task names: those without tasks of
 * their own, sorted. */
const core::BlueprintTask &blueprint_task(const std::string &name) {
  const auto &tasks = core::BlueprintParser::getTasksMapRef();
  const auto found = tasks.find(name);
  if (found != tasks.end()) return found->second;

  std::vector<std::string> names;
  for (const auto &kv : tasks) {
    const std::string prefix = kv.first + ".";
    const bool group = std::any_of(tasks.begin(), tasks.end(),
        [&](const auto &other) {
          return other.first.compare(0, prefix.size(), prefix) == 0;
        });
    if (!group) names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());

  std::string msg = "unknown task '" + name + "'";
  const auto same = std::find_if(names.begin(), names.end(),
      [&](const std::string &n) { return boost::iequals(n, name); });
  if (same != names.end())
    msg += ": task names are matched with case, and the task is spelt '" +
           *same + "'.";
  else
    msg += ".";
  msg += " The tasks:";
  for (std::size_t i = 0; i < names.size(); ++i)
    msg += (i == 0 ? " " : ", ") + names[i];
  throw std::invalid_argument(msg + ".");
}

}  // namespace

std::vector<Task> TaskFactory::createTasksFromInput(const ITaskInput &taskInput)
{
  // The base context: the key-value data of the input (the command-line
  // options). A configuration file it names (CONFIG, -c) is parsed, the
  // files it includes with it, and the base context merged over it: the
  // command line overrides the file. The result is the task of one command.

  // prepare the result vector
  std::vector<Task> tasksResult;

  // Step 0: Validate the taskInput key-value pairs against the blueprint.
  // When taskInput comes from CLI, it is already validated by the CLI11 parser.
  // But we validate anyway so class is generic.
  const auto &bpMap = tadah::core::BlueprintParser::getKeysMapRef();
  for (const auto &kvPair : taskInput.getKeyValueData()) {
    const std::string &longKey = kvPair.first;
    std::vector<std::string> keyTokens;
    boost::split(keyTokens, longKey, boost::is_any_of("."));
    std::string key = keyTokens.back(); // last token is the key

    const std::vector<std::string> &values = kvPair.second;
    if (values.empty()) {
      continue; // skip empty values
    }
    bool valid_schema = false;
    const core::BlueprintKey *keyInfo = nullptr;
    try {
      keyInfo = &bpMap.at(core::to_upper(key));
      valid_schema = core::CoreUserConfigParser::validateWithBlueprint(key, values, *keyInfo);
    } catch (const std::out_of_range &e) {
      // The parser's form of an unknown key, with the closest key.
      throw std::runtime_error(
          "Invalid key " + key + " (closest key: " +
          core::BlueprintParser::findClosestKey(key, bpMap) + ")");
    }
    // An option the task retires (tadah data balance --rescale), whatever
    // the key of that name is to the other commands: why, what replaces it
    // and its conversion.
    if (const core::BlueprintKey *option =
            retired_task_option(taskInput.getTaskName(), key))
      throw std::runtime_error(
          retired_task_option_text(taskInput.getTaskName(), *option));
    // A retired key (its command-line option is kept to say so): what
    // replaces it and its conversion, before its value is used.
    if (keyInfo->isRetired()) {
      const std::string name = keyInfo->cli_long_flag.empty()
                                   ? core::to_upper(key)
                                   : keyInfo->cli_long_flag;
      throw std::runtime_error(
          name + ": " + core::retired_key_text(core::to_upper(key), *keyInfo));
    }
    if (!valid_schema) {
      // Values joined by a comma or a '+' (one command-line word), which no
      // reader splits at: name the separator and the values as words.
      const std::string text = core::concat(values);
      std::string respaced;
      const std::string seps = core::CoreUserConfigParser::misusedSeparators(
          text, *keyInfo, respaced);
      // The option as it is given ("--percent"), else the key.
      const std::string name = keyInfo->cli_long_flag.empty()
                                   ? core::to_upper(key)
                                   : keyInfo->cli_long_flag;
      if (!seps.empty())
        throw std::runtime_error(
            name + " " + text + ": '" + text + "' separates values with " +
            seps + ", which is not a value separator; the values are "
            "separate words: " + respaced);
      std::string forms;
      for (std::size_t i = 0; i < keyInfo->value_format.size(); ++i)
        forms += (i == 0 ? "" : " or ") + keyInfo->value_format[i];
      throw std::runtime_error(name + " " + text + ": " + name + " takes " +
                               forms);
    }
  }

  // Step 1: Build a "base" context from the key-value data in the input.
  tadah::core::Context baseContext;
  auto kvmap = taskInput.getKeyValueData();
  for (auto &kv : kvmap) {
    std::string upKey = tadah::core::to_upper(kv.first);
    for (auto &val : kv.second) {
      baseContext.add<std::string>(upKey, val);
    }
  }

  // Step 2: the input names a blueprint task, or is refused here, before
  // any configuration file is read; a group of tasks (DATA) is no task:
  // only its tasks run (the command line refuses a group alone itself).
  (void)blueprint_task(taskInput.getTaskName());
  for (const auto &kv : tadah::core::BlueprintParser::getTasksMapRef()) {
    if (kv.first.find(taskInput.getTaskName() + ".") == 0) {
      throw std::runtime_error(taskInput.getTaskName() +
                               " is a group of tasks, not a task");
    }
  }

  // Step 3: the configuration file given with CONFIG (-c), read by CORE's
  // parser, the files it includes with CONFIG too. A potential's lines of
  // the keys of its training record but DBFILE -- those that prepared the
  // training data of its own fit and the settings of the run that wrote it
  // (VALIDATION, HPOTARGET, CHECK, VERBOSE, NUMERIC, OUTFILE, AUDIT,
  // AUDIT_SAMPLE, HEALTH_LOG) -- are not read, wherever the file and its
  // CONFIG includes put it: every other line, and the command line, is the
  // run's. A key that only other commands read (an option of theirs such
  // as ANALYTICS of tadah predict, or a key of their configuration file
  // such as REFIT_MODE of tadah refit) is refused at its line, as a file of
  // this command (CoreUserConfigParser::set_command), and not read from a
  // potential, with a notice.
  core::Context fileContext;
  std::string configFile;
  std::vector<core::SkippedKeyLine> skipped;
  if (baseContext.has_stored("CONFIG")) {
    configFile = baseContext.get<std::string>("CONFIG");
    core::CoreUserConfigParser parser;
    parser.set_command(taskInput.getTaskName());
    fileContext = parser.parse(configFile, potential_line_not_read, skipped);
    baseContext.remove("CONFIG");
  }

  // Where the task's keys come from (KeyProvenance): the command-line
  // options, the configuration file and (below) the defaults.
  tasksResult.emplace_back(Task(
      taskInput.getTaskName(),
      configFile.empty() ? baseContext
                         : core::Context::merge(fileContext, baseContext)));

  // Add default keys to the context if not provided by the user
  for (std::size_t t = 0; t < tasksResult.size(); ++t) {
    auto &task = tasksResult[t];
    const std::string &taskName = task.getTaskName();
    core::Context &taskContext = task.getContext();
    const core::BlueprintTask &bp = blueprint_task(taskName);
    std::vector<std::string> defaulted;  // the keys added here

    // dealt with options keys
    for (const auto &bpKey : bp.blueprintKeys) {
      if (bpKey.default_value.empty()) {
        continue; // skip if no default value
      }
      std::vector<std::string> tokens;
      boost::split(tokens, bpKey.keyName, boost::is_any_of("."));
      // contains() inserts the blueprint default itself: a key not stored
      // before it ends up holding the default either way.
      if (!taskContext.has_stored(tokens.back()))
        defaulted.push_back(core::to_upper(tokens.back()));
      if (!taskContext.contains(tokens.back())) {
        for (size_t i = 0; i < bpKey.default_value.size(); ++i) {
          size_t const ni = (i < bpKey.value_N.size()) ? i : bpKey.value_N.size() - 1;
          taskContext.add<std::string>(tokens.back(), bpKey.default_value[i], std::abs(bpKey.value_N[ni]));
        }
      }
    }

    // also deal with groups
    for (const auto &group : bp.groups) {
      for (const auto &bpKey : group.blueprintKeys) {
        if (bpKey.default_value.empty()) {
          continue; // skip if no default value
        }
        std::vector<std::string> tokens;
        boost::split(tokens, bpKey.keyName, boost::is_any_of("."));
        if (!taskContext.has_stored(tokens.back()))
          defaulted.push_back(core::to_upper(tokens.back()));
        if (!taskContext.contains(tokens.back())) {
          for (size_t i = 0; i < bpKey.default_value.size(); ++i) {
            size_t const ni = (i < bpKey.value_N.size()) ? i : bpKey.value_N.size() - 1;
            taskContext.add<std::string>(tokens.back(), bpKey.default_value[i], std::abs(bpKey.value_N[ni]));
          }
        }
      }
    }

    task.setKeyProvenance(KeyProvenance(baseContext, fileContext, configFile,
                                        std::move(defaulted), skipped));
  }

  return tasksResult;
}

} // namespace mlip
} // namespace tadah
