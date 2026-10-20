#include <tadah/mlip/task_factory.h>
#include <tadah/mlip/retired_options.h>
#include <tadah/mlip/mlip_user_config_parser.h>
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
 * line and a task file's parser pass only blueprint names) is a
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

  std::string msg =
      "TaskFactory::createTasksFromInput - unknown task '" + name + "'";
  const auto same = std::find_if(names.begin(), names.end(),
      [&](const std::string &n) { return boost::iequals(n, name); });
  if (same != names.end())
    msg += ": task names are case-sensitive, did you mean '" + *same + "'?";
  else
    msg += ".";
  msg += " The tasks:";
  for (std::size_t i = 0; i < names.size(); ++i)
    msg += (i == 0 ? " " : ", ") + names[i];
  throw std::invalid_argument(msg + ".");
}

}  // namespace

TaskFactory::TaskFactory(std::shared_ptr<MLIPUserConfigParser> parser)
  : parser_(std::move(parser))
{}
std::vector<Task> TaskFactory::createTasksFromInput(const ITaskInput &taskInput)
{
  // Definitions:
  // base context: the context built from the key-value data in the taskInput
  // global context: the non task section found in the task file
  // task context: the context built from a task file
  //
  // The CONFIG key is always expanded and its content is injected into the context.
  // Currently it will override the base context keys 
  //
  // taskInput is either fully specified task from CLI, for example:
  // tadah train -c config.train  <- results in a single Task
  // or a task file, for example:
  // tadah --task tasks.tadah <- results in one or more Tasks

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
      // suggest closest key match then throw
      const std::string closestKey = core::BlueprintParser::findClosestKey(key, bpMap);
      core::Logger::instance().log(core::LogLevel::ERROR,
          "TaskFactory::createTasksFromInput - Invalid key: " + key +
          ". Closest match: " + closestKey);
      throw std::runtime_error("TaskFactory::createTasksFromInput - Invalid blueprint key: " + key);
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
      throw std::runtime_error(
          "TaskFactory::createTasksFromInput - Invalid key-value pair: " + key +
          " -> " + text +
          (seps.empty() ? std::string()
                        : " ('" + text + "' separates values with " + seps +
                              ", which is not a value separator: give them "
                              "as separate words, " + respaced + ")"));
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

  // Step 2: Parse files given in CLI. The parser deals gracefully with nested CONFIG keys.
  // There is no option for nested TASK keys.
  // The global section of task files is merged with tasks section.
  // The task section has priority over the global section.
  bool hasConfig = baseContext.contains("CONFIG");  // --config given in CLI
  bool hasTask = baseContext.contains("TASK");  // --task given in CLI
  // Without a --task file the input names the task: a blueprint task, or
  // refused here, before any configuration file is read.
  if (!hasTask) (void)blueprint_task(taskInput.getTaskName());
  std::vector<std::pair<std::string, core::Context>> configVec;
  // The lines of each section of configVec that its parse did not read.
  std::vector<std::vector<core::SkippedKeyLine>> skippedVec;
  std::string taskFile;   // the --task file, for its error
  std::string baseContextConfigFile;  // the --config file, for KeyProvenance
  if (hasConfig || hasTask) {
    std::string userFile = hasConfig ? baseContext.get<std::string>("CONFIG") : baseContext.get<std::string>("TASK");
    if (hasTask) taskFile = userFile;
    else baseContextConfigFile = userFile;
    // parser returns a vec of <taskName, Context> so we can keep multiple
    // tasks of the same type. A potential's lines of the keys of its
    // training record but DBFILE -- those that prepared the training data
    // of its own fit and the settings of the run that wrote it
    // (VALIDATION, HPOTARGET, CHECK, VERBOSE, NUMERIC, OUTFILE, AUDIT,
    // AUDIT_SAMPLE, HEALTH_LOG) -- are not read, wherever the file and its
    // CONFIG includes put it: every other line, and the command line, is the
    // run's.
    configVec = parser_->parse(userFile, /*mergeFlag=*/true,
                               potential_line_not_read, skippedVec);
    hasConfig ? baseContext.remove("CONFIG") : baseContext.remove("TASK");
  }

  // Step 3:  For non --task input we should only allow fully specified command to proceed
  const auto &blueprintTasks = tadah::core::BlueprintParser::getTasksMapRef();
  if (!hasTask) {
    for (const auto &kv : blueprintTasks) {
      const std::string &taskName = kv.second.taskName;
      // if map contains taskInput.getTaskName()+"." then it is not a valid task
      // i.e. there exist another task of higher level
      if (taskName.find(taskInput.getTaskName() + ".") == 0) {
        throw std::runtime_error("TaskFactory::createTasksFromInput - "
            "taskInput.getTaskName() is not a valid task: " + taskInput.getTaskName());
      }
    }
  }

  // Where each task's keys come from (KeyProvenance): the command-line
  // options, the configuration or task file, and (below) the defaults.
  struct Sources {
    core::Context command_line;
    core::Context file;
    std::string file_path;
    KeyOrigin::Kind file_kind;
    std::vector<core::SkippedKeyLine> skipped;  // the file's lines not read
  };
  std::vector<Sources> sources;

  // the taskInput.getTaskName() as a single task. A --task file's tasks are
  // its TASK sections (below).
  if (configVec.empty() && !hasTask) {
    tasksResult.emplace_back(Task(taskInput.getTaskName(), baseContext));
    sources.push_back({baseContext, core::Context(), std::string(),
                       KeyOrigin::Kind::ConfigFile, {}});
  }

  if (hasTask) {
    // remove the GLOBAL section from the configVec for --task input
    if (!configVec.empty() && configVec.front().first == "GLOBAL") {
      configVec.erase(configVec.begin());
      skippedVec.erase(skippedVec.begin());
    }
    // A task file without a TASK section has nothing to run: say so rather
    // than run nothing (or read the first element of an empty list).
    if (configVec.empty()) {
      throw std::runtime_error(
          "TaskFactory::createTasksFromInput - the task file '" + taskFile +
          "' has no TASK section (a line 'TASK <name>', with or without key "
          "lines after it)");
    }
    // baseContext should be empty by now, we can ignore it.
    // iterate configVec and copy the contexts to the tasksResult
    for (std::size_t i = 0; i < configVec.size(); ++i) {
      const std::string &taskName = configVec[i].first;
      const core::Context &taskCtx = configVec[i].second;
      // TODO only add the tasks allowed by the subcommand
      tasksResult.emplace_back(Task(taskName, taskCtx));
      sources.push_back({core::Context(), taskCtx, taskFile,
                         KeyOrigin::Kind::TaskFile, skippedVec.at(i)});
    }
  } else if (hasConfig && !configVec.empty()) {
    const core::Context &globalCtx = configVec.front().second;
    const core::Context mergedCtx = core::Context::merge(globalCtx, baseContext);
    tasksResult.emplace_back(Task(taskInput.getTaskName(), mergedCtx));
    sources.push_back({baseContext, globalCtx,
                       baseContextConfigFile,
                       KeyOrigin::Kind::ConfigFile, skippedVec.front()});

  } 

  if (tasksResult.empty()) {
    // No final-level tasks found, log a message and return an empty list.
    core::Logger::instance().log(core::LogLevel::DEBUG,
        "createTasksFromInput: No final-level tasks found. Returning empty list.");
  }

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

    const Sources &src = sources.at(t);
    task.setKeyProvenance(KeyProvenance(src.command_line, src.file,
                                        src.file_path, src.file_kind,
                                        std::move(defaulted), src.skipped));
  }

  return tasksResult;
}

} // namespace mlip
} // namespace tadah
