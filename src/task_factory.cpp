#include <tadah/mlip/task_factory.h>
#include <tadah/mlip/mlip_user_config_parser.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/context.h>
#include <tadah/core/logger.h>
#include <tadah/core/utils.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/blueprint_validator.h>
#include <stdexcept>
#include <algorithm>
#include <boost/algorithm/string.hpp>

namespace tadah {
namespace mlip {
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
    try {
      valid_schema = core::CoreUserConfigParser::validateWithBlueprint(key, values, bpMap.at(core::to_upper(key)));
    } catch (const std::out_of_range &e) {
      // suggest closest key match then throw
      const std::string closestKey = core::BlueprintParser::findClosestKey(key, bpMap);
      core::Logger::instance().log(core::LogLevel::ERROR,
          "TaskFactory::createTasksFromInput - Invalid key: " + key +
          ". Closest match: " + closestKey);
      throw std::runtime_error("TaskFactory::createTasksFromInput - Invalid blueprint key: " + key);
    }
    if (!valid_schema) {
      throw std::runtime_error("TaskFactory::createTasksFromInput - Invalid key-value pair: " + key + " -> " + core::concat(values));
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
  std::vector<std::pair<std::string, core::Context>> configVec;
  if (hasConfig || hasTask) {
    std::string userFile = hasConfig ? baseContext.get<std::string>("CONFIG") : baseContext.get<std::string>("TASK");
    // parser returns a vec of <taskName, Context> so we can keep multiple tasks of the same type
    configVec = parser_->parse(userFile, /*mergeFlag=*/true);
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

  // the taskInput.getTaskName() as a single task.
  if (configVec.empty()) {
    tasksResult.emplace_back(Task(taskInput.getTaskName(), baseContext));
  }

  if (hasTask) {
    // remove the GLOBAL section from the configVec for --task input
    if (configVec.begin()->first == "GLOBAL") {
      configVec.erase(configVec.begin());
    }
    // baseContext should be empty by now, we can ignore it.
    // iterate configVec and copy the contexts to the tasksResult
    for (const auto &cv : configVec) {
      const std::string &taskName = cv.first;
      const core::Context &taskCtx = cv.second;
      // TODO only add the tasks allowed by the subcommand
      tasksResult.emplace_back(Task(taskName, taskCtx));
    }
  } else if (hasConfig) {
    const core::Context &globalCtx = configVec.begin()->second;
    const core::Context mergedCtx = core::Context::merge(globalCtx, baseContext);
    tasksResult.emplace_back(Task(taskInput.getTaskName(), mergedCtx));

  } 

  if (tasksResult.empty()) {
    // No final-level tasks found, log a message and return an empty list.
    core::Logger::instance().log(core::LogLevel::DEBUG,
        "createTasksFromInput: No final-level tasks found. Returning empty list.");
  }

  // Add default keys to the context if not provided by the user
  for (auto &task : tasksResult) {
    const std::string &taskName = task.getTaskName();
    core::Context &taskContext = task.getContext();
    const auto bp = blueprintTasks.find(taskName);

    // dealt with options keys
    for (const auto &bpKey : bp->second.blueprintKeys) {
      if (bpKey.default_value.empty()) {
        continue; // skip if no default value
      }
      std::vector<std::string> tokens;
      boost::split(tokens, bpKey.keyName, boost::is_any_of("."));
      if (!taskContext.contains(tokens.back())) {
        size_t const nv = std::max(bpKey.default_value.size(),
            bpKey.value_N.size());
        for (size_t i = 0; i < nv; ++i) {
          taskContext.add<std::string>(tokens.back(), bpKey.default_value[i], std::abs(bpKey.value_N[i]));
        }
      }
    }

    // also deal with groups
    for (const auto &group : bp->second.groups) {
      for (const auto &bpKey : group.blueprintKeys) {
        if (bpKey.default_value.empty()) {
          continue; // skip if no default value
        }
        std::vector<std::string> tokens;
        boost::split(tokens, bpKey.keyName, boost::is_any_of("."));
        if (!taskContext.contains(tokens.back())) {
          size_t const nv = std::max(bpKey.default_value.size(),
              bpKey.value_N.size());
          for (size_t i = 0; i < nv; ++i) {
            taskContext.add<std::string>(tokens.back(), bpKey.default_value[i], std::abs(bpKey.value_N[i]));
          }
        }
      }
    }
  }

  return tasksResult;
}

} // namespace mlip
} // namespace tadah
