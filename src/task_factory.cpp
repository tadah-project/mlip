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

TaskFactory::TaskFactory(const MLIPUserConfigParser &parser)
  : parser_(parser)
{}

std::vector<Task> TaskFactory::createTasksFromInput(const ITaskInput &taskInput)
{
  // Definitions:
  // base context: the context built from the key-value data in the taskInput
  // global context: the non task section found in the optional user config file
  // task context: the context built for each TASK section in the user config

  // The order of merging is important:
  // 1. base context has the highest priority
  // 2. task context has the second priority
  // 3. global context has the lowest priority

  // prepare the result vector
  std::vector<Task> tasksResult;

  // Step 0: Validate the key-value data against the blueprint schema.
  const auto &bpMap = tadah::core::BlueprintParser::getKeysMapRef();
  for (const auto &kvPair : taskInput.getKeyValueData()) {
    const std::string &key = kvPair.first;
    // std::cout << "DEBUG: Step 0: Validating key-value pair: " << key << "\n";
    const std::vector<std::string> &values = kvPair.second;
    if (values.empty()) {
      continue; // skip empty values
    }
    bool valid_schema = false;
    try {
      valid_schema = core::CoreUserConfigParser::validateWithBlueprint(key, values, bpMap.at(core::to_upper(key)));
    } catch (const std::out_of_range &e) {
      throw std::runtime_error("TaskFactory::createTasksFromInput - Invalid blueprint key: " + key);
    }
    if (!valid_schema) {
      throw std::runtime_error("TaskFactory::createTasksFromInput - Invalid key-value pair: " + key + " -> " + core::concat(values));
    }
  }
  // std::cout << "DEBUG: Step 0: Key-value data validated against blueprint schema.\n";

  // Step 1: Build a "base" context from the key-value data in the input.
  tadah::core::Context baseContext;
  auto kvmap = taskInput.getKeyValueData();
  for (auto &kv : kvmap) {
    std::string upKey = tadah::core::to_upper(kv.first);
    // if (upKey == "TASK") continue;
    for (auto &val : kv.second) {
      baseContext.add<std::string>(upKey, val);
    }
  }
  // std::cout << "DEBUG: Step 1: Base context built from key-value data.\n";

  // Step 2: Check if a "MyTask.config" key is present in the context
  // TODO Shouldn't this be MyTask.task key?
  bool hasConfig = baseContext.contains("CONFIG");
  bool hasTask = baseContext.contains("TASK");
  // std::cout << "DEBUG: Step 2: Checking for config key: " << configKey << "\n";
  // if (hasConfig) {
    // std::cout << "DEBUG: Step 2: Config key found.\n";
  // } else {
    // std::cout << "DEBUG: Step 2: No config key found.\n";
  // }

  // Step 3: If there is a user config file, parse it to obtain a map of contexts.
  std::unordered_map<std::string, core::Context> configMap;
  if (hasConfig || hasTask) {
    std::string userFile = hasConfig ? baseContext.get<std::string>("CONFIG") : baseContext.get<std::string>("TASK");
    // std::cout << "DEBUG: Step 3: User config file found: " << userConfigFile << "\n";
    configMap = parser_.parse(userFile, /*mergeFlag=*/true);
    // remove CONFIG key from the base context
    hasConfig ? baseContext.remove("CONFIG") : baseContext.remove("TASK");
  }
  // std::cout << "DEBUG: Step 3: User config file parsed. Number of sections: " << configMap.size() << "\n";

  // Step 4: 
  // check taskInput.getTaskName() against blueprint tasks if it is the highest level task.
  // Iterate all possible task names in the blueprint if there is no task of higher level.
  // We will add taskInput.getTaskName() to the tasksResult if it is a valid task.
  // Also if taskInput.getTaskName() is in fact higher level task then we do not
  // expect more tasks in the config file. We should throw in this case?
  const auto &blueprintTasks = tadah::core::BlueprintParser::getTasksMapRef();
  bool taskInputIsValidTask = true;

  if (!hasTask) {
    // TASKS in a --task file are validated by the MLIPUserConfigParser
    for (const auto &kv : blueprintTasks) {
      const std::string &taskName = kv.second.taskName;
      // if map contains taskInput.getTaskName()+"." then it is not a valid task
      // i.e. there exist another task of higher level
      if (taskName.find(taskInput.getTaskName() + ".") == 0) {
        // std::cout << "DEBUG: Step 4: Checking task name: " << taskName << "\n";
        taskInputIsValidTask = false;
        break;
      }
    }
  }
  // std::cout << "DEBUG: Step 4: Task input is " << (taskInputIsValidTask ? "valid" : "invalid") << "\n";

  // Step 5: if taskInputIsValidTask is true than we check configMap
  // Map is either empty or contain a single GLOBAL section.
  // If it contains anything else we throw an error.
  if (taskInputIsValidTask) {
    if (!hasTask && configMap.size() > 1) {
      throw std::runtime_error("TaskFactory::createTasksFromInput - "
                               "taskInput.getTaskName() is a valid task but configMap contains more than one section.");
    }
    if (configMap.size() == 1) {
      auto it = configMap.find("GLOBAL");
      if (it == configMap.end()) {
        throw std::runtime_error("TaskFactory::createTasksFromInput - "
                                 "taskInput.getTaskName() is a valid task but configMap contains no GLOBAL section.");
      }
    }
    // If configMap is empty, we can skip the rest of the code and returns
    // the taskInput.getTaskName() as a single task.
    if (configMap.empty()) {
      tasksResult.emplace_back(Task(taskInput.getTaskName(), baseContext));
      return tasksResult;
    }
    // otherwise me merge the baseContext with the GLOBAL section and return.
    if (hasTask) {
      // remove the GLOBAL section from the configMap
      configMap.erase("GLOBAL");
      // baseContext should be empty by now, we can ignore it.
      // iterate configMap and copy the contexts to the tasksResult
      for (const auto &cm : configMap) {
        const std::string &taskName = cm.first;
        const core::Context &taskCtx = cm.second;
        // std::cout << "DEBUG: Step 5: Adding task by: " << taskInput.getTaskName() << "\n";
        // std::cout << "DEBUG: Step 5: Adding task: " << taskName << "\n";
        // TODO only add the tasks allowed by the subcommand
        tasksResult.emplace_back(Task(taskName, taskCtx));
      }
    } else {
      const core::Context &globalCtx = configMap["GLOBAL"];
      const core::Context mergedCtx = core::Context::merge(globalCtx, baseContext);
      tasksResult.emplace_back(Task(taskInput.getTaskName(), mergedCtx));
    }
    return tasksResult;
  }
  // std::cout << "DEBUG: Step 5: Task input is not a valid task.\n";

  // Step 6. If we got here, it means taskInput.getTaskName() is not a valid task.
  // We simple merge the baseContext with each task section in the configMap except GLOBAL.
  for (const auto &cm : configMap) {
    if (cm.first == "GLOBAL") {
      continue; // skip the GLOBAL section
    }
    std::string taskName = cm.first;
    // check is higher level task (it should be!) throw otherwise!
    for (const auto &bp : blueprintTasks) {
      const std::string &bpTaskName = bp.second.taskName;
      if (bpTaskName.find(taskName+ ".") == 0) {
        throw std::runtime_error("TaskFactory::createTasksFromInput - "
                                 "taskInput.getTaskName() is not a valid task but configMap contains a section for it.");
      }
    }

    const core::Context mergedCtx = core::Context::merge(cm.second, baseContext); 
    tasksResult.emplace_back(Task(taskInput.getTaskName(), mergedCtx));
  }
  // std::cout << "DEBUG: Step 6: Merged contexts for each task section in the configMap.\n";
  if (tasksResult.empty()) {
    // No final-level tasks found, log a message and return an empty list.
    core::Logger::instance().log(core::LogLevel::DEBUG,
                                 "createTasksFromInput: No final-level tasks found. Returning empty list.");
  }
  // Step 7: Return the tasksResult vector
  return tasksResult;
}

} // namespace mlip
} // namespace tadah
