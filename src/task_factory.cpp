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
  // base_task context: the context built for each TASK section in the user config
  // task context: merged base with optional if present global and task sections
  // The order of merging is important:
  // 1. base context has the highest priority
  // 2. base_task context has the second priority
  // 3. global context has the lowest priority

  // The final task context specifies the exact command to be run
  // and all the required parameters.
  //
  // MLIPUserConfigParser
  //
  // TODO cases we want to consider:
  // 1. If taskInput contains only a single config file:
  //    - It is possible that the user config file contains no further task sections. In this case the taskInput.getTaskName() is the only and final task. In this case MLIPUserConfigParser returns a single context.
  //    - If user config file contains multiple task sections, we need to create a task for each section. In this case getTaskName() is not used at all.  How do we get task names in this case?  In this case MLIPUserConfigParser returns a vector of contexts of size>=2. The first context could be either a task or a global section.

  // validate the key-value data
  const auto &bpMap = tadah::core::BlueprintParser::getKeysMapRef();
  for (const auto &kvPair : taskInput.getKeyValueData()) {
    const std::string &key = kvPair.first;
    const std::vector<std::string> &values = kvPair.second;
    if (values.empty()) {
      continue; // skip empty values
    }
    bool valid_schema = core::CoreUserConfigParser::validateWithBlueprint(key, values, bpMap.at(core::to_upper(key)));
    if (!valid_schema) {
      throw std::runtime_error("TaskFactory::createTasksFromInput - Invalid key-value pair: " + key + " -> " + core::concat(values));
    }
  }

  // Prepare a result vector
  // Depending on the task input, we may have multiple tasks to return.
  std::vector<Task> result;

  // Step 1: Build a "base" context from the key-value data in the input.
  tadah::core::Context baseContext;
  auto kvmap = taskInput.getKeyValueData();
  for (auto &kv : kvmap) {
    std::string upKey = tadah::core::to_upper(kv.first);
    for (auto &val : kv.second) {
      baseContext.add<std::string>(upKey, val);
    }
  }

  // Step 2: Check if a "MyTask.config" key is present in the context
  std::string configKey = tadah::core::to_upper(taskInput.getTaskName() + ".config");
  bool hasConfig = baseContext.contains(configKey);

  // TODO modify from here
  std::vector<Task> tasksFromConfigs;
  if (hasConfig) {
    // The user config file path is stored as the first string for that key.
    std::string userConfigFile = baseContext.get<std::string>(configKey);

    // Parse config sections via MLIPUserConfigParser
    std::vector<tadah::core::Context> configSections;
    try {
      configSections = parser_.parse(userConfigFile, true /*merge*/);
    } catch(const std::exception &e) {
      throw std::runtime_error(std::string("TaskFactory::createTasksFromInput - ")
                            + "Error parsing config file: " + e.what());
    }

    // If configSections is empty, possibly no valid sections found
    if (configSections.empty()) {
      throw std::runtime_error("TaskFactory::createTasksFromInput - No valid sections in the user config file.");
    }

    // Deal with global section if it exists
    // followed by the task-specific sections.

    // For each section, merge base & override with CLI again (if needed)
    // TODO this is wrong order
    for (auto &sectionCtx : configSections) {
      // Overwrite with baseContext's key-value data
      for (const auto &kvPair : baseContext.getMapRef()) {
        const std::string &key = kvPair.first;
        // Remove any existing references in the config section
        sectionCtx.remove(key);
        // Re-add from base
        for (auto &val : kvPair.second) {
          sectionCtx.add<std::string>(key, val);
        }
      }
      // Create a Task out of this merged context
      Task t(taskInput.getTaskName(), sectionCtx);
      tasksFromConfigs.push_back(t);
    }
  }

  // Step 3: If there was no config, or if config yielded no tasks, create a single Task.
  if (tasksFromConfigs.empty()) {
    Task singleTask(taskInput.getTaskName(), baseContext);
    result.push_back(singleTask);
  } else {
    result = std::move(tasksFromConfigs);
  }

  return result;
}

} // namespace mlip
} // namespace tadah
