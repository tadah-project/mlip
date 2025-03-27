/*#include <tadah/mlip/task_manager.h>*/
/*#include <tadah/core/logger.h>*/
/*#include <fstream>*/
/*#include <sstream>*/
/*#include <stdexcept>*/
/*#include <algorithm>*/
/**/
/*namespace tadah {*/
/*namespace mlip {*/
/**/
/*using tadah::core::Context;*/
/*using tadah::core::ConfigValue;*/
/**/
/**
 * @brief Constructs the TaskManager with a global Context.
 */
/*TaskManager::TaskManager(const Context &globalContext)*/
/*  : globalContext_(globalContext) {}*/
/**/
/**
 * @brief Parses the user configuration and creates tasks.
 *
 * The user_config file is assumed to have global settings first, followed by task blocks.
 * Each task block starts with the keyword "task" followed by the command.
 * The rest of the lines until the next "task" are treated as key/value pairs.
 */
/*void TaskManager::parseUserConfig(const std::string &filename) {*/
/*  // Use the internal function to obtain a vector of tasks.*/
/*  std::vector<Task> parsedTasks = parseUserConfigInternal(filename);*/
/*  // Convert tasks to shared pointers and add them.*/
/*  for (const auto &t : parsedTasks) {*/
/*    tasks_.push_back(std::make_shared<Task>(t));*/
/*  }*/
/*}*/
/**/
/**
 * @brief Adds a Task.
 */
/*void TaskManager::addTask(const Task &task) {*/
/*  tasks_.push_back(std::make_shared<Task>(task));*/
/*}*/

/**
 * @brief Modifies a Task identified by its ID.
 */
/*void TaskManager::modifyTask(const std::string &id, const Task &task) {*/
/*  auto it = std::find_if(tasks_.begin(), tasks_.end(),*/
/*    [&id](const std::shared_ptr<Task>& t) {*/
/*      return t->getID() == id;*/
/*    });*/
/*  if (it == tasks_.end()) {*/
/*    throw std::runtime_error("Task ID not found: " + id);*/
/*  }*/
/*  // Replace the task with the new version.*/
/*  *it = std::make_shared<Task>(task);*/
/*}*/
/**/
/**
 * @brief Removes a Task by its ID.
 */
/*void TaskManager::removeTask(const std::string &id) {*/
/*  auto it = std::remove_if(tasks_.begin(), tasks_.end(),*/
/*    [&id](const std::shared_ptr<Task>& t) {*/
/*      return t->getID() == id;*/
/*    });*/
/*  if (it == tasks_.end()) {*/
/*    throw std::runtime_error("Task ID not found: " + id);*/
/*  }*/
/*  tasks_.erase(it, tasks_.end());*/
/*}*/
/**/
/**
 * @brief Retrieves a Task by its ID.
 */
/*std::shared_ptr<Task> TaskManager::getTaskByID(const std::string &id) const {*/
/*  auto it = std::find_if(tasks_.begin(), tasks_.end(),*/
/*    [&id](const std::shared_ptr<Task>& t) {*/
/*      return t->getID() == id;*/
/*    });*/
/*  if (it == tasks_.end()) {*/
/*    throw std::runtime_error("Task ID not found: " + id);*/
/*  }*/
/*  return *it;*/
/*}*/
/**/
/**
 * @brief Retrieves Tasks by type.
 */
/*std::vector<std::shared_ptr<Task>> TaskManager::getTasksByType(const std::string &taskType) const {*/
/*  std::vector<std::shared_ptr<Task>> result;*/
/*  for (const auto &t : tasks_) {*/
/*    if (t->getTaskType() == taskType)*/
/*      result.push_back(t);*/
/*  }*/
/*  return result;*/
/*}*/
/**/
/**
 * @brief Returns all tasks.
 */
/*std::vector<std::shared_ptr<Task>> TaskManager::getAllTasks() const {*/
/*  return tasks_;*/
/*}*/
/**/
/**
 * @brief Internal parser that reads a user configuration file and returns tasks.
 *
 * A simplistic parser is implemented for demonstration.
 */
/*std::vector<Task> TaskManager::parseUserConfigInternal(const std::string &filename) {*/
/*  std::vector<Task> parsedTasks;*/
/*  std::ifstream infile(filename);*/
/*  if (!infile.is_open())*/
/*    throw std::runtime_error("Cannot open user config file: " + filename);*/
/**/
/*  std::string line;*/
/*  std::string currentCommand;*/
/*  std::stringstream taskStream;*/
/*  bool inTaskBlock = false;*/
/**/
/*  while (std::getline(infile, line)) {*/
/*    // Ignore blank lines and full-line comments*/
/*    if (line.empty() || line[0] == '#') continue;*/
/**/
/*    // Check for task block marker ("task" keyword).*/
/*    if (line.substr(0, 4) == "task") {*/
/*      // If already in a task block, finalize the previous task.*/
/*      if (inTaskBlock) {*/
/*        Task t(currentCommand, globalContext_);*/
/*        std::string optionLine;*/
/*        while (std::getline(taskStream, optionLine)) {*/
/*          if(optionLine.empty() || optionLine[0]=='#') continue;*/
/*          auto pos = optionLine.find(' ');*/
/*          if (pos != std::string::npos) {*/
/*            std::string key = optionLine.substr(0, pos);*/
/*            std::string value = optionLine.substr(pos+1);*/
/*            // For simplicity, store as string-type ConfigValue.*/
/*            t.setOption(key, tadah::core::ConfigValue(value));*/
/*          }*/
/*        }*/
/*        parsedTasks.push_back(t);*/
/*        taskStream.str("");*/
/*        taskStream.clear();*/
/*      }*/
/*      inTaskBlock = true;*/
/*      // Parse the command (everything after "task ")*/
/*      currentCommand = line.substr(5);*/
/*    } else if (inTaskBlock) {*/
/*      // Accumulate key/value lines for the current task.*/
/*      taskStream << line << "\n";*/
/*    }*/
/*  }*/
/*  // Finalize last task if any.*/
/*  if (inTaskBlock) {*/
/*    Task t(currentCommand, globalContext_);*/
/*    std::string optionLine;*/
/*    while (std::getline(taskStream, optionLine)) {*/
/*      if(optionLine.empty() || optionLine[0]=='#') continue;*/
/*      auto pos = optionLine.find(' ');*/
/*      if (pos != std::string::npos) {*/
/*        std::string key = optionLine.substr(0, pos);*/
/*        std::string value = optionLine.substr(pos+1);*/
/*        t.setOption(key, tadah::core::ConfigValue(value));*/
/*      }*/
/*    }*/
/*    parsedTasks.push_back(t);*/
/*  }*/
/*  return parsedTasks;*/
/*}*/
/**/
/*} // namespace mlip*/
/*} // namespace tadah*/
/**/
