#ifndef TADAH_MLIP_TASK_MANAGER_H
#define TADAH_MLIP_TASK_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <tadah/mlip/task.h>
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/context.h>

namespace tadah {
namespace mlip {

/**
 * @brief TaskManager creates and manages tasks based on the user configuration.
 *
 * TaskManager uses a dedicated parser to read the user_config file
 * and produces Task objects in the order encountered. It offers functions
 * to add, modify, remove, and retrieve tasks using shared pointers.
 */
class TaskManager {
public:
  /**
   * @brief Constructs a TaskManager.
   *
   * The TaskManager uses a provided global Context.
   *
   * @param globalContext The global configuration context.
   */
  explicit TaskManager(const tadah::core::Context &globalContext);

  /**
   * @brief Parses a user configuration file and creates tasks.
   *
   * This function reads the user_config file,
   * validates it against the blueprint, and spawns Task objects.
   *
   * @param filename The user configuration filename.
   */
  void parseUserConfig(const std::string &filename);

  /**
   * @brief Adds a Task to the TaskManager.
   *
   * @param task The Task to add.
   */
  void addTask(const Task &task);

  /**
   * @brief Modifies an existing task identified by its ID.
   *
   * @param id The task ID.
   * @param task A mutable reference to a Task with modifications.
   * @throws std::runtime_error if the task ID is not found.
   */
  void modifyTask(const std::string &id, const Task &task);

  /**
   * @brief Removes a task by its ID.
   *
   * @param id The task ID.
   * @throws std::runtime_error if the task ID is not found.
   */
  void removeTask(const std::string &id);

  /**
   * @brief Retrieves a task by its ID.
   *
   * @param id The task ID.
   * @return A shared pointer to the Task.
   * @throws std::runtime_error if the task ID is not found.
   */
  std::shared_ptr<Task> getTaskByID(const std::string &id) const;

  /**
   * @brief Retrieves tasks by their type.
   *
   * @param taskType The task type (as extracted from the command).
   * @return A vector of shared pointers to tasks of the given type.
   */
  std::vector<std::shared_ptr<Task>> getTasksByType(const std::string &taskType) const;

  /**
   * @brief Returns all the tasks.
   *
   * @return A vector of shared pointers to all tasks.
   */
  std::vector<std::shared_ptr<Task>> getAllTasks() const;

private:
  tadah::core::Context globalContext_;
  std::vector<std::shared_ptr<Task>> tasks_;

  /**
   * @brief Internal function to parse a user_config file.
   *
   * Assumes that the file is in a simple key/value format with task blocks.
   *
   * @param filename The user configuration file.
   * @return A vector of Task objects in order of appearance.
   */
  std::vector<Task> parseUserConfigInternal(const std::string &filename);
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_TASK_MANAGER_H

