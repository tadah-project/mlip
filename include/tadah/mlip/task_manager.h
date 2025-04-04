#ifndef TADAH_MLIP_TASK_MANAGER_H
#define TADAH_MLIP_TASK_MANAGER_H

#include <vector>
#include <tadah/mlip/task.h>

namespace tadah {
namespace mlip {

class TaskManager {
public:
  TaskManager() = default;
  ~TaskManager() = default;

  /**
   * @brief Stores a task in the internal list
   */
  void store(const Task &task);

  /**
   * @brief Stores a task in the internal list
   */
  void store(const std::vector<Task> &tasks);

  /**
   * @brief Return all tasks
   */
  const std::vector<Task>& getTasks() const;

  /**
   * @brief Clear or remove tasks
   */
  void clear();

  // Additional methods: removeTaskById, findTaskByName, runAll, etc.

private:
  std::vector<Task> tasks_;
};

} // namespace mlip
} // namespace tadah

#endif // TADAH_MLIP_TASK_MANAGER_H
