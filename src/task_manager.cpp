#include <tadah/mlip/task_manager.h>

namespace tadah {
namespace mlip {

void TaskManager::store(const Task &task) {
  tasks_.push_back(task);
}

void TaskManager::store(const std::vector<Task> &tasks) {
  for (const auto &task : tasks) {
    tasks_.push_back(task);
  }
}

std::vector<Task>& TaskManager::getTasks() {
  return tasks_;
}

void TaskManager::clear() {
  tasks_.clear();
}

} // namespace mlip
} // namespace tadah
